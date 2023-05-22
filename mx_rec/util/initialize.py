# coding: UTF-8
import json
import logging
import os
import psutil

import mxrec_pybind
import mx_rec.util.constants
from mx_rec.util.constants import LOCAL_RANK_SIZE, MAX_DEVICE_NUM_LOCAL_MACHINE, DEFAULT_DEVICE_NUM_LOCAL_MACHINE, \
    ASCEND_GLOBAL_HASHTABLE_COLLECTION
from mx_rec.util.ops import import_host_pipeline_ops


class ConfigInitializer:
    _single_instance = None
    customized_ops = None
    host_pipeline_ops = import_host_pipeline_ops()

    def __init__(self, use_mpi, **kwargs):
        self._use_mpi = use_mpi
        self._ascend_global_hashtable_collection = ASCEND_GLOBAL_HASHTABLE_COLLECTION
        self._comm = None
        self._asc_manager = None
        self._mpi = None
        self._is_frozen = False
        self._train_interval = None
        self._eval_steps = None
        self._prefetch_batch_number = None
        self._if_load = None
        self._table_instance_dict = dict()
        self._dangling_table = []
        self._name_to_var_dict = dict()
        self._table_name_set = set()
        self._table_name_to_feature_spec = dict()
        self._feature_spec_dict = dict()
        self._training_mode_channel_dict = dict()
        self._rank_to_device_dict = dict()
        self._initializer_dict = {}
        self._optimizer_instance = None

        if self._use_mpi:
            logging.debug(f"Using mpi to launch task.")
            from mpi4py import MPI
            self._mpi = MPI
            self._comm = MPI.COMM_WORLD
            self._rank_id = self._comm.Get_rank()
            self._rank_size = self._comm.Get_size()
        else:
            self._rank_id = kwargs.get("rank_id")
            self._rank_size = kwargs.get("rank_size")

        if os.getenv("RANK_TABLE_FILE"):
            self.parse_hccl_json()
        else:
            self.set_device_dict()
        self.check_parameters()
        self.train_interval = kwargs.get("train_interval", -1)
        self.eval_steps = kwargs.get("eval_steps", -1)
        self.prefetch_batch_number = kwargs.get("prefetch_batch_number", 1)
        self.if_load = kwargs.get("if_load", False)
        if_dynamic = kwargs.get("use_dynamic", 1)

        self.use_static = 0 if if_dynamic == 1 else 1
        self.use_hot = kwargs.get("use_hot", True)
        self.use_dynamic_expansion = kwargs.get("use_dynamic_expansion", False)
        if kwargs.get("bind_cpu", True):
            bind_cpu(self._rank_id, self._rank_size)

    def __del__(self):
        self.terminate()

    def terminate(self):
        if self._asc_manager is not None:
            self.del_asc_manager()

        if self._mpi:
            self._mpi.Finalize()
            logging.debug("MPI has been destroyed.")

    def insert_feature_spec(self, feature, is_training):
        self._feature_spec_dict[feature.name] = feature
        if feature.table_name not in self._table_name_to_feature_spec:
            self._table_name_to_feature_spec[feature.table_name] = {True: [], False: []}
        self._table_name_to_feature_spec[feature.table_name][is_training].append(feature)

    def get_feature_spec(self, key):
        return self._feature_spec_dict.get(key)

    @property
    def feature_spec_dict(self):
        return self._feature_spec_dict

    @property
    def table_name_set(self):
        return self._table_name_set

    @property
    def table_name_to_feature_spec(self):
        return self._table_name_to_feature_spec

    def parse_hccl_json(self):
        rank_table_path = os.getenv("RANK_TABLE_FILE")
        if not os.path.exists(rank_table_path):
            raise FileExistsError(f"Target_hccl_json_dir {rank_table_path} does not exist when reading.")
        with open(rank_table_path, "r", encoding="utf-8") as file:
            table_hccl = json.load(file)
            if "server_list" not in table_hccl:
                raise AttributeError(f"Lack of attribute server_list.")
            if not table_hccl["server_list"]:
                raise ValueError(f"Server_list is empty.")
            if "device" not in table_hccl["server_list"][0]:
                raise AttributeError(f"Lack of attribute device.")

        for server_list in table_hccl.get("server_list"):
            devices = server_list.get("device")
            if devices is None:
                raise ValueError("device is empty")
            for device in devices:
                if "rank_id" not in device or not device["rank_id"].isdigit():
                    raise ValueError(f"hccl_json rank_id wrong.")
                rank_id = int(device["rank_id"])
                if "device_id" not in device or not device["device_id"].isdigit():
                    raise ValueError(f"hccl_json device_id wrong.")
                device_id = mxrec_pybind.get_logic_id(int(device["device_id"]))
                if device_id > 16:
                    raise ValueError(f"get logic id from physic id fail.")
                self._rank_to_device_dict[rank_id] = device_id

    def set_device_dict(self):
        ascend_visible_devices = os.getenv("ASCEND_VISIBLE_DEVICES")
        if not ascend_visible_devices:
            raise ValueError("env variable ascend_visible_devices is null.")
        if "-" in ascend_visible_devices:
            rank_start = int(ascend_visible_devices.strip().split("-")[0])
            device_list = [i for i in range(rank_start, int(ascend_visible_devices.strip().split("-")[-1]))]
        elif "," in ascend_visible_devices:
            device_list = list(map(int, ascend_visible_devices.strip().split(",")))
        elif ascend_visible_devices in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            device_list = [int(ascend_visible_devices.strip())]
        else:
            raise ValueError("invalid env variable ascend_visible_devices.")
        rank_size = int(os.getenv("CM_WORKER_SIZE"))
        self._rank_to_device_dict[0] = int(os.getenv("CM_CHIEF_DEVICE"))
        device_list.pop(int(os.getenv("CM_CHIEF_DEVICE")))
        if rank_size:
            local_rank_size = rank_size if rank_size < 8 else 8
            for device_index in range(local_rank_size - 1):
                device_id = mxrec_pybind.get_logic_id(int(device_list[device_index]))
                if device_id > 16:
                    raise ValueError(f"get logic id from physic id fail.")
                self._rank_to_device_dict[device_index + 1] = device_id
        else:
            raise ValueError("get CM_WORKER_SIZE failed.")

    def insert_training_mode_channel_id(self, is_training):
        if is_training not in self._training_mode_channel_dict:
            # mx_rec has 2 channel for data input. it would bind channel_id to training mode recorded in dict.
            self._training_mode_channel_dict[is_training] = len(self._training_mode_channel_dict)

    def get_training_mode_channel_id(self, is_training):
        return self._training_mode_channel_dict.get(is_training)

    def insert_dangling_table(self, name):
        if name in  self._dangling_table:
            return
        self._dangling_table.append(name)

    @property
    def dangling_table(self):
        return self._dangling_table

    def insert_table_instance(self, name, key, instance):
        if key in self._table_instance_dict:
            raise KeyError(f"Given key {key} has been used.")

        if name in self._table_name_set:
            raise ValueError(f"Duplicated hashtable name '{name}' was used.")

        logging.debug(f"Record one hash table, with name: {name}, key: {key}.")
        self._table_name_set.add(name)
        if name not in self._table_name_to_feature_spec:
            self._table_name_to_feature_spec[name] = {True: [], False: []}
        self._name_to_var_dict[name] = key
        self._table_instance_dict[key] = instance

    def get_table_instance(self, key):
        if key not in self._table_instance_dict:
            raise KeyError(f"Given key does not exist.")

        return self._table_instance_dict.get(key)

    def get_table_instance_by_name(self, table_name):
        if table_name not in self._name_to_var_dict:
            raise KeyError(f"Given table name does not exist.")

        key = self._name_to_var_dict.get(table_name)
        return self._table_instance_dict.get(key)

    @property
    def table_instance_dict(self):
        return self._table_instance_dict

    def insert_optimizer(self, optimizer):
        self._optimizer_instance = optimizer

    @property
    def optimizer_instance(self):
        return self._optimizer_instance

    def check_parameters(self):
        if not isinstance(self._use_mpi, bool):
            raise ValueError(f"Arg use_mpi must be a boolean.")

        if not isinstance(self.rank_id, int) or not isinstance(self.rank_size, int):
            raise ValueError(f"Args rank_size and rank_id must be integers. {self.rank_id} {self.rank_size}")

        if self.rank_id < 0:
            raise ValueError(f"Arg rank_id must be larger than 0, which is {self.rank_id} now.")

        if self.rank_size < 1:
            raise ValueError(f"Arg rank_size must be larger than 1, which is {self.rank_size} now.")

        if self.rank_id >= self.rank_size:
            raise ValueError(f"Rank_id must be within the range from 0 to rank_size.")

    def freeze(self):
        self._is_frozen = True

    def unfreeze(self):
        self._is_frozen = False

    @property
    def is_frozen(self):
        return self._is_frozen

    @property
    def name_to_var_dict(self):
        return self._name_to_var_dict

    def set_asc_manager(self, manager):
        from mxrec_pybind import HybridMgmt
        if not isinstance(manager, HybridMgmt):
            raise ValueError(f"Given manager must be the instance of {HybridMgmt}, which is {type(manager)} "
                             f"type currently.")
        self._asc_manager = manager
        self.freeze()

    def get_asc_manager(self):
        return self._asc_manager

    def del_asc_manager(self):
        self.delete_initializers()
        self._asc_manager.destroy()
        self._asc_manager = None
        self.unfreeze()
        logging.debug("ASC manager has been destroyed.")

    @property
    def use_mpi(self):
        return self._use_mpi

    @property
    def rank_size(self):
        return self._rank_size

    @property
    def rank_id(self):
        return self._rank_id

    @property
    def device_id(self):
        if self._rank_id not in self._rank_to_device_dict:
            raise KeyError(f"rank id not in rank_to_device_dict. {self._rank_id} {self._rank_to_device_dict}")
        return self._rank_to_device_dict[self._rank_id]

    @staticmethod
    def get_instance():
        if ConfigInitializer._single_instance is None:
            raise EnvironmentError("Please init the environment for mx_rec at first.")

        return ConfigInitializer._single_instance

    @staticmethod
    def set_instance(use_mpi, **kwargs):
        if ConfigInitializer._single_instance is not None:
            raise EnvironmentError("ConfigInitializer has been initialized once, twice initialization was forbidden.")

        ConfigInitializer._single_instance = ConfigInitializer(use_mpi, **kwargs)

    @property
    def train_interval(self):
        return self._train_interval

    @property
    def eval_steps(self):
        return self._eval_steps

    @train_interval.setter
    def train_interval(self, interval):
        check_step(interval)
        self._train_interval = interval

    @eval_steps.setter
    def eval_steps(self, steps):
        check_step(steps)
        self._eval_steps = steps

    @property
    def prefetch_batch_number(self):
        return self._prefetch_batch_number

    @prefetch_batch_number.setter
    def prefetch_batch_number(self, number):
        check_step(number, 1)
        self._prefetch_batch_number = number

    @property
    def if_load(self):
        return self._if_load

    @if_load.setter
    def if_load(self, flag):
        if not isinstance(flag, bool):
            raise TypeError(f"Flag if load should be a boolean.")

        self._if_load = flag

    @property
    def ascend_global_hashtable_collection(self):
        return self._ascend_global_hashtable_collection

    @ascend_global_hashtable_collection.setter
    def ascend_global_hashtable_collection(self, name):
        if not isinstance(name, str):
            raise TypeError(f"collection name '{name}' must be a string.")
        self._ascend_global_hashtable_collection = name

    def get_initializer(self, is_training):
        return self._initializer_dict.get(is_training)

    def set_initializer(self, is_training, initializer):
        if not isinstance(is_training, bool):
            raise ValueError(f"Given key must be a boolean, but got {is_training}.")

        self._initializer_dict[is_training] = initializer

    def delete_initializers(self):
        self._initializer_dict = {}


def set_ascend_global_hashtable_collection(name=ASCEND_GLOBAL_HASHTABLE_COLLECTION):
    ConfigInitializer.get_instance().ascend_global_hashtable_collection = name


def get_ascend_global_hashtable_collection():
    return ConfigInitializer.get_instance().ascend_global_hashtable_collection


def check_step(param, min_value=-1):
    if not isinstance(param, int):
        raise TypeError("Given param must be an integer.")

    if param < min_value:
        raise ValueError(f"Valid value range is larger than or equals to {min_value}.")

    if param == 0:
        raise ValueError("Arg train_interval or eval_steps cannot equal to 0.")


def init(use_mpi, **kwargs):
    ConfigInitializer.set_instance(use_mpi, **kwargs)
    set_ascend_env()


def is_mpi_in_use():
    return ConfigInitializer.get_instance().use_mpi


def get_rank_size():
    return ConfigInitializer.get_instance().rank_size


def get_rank_id():
    return ConfigInitializer.get_instance().rank_id


def get_device_id():
    return ConfigInitializer.get_instance().device_id


def set_asc_manager(manager):
    ConfigInitializer.get_instance().set_asc_manager(manager)


def trigger_evict():
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")

    ConfigInitializer.get_instance().get_asc_manager().evict()
    logging.debug("Feature evict is triggered by ops.")


def clear_channel(is_train_channel=False):
    if not isinstance(is_train_channel, bool):
        raise ValueError("Arg is_train_channel should be a boolean.")
    channel_id = get_training_mode_channel_id(is_train_channel)
    logging.info(f"clear channel: {channel_id}")

    return ConfigInitializer.get_instance().host_pipeline_ops.clear_channel(channel_id)


def is_asc_manager_initialized():
    return ConfigInitializer.get_instance().get_asc_manager() is not None


def save_host_data(root_dir):
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")

    ConfigInitializer.get_instance().get_asc_manager().save(root_dir)
    logging.debug("Data from host pipeline has been saved.")


def restore_host_data(root_dir):
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")

    if not ConfigInitializer.get_instance().get_asc_manager().load(root_dir):
        terminate_config_initializer()
        raise TypeError("Asc load data does not match usr setups, \
        please re-consider if you want to restore from this dir")
    logging.debug("Data from host pipeline has been restored.")


def destroy_asc_manager():
    initializer = ConfigInitializer.get_instance()
    if initializer.get_asc_manager() is not None:
        logging.debug("start destroy asc manager...")
        initializer.del_asc_manager()
    else:
        logging.warning("ASC manager does not exist, please check your code.")


def is_asc_frozen():
    return ConfigInitializer.get_instance().is_frozen


def export_table_name_set():
    return ConfigInitializer.get_instance().table_name_set


def get_host_pipeline_ops():
    return ConfigInitializer.host_pipeline_ops


def get_customized_ops():
    return ConfigInitializer.customized_ops


def get_train_interval():
    return ConfigInitializer.get_instance().train_interval


def get_eval_steps():
    return ConfigInitializer.get_instance().eval_steps


def set_train_interval(interval):
    ConfigInitializer.get_instance().train_interval = interval


def set_eval_steps(steps):
    ConfigInitializer.get_instance().eval_steps = steps


def get_prefetch_batch_number():
    return ConfigInitializer.get_instance().prefetch_batch_number


def set_prefetch_batch_number(number):
    ConfigInitializer.get_instance().prefetch_batch_number = number


def get_table_instance(key):
    return ConfigInitializer.get_instance().get_table_instance(key)


def get_table_instance_by_name(table_name):
    return ConfigInitializer.get_instance().get_table_instance_by_name(table_name)


def insert_dangling_table(table_name):
    ConfigInitializer.get_instance().insert_dangling_table(table_name)


def insert_table_instance(name, key, instance):
    ConfigInitializer.get_instance().insert_table_instance(name, key, instance)


def export_table_instances():
    return ConfigInitializer.get_instance().table_instance_dict


def export_dangling_table():
    return ConfigInitializer.get_instance().dangling_table


def insert_optimizer(optimizer):
    ConfigInitializer.get_instance().insert_optimizer(optimizer)


def export_optimizer():
    return ConfigInitializer.get_instance().optimizer_instance


def insert_feature_spec(feature, is_training):
    ConfigInitializer.get_instance().insert_feature_spec(feature, is_training)


def get_feature_spec(key):
    return ConfigInitializer.get_instance().get_feature_spec(key)


def insert_training_mode_channel_id(is_training):
    ConfigInitializer.get_instance().insert_training_mode_channel_id(is_training)


def get_training_mode_channel_id(is_training):
    return ConfigInitializer.get_instance().get_training_mode_channel_id(is_training)


def export_feature_spec():
    return ConfigInitializer.get_instance().feature_spec_dict


def set_if_load(if_load):
    ConfigInitializer.get_instance().if_load = if_load


def get_if_load():
    return ConfigInitializer.get_instance().if_load


def get_use_static():
    return ConfigInitializer.get_instance().use_static


def get_use_hot():
    return ConfigInitializer.get_instance().use_hot


def get_use_dynamic_expansion():
    return ConfigInitializer.get_instance().use_dynamic_expansion


def terminate_config_initializer():
    ConfigInitializer.get_instance().terminate()


def get_name_to_var_dict():
    return ConfigInitializer.get_instance().name_to_var_dict


def get_initializer(is_training):
    return ConfigInitializer.get_instance().get_initializer(is_training)


def set_initializer(is_training, initializer):
    ConfigInitializer.get_instance().set_initializer(is_training, initializer)


def set_ascend_table_name_must_contain(name="merged"):
    mx_rec.util.constants.ASCEND_TABLE_NAME_MUST_CONTAIN = name


def set_ascend_env():
    """
    配置昇腾相关的参数和环境变量，生成hccl配置
    """
    rank = get_rank_id()
    rank_size = get_rank_size()
    local_rank_size = 8

    os.environ["MOX_USE_NPU"] = "1"
    os.environ["FUSION_TENSOR_SIZE"] = "2000000000"
    os.environ["MOX_USE_TF_ESTIMATOR"] = "0"
    os.environ["MOX_USE_TDT"] = "1"
    os.environ["HEARTBEAT"] = "1"
    os.environ["CONITNUE_TRAIN"] = "true"

    os.environ["RANK_ID"] = str(rank)

    device_id = str(get_device_id())
    os.environ["DEVICE_ID"] = device_id
    os.environ["ASCEND_DEVICE_ID"] = device_id
    os.environ["DEVICE_INDEX"] = device_id

    if os.getenv("RANK_TABLE_FILE"):
        os.environ["RANK_SIZE"] = str(rank_size)
    else:
        import socket
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        os.environ["CM_WORKER_IP"] = host_ip
    os.environ["HCCL_CONNECT_TIMEOUT"] = "1200"

    os.environ["JOB_ID"] = "10086"
    os.environ["SOC_VERSION"] = "Ascend910"
    os.environ["GE_AICPU_FLAG"] = "1"
    os.environ["NEW_GE_FE_ID"] = "1"
    os.environ["EXPERIMENTAL_DYNAMIC_PARTITION"] = "1"
    os.environ["ENABLE_FORCE_V2_CONTROL"] = "1"

    logging.debug(f"Ascend env has been set.")


def bind_cpu(rank_id: int, rank_size: int = None):
    """
    以均衡的方式为每个进程绑定CPU
    :param rank_id:当前进程的rank_id
    :return:
    """
    from multiprocessing import cpu_count
    import math

    try:
        local_rank_size = int(os.getenv(LOCAL_RANK_SIZE)) if rank_size is None else rank_size
    except (ValueError, TypeError):
        logging.warning(f"no valid LOCAL_RANK_SIZE was set. {DEFAULT_DEVICE_NUM_LOCAL_MACHINE} is set as default value")
        local_rank_size = DEFAULT_DEVICE_NUM_LOCAL_MACHINE

    if not (1 <= local_rank_size <= MAX_DEVICE_NUM_LOCAL_MACHINE):
        logging.warning(f"LOCAL_RANK_SIZE should be between 1 and {MAX_DEVICE_NUM_LOCAL_MACHINE}. "
                        f"{DEFAULT_DEVICE_NUM_LOCAL_MACHINE} is set as default value")
        local_rank_size = DEFAULT_DEVICE_NUM_LOCAL_MACHINE

    total_cpu = cpu_count()
    avg_count = math.ceil(total_cpu / local_rank_size)
    max_index = total_cpu - 1
    start = rank_id * avg_count
    cpu_list = [start + i for i in range(avg_count) if start + i <= max_index]

    process = psutil.Process()
    try:
        process.cpu_affinity(cpu_list)
    except IndexError:
        logging.error(f"failed to bind cpu for rank {rank_id}: {cpu_list}")

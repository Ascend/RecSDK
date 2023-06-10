#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import json
import logging
import os
from collections import defaultdict

import mxrec_pybind
import psutil

from mx_rec.constants.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION, VALID_DEVICE_ID_LIST, LOCAL_RANK_SIZE, \
    MAX_DEVICE_NUM_LOCAL_MACHINE, DEFAULT_DEVICE_NUM_LOCAL_MACHINE, HASHTABLE_COLLECTION_NAME_LENGTH
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.validator.validator import RankInfoValidator, StringValidator


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
        self._removing_var_list = []
        self._name_to_var_dict = dict()
        self._table_name_set = set()
        self._table_name_to_feature_spec = dict()
        self._feature_spec_dict = dict()
        self._training_mode_channel_dict = dict()
        self._rank_to_device_dict = dict()
        self._initializer_dict = {}
        self._optimizer_instance = None
        self._is_graph_modify_hook_running = False
        self._modify_graph = False
        self._is_terminated = False

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
            self.set_hccl_info_without_json()
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

    @property
    def is_graph_modify_hook_running(self):
        return self._is_graph_modify_hook_running

    @property
    def modify_graph(self):
        return self._modify_graph

    @property
    def feature_spec_dict(self):
        return self._feature_spec_dict

    @property
    def table_name_set(self):
        return self._table_name_set

    @property
    def table_name_to_feature_spec(self):
        return self._table_name_to_feature_spec

    @property
    def table_instance_dict(self):
        return self._table_instance_dict

    @property
    def optimizer_instance(self):
        return self._optimizer_instance

    @property
    def is_frozen(self):
        return self._is_frozen

    @property
    def name_to_var_dict(self):
        return self._name_to_var_dict

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

    @property
    def train_interval(self):
        return self._train_interval

    @property
    def eval_steps(self):
        return self._eval_steps

    @property
    def prefetch_batch_number(self):
        return self._prefetch_batch_number

    @property
    def if_load(self):
        return self._if_load

    @property
    def ascend_global_hashtable_collection(self):
        return self._ascend_global_hashtable_collection

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

    def terminate(self):
        if self._is_terminated:
            logging.warning("The initializer has already been released once, please do not release it again.")
            return

        if self._asc_manager is not None:
            self.del_asc_manager()

        if self._mpi:
            self._mpi.Finalize()
            logging.debug("MPI has been destroyed.")

        self._is_terminated = True

    def insert_feature_spec(self, feature, is_training):
        self._feature_spec_dict[feature.name] = feature
        if feature.table_name not in self._table_name_to_feature_spec:
            self._table_name_to_feature_spec[feature.table_name] = {True: [], False: []}
        self._table_name_to_feature_spec[feature.table_name][is_training].append(feature)

    def get_feature_spec(self, key):
        return self._feature_spec_dict.get(key)

    def parse_hccl_json(self):
        rank_table_path = os.path.realpath(os.getenv("RANK_TABLE_FILE"))
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

    def set_hccl_info_without_json(self):
        """
        Used for no rank table file configured training situation.
        Now, only less than or equal 8p training job is supported.
        :return: None
        """
        RankInfoValidator().check_visible_devices()
        ascend_visible_devices = os.getenv("ASCEND_VISIBLE_DEVICES")
        device_list = []
        try:
            if "-" in ascend_visible_devices:
                split_devices = ascend_visible_devices.strip().split("-")
                if len(split_devices) >= 1:
                    rank_start = int(split_devices[0])
                    device_list = list(range(rank_start, int(ascend_visible_devices.strip().split("-")[-1]) + 1))
            elif "," in ascend_visible_devices:
                device_list = list(map(int, ascend_visible_devices.strip().split(","))).sort()
            elif ascend_visible_devices in VALID_DEVICE_ID_LIST:
                device_list = [int(ascend_visible_devices.strip())]
            else:
                raise ValueError("invalid env variable ascend_visible_devices.")
        except ValueError as error:
            raise ValueError("Invalid env variable ascend_visible_devices, no valid device id is configured. "
                             "Please refer to the document https://www.hiascend.com/document/detail/zh/"
                             "CANNCommunityEdition/63RC2alpha002/ptmoddevg/ptmigr/ptmigr_0151.html for "
                             "the correct configuration method.") from error
        except IndexError as error:
            raise IndexError(
                f"Index of ascend_visible_devices {ascend_visible_devices.strip().split('-')[-1]} is out of range") \
                from error

        chief_device = os.getenv("CM_CHIEF_DEVICE")
        rank_size = os.getenv("CM_WORKER_SIZE")
        if int(rank_size) != len(device_list):
            raise ValueError(f"Rank size {rank_size} is different from device num {len(device_list)}.")
        try:
            self._rank_to_device_dict[0] = int(chief_device)
            device_list.pop(int(chief_device))
        except IndexError as err:
            raise IndexError(
                f"Config CM_CHIEF_DEVICE {chief_device} not in training container device list {device_list}.") from err
        except ValueError as err:
            raise ValueError("CM_WORKER_SIZE or CM_CHIEF_DEVICE uncorrected configured.") from err

        for device_idx in device_list:
            device_id = mxrec_pybind.get_logic_id(int(device_idx))
            if device_id > 16:
                raise ValueError(f"get logic id from physic id fail.")
            index = device_list.index(device_idx)
            self._rank_to_device_dict[index + 1] = device_id

    def insert_training_mode_channel_id(self, is_training):
        if is_training not in self._training_mode_channel_dict:
            # mx_rec has 2 channel for data input. it would bind channel_id to training mode recorded in dict.
            self._training_mode_channel_dict[is_training] = len(self._training_mode_channel_dict)

    def get_training_mode_channel_id(self, is_training):
        return self._training_mode_channel_dict.get(is_training)

    def insert_dangling_table(self, name):
        if name not in self._dangling_table:
            self._dangling_table.append(name)

    def insert_removing_var_list(self, name):
        if name not in self._removing_var_list:
            self._removing_var_list.append(name)

    @property
    def dangling_table(self):
        return self._dangling_table

    @property
    def removing_var_list(self):
        return self._removing_var_list

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

    def insert_optimizer(self, optimizer):
        self._optimizer_instance = optimizer

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

    @train_interval.setter
    def train_interval(self, interval):
        check_step(interval)
        self._train_interval = interval

    @eval_steps.setter
    def eval_steps(self, steps):
        check_step(steps)
        self._eval_steps = steps

    @prefetch_batch_number.setter
    def prefetch_batch_number(self, number):
        check_step(number, 1)
        self._prefetch_batch_number = number

    @if_load.setter
    def if_load(self, flag):
        if not isinstance(flag, bool):
            raise TypeError(f"Flag if load should be a boolean.")

        self._if_load = flag

    @is_graph_modify_hook_running.setter
    def is_graph_modify_hook_running(self, is_hook_running):
        if not isinstance(is_hook_running, bool):
            raise TypeError(f"is_hook_running should be a boolean.")

        self._is_graph_modify_hook_running = is_hook_running

    @modify_graph.setter
    def modify_graph(self, is_modify_graph):
        if not isinstance(is_modify_graph, bool):
            raise TypeError(f"is_modify_graph should be a boolean.")

        self._modify_graph = is_modify_graph

    @ascend_global_hashtable_collection.setter
    def ascend_global_hashtable_collection(self, name):
        string_validator = StringValidator(name, max_len=HASHTABLE_COLLECTION_NAME_LENGTH, min_len=1)
        if not string_validator.check_string_length().check_whitelist().is_valid():
            raise ValueError(string_validator.msg)
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


def get_is_graph_modify_hook_running():
    return ConfigInitializer.get_instance().is_graph_modify_hook_running


def set_is_graph_modify_hook_running(is_running):
    ConfigInitializer.get_instance().is_graph_modify_hook_running = is_running


def get_modify_graph():
    return ConfigInitializer.get_instance().modify_graph


def set_modify_graph(is_modify_graph):
    ConfigInitializer.get_instance().modify_graph = is_modify_graph


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


def insert_removing_var_list(var_name):
    ConfigInitializer.get_instance().insert_removing_var_list(var_name)


def insert_table_instance(name, key, instance):
    ConfigInitializer.get_instance().insert_table_instance(name, key, instance)


def export_table_instances():
    return ConfigInitializer.get_instance().table_instance_dict


def export_dangling_table():
    return ConfigInitializer.get_instance().dangling_table


def export_removing_var_list():
    return ConfigInitializer.get_instance().removing_var_list


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
    mx_rec.constants.constants.ASCEND_TABLE_NAME_MUST_CONTAIN = name


def set_ascend_env():
    """
    配置昇腾相关的参数和环境变量，生成hccl配置
    """
    rank = get_rank_id()
    rank_size = get_rank_size()

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


def get_available_cpu_num_and_range():
    """
    获取当前环境可用的cpu数量和numa范围
    Returns:

    """
    cpu_available = os.sched_getaffinity(os.getpid())  # 获取可被绑定的核心

    is_ok = True
    cpu_pkg_id_file = "/sys/devices/system/cpu/cpu{}/topology/physical_package_id"
    pkg_id2cpu_list = defaultdict(list)
    for cpu in cpu_available:
        f_path = cpu_pkg_id_file.format(cpu)
        if not os.path.exists(f_path):
            logging.warning(f"failed to get numa node of cpu: {cpu}")
            is_ok = False
            break
        with open(f_path, "r", encoding="utf-8") as f_in:
            pkg_id = f_in.readline().strip()
            pkg_id2cpu_list[pkg_id].append(cpu)

    def parse_range(cpu_list, cpu_range):
        sorted_cpu_list = sorted(cpu_list)
        pre_cpu = sorted_cpu_list[0]
        cpu_range.append([pre_cpu])

        for sorted_cpu in sorted_cpu_list[1:]:
            if sorted_cpu - pre_cpu != 1:
                cpu_range[-1].append(pre_cpu)
                cpu_range.append([sorted_cpu])
            pre_cpu = sorted_cpu

        if len(cpu_range[-1]) == 1:
            cpu_range[-1].append(pre_cpu)

    valid_cpu_range_list = []
    if is_ok:
        logging.info(f"available numa node num: {len(pkg_id2cpu_list)}")
        for _, part_cpu_list in pkg_id2cpu_list.items():
            parse_range(part_cpu_list, valid_cpu_range_list)
    else:
        parse_range(list(cpu_available), valid_cpu_range_list)
    return len(cpu_available), valid_cpu_range_list


def bind_cpu(rank_id: int, rank_size: int = None):
    """
    以均衡的方式为每个进程绑定CPU
    :param rank_id:当前进程的rank_id
    :param rank_size: 进程数
    :return:
    """
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

    total_cpu, cpu_range_list = get_available_cpu_num_and_range()
    avg_count = math.ceil(total_cpu / local_rank_size)
    while True:
        if avg_count == 0:
            logging.warning(f"not enough cpu to bind. cpu num: {total_cpu}, range: {cpu_range_list}")
            return

        max_split = 0
        for cpu_range in cpu_range_list:
            max_split += (cpu_range[1] - cpu_range[0] + 1) // avg_count
        if max_split >= local_rank_size:
            break
        avg_count -= 1

    candidate_list = []
    for cpu_range in cpu_range_list:
        start = cpu_range[0]
        splits = (cpu_range[1] - cpu_range[0] + 1) // avg_count
        candidate_range = [list(range(start + i * avg_count, start + ((i + 1) * avg_count))) for i in range(splits)]
        candidate_list.extend(candidate_range)

    cpu_list = candidate_list[rank_id]

    process = psutil.Process()
    try:
        process.cpu_affinity(cpu_list)
    except IndexError:
        logging.error(f"failed to bind cpu for rank {rank_id}: {cpu_list}")
    logging.info(f"bind cpu for rank {rank_id}: {cpu_list}")
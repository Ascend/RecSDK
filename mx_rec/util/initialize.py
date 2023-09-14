#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
from collections import defaultdict
import dataclasses
import json

import psutil

import mx_rec.constants.constants
from mx_rec.constants.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION, HASHTABLE_COLLECTION_NAME_LENGTH, \
    TRAIN_CHANNEL_ID, EVAL_CHANNEL_ID, MIN_SIZE, MAX_CONFIG_SIZE, MAX_RANK_SIZE, MAX_INT32, TFDevice, Flag
from mx_rec.util.communication.hccl_mgmt import parse_hccl_json, set_hccl_info_without_json
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.validator.validator import StringValidator, FileValidator, para_checker_decorator, ClassValidator, \
    IntValidator, ValueCompareValidator
from mx_rec.util.atomic import AtomicInteger
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.log import logger


class ConfigInitializer:
    _single_instance = None
    customized_ops = None
    host_pipeline_ops = import_host_pipeline_ops()

    @para_checker_decorator(check_option_list=[
        ("use_mpi", ClassValidator, {"classes": (bool, )}),
        ("train_steps", IntValidator, {"min_value": -1, "max_value": MAX_INT32}),
        ("eval_steps", IntValidator, {"min_value": -1, "max_value": MAX_INT32}),
        (["train_steps", "eval_steps"], ValueCompareValidator, {"target": 0},
         ["check_at_least_one_not_equal_to_target"]),
        ("if_load", ClassValidator, {"classes": (bool, )}),
        ("use_dynamic", ClassValidator, {"classes": (bool, )}),
        ("use_hot", ClassValidator, {"classes": (bool, )}),
        ("use_dynamic_expansion", ClassValidator, {"classes": (bool, )}),
        ("bind_cpu", ClassValidator, {"classes": (bool, )}),
    ])
    def __init__(self, use_mpi=True, **kwargs):
        self._use_mpi = use_mpi
        self._rank_id = kwargs.get("rank_id", 0)
        self._rank_size = kwargs.get("rank_size", 1)
        self._ascend_global_hashtable_collection = ASCEND_GLOBAL_HASHTABLE_COLLECTION
        self._comm = None
        self._asc_manager = None
        self._mpi = None
        self._is_frozen = False
        self._train_steps = None
        self._eval_steps = None
        self._save_steps = None
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
        self._bool_gauge_set = set()
        self._optimizer_instance = None
        self._is_graph_modify_hook_running = False
        self._modify_graph = False
        self._is_terminated = False
        self._is_last_round = False
        self._run_times = AtomicInteger()
        self._merged_multi_lookup = dict()
        self._target_batch = dict()
        self._iterator_type = ""
        self._sparse_dir = ""

        if self._use_mpi:
            logger.debug(f"Using mpi to launch task.")
            from mpi4py import MPI
            self._mpi = MPI
            self._comm = MPI.COMM_WORLD
            self._rank_id = self._comm.Get_rank()
            self._rank_size = self._comm.Get_size()
        else:
            raise ValueError("only mpi is supported for launching task.")

        self._rank_to_device_dict, self._local_rank_size = parse_hccl_json() if global_env.rank_table_file else \
            set_hccl_info_without_json(visible_devices=global_env.ascend_visible_devices,
                                       rank_size=global_env.cm_worker_size,
                                       chief_device=global_env.cm_chief_device)
        self.train_steps = kwargs.get("train_steps", -1)
        self.eval_steps = kwargs.get("eval_steps", -1)
        self.save_steps = kwargs.get("save_steps", -1)

        self.if_load = kwargs.get("if_load", False)

        self.use_static = not kwargs.get("use_dynamic", True)
        self.use_hot = kwargs.get("use_hot", True)
        self.use_dynamic_expansion = kwargs.get("use_dynamic_expansion", False)
        if kwargs.get("bind_cpu", True):
            bind_cpu(self._rank_id, self._local_rank_size)
        self.enable_table_merge = True if global_env.tf_device == TFDevice.NPU.value else False
        # 两个通道的sparse look id，用于通讯的标识
        self.notify_hybrid_channel_sparse_id = [0, 0]
        self.stat_on = (global_env.stat_on == Flag.TRUE.value)

    def __del__(self):
        self.terminate()

    @property
    def iterator_type(self):
        return self._iterator_type

    @property
    def local_rank_size(self):
        return self._local_rank_size

    @property
    def merged_multi_lookup(self):
        return self._merged_multi_lookup

    @property
    def target_batch(self):
        return self._target_batch

    @property
    def is_last_round(self):
        return self._is_last_round

    @property
    def run_times(self):
        return self._run_times

    @property
    def bool_gauge_set(self):
        return self._bool_gauge_set

    @property
    def is_graph_modify_hook_running(self):
        return self._is_graph_modify_hook_running

    @property
    def modify_graph(self):
        return self._modify_graph

    @property
    def sparse_dir(self):
        return self._sparse_dir

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
    def train_steps(self):
        return self._train_steps

    @property
    def eval_steps(self):
        return self._eval_steps

    @property
    def save_steps(self):
        return self._save_steps

    @property
    def if_load(self):
        return self._if_load

    @property
    def ascend_global_hashtable_collection(self):
        return self._ascend_global_hashtable_collection

    @property
    def dangling_table(self):
        return self._dangling_table

    @property
    def removing_var_list(self):
        return self._removing_var_list

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
        logger.info("python process run into terminate")
        if self._is_terminated:
            logger.warning("The initializer has already been released once, please do not release it again.")
            return

        if self._asc_manager is not None:
            self.del_asc_manager()
        logger.info("python process run terminate success")

        self._is_terminated = True
        ConfigInitializer._single_instance = None

    def insert_feature_spec(self, feature, is_training):
        self._feature_spec_dict[feature.name] = feature
        if feature.table_name not in self._table_name_to_feature_spec:
            self._table_name_to_feature_spec[feature.table_name] = {True: [], False: []}
        self._table_name_to_feature_spec[feature.table_name][is_training].append(feature)

    def get_feature_spec(self, key):
        return self._feature_spec_dict.get(key)

    def insert_training_mode_channel_id(self, is_training):
        if is_training not in self._training_mode_channel_dict:
            # mx_rec has 2 channel for data input.
            # train_model bind to channel TRAIN_CHANNEL_ID
            # eval_model bind to channel EVAL_CHANNEL_ID
            self._training_mode_channel_dict[is_training] = TRAIN_CHANNEL_ID if is_training else EVAL_CHANNEL_ID

    def get_training_mode_channel_id(self, is_training):
        return self._training_mode_channel_dict.get(is_training)

    def insert_dangling_table(self, name):
        if name not in self._dangling_table:
            self._dangling_table.append(name)

    def insert_removing_var_list(self, name):
        if name not in self._removing_var_list:
            self._removing_var_list.append(name)

    def insert_table_instance(self, name, key, instance):
        if key in self._table_instance_dict:
            raise KeyError(f"Given key {key} has been used.")

        if name in self._table_name_set:
            raise ValueError(f"Duplicated hashtable name '{name}' was used.")

        logger.debug("Record one hash table, with name: %s, key: %s.", name, key)
        self._table_name_set.add(name)
        if name not in self._table_name_to_feature_spec:
            self._table_name_to_feature_spec[name] = {True: [], False: []}
        self._name_to_var_dict[name] = key
        self._table_instance_dict[key] = instance
        if self.stat_on:
            logger.info("[StatInfo] current_table_num %s", len(self._table_instance_dict))

    def insert_bool_gauge(self, name):
        if not isinstance(name, str):
            raise TypeError(f"bool gauge name '{name}' should be str.")

        self._bool_gauge_set.add(name)

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
        logger.debug("ASC manager has been destroyed.")

    @iterator_type.setter
    def iterator_type(self, iterator_type):
        if not isinstance(iterator_type, str):
            raise TypeError(f"iterator_type `{iterator_type}` should be str.")

        self._iterator_type = iterator_type

    @train_steps.setter
    def train_steps(self, step: int):
        check_step(step)
        self._train_steps = step

    @eval_steps.setter
    def eval_steps(self, steps):
        check_step(steps)
        self._eval_steps = steps

    @save_steps.setter
    def save_steps(self, steps):
        check_step(steps)
        self._save_steps = steps

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

    @sparse_dir.setter
    def sparse_dir(self, sparse_dir):
        if not isinstance(sparse_dir, str):
            raise TypeError(f"sparse_dir should be str.")

        self._sparse_dir = sparse_dir

    @is_last_round.setter
    def is_last_round(self, last_round):
        if not isinstance(last_round, bool):
            raise TypeError(f"last_round should be a boolean.")

        self._is_last_round = last_round

    @ascend_global_hashtable_collection.setter
    def ascend_global_hashtable_collection(self, name):
        string_validator = StringValidator(name="hashtable_collection", value=name,
                                           max_len=HASHTABLE_COLLECTION_NAME_LENGTH, min_len=1)
        if not string_validator.check_string_length().check_whitelist().is_valid():
            raise ValueError(string_validator.msg)
        self._ascend_global_hashtable_collection = name

    def get_initializer(self, is_training):
        return self._initializer_dict.get(is_training)

    def set_initializer(self, is_training, initializer):
        if not isinstance(is_training, bool):
            raise ValueError(f"Given key must be a boolean, but got {is_training}.")

        self._initializer_dict[is_training] = initializer

    def insert_merged_multi_lookup(self, is_training, value=True):
        if not isinstance(is_training, bool):
            raise TypeError(f"Given key must be a boolean, but got {is_training} for `merged_multi_lookup`.")

        self._merged_multi_lookup[is_training] = value

    def get_merged_multi_lookup(self, is_training):
        return self._merged_multi_lookup.get(is_training)

    def set_target_batch(self, is_training, batch):
        if not isinstance(is_training, bool):
            raise TypeError(f"Given key must be a boolean, but got {is_training} for `target_batch`.")

        self._target_batch[is_training] = batch

    def get_target_batch(self, is_training):
        return self._target_batch.get(is_training)

    def delete_initializers(self):
        self._initializer_dict = {}


@para_checker_decorator(check_option_list=[
    ("name", ClassValidator, {"classes": (str, type(None))})
])
def set_ascend_global_hashtable_collection(name=ASCEND_GLOBAL_HASHTABLE_COLLECTION):
    ConfigInitializer.get_instance().ascend_global_hashtable_collection = name


def get_ascend_global_hashtable_collection():
    return ConfigInitializer.get_instance().ascend_global_hashtable_collection


def check_step(param, min_value=-1):
    if not isinstance(param, int):
        raise TypeError("Given param must be an integer.")

    if param < min_value:
        raise ValueError(f"Valid value range is larger than or equals to {min_value}.")


def init(use_mpi, **kwargs):
    logger.info("The environment variables set for mxRec is: %s",
                json.dumps(dataclasses.asdict(global_env), ensure_ascii=False))
    ConfigInitializer.set_instance(use_mpi, **kwargs)
    set_ascend_env()


def get_is_graph_modify_hook_running():
    return ConfigInitializer.get_instance().is_graph_modify_hook_running


def set_is_graph_modify_hook_running(is_running):
    ConfigInitializer.get_instance().is_graph_modify_hook_running = is_running


def get_run_times():
    return ConfigInitializer.get_instance().run_times


def increase_run_times():
    ConfigInitializer.get_instance().run_times.increase()


def get_is_last_round():
    return ConfigInitializer.get_instance().is_last_round


def set_is_last_round(last_round):
    ConfigInitializer.get_instance().is_last_round = last_round


def get_bool_gauge_set():
    return ConfigInitializer.get_instance().bool_gauge_set


def insert_bool_gauge(name):
    ConfigInitializer.get_instance().insert_bool_gauge(name)


def get_modify_graph():
    return ConfigInitializer.get_instance().modify_graph


def set_modify_graph(is_modify_graph):
    ConfigInitializer.get_instance().modify_graph = is_modify_graph


def set_sparse_dir(sparse_dir):
    ConfigInitializer.get_instance().sparse_dir = sparse_dir


def get_sparse_dir():
    return ConfigInitializer.get_instance().sparse_dir


def get_rank_size():
    return ConfigInitializer.get_instance().rank_size


def get_rank_id():
    return ConfigInitializer.get_instance().rank_id


def get_device_id():
    return ConfigInitializer.get_instance().device_id


def set_asc_manager(manager):
    ConfigInitializer.get_instance().set_asc_manager(manager)


def get_asc_manager():
    return ConfigInitializer.get_instance().get_asc_manager()


def trigger_evict():
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")

    if ConfigInitializer.get_instance().get_asc_manager().evict():
        logger.debug("Feature evict is triggered by ops.")
        return True
    logger.warning("Feature evict not success, skip this time!")
    return False


def clear_channel(is_train_channel=False):
    if not isinstance(is_train_channel, bool):
        raise ValueError("Arg is_train_channel should be a boolean.")
    channel_id = get_training_mode_channel_id(is_train_channel)
    logger.info("clear channel: %s", channel_id)

    return ConfigInitializer.get_instance().host_pipeline_ops.clear_channel(channel_id)


def is_asc_manager_initialized():
    return ConfigInitializer.get_instance().get_asc_manager() is not None


def get_host_data(table_name):
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")
    logger.debug("start to get host data.")
    return ConfigInitializer.get_instance().get_asc_manager().send(table_name)


def send_host_data(key_offset_map):
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")
    ConfigInitializer.get_instance().get_asc_manager().receive(key_offset_map)
    logger.debug("Data has been send to the host pipeline.")


def save_host_data(root_dir):
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")

    ConfigInitializer.get_instance().get_asc_manager().save(root_dir)
    logger.debug("Data from host pipeline has been saved.")


def restore_host_data(root_dir):
    if not is_asc_manager_initialized():
        raise RuntimeError("ASC manager does not exist.")

    if not ConfigInitializer.get_instance().get_asc_manager().load(root_dir):
        terminate_config_initializer()
        raise TypeError("Asc load data does not match usr setups, \
        please re-consider if you want to restore from this dir")
    logger.debug("Data from host pipeline has been restored.")


def destroy_asc_manager():
    initializer = ConfigInitializer.get_instance()
    if initializer.get_asc_manager() is not None:
        logger.debug("start destroy asc manager...")
        initializer.del_asc_manager()
    else:
        logger.warning("ASC manager does not exist, please check your code.")


def is_asc_frozen():
    return ConfigInitializer.get_instance().is_frozen


def export_table_name_set():
    return ConfigInitializer.get_instance().table_name_set


def get_host_pipeline_ops():
    return ConfigInitializer.host_pipeline_ops


def get_customized_ops():
    return ConfigInitializer.customized_ops


def get_train_steps():
    return ConfigInitializer.get_instance().train_steps


def get_eval_steps():
    return ConfigInitializer.get_instance().eval_steps


def get_save_steps():
    return ConfigInitializer.get_instance().save_steps


def set_train_steps(steps: int):
    ConfigInitializer.get_instance().train_steps = steps


def set_eval_steps(steps: int):
    ConfigInitializer.get_instance().eval_steps = steps


def set_save_steps(steps: int):
    ConfigInitializer.get_instance().save_steps = steps


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


def export_table_num():
    return len(ConfigInitializer.get_instance().table_instance_dict)


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


@para_checker_decorator(check_option_list=[
    ("if_load", ClassValidator, {"classes": (bool, )})
])
def set_if_load(if_load):
    ConfigInitializer.get_instance().if_load = if_load


def get_if_load():
    return ConfigInitializer.get_instance().if_load


def get_use_static():
    return ConfigInitializer.get_instance().use_static


def get_stat_on():
    return ConfigInitializer.get_instance().stat_on
    

def get_use_hot():
    return ConfigInitializer.get_instance().use_hot


def get_enable_table_merge():
    return ConfigInitializer.get_instance().enable_table_merge


def get_use_dynamic_expansion():
    return ConfigInitializer.get_instance().use_dynamic_expansion


def terminate_config_initializer():
    ConfigInitializer.get_instance().terminate()


def get_name_to_var_dict():
    return ConfigInitializer.get_instance().name_to_var_dict


@para_checker_decorator(check_option_list=[
    ("is_training", ClassValidator, {"classes": (bool, )})
])
def get_initializer(is_training):
    return ConfigInitializer.get_instance().get_initializer(is_training)


def set_initializer(is_training, initializer):
    ConfigInitializer.get_instance().set_initializer(is_training, initializer)


def set_ascend_table_name_must_contain(name="merged"):
    mx_rec.constants.constants.ASCEND_TABLE_NAME_MUST_CONTAIN = name


def insert_merged_multi_lookup(is_training: bool, value: bool = True):
    """
    记录自动改图模式下是否调用了合并lookup的函数.
    Args:
        is_training: 当前是否为训练模式，训练模式为True，否则为False
        value: 是否调用了合并lookup的函数, 调用了为True，否则为False
    Returns: None
    """
    ConfigInitializer.get_instance().insert_merged_multi_lookup(is_training, value)


def get_merged_multi_lookup(is_training: bool) -> bool:
    """
    返回自动改图模式下是否调用了合并lookup函数的记录.
    Args:
        is_training: 当前是否为训练模式，训练模式为True，否则为False
    Returns: 调用记录，调用了为True，否则为False
    """
    return ConfigInitializer.get_instance().get_merged_multi_lookup(is_training)


def set_target_batch(is_training: bool, batch: dict):
    """
    记录自动改图模式下生成新数据集中的batch.
    Args:
        is_training: 当前是否为训练模式，训练模式为True，否则为False
        batch: 数据集中的batch
    Returns: None
    """
    ConfigInitializer.get_instance().set_target_batch(is_training, batch)


def get_target_batch(is_training: bool) -> dict:
    """
    返回自动改图模式下生成新数据集中batch的记录.
    Args:
        is_training: 当前是否为训练模式，训练模式为True，否则为False
    Returns: 新数据集中的batch
    """
    return ConfigInitializer.get_instance().get_target_batch(is_training)


def get_iterator_type() -> str:
    """
    返回数据集的迭代器类型.
    Returns: 数据集的迭代器类型
    """
    return ConfigInitializer.get_instance().iterator_type


def get_local_rank_size() -> int:
    """
    获取当前worker参与任务的进程数
    Returns:
    """
    return ConfigInitializer.get_instance().local_rank_size


def set_iterator_type(iterator_type: str):
    """
    记录数据集的迭代器类型.
    Args:
        iterator_type: 数据集的迭代器类型
    Returns: None
    """
    ConfigInitializer.get_instance().iterator_type = iterator_type


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

    if global_env.rank_table_file:
        os.environ["RANK_SIZE"] = str(rank_size)
    os.environ["HCCL_CONNECT_TIMEOUT"] = "1200"

    os.environ["JOB_ID"] = "10086"
    os.environ["SOC_VERSION"] = "Ascend910"
    os.environ["GE_AICPU_FLAG"] = "1"
    os.environ["NEW_GE_FE_ID"] = "1"
    os.environ["EXPERIMENTAL_DYNAMIC_PARTITION"] = "1"
    os.environ["ENABLE_FORCE_V2_CONTROL"] = "1"

    logger.debug(f"Ascend env has been set.")


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
            logger.warning("failed to get numa node of cpu: %s", cpu)
            is_ok = False
            break

        with open(f_path, "r", encoding="utf-8") as f_in:
            # check whether file is valid
            file_validator = FileValidator("cpu_topology_file", f_path)
            # 1.check whether f_path is soft link
            file_validator.check_not_soft_link()
            # 2.check file size
            file_validator.check_file_size(MAX_CONFIG_SIZE, MIN_SIZE)
            file_validator.check()
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
        logger.info("available numa node num: %s", len(pkg_id2cpu_list))
        for _, part_cpu_list in pkg_id2cpu_list.items():
            parse_range(part_cpu_list, valid_cpu_range_list)
    else:
        parse_range(list(cpu_available), valid_cpu_range_list)
    return len(cpu_available), valid_cpu_range_list


def bind_cpu(rank_id: int, local_rank_size: int):
    """
    以均衡的方式为每个进程绑定CPU
    :param rank_id:当前进程的rank_id
    :param local_rank_size: 当前worker进程数
    :return:
    """
    import math

    total_cpu, cpu_range_list = get_available_cpu_num_and_range()
    avg_count = math.ceil(total_cpu / local_rank_size)
    while True:
        if avg_count == 0:
            logger.warning(f"not enough cpu to bind. cpu num: %s, range: %s", total_cpu, cpu_range_list)
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
        logger.error("failed to bind cpu for rank %s: %s", rank_id, cpu_list)
    logger.info("bind cpu for rank %s: %s", rank_id, cpu_list)
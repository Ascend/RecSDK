#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from typing import Union, Optional

import tensorflow as tf

from mx_rec.constants.constants import MAX_VOCABULARY_SIZE, MULTI_LOOKUP_TIMES
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.util.communication.hccl_ops import get_rank_size
from mx_rec.util.initialize import ConfigInitializer


def check_emb_init_params(is_hbm: bool, embedding_size: tf.TensorShape):
    """
    校验稀疏表的初始化参数.

    Args:
        is_hbm: 是否为HBM模式
        embedding_size: 稀疏表维度大小

    Returns: None
    """
    if ConfigInitializer.get_instance().hybrid_manager_config.freeze:
        raise EnvironmentError("Emb cache management has been established, you cannot build new hash table.")

    if not is_hbm and ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError("DDR/SSD mode do not support embedding dynamic expansion for now.")

    if embedding_size.ndims != 1:
        raise ValueError("Parameter 'embedding_size' can only be one dim shape.")


def check_emb_lookup_params(table_params: dict, feature_spec: Union[tf.Tensor, FeatureSpec], send_count: Optional[int],
                            is_training: bool):
    """
    校验稀疏表此次lookup的参数.

    Args:
        table_params: 稀疏表参数字典
        feature_spec: 稀疏表次数lookup的tensor或tensor的包装类
        send_count: all2all通信参数
        is_training: 当前流程是训练还是推理

    Returns: None
    """
    # check FeatureSpec
    if isinstance(feature_spec, FeatureSpec):
        if not feature_spec.initialized:
            raise RuntimeError("Feature Spec has not been initialized.")
        if is_training not in feature_spec.pipeline_mode:
            raise RuntimeError(f"You have not config feature for is training mode '{is_training}', please config "
                               f"feature with func sparse_lookup at first.")

    # check max vocabulary size
    slice_device_vocabulary_size = table_params.get("slice_device_vocabulary_size")
    slice_host_vocabulary_size = table_params.get("slice_host_vocabulary_size")
    table_name = table_params.get("table_name")
    if slice_host_vocabulary_size > MAX_VOCABULARY_SIZE:
        raise ValueError(f"given host_vocabulary_size was too big for table "
                         f"'{table_name}', in which slice_device_vocabulary_size was "
                         f"{slice_device_vocabulary_size} and slice_host_vocabulary_size was "
                         f"{slice_host_vocabulary_size}.")

    if not ConfigInitializer.get_instance().use_static:
        return

    # check send count
    if not (isinstance(send_count, int) and send_count > 0):
        raise ValueError("Send count must be a integer which is larger than 0.")

    if table_params.get("is_hbm") or ConfigInitializer.get_instance().use_dynamic_expansion:
        return

    # check vocabulary size with send count
    rank_size = get_rank_size()
    if slice_device_vocabulary_size < send_count * rank_size:
        raise ValueError(f"Given device_vocabulary_size was too small for table '{table_name}', "
                         f"in which slice_device_vocabulary_size was {slice_device_vocabulary_size} "
                         f"and send_count({send_count}) * rank_size({rank_size}) was "
                         f"{send_count * rank_size}.")

    if slice_host_vocabulary_size < send_count * rank_size:
        raise ValueError(f"Given host_vocabulary_size was too small for table '{table_name}', "
                         f"in which slice_host_vocabulary_size was {slice_host_vocabulary_size} "
                         f"and send_count({send_count}) * rank_size({rank_size}) was "
                         f"{send_count * rank_size}.")


def check_emb_multi_lookup_times(lookup_times: int, table_name: str):
    """
    校验稀疏表一表多查的次数.

    Args:
        lookup_times: 稀疏表lookup的次数
        table_name: 稀疏表名

    Returns: None
    """
    if lookup_times > MULTI_LOOKUP_TIMES:
        raise RuntimeError(f"The number of multiple sparse lookup for a table ({table_name}) is "
                           f"{MULTI_LOOKUP_TIMES}, and current times is {lookup_times}.")

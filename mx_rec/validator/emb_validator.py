#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from typing import Union, Optional, List

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


def check_emb_lookup_params(
        table_params: dict, feature_spec: Union[tf.Tensor, FeatureSpec], send_count: Optional[int], is_training: bool
):
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
            raise RuntimeError(
                f"You have not config feature for is training mode '{is_training}', please config "
                f"feature with func sparse_lookup at first."
            )

    # check max vocabulary size
    slice_device_vocabulary_size = table_params.get("slice_device_vocabulary_size")
    slice_host_vocabulary_size = table_params.get("slice_host_vocabulary_size")
    table_name = table_params.get("table_name")
    if slice_host_vocabulary_size > MAX_VOCABULARY_SIZE:
        raise ValueError(
            f"Given host_vocabulary_size was too big for table "
            f"'{table_name}', in which slice_device_vocabulary_size was "
            f"{slice_device_vocabulary_size} and slice_host_vocabulary_size was "
            f"{slice_host_vocabulary_size}."
        )

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
        raise ValueError(
            f"Given device_vocabulary_size was too small for table '{table_name}', "
            f"in which slice_device_vocabulary_size was {slice_device_vocabulary_size} "
            f"and it must be bigger than send_count({send_count}) * rank_size({rank_size}): "
            f"{send_count * rank_size}, please increase [device vocabSize] in [create_table] interface."
        )

    if slice_host_vocabulary_size < send_count * rank_size:
        raise ValueError(
            f"Given host_vocabulary_size was too small for table '{table_name}', "
            f"in which slice_host_vocabulary_size was {slice_host_vocabulary_size} "
            f"and it must be bigger than send_count({send_count}) * rank_size({rank_size}): "
            f"{send_count * rank_size}, please increase [host vocabSize] in [create_table] interface."
        )


def check_emb_multi_lookup_times(lookup_times: int, table_name: str):
    """
    校验稀疏表一表多查的次数.

    Args:
        lookup_times: 稀疏表lookup的次数
        table_name: 稀疏表名

    Returns: None
    """
    if lookup_times > MULTI_LOOKUP_TIMES:
        raise RuntimeError(
            f"The number of multiple sparse lookup for a table ({table_name}) is "
            f"{MULTI_LOOKUP_TIMES}, and current times is {lookup_times}."
        )


def check_and_format_emb_padding_keys(
        padding_keys: Optional[Union[int, List[int]]],
        padding_keys_mask: bool,
        padding_keys_len: Optional[int],
) -> List[int]:
    """
    Check and set the padding keys parameters.

    Args:
        padding_keys: Upper-layer services must ensure that the padding keys are included in the incoming ids.
        padding_keys_mask: Whether the embedding value corresponding to the padding key is updated;
                           `True` indicates that the embedding value is not updated.
        padding_keys_len: Indicates the feature length of the corresponding ids. This parameter is mandatory
                          if padding keys are specified.

    Returns: The padding keys after formatting.

    """

    if padding_keys is None:
        if padding_keys_mask:
            raise ValueError("Padding keys mask be False when padding keys is None.")
        return []

    if not padding_keys_mask:
        raise ValueError("Padding keys mask be True when padding keys is not None.")

    if padding_keys_len is None:
        raise ValueError("Padding keys length cannot be None when padding keys is not None.")

    if isinstance(padding_keys, int):
        padding_keys = [padding_keys]
        return padding_keys

    # Remove duplicates.
    padding_keys = list(set(padding_keys))
    return padding_keys


def check_padding_keys_global_params():
    """
    Check the global parameters of the padding keys.

    Returns: None

    """

    padding_keys_mask_list = []
    table_instance_dict = ConfigInitializer.get_instance().sparse_embed_config.table_instance_dict
    for _, table_instance in table_instance_dict.items():
        if table_instance.padding_keys_mask and table_instance.is_dp:
            raise RuntimeError("The padding keys mode does not yet support dp mode.")

        if table_instance.padding_keys_mask and not table_instance.is_grad:
            raise RuntimeError("The padding keys mode should not be used together with the no grad mode.")

        padding_keys_mask_list.append(table_instance.padding_keys_mask)

    if all(padding_keys_mask_list) and not ConfigInitializer.get_instance().use_static:
        raise RuntimeError("When the padding keys mask of all tables is True, it should be set to static shape mode.")

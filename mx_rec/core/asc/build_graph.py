#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Tuple

import tensorflow as tf

import mxrec_pybind
from mx_rec.constants.constants import ASCAnchorAttr
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.util.log import logger
from mx_rec.core.asc.swap_args import SwapArgs, SwapDataType


@dataclass
class SwapInfo:
    swap_in_len: int = 0
    swap_in_pos: List[tf.Tensor] = field(default_factory=lambda: [])
    swap_out_len: int = 0
    swap_out_pos: List[tf.Tensor] = field(default_factory=lambda: [])


def get_restore_vector(config):
    logger.debug('Channel %s_restore_%s was built for getnext', config.get(ASCAnchorAttr.TABLE_NAME.value),
                 config.get(ASCAnchorAttr.CHANNEL_ID.value))
    if config.get("is_hbm"):
        if not isinstance(config.get("emb_size"), int) or config.get("emb_size") < 1:
            raise TypeError(f"emb_size must be a int")
        if config.get("emb_size") < 1:
            raise ValueError(f"emb_size is less than 1")
        emb_size = config.get("emb_size")
    else:
        if not isinstance(config.get("ext_emb_size"), int) or config.get("ext_emb_size") < 1:
            raise TypeError("ext_emb_size must be a int")
        if config.get("ext_emb_size") < 1:
            raise ValueError("ext_emb_size is less than 1")
        emb_size = config.get("emb_size")

    if ConfigInitializer.get_instance().use_static:
        restore_size = config.get("batch_size") * config.get("feat_cnt")
        device_id = int(config.get("device_id"))
        hot_size = int(mxrec_pybind.get_ub_hot_size(device_id) / emb_size)
    else:
        restore_size = None
        hot_size = None

    with tf.compat.v1.variable_scope(config.get(ASCAnchorAttr.TABLE_NAME.value), reuse=tf.compat.v1.AUTO_REUSE):
        restore_vector, hot_pos = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32, tf.int32],
            output_shapes=[restore_size, [hot_size]],
            channel_name=f'{config.get(ASCAnchorAttr.TABLE_NAME.value)}'
                         f'_restore_{config.get(ASCAnchorAttr.CHANNEL_ID.value)}')

    return restore_vector, hot_pos


def get_id_offsets(max_lookup_vec_size: int, config: dict) -> Tuple[int, SwapInfo]:
    logger.debug('Channel %s_lookup_%s was built for getnext', config.get(ASCAnchorAttr.TABLE_NAME.value),
                 config.get(ASCAnchorAttr.CHANNEL_ID.value))
    # 自动扩容当前只支持HBM模式，默认没有换入换出
    swap_info = SwapInfo()

    with tf.compat.v1.variable_scope(config.get(ASCAnchorAttr.TABLE_NAME.value), reuse=tf.compat.v1.AUTO_REUSE):
        if config.get("use_dynamic_expansion"):
            [id_offsets] = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int64],
                output_shapes=[[max_lookup_vec_size]],
                channel_name=f'{config.get(ASCAnchorAttr.TABLE_NAME.value)}'
                             f'_lookup_{config.get(ASCAnchorAttr.CHANNEL_ID.value)}')
            return id_offsets, swap_info

        [id_offsets] = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32],
            output_shapes=[[max_lookup_vec_size]],
            channel_name=f'{config.get(ASCAnchorAttr.TABLE_NAME.value)}'
                         f'_lookup_{config.get(ASCAnchorAttr.CHANNEL_ID.value)}')
        if config.get("is_hbm"):
            return id_offsets, swap_info

        swap_channel = f'{config.get(ASCAnchorAttr.TABLE_NAME.value)}_swap_{config.get(ASCAnchorAttr.CHANNEL_ID.value)}'
        (
            swap_info.swap_in_pos,
            swap_info.swap_out_pos,
            swap_info.swap_in_len,
            swap_info.swap_out_len,
        ) = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32, tf.int32, tf.int32, tf.int32],
            output_shapes=[[max_lookup_vec_size], [max_lookup_vec_size], [], []],
            channel_name=swap_channel,
        )
        logger.debug('Channel %s_swap_%s was built for getnext', config.get(ASCAnchorAttr.TABLE_NAME.value),
                     config.get(ASCAnchorAttr.CHANNEL_ID.value))
    return id_offsets, swap_info


def get_all2all_args(use_static: bool, config: dict) -> Optional[list]:
    """
    Get all2all parameters for dynamic condition
    :param use_static: dynamic or static
    :param config: embedding config
    :return: all2all parametrs
    """
    all2all_args = None
    if use_static:
        return all2all_args

    with tf.compat.v1.variable_scope(config.get(ASCAnchorAttr.TABLE_NAME.value), reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope("all2all"):
            logger.debug('Channel %s_a2a_%s was built for getnext', config.get(ASCAnchorAttr.TABLE_NAME.value),
                         config.get(ASCAnchorAttr.CHANNEL_ID.value))
            all2all_args = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int64],
                output_shapes=[[config.get("rank_size"), config.get("rank_size")]],
                channel_name=f'{config.get(ASCAnchorAttr.TABLE_NAME.value)}'
                             f'_all2all_{config.get(ASCAnchorAttr.CHANNEL_ID.value)}',
                name="a2a_get_next")[0] * config.get("emb_size")

    return all2all_args


def get_preprocessed_tensor_for_asc(table, config):
    use_static = ConfigInitializer.get_instance().use_static
    max_lookup_vec_size = None
    if use_static:
        max_lookup_vec_size = config.get("send_count") * config.get("rank_size") if not config.get("is_dp") else (
            config.get("send_count"))

    with tf.compat.v1.variable_scope("restore_vector"):
        restore_vector, hot_pos = get_restore_vector(config)

    with tf.compat.v1.variable_scope("id_offsets"):
        id_offsets, swap_info = get_id_offsets(max_lookup_vec_size, config)

    if not config.get("is_hbm"):
        # 一表多查时，会多次进入get_preprocessed_tensor_for_asc，最后一次大查询替换map的key-value即可
        swap_args = SwapArgs()

        swap_args.set_data(SwapDataType.CONFIG.value, var_name=config.get(ASCAnchorAttr.TABLE_NAME.value),
                           var_channel=config.get(ASCAnchorAttr.CHANNEL_ID.value), config=config, swap_info=swap_info)

    result = {
        'restore_vector': restore_vector,
        'hot_pos': hot_pos,
        'id_offsets': id_offsets,
    }

    if not config.get("is_dp"):
        all2all_args = get_all2all_args(use_static, config)
        result.update({'all2all_args': all2all_args})

    return result

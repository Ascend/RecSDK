#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging

import tensorflow as tf

import mxrec_pybind
from mx_rec.util.initialize import get_use_static
from mx_rec.util.tf_version_adapter import npu_ops


def get_restore_vector(config):
    logging.debug(f'Channel {config.get("table_name")}_restore_{config.get("channel_id")} was built for getnext')
    if config.get("skip_emb_transfer"):
        if not isinstance(config.get("emb_size"), int) or config.get("emb_size") < 1:
            raise TypeError(f"emb_size must be a int")
        if config.get("emb_size") < 1:
            raise ValueError(f"emb_size is less than 1")
        emb_size = config.get("emb_size")
    else:
        if not isinstance(config.get("ext_emb_size"), int) or config.get("ext_emb_size") < 1:
            raise TypeError(f"ext_emb_size must be a int")
        if config.get("ext_emb_size") < 1:
            raise ValueError(f"ext_emb_size is less than 1")
        emb_size = config.get("ext_emb_size")

    use_hot = config.get("use_hot")
    hot_pos = None

    if get_use_static():
        restore_size = config.get("batch_size") * config.get("feat_cnt")
    else:
        restore_size = None

    with tf.compat.v1.variable_scope(config.get("table_name"), reuse=tf.compat.v1.AUTO_REUSE):
        if use_hot:
            device_id = int(config.get("device_id"))
            hot_size = int(mxrec_pybind.get_ub_hot_size(device_id) / emb_size)
            restore_vector, hot_pos = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int32, tf.int32],
                output_shapes=[restore_size, [hot_size]],
                channel_name=f'{config.get("table_name")}_restore_{config.get("channel_id")}'
            )
        else:
            restore_vector = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int32],
                output_shapes=[restore_size],
                channel_name=f'{config.get("table_name")}_restore_{config.get("channel_id")}')[0]

    return restore_vector, hot_pos


def get_id_offsets(max_lookup_vec_size, config):
    logging.debug(f'Channel {config.get("table_name")}_lookup_{config.get("channel_id")} was built for getnext')
    # 自动扩容当前只支持HBM模式，默认没有换入换出
    with tf.compat.v1.variable_scope(config.get("table_name"), reuse=tf.compat.v1.AUTO_REUSE):
        if config.get("use_dynamic_expansion"):
            [id_offsets] = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int64],
                output_shapes=[[max_lookup_vec_size]],
                channel_name=f'{config.get("table_name")}_lookup_{config.get("channel_id")}')
            return id_offsets, [], 0

        [id_offsets] = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32],
            output_shapes=[[max_lookup_vec_size]],
            channel_name=f'{config.get("table_name")}_lookup_{config.get("channel_id")}')
        if config.get("skip_emb_transfer"):
            return id_offsets, [], 0
        swap_pos, swap_len = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32, tf.int32],
            output_shapes=[[max_lookup_vec_size], []],
            channel_name=f'{config.get("table_name")}_swap_{config.get("channel_id")}')
    return id_offsets, swap_pos, swap_len


def get_all2all_args(use_static: bool, config: dict) -> list:
    """
    Get all2all parameters for dynamic condition
    :param use_static: dynamic or static
    :param config: embedding config
    :return: all2all parametrs
    """
    all2all_args = None
    with tf.compat.v1.variable_scope(config.get("table_name"), reuse=tf.compat.v1.AUTO_REUSE):
        if not use_static:
            with tf.compat.v1.variable_scope("all2all"):
                logging.debug(
                    f'Channel {config.get("table_name")}_a2a_{config.get("channel_id")} was built for getnext')
                all2all_args = npu_ops.gen_npu_ops.get_next(
                    output_types=[tf.int64],
                    output_shapes=[[config.get("rank_size"), config.get("rank_size")]],
                    channel_name=f'{config.get("table_name")}_all2all_{config.get("channel_id")}',
                    name="a2a_get_next")[0] * config.get("emb_size")

    return all2all_args


def get_preprocessed_tensor_for_asc(table, config):
    use_static = get_use_static()
    max_lookup_vec_size = None
    if use_static:
        max_lookup_vec_size = config.get("send_count") * config.get("rank_size")

    with tf.compat.v1.variable_scope("restore_vector"):
        restore_vector, hot_pos = get_restore_vector(config)

    with tf.compat.v1.variable_scope("id_offsets"):
        id_offsets, swap_pos, swap_len = get_id_offsets(max_lookup_vec_size, config)

    all2all_args = get_all2all_args(use_static, config)

    if config.get("skip_emb_transfer"):
        swap_in = [tf.no_op()]
    else:
        with tf.compat.v1.variable_scope("h2d_emb"):
            logging.debug(f'Channel {config.get("table_name")}_h2d_{config.get("channel_id")} was built for getnext')
            h2d_emb = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.float32],
                output_shapes=[[max_lookup_vec_size, config.get("ext_emb_size")]],
                channel_name=f'{config.get("table_name")}_h2d_{config.get("channel_id")}')[0]
        logging.debug(f"h2d_emb shape: {h2d_emb}")
        if not isinstance(table, list):
            raise RuntimeError("When enable emb_transfer, optimizer should have slots")
        if use_static:
            swap_pos = swap_pos[0:swap_len]
            h2d_emb = h2d_emb[0:swap_len, :]
        swap_outs = [tf.gather(one_table, swap_pos) for one_table in table]
        swap_out = tf.concat(swap_outs, axis=1)
        logging.debug(
            f'Channel {config.get("table_name")}_d2h_{config.get("channel_id")} was built for op outfeed.')
        swap_out_op = npu_ops.outfeed_enqueue_op(
            channel_name=f'{config.get("table_name")}_d2h_{config.get("channel_id")}', inputs=[swap_out])
        with tf.control_dependencies([swap_out_op]):
            nd_swap_pos = tf.expand_dims(swap_pos, 1)
            table_num = len(table)
            h2d_emb_split = tf.split(h2d_emb, table_num, axis=1)
            swap_in = [tf.compat.v1.scatter_nd_update(table[i], nd_swap_pos, h2d_emb_split[i])
                       for i in range(len(table))]
    result = {
        'restore_vector': restore_vector,
        'hot_pos': hot_pos,
        'id_offsets': id_offsets,
        'swap_in': swap_in,
        'all2all_args': all2all_args,
    }
    return result

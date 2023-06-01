#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging

import tensorflow as tf

from mx_rec.constants.constants import MxRecMode
from mx_rec.util.initialize import get_rank_id, get_device_id, get_rank_size, set_asc_manager, \
    is_asc_manager_initialized, get_train_interval, get_eval_steps, get_prefetch_batch_number, \
    export_table_instances, export_feature_spec, get_if_load, get_training_mode_channel_id, get_use_static, \
    get_use_hot, get_use_dynamic_expansion, export_optimizer, export_dangling_table
from mx_rec.core.asc.helper import find_dangling_table


def check_dangling_table():
    dangling_table = export_dangling_table()
    if not dangling_table:
        dangling_table = find_dangling_table([table_instance.table_name
                                              for _, table_instance in export_table_instances().items()])
    return dangling_table


def generate_table_info_list():
    from mxrec_pybind import EmbInfo
    from mx_rec.constants.constants import ASCEND_TABLE_NAME_MUST_CONTAIN
    # table_name is corresponding to channel_name which is in used in operator gen_npu_ops.get_next
    table_info_list = []

    # check whether DDR is enabled or disabled for all tables.
    host_voc_sizes = [table_instance.host_vocabulary_size for table_instance in export_table_instances().values()]
    total_host_voc_size = sum(host_voc_sizes)
    if total_host_voc_size != 0 and 0 in host_voc_sizes:
        raise ValueError(f"The host-side DDR function of all tables must be used or not used at the same time. "
                         f"However, host voc size of each table is {host_voc_sizes}.")

    optimizer = export_optimizer()
    # generate table info
    dangling_table = check_dangling_table()

    for _, table_instance in export_table_instances().items():
        # When dynamic expansion mode, ext_emb_size is set by optimizer
        if optimizer is not None:
            table_instance.ext_emb_size = table_instance.scalar_emb_size * (1 + optimizer.slot_num)
            logging.debug(f"ext_emb_size is reset to be {table_instance.ext_emb_size} for EmbInfo")

        if table_instance.table_name in dangling_table:
            logging.info(f"Found dangling table: {table_instance.table_name} "
                         f"which does not need to be provided to the EmbInfo.")
            continue

        rec_mode_asc_flag = table_instance.mode == MxRecMode.ASC
        static_shape_rec_flag = rec_mode_asc_flag and get_use_static() and table_instance.send_count > 0
        dynamic_shape_rec_flag = rec_mode_asc_flag and not get_use_static()
        if static_shape_rec_flag or dynamic_shape_rec_flag:
            logging.debug(f"table_instance.slice_device_vocabulary_size: {table_instance.slice_device_vocabulary_size}")
            logging.debug(f"table_instance.slice_host_vocabulary_size: {table_instance.slice_host_vocabulary_size}")
            if table_instance.modify_graph and len(table_instance.channel_name_list) > 1 \
                    and table_instance.slice_host_vocabulary_size > 0:
                raise RuntimeError(f"In the case of modify graph, multiple lookups of a table are currently "
                                   f"only compatible with HBM mode.")
            if len(table_instance.channel_name_list) == 1:
                ids_channel_name = table_instance.channel_name_list[0]
                table_instance.channel_name_list = [table_instance.table_name]
                try:
                    table_instance.send_count_map.pop(ids_channel_name)
                    table_instance.send_count_map[table_instance.table_name] = table_instance.send_count
                except KeyError as error:
                    raise KeyError(f"ids_channel_name '{ids_channel_name}' not in send_count_map "
                                   f"'{table_instance.send_count_map}'") from error
            logging.debug(f"table_instance, table_name: {table_instance.table_name}, channel_name_list: "
                          f"{table_instance.channel_name_list}, send_count_map: {table_instance.send_count_map}")
            table_info = EmbInfo(table_instance.table_name, table_instance.send_count, table_instance.scalar_emb_size,
                                 table_instance.ext_emb_size, table_instance.modify_graph,
                                 table_instance.channel_name_list,
                                 [table_instance.slice_device_vocabulary_size,
                                  table_instance.slice_host_vocabulary_size],
                                 [matched_emb_initializer(table_instance)] +
                                 matched_opt_slot_initializers(table_instance),
                                 table_instance.send_count_map)
            table_info_list.append(table_info)

    return table_info_list


def matched_emb_initializer(tabel_info):
    from mxrec_pybind import InitializeInfo, ConstantInitializerInfo, NormalInitializerInfo
    initializer_case_map = {"tf1/tf2_constant_initializer":
                                isinstance(tabel_info.emb_initializer, tf.keras.initializers.Constant) or
                                isinstance(tabel_info.emb_initializer, tf.constant_initializer),
                            "tf1/tf2_random_normal_initializer":
                                isinstance(tabel_info.emb_initializer, tf.keras.initializers.RandomNormal) or
                                isinstance(tabel_info.emb_initializer, tf.random_normal_initializer),
                            "tf1_truncated_normal_initializer":
                                tf.__version__.startswith("1") and
                                (isinstance(tabel_info.emb_initializer, tf.truncated_normal_initializer) or
                                 isinstance(tabel_info.emb_initializer, tf.keras.initializers.TruncatedNormal)),
                            "tf2_truncated_normal_initializer":
                                tf.__version__.startswith("2") and
                                isinstance(tabel_info.emb_initializer, tf.keras.initializers.TruncatedNormal),
                            }
    if initializer_case_map.get("tf1/tf2_constant_initializer"):
        initializer = InitializeInfo(name="constant_initializer", start=0, len=tabel_info.scalar_emb_size,
                                     constant_initializer_info=ConstantInitializerInfo(
                                         constant_val=tabel_info.emb_initializer.value))
    elif initializer_case_map.get("tf1/tf2_random_normal_initializer"):
        random_seed = 0 if tabel_info.emb_initializer.seed is None else tabel_info.emb_initializer.seed
        initializer = InitializeInfo(name="random_normal_initializer", start=0, len=tabel_info.scalar_emb_size,
                                     normal_initializer_info=NormalInitializerInfo(
                                         mean=tabel_info.emb_initializer.mean,
                                         stddev=tabel_info.emb_initializer.stddev,
                                         seed=random_seed
                                     ))
    elif initializer_case_map.get("tf1_truncated_normal_initializer") or \
            initializer_case_map.get("tf2_truncated_normal_initializer"):
        random_seed = 0 if tabel_info.emb_initializer.seed is None else tabel_info.emb_initializer.seed
        initializer = InitializeInfo(name="truncated_normal_initializer", start=0, len=tabel_info.scalar_emb_size,
                                     normal_initializer_info=NormalInitializerInfo(
                                         mean=tabel_info.emb_initializer.mean,
                                         stddev=tabel_info.emb_initializer.stddev,
                                         seed=random_seed
                                     ))
    else:
        initializer = InitializeInfo(name="truncated_normal_initializer", start=0, len=tabel_info.scalar_emb_size,
                                     normal_initializer_info=NormalInitializerInfo(
                                         mean=0.0,
                                         stddev=1.0,
                                         seed=0
                                     ))
    return initializer


def matched_opt_slot_initializers(table_instance):
    from mxrec_pybind import InitializeInfo, ConstantInitializerInfo

    start_index = table_instance.scalar_emb_size
    slot_initializers = []

    for optimizer in table_instance.optimizer_instance_list:
        for slot_init_value in optimizer.get_slot_init_values():
            slot_initializer = InitializeInfo(name="constant_initializer",
                                              start=start_index,
                                              len=table_instance.scalar_emb_size,
                                              constant_initializer_info=ConstantInitializerInfo(
                                                  constant_val=slot_init_value
                                              ))
            slot_initializers.append(slot_initializer)
            start_index += table_instance.scalar_emb_size

    return slot_initializers


def generate_threshold_list():
    from mxrec_pybind import ThresholdValue
    threshold_list = []

    for _, feature_spec in export_feature_spec().items():
        if feature_spec.eviction_threshold:
            threshold = ThresholdValue(feature_spec.table_name,
                                       feature_spec.access_threshold,
                                       feature_spec.eviction_threshold)
            threshold_list.append(threshold)
            continue
        if feature_spec.access_threshold:
            threshold = ThresholdValue(feature_spec.table_name,
                                       feature_spec.access_threshold,
                                       -1)
            threshold_list.append(threshold)

    return threshold_list


def initialize_emb_cache(table_info_list, threshold_list):
    from mxrec_pybind import HybridMgmt, RankInfo, USE_STATIC, USE_HOT, USE_DYNAMIC_EXPANSION

    rank_id = get_rank_id()
    device_id = get_device_id()
    rank_size = get_rank_size()
    evaluate_stride = get_train_interval()
    eval_steps = get_eval_steps()
    n_batch_to_prefetch = get_prefetch_batch_number()
    if_load = get_if_load()
    option = 0
    if get_use_static():
        option = option | USE_STATIC
    if get_use_hot():
        option = option | USE_HOT
    if get_use_dynamic_expansion():
        option = option | USE_DYNAMIC_EXPANSION

    if get_training_mode_channel_id(is_training=False) == 0:
        rank_info = RankInfo(rank_id, device_id, rank_size, option, n_batch_to_prefetch,
                             [eval_steps, evaluate_stride])
    else:
        rank_info = RankInfo(rank_id, device_id, rank_size, option, n_batch_to_prefetch,
                             [evaluate_stride, eval_steps])

    emb_cache = HybridMgmt()
    if threshold_list:
        emb_cache.initialize(rank_info=rank_info, emb_info=table_info_list, if_load=if_load,
                             threshold_values=threshold_list)
    else:
        emb_cache.initialize(rank_info=rank_info, emb_info=table_info_list, if_load=if_load)

    set_asc_manager(emb_cache)
    logging.info("Preprocessing has been sunk into the host pipeline.")
    logging.debug(f"Flag if load is {if_load}.")
    logging.debug(f"n_batch_to_prefetch is {n_batch_to_prefetch}.")
    logging.debug(f"evaluate_stride is {evaluate_stride}.")
    logging.debug(f"eval_steps is {eval_steps}.")
    logging.debug(f"threshold_values are {threshold_list}.")


def start_asc_pipeline():
    table_info_list = generate_table_info_list()
    threshold_list = generate_threshold_list()
    if not table_info_list:
        logging.warning(f"table_info_list is empty")
    if not is_asc_manager_initialized() and table_info_list:
        initialize_emb_cache(table_info_list, threshold_list)

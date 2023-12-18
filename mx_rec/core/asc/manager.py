#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import tensorflow as tf

from mxrec_pybind import InitializeInfo, ConstantInitializerInfo, NormalInitializerInfo, EmbInfo, EmbInfoParams, \
    ThresholdValue, HybridMgmt, RankInfo, USE_STATIC, USE_HOT, USE_DYNAMIC_EXPANSION

from mx_rec.util.initialize import get_rank_id, get_device_id, get_rank_size, set_asc_manager, \
    is_asc_manager_initialized, get_train_steps, get_eval_steps, get_save_steps, \
    export_table_instances, export_feature_spec, get_if_load, get_use_static, \
    get_use_hot, get_stat_on, get_use_dynamic_expansion, export_optimizer, export_dangling_table, export_table_num
from mx_rec.core.asc.merge_table import find_dangling_table, should_skip
from mx_rec.util.log import logger


def check_dangling_table():
    """
    If the dangling_table list is empty(maybe feature_spec mode), try to find again
    :return: list of dangling_table
    """
    dangling_table = export_dangling_table()
    if not dangling_table:
        dangling_table = find_dangling_table([table_instance.table_name
                                              for _, table_instance in export_table_instances().items()])
    return dangling_table


def generate_table_info_list():
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
            logger.debug("ext_emb_size is reset to be %s for EmbInfo", table_instance.ext_emb_size)
        skip = should_skip(table_instance.table_name)
        if table_instance.table_name in dangling_table or skip:
            logger.info("skip table %s: %s which does not need to be provided to the EmbInfo.",
                        skip, table_instance.table_name)
            continue

        static_shape_rec_flag = get_use_static() and table_instance.send_count > 0
        dynamic_shape_rec_flag = not get_use_static()
        if static_shape_rec_flag or dynamic_shape_rec_flag:
            logger.debug("table_instance.slice_device_vocabulary_size: %s",
                         table_instance.slice_device_vocabulary_size)
            logger.debug("table_instance.slice_host_vocabulary_size: %s", table_instance.slice_host_vocabulary_size)
            logger.debug("table_instance.slice_ssd_vocabulary_size: %s", table_instance.slice_ssd_vocabulary_size)
            logger.debug("EmbInfoParams: The table name is %s, and the value of `is_grad` in this table is %s.",
                         table_instance.table_name, table_instance.is_grad)
            params = EmbInfoParams(table_instance.table_name, table_instance.send_count, table_instance.scalar_emb_size,
                                   table_instance.ext_emb_size, table_instance.is_save, table_instance.is_grad)
            table_info = EmbInfo(params,
                                 [table_instance.slice_device_vocabulary_size,
                                  table_instance.slice_host_vocabulary_size, table_instance.slice_ssd_vocabulary_size],
                                 [matched_emb_initializer(table_instance)] +
                                 matched_opt_slot_initializers(table_instance), table_instance.ssd_data_path)
            table_info_list.append(table_info)

    return table_info_list


def matched_constant_initializer(tabel_info):
    init_param = tabel_info.init_param
    logger.debug("constant_initializer, tabel: %s, initK is %s.", tabel_info.table_name, init_param)
    return InitializeInfo(name="constant_initializer", start=0, len=tabel_info.scalar_emb_size,
                          constant_initializer_info=ConstantInitializerInfo(
                              constant_val=tabel_info.emb_initializer.value, initK=init_param))


def matched_random_normal_initializer(tabel_info):
    random_seed = 0 if tabel_info.emb_initializer.seed is None else tabel_info.emb_initializer.seed
    init_param = tabel_info.init_param
    logger.debug("random_normal_initializer, tabel: %s, initK is %s.", tabel_info.table_name, init_param)
    return InitializeInfo(name="random_normal_initializer", start=0, len=tabel_info.scalar_emb_size,
                          normal_initializer_info=NormalInitializerInfo(
                              mean=tabel_info.emb_initializer.mean,
                              stddev=tabel_info.emb_initializer.stddev,
                              seed=random_seed,
                              initK=init_param
                          ))


def matched_truncated_normal_initializer(tabel_info):
    random_seed = 0 if tabel_info.emb_initializer.seed is None else tabel_info.emb_initializer.seed
    init_param = tabel_info.init_param
    logger.debug("truncated_normal_initializer, tabel: %s, initK is %s.", tabel_info.table_name, init_param)
    return InitializeInfo(name="truncated_normal_initializer", start=0, len=tabel_info.scalar_emb_size,
                          normal_initializer_info=NormalInitializerInfo(
                              mean=tabel_info.emb_initializer.mean,
                              stddev=tabel_info.emb_initializer.stddev,
                              seed=random_seed,
                              initK=init_param
                          ))


def matched_emb_initializer(tabel_info):
    initializer_case_map = {
        "tf1/tf2_constant_initializer":
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
        initializer = matched_constant_initializer(tabel_info)
    elif initializer_case_map.get("tf1/tf2_random_normal_initializer"):
        initializer = matched_random_normal_initializer(tabel_info)
    elif initializer_case_map.get("tf1_truncated_normal_initializer") or \
            initializer_case_map.get("tf2_truncated_normal_initializer"):
        initializer = matched_truncated_normal_initializer(tabel_info)
    else:
        initializer = InitializeInfo(name="truncated_normal_initializer", start=0, len=tabel_info.scalar_emb_size,
                                     normal_initializer_info=NormalInitializerInfo(
                                         mean=0.0,
                                         stddev=1.0,
                                         seed=0
                                     ))
    return initializer


def matched_opt_slot_initializers(table_instance):
    start_index = table_instance.scalar_emb_size
    slot_initializers = []
    logger.debug("matched_opt_slot_initializers, scalar emb size:%s,  optimizer_instance_list size:%s",
                 table_instance.ext_emb_size, len(table_instance.optimizer_instance_list))
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
    threshold_list = []

    for _, feature_spec in export_feature_spec().items():
        coef = 1 if feature_spec.faae_coefficient is None else feature_spec.faae_coefficient
        if feature_spec.eviction_threshold:
            threshold = ThresholdValue(feature_spec.table_name,
                                       feature_spec.access_threshold,
                                       feature_spec.eviction_threshold,
                                       coef,
                                       True)
            threshold_list.append(threshold)
            continue
        if feature_spec.access_threshold:
            threshold = ThresholdValue(feature_spec.table_name,
                                       feature_spec.access_threshold,
                                       -1,
                                       coef,
                                       True)
            threshold_list.append(threshold)

    return threshold_list


def initialize_emb_cache(table_info_list, threshold_list):
    rank_id = get_rank_id()
    device_id = get_device_id()
    rank_size = get_rank_size()
    train_steps = get_train_steps()
    eval_steps = get_eval_steps()
    save_steps = get_save_steps()

    if_load = get_if_load()
    option = 0
    if get_use_static():
        option = option | USE_STATIC
    if get_use_hot():
        option = option | USE_HOT
    if get_use_dynamic_expansion():
        option = option | USE_DYNAMIC_EXPANSION

    # [train_steps, eval_steps, save_steps] pass step information to HybridMgmt for data process loop
    rank_info = RankInfo(rank_id, device_id, rank_size, option, [train_steps, eval_steps, save_steps])

    emb_cache = HybridMgmt()

    is_initialized = emb_cache.initialize(rank_info=rank_info, emb_info=table_info_list, if_load=if_load,
                                          threshold_values=threshold_list)

    if is_initialized is False:
        logger.error("Failed to init emb_cache!")
        raise RuntimeError("emb_cache has not been initialized successfully.")

    set_asc_manager(emb_cache)
    logger.info("Preprocessing has been sunk into the host pipeline.")
    logger.debug("Flag if load is %s.", if_load)
    logger.debug("train_steps is %s.", train_steps)
    logger.debug("eval_steps is %s.", eval_steps)
    logger.debug("threshold_values are %s.", threshold_list)


def start_asc_pipeline():
    table_info_list = generate_table_info_list()
    threshold_list = generate_threshold_list()

    if not table_info_list:
        logger.error("table_info_list is empty!")
        raise RuntimeError("table_info_list is empty!")
    if get_stat_on():
        logger.info("[StatInfo] current_table_num %s", export_table_num())
    if not is_asc_manager_initialized() and table_info_list:
        initialize_emb_cache(table_info_list, threshold_list)

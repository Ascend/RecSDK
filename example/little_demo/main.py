#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os
import warnings
from glob import glob

import tensorflow as tf
from config import Config
from dataset import generate_dataset
from optimizer import create_dense_and_sparse_optimizer
from model import MyModel
from run_mode import RunMode, UseMode

from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
from mx_rec.constants.constants import MxRecMode, ASCEND_TIMESTAMP
from mx_rec.util.initialize import get_rank_id, init, terminate_config_initializer, set_if_load, get_rank_size
from mx_rec.util.variable import get_dense_and_sparse_variable
from mx_rec.constants.constants import ApplyGradientsStrategy

tf.compat.v1.disable_eager_execution()


def make_batch_and_iterator(is_training, feature_spec_list=None,
                            use_timestamp=False, dump_graph=False, batch_number=100):
    dataset = generate_dataset(cfg, use_timestamp=use_timestamp, batch_number=batch_number)
    if not MODIFY_GRAPH_FLAG:
        insert_fn = get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=is_training, dump_graph=dump_graph)
        dataset = dataset.map(insert_fn)
    dataset = dataset.prefetch(100)
    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    return batch, iterator


def model_forward(input_list, batch, is_train, modify_graph, config_dict=None):
    embedding_list = []
    feature_list, hash_table_list, send_count_list = input_list
    for feature, hash_table, send_count in zip(feature_list, hash_table_list, send_count_list):
        access_and_evict_config = None
        if isinstance(config_dict, dict):
            access_and_evict_config = config_dict.get(hash_table.table_name)
        embedding = sparse_lookup(hash_table, feature, send_count, dim=None, is_train=is_train,
                                  access_and_evict_config=access_and_evict_config,
                                  name=hash_table.table_name + "_lookup", modify_graph=modify_graph, batch=batch)

        reduced_embedding = tf.reduce_sum(embedding, axis=1, keepdims=False)
        embedding_list.append(reduced_embedding)

    my_model = MyModel()
    my_model(embedding_list, batch["label_0"], batch["label_1"])
    return my_model


def build_graph(hash_table_list, is_train, feature_spec_list=None, config_dict=None, batch_number=100):
    batch, iterator = make_batch_and_iterator(is_train, feature_spec_list=feature_spec_list,
                                              use_timestamp=USE_TIMESTAMP, dump_graph=is_train,
                                              batch_number=batch_number)
    if MODIFY_GRAPH_FLAG:
        input_list = [[batch["user_ids"], batch["item_ids"]],
                      [hash_table_list[0], hash_table_list[1]],
                      [cfg.user_send_cnt, cfg.item_send_cnt]]
        if use_multi_lookup:
            input_list = [[batch["user_ids"], batch["item_ids"], batch["user_ids"], batch["item_ids"]],
                          [hash_table_list[0], hash_table_list[0], hash_table_list[0], hash_table_list[1]],
                          [cfg.user_send_cnt, cfg.item_send_cnt, cfg.user_send_cnt, cfg.item_send_cnt]]
        if USE_TIMESTAMP:
            tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, batch["timestamp"])
        model = model_forward(input_list, batch,
                              is_train=is_train, modify_graph=True, config_dict=config_dict)
    else:
        input_list = [feature_spec_list,
                      [hash_table_list[0], hash_table_list[1]],
                      [cfg.user_send_cnt, cfg.item_send_cnt]]
        if use_multi_lookup:
            input_list = [feature_spec_list,
                          [hash_table_list[0], hash_table_list[1], hash_table_list[0], hash_table_list[0]],
                          [cfg.user_send_cnt, cfg.item_send_cnt, cfg.user_send_cnt, cfg.item_send_cnt]]
        model = model_forward(input_list, batch,
                              is_train=is_train, modify_graph=False, config_dict=config_dict)

    return iterator, model


def create_feature_spec_list(use_timestamp=False):
    access_threshold = cfg.access_threshold if use_timestamp else None
    eviction_threshold = cfg.eviction_threshold if use_timestamp else None
    feature_spec_list = [FeatureSpec("user_ids", feat_count=cfg.user_feat_cnt, table_name="user_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold),
                         FeatureSpec("item_ids", feat_count=cfg.item_feat_cnt, table_name="item_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold)]
    if use_multi_lookup:
        feature_spec_list.extend([FeatureSpec("user_ids", feat_count=cfg.user_feat_cnt, table_name="user_table",
                                              access_threshold=access_threshold,
                                              eviction_threshold=eviction_threshold),
                                  FeatureSpec("item_ids", feat_count=cfg.item_feat_cnt, table_name="user_table",
                                              access_threshold=access_threshold,
                                              eviction_threshold=eviction_threshold)])
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")

    use_mode = UseMode.mapping(os.getenv("USE_MODE"))
    mode = MxRecMode.mapping(os.getenv("MXREC_MODE"))
    TRAIN_STEPS = 100
    EVAL_INTERVAL = 100
    EVAL_STEPS = 10
    SAVING_INTERVAL = 100

    # get init configuration
    try:
        use_mpi = bool(int(os.getenv("USE_MPI", 1)))
        use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
        use_hot = bool(int(os.getenv("USE_HOT", 0)))
        use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
        use_multi_lookup = bool(int(os.getenv("USE_MULTI_LOOKUP", 1)))
        MODIFY_GRAPH_FLAG = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
        USE_TIMESTAMP = bool(int(os.getenv("USE_TIMESTAMP", 0)))
    except ValueError as err:
        raise ValueError(f"please correctly config USE_MPI or USE_DYNAMIC or USE_HOT or USE_DYNAMIC_EXPANSION or "
                         f"USE_MULTI_LOOKUP or USE_MODIFY_GRAPH or USE_TIMESTAMP only 0 or 1 is supported.") from err

    # nbatch function needs to be used together with the prefetch and host_vocabulary_size != 0
    init(use_mpi=use_mpi,
         train_steps=TRAIN_STEPS,
         eval_steps=EVAL_STEPS,
         prefetch_batch_number=1,
         use_dynamic=use_dynamic,
         use_hot=use_hot,
         use_dynamic_expansion=use_dynamic_expansion)
    IF_LOAD = False
    rank_id = get_rank_id()
    filelist = glob(f"./saved-model/sparse-model-{rank_id}-0")
    if filelist:
        IF_LOAD = True
    set_if_load(IF_LOAD)

    cfg = Config()
    # access_threshold unit counts; eviction_threshold unit seconds
    ACCESS_AND_EVICT = None
    if USE_TIMESTAMP:
        config_for_user_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold)
        config_for_item_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold)
        ACCESS_AND_EVICT = dict(user_table=config_for_user_table, item_table=config_for_item_table)
    train_feature_spec_list = create_feature_spec_list(use_timestamp=USE_TIMESTAMP)
    eval_feature_spec_list = create_feature_spec_list(use_timestamp=USE_TIMESTAMP)

    optimizer_list = [create_dense_and_sparse_optimizer(cfg)]
    sparse_optimizer_list = [sparse_optimizer for dense_optimizer, sparse_optimizer in optimizer_list]

    # 如需验证DDR模式，请按照key数量、batch unique数量合理设置device与host表大小。
    # 验证DDR的配置参考：数据集key总量大于device表，小于device+host；一个batch的unique key数量小于device表。
    user_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.user_hashtable_dim]),
                                  name='user_table',
                                  emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                                  device_vocabulary_size=cfg.user_vocab_size * 10,
                                  host_vocabulary_size=0,
                                  optimizer_list=sparse_optimizer_list,
                                  mode=mode,
                                  all2all_gradients_op="sum_gradients_and_div_by_ranksize",
                                  apply_gradients_strategy = ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY)

    item_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.item_hashtable_dim]),
                                  name='item_table',
                                  emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                                  device_vocabulary_size=cfg.item_vocab_size * 10,
                                  host_vocabulary_size=0,
                                  optimizer_list=sparse_optimizer_list,
                                  mode=mode)

    train_iterator, train_model = build_graph([user_hashtable, item_hashtable], is_train=True,
                                              feature_spec_list=train_feature_spec_list,
                                              config_dict=ACCESS_AND_EVICT,
                                              batch_number=TRAIN_STEPS * get_rank_size())
    eval_iterator, eval_model = build_graph([user_hashtable, item_hashtable], is_train=False,
                                            feature_spec_list=eval_feature_spec_list,
                                            config_dict=ACCESS_AND_EVICT,
                                            batch_number=EVAL_STEPS * get_rank_size())
    dense_variables, sparse_variables = get_dense_and_sparse_variable()

    run_mode = RunMode(
        MODIFY_GRAPH_FLAG, optimizer_list, train_model, eval_model, train_iterator, eval_iterator,
        TRAIN_STEPS, EVAL_STEPS
    )

    # start host pipeline
    if not MODIFY_GRAPH_FLAG:
        start_asc_pipeline()
    # start modify graph
    if MODIFY_GRAPH_FLAG and use_mode != UseMode.TRAIN:
        logging.info("start to modifying graph")
        modify_graph_and_start_emb_cache(dump_graph=True)

    if use_mode == UseMode.TRAIN:
        run_mode.train(EVAL_INTERVAL, SAVING_INTERVAL)
    elif use_mode == UseMode.PREDICT:
        run_mode.predict()

    terminate_config_initializer()
    logging.info("Demo done!")

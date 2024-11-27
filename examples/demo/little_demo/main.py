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

import os
import shutil
import warnings
from glob import glob

import numpy as np
import tensorflow as tf

from mx_rec.constants.constants import ASCEND_TIMESTAMP
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
from mx_rec.util.communication.hccl_ops import get_rank_size
from mx_rec.util.initialize import init, terminate_config_initializer
from mx_rec.util.variable import get_dense_and_sparse_variable

from config import (GLOBAL_RANDOM_SEED, MODIFY_GRAPH_FLAG, MULTI_LOOKUP_TIMES,
                    PRECISION_CHECK, USE_DETERMINISTIC, USE_DYNAMIC,
                    USE_DYNAMIC_EXPANSION, USE_MULTI_LOOKUP, USE_ONE_SHOT,
                    USE_TIMESTAMP, USE_DP, USE_TUPLE_DATA_FORMAT, USE_PADDING_KEYS, Config, CacheModeEnum)
from dataset import generate_dataset, generate_tuple_data_format_func
from demo_logger import logger
from model import MyModel
from optimizer import create_dense_and_sparse_optimizer
from run_mode import RunMode, UseMode
from utils import GLOBAL_RANK_SIZE, PRECISION_DUMP_STEP, PrecisionDumpInfo

tf.compat.v1.disable_eager_execution()

_SSD_SAVE_PATH = ["ssd_data"]  # user should make sure directory exist and clean before training


def make_batch_and_iterator(is_training, feature_spec_list=None,
                            use_timestamp=False, dump_graph=False, batch_number=100):
    dataset = generate_dataset(cfg, use_timestamp=use_timestamp, batch_number=batch_number)
    if USE_TUPLE_DATA_FORMAT:
        dataset = dataset.map(generate_tuple_data_format_func)
    if not MODIFY_GRAPH_FLAG:
        insert_fn = get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=is_training, dump_graph=dump_graph)
        dataset = dataset.map(insert_fn)
    dataset = dataset.prefetch(100)
    if USE_ONE_SHOT:
        iterator = dataset.make_one_shot_iterator()
    else:
        iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    return batch, iterator


def model_forward(input_list, batch, is_train, modify_graph, config_dict=None):
    embedding_list = []
    feature_list, hash_table_list, send_count_list, is_grad_list, dim_list = input_list
    for feature, hash_table, send_count, is_grad, dim in zip(feature_list, hash_table_list, send_count_list,
                                                             is_grad_list, dim_list):
        access_and_evict_config = None
        if isinstance(config_dict, dict):
            access_and_evict_config = config_dict.get(hash_table.table_name)

        # The modify graph mode does not require a batch to be passed in.
        batch_param = None
        if not MODIFY_GRAPH_FLAG:
            batch_param = batch

        embedding = sparse_lookup(hash_table, feature, send_count, is_train=is_train,
                                  access_and_evict_config=access_and_evict_config, is_grad=is_grad,
                                  name=hash_table.table_name + "_lookup", modify_graph=modify_graph, batch=batch_param,
                                  serving_default_value=tf.ones(shape=(dim), dtype=tf.float32) * 2)

        reduced_embedding = tf.reduce_sum(embedding, axis=1, keepdims=False)
        embedding_list.append(reduced_embedding)

    my_model = MyModel()
    if USE_TUPLE_DATA_FORMAT:
        label_0 = batch[1]["label_0"]
        label_1 = batch[1]["label_1"]
    else:
        label_0 = batch["label_0"]
        label_1 = batch["label_1"]
    my_model(embedding_list, label_0, label_1)
    return my_model


def build_graph(hash_table_list, is_train, feature_spec_list=None, config_dict=None, batch_number=100):
    batch, iterator = make_batch_and_iterator(is_train, feature_spec_list=feature_spec_list,
                                              use_timestamp=USE_TIMESTAMP, dump_graph=is_train,
                                              batch_number=batch_number)
    if MODIFY_GRAPH_FLAG:
        if USE_TUPLE_DATA_FORMAT:
            user_ids = batch[0]["user_ids"]
            item_ids = batch[0]["item_ids"]
        else:
            user_ids = batch["user_ids"]
            item_ids = batch["item_ids"]

        input_list = [[user_ids, item_ids],
                      [hash_table_list[0], hash_table_list[1]],
                      [cfg.user_send_cnt, cfg.item_send_cnt],
                      [True, True],
                      [cfg.user_hashtable_dim, cfg.item_hashtable_dim]]
        if USE_MULTI_LOOKUP:
            # add `MULTI_LOOKUP_TIMES` times
            for i, _ in enumerate(input_list):
                input_list[i].extend([input_list[i][0]] * MULTI_LOOKUP_TIMES)
        if USE_TIMESTAMP:
            if not USE_TUPLE_DATA_FORMAT:
                tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, batch["timestamp"])
            else:
                tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, batch[0]["timestamp"])
        model = model_forward(input_list, batch,
                              is_train=is_train, modify_graph=True, config_dict=config_dict)
    else:
        input_list = [feature_spec_list,
                      [hash_table_list[0], hash_table_list[1]],
                      [cfg.user_send_cnt, cfg.item_send_cnt],
                      [True, True],
                      [cfg.user_hashtable_dim, cfg.item_hashtable_dim]]
        if USE_MULTI_LOOKUP:
            # add `MULTI_LOOKUP_TIMES` times
            for i, _ in enumerate(input_list):
                if i == 0:
                    continue
                input_list[i].extend([input_list[i][0]] * MULTI_LOOKUP_TIMES)

        model = model_forward(input_list, batch,
                              is_train=is_train, modify_graph=False, config_dict=config_dict)

    return iterator, model, batch


def create_feature_spec_list(use_timestamp=False):
    access_threshold = cfg.access_threshold if use_timestamp else None
    eviction_threshold = cfg.eviction_threshold if use_timestamp else None
    feature_spec_list = [FeatureSpec("user_ids", table_name="user_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold,
                                     faae_coefficient=1),
                         FeatureSpec("item_ids", table_name="item_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold,
                                     faae_coefficient=4)]
    if USE_MULTI_LOOKUP:
        # add `MULTI_LOOKUP_TIMES` times
        for _ in range(MULTI_LOOKUP_TIMES):
            feature_spec_list.append(FeatureSpec("user_ids", table_name="user_table",
                                                 access_threshold=access_threshold,
                                                 eviction_threshold=eviction_threshold,
                                                 faae_coefficient=1))
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list


def _del_related_dir(del_path: str) -> None:
    if not os.path.isabs(del_path):
        del_path = os.path.join(os.getcwd(), del_path)
    dirs = glob(del_path)
    for sub_dir in dirs:
        shutil.rmtree(sub_dir, ignore_errors=True)
        logger.info(f"delete dir:{sub_dir}")


def _clear_saved_model() -> None:
    _del_related_dir("/root/ascend/log/*")
    _del_related_dir("kernel*")
    _del_related_dir("export_graph")

    mode = UseMode.mapping(os.getenv("USE_MODE"))
    if mode != UseMode.TRAIN:
        return
    logger.info("current mode is train, will delete previous saved model data if exist.")
    _del_related_dir("saved-model")

    if not (os.getenv("CACHE_MODE", "") == CacheModeEnum.SSD.value):
        return
    logger.info("current cache mode is SSD, and file overwrite is not allowed in SSD mode, deleting exist directory"
                " then create empty directory for this use case.")
    for sub_path in _SSD_SAVE_PATH:
        _del_related_dir(sub_path)
        os.makedirs(sub_path, mode=0o550, exist_ok=True)
        logger.info(f"Create dir:{sub_path}")


def index_initializer(shape, dtype=None, partition_info=None):
    # shape 是一个元组，表示张量的形状，例如 (rows, cols)
    rows, cols = shape
    # 创建一个与shape相同大小的列表，用于存储初始化值
    values = [[i * 1e06 + j * 1e-20 for j in range(cols)] for i in range(rows)]
    # 将列表转换为numpy数组
    return tf.constant(values, dtype=dtype)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")
    _clear_saved_model()

    use_mode = UseMode.mapping(os.getenv("USE_MODE"))
    # 最大数据集生成数量
    MAX_DATASET_GENERATE_TRAIN = 200
    MAX_DATASET_GENERATE_EVAL = 10
    # 最大训练的步数
    MAX_TRAIN_STEPS = 200
    # 训练多少步切换为评估
    TRAIN_STEPS = 100
    # 评估多少步切换为训练
    EVAL_STEPS = 10
    # 训练多少步进行保存
    SAVING_INTERVAL = 100

    task_config = {"use_dynamic": USE_DYNAMIC, "use_dynamic_expansion": USE_DYNAMIC_EXPANSION,
                   "use_multi_lookup": USE_MULTI_LOOKUP, "modify_graph_flag": MODIFY_GRAPH_FLAG,
                   "use_timestamp": USE_TIMESTAMP, "use_one_shot": USE_ONE_SHOT,
                   "use_deterministic": USE_DETERMINISTIC, "multi_lookup_times": MULTI_LOOKUP_TIMES,
                   "use_dp": USE_DP, "use_tuple_data_format": USE_TUPLE_DATA_FORMAT}
    if PRECISION_CHECK:
        task_config["precision_dump_step"] = PRECISION_DUMP_STEP
        task_config["global_rank_size"] = GLOBAL_RANK_SIZE
        PrecisionDumpInfo.add_item(key="task_config", value=task_config)

    if USE_DETERMINISTIC:
        np.random.seed(GLOBAL_RANDOM_SEED)
        tf.compat.v2.random.set_seed(GLOBAL_RANDOM_SEED)

    if_load = False
    save_path = "./saved-model"
    model_file = []
    if use_mode in [UseMode.PREDICT, UseMode.LOAD_AND_TRAIN]:
        load_path_pattern = os.path.join(save_path, "sparse-model-*")
        model_file = glob(load_path_pattern)
        if len(model_file) == 0:
            raise ValueError(f"get USE_MODE:{use_mode}, but no model file exist at:{load_path_pattern}")
        if_load = True

    # nbatch function needs to be used together with the prefetch and host_vocabulary_size != 0
    init(train_steps=TRAIN_STEPS,
         eval_steps=EVAL_STEPS,
         save_steps=SAVING_INTERVAL,
         max_steps=MAX_TRAIN_STEPS,
         use_dynamic=USE_DYNAMIC,
         use_dynamic_expansion=USE_DYNAMIC_EXPANSION,
         if_load=if_load)

    cfg = Config()
    # multi lookup config, batch size: 32 * 128 = 4096
    if USE_MULTI_LOOKUP and MULTI_LOOKUP_TIMES > 2:
        cfg.batch_size = 32

    # access_threshold unit counts; eviction_threshold unit seconds
    ACCESS_AND_EVICT = None
    if USE_TIMESTAMP:
        config_for_user_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold,
                                     faae_coefficient=1)
        config_for_item_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold,
                                     faae_coefficient=4)
        ACCESS_AND_EVICT = dict(user_table=config_for_user_table, item_table=config_for_item_table)

    train_feature_spec_list = None
    eval_feature_spec_list = None

    if not MODIFY_GRAPH_FLAG:
        train_feature_spec_list = create_feature_spec_list(use_timestamp=USE_TIMESTAMP)
        eval_feature_spec_list = create_feature_spec_list(use_timestamp=USE_TIMESTAMP)

    optimizer_list = [create_dense_and_sparse_optimizer(cfg)]

    # 如需验证DDR模式，请按照key数量、batch unique数量合理设置device与host表大小。
    # 验证DDR的配置参考：建议跑dynamic避免调参。数据集key总量大于device表，小于device+host；一个batch的unique key数量小于device表。
    # 验证SSD的配置参考：建议跑dynamic避免调参。数据集key总量大于device+host；一个batch的unique key数量小于device表。
    hbm_test_cfg = {"device_vocabulary_size": cfg.user_vocab_size, "host_vocabulary_size": 0}
    ddr_test_cfg = {"device_vocabulary_size": int(cfg.user_vocab_size * 0.4),
                    "host_vocabulary_size": int(cfg.user_vocab_size * 1.0)}
    ssd_test_cfg = {
        "device_vocabulary_size": int(cfg.user_vocab_size * 0.4),
        "host_vocabulary_size": int(cfg.user_vocab_size * 0.8),
        "ssd_vocabulary_size": int(cfg.user_vocab_size * 1.8), "ssd_data_path": _SSD_SAVE_PATH
    }
    cache_mode_dict = {CacheModeEnum.HBM.value: hbm_test_cfg, CacheModeEnum.DDR.value: ddr_test_cfg,
                       CacheModeEnum.SSD.value: ssd_test_cfg}

    cache_mode = os.getenv("CACHE_MODE")
    if cache_mode not in cache_mode_dict.keys():
        raise ValueError(f"cache mode must in {list(cache_mode_dict.keys())}, get:{cache_mode}")
    if cache_mode in [CacheModeEnum.DDR.value, CacheModeEnum.SSD.value] and not USE_DYNAMIC:
        logger.warning("when cache_mode in [DDR, SSD], suggest use_dynamic=true to avoid tuning size parameter")

    emb_initializer = tf.compat.v1.constant_initializer(0.1) if USE_DETERMINISTIC or PRECISION_CHECK \
        else tf.compat.v1.truncated_normal_initializer()
    user_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.user_hashtable_dim]),
                                  name='user_table',
                                  emb_initializer=emb_initializer,
                                  all2all_gradients_op="sum_gradients_and_div_by_ranksize",
                                  is_dp=USE_DP,
                                  **cache_mode_dict[cache_mode])

    padding_keys_config = {}
    if USE_PADDING_KEYS:
        padding_keys_config = {"padding_keys": cfg.padding_keys,
                               "padding_keys_mask": True,
                               "padding_keys_len": cfg.batch_size * cfg.item_feat_cnt}
    item_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.item_hashtable_dim]),
                                  name='item_table',
                                  emb_initializer=emb_initializer,
                                  padding_keys=padding_keys_config.get("padding_keys", None),
                                  padding_keys_mask=padding_keys_config.get("padding_keys_mask", False),
                                  padding_keys_len=padding_keys_config.get("padding_keys_len", None),
                                  **cache_mode_dict[cache_mode])

    # 在predict的场景下，train model不需要被执行
    train_iterator = None
    train_model = None
    train_batch = None
    table_list = [user_hashtable, item_hashtable]

    if use_mode in [UseMode.TRAIN, UseMode.LOAD_AND_TRAIN]:
        train_iterator, train_model, train_batch = build_graph(
            table_list, is_train=True,
            feature_spec_list=train_feature_spec_list,
            config_dict=ACCESS_AND_EVICT,
            batch_number=MAX_DATASET_GENERATE_TRAIN * get_rank_size()
        )

    eval_iterator, eval_model, eval_batch = build_graph(table_list, is_train=False,
                                                        feature_spec_list=eval_feature_spec_list,
                                                        config_dict=ACCESS_AND_EVICT,
                                                        batch_number=MAX_DATASET_GENERATE_EVAL * get_rank_size())
    dense_variables, sparse_variables = get_dense_and_sparse_variable()

    params = {"train_batch": train_batch, "eval_batch": eval_batch, "use_one_shot": USE_ONE_SHOT}

    run_mode = RunMode(
        MODIFY_GRAPH_FLAG, USE_TIMESTAMP, table_list, optimizer_list, train_model, eval_model, train_iterator,
        eval_iterator, MAX_TRAIN_STEPS, EVAL_STEPS, params
    )

    # start host pipeline
    if not MODIFY_GRAPH_FLAG:
        start_asc_pipeline()

    # start modify graph
    if MODIFY_GRAPH_FLAG and use_mode not in [UseMode.TRAIN, UseMode.LOAD_AND_TRAIN]:
        logger.info("start to modifying graph")
        modify_graph_and_start_emb_cache(dump_graph=True)

    if use_mode in [UseMode.TRAIN, UseMode.LOAD_AND_TRAIN]:
        run_mode.train(TRAIN_STEPS, SAVING_INTERVAL, if_load, model_file)
    elif use_mode == UseMode.PREDICT:
        run_mode.predict(model_file)

    terminate_config_initializer()
    logger.info("Demo done!")

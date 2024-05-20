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

import enum
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
from mx_rec.util.log import logger
from mx_rec.util.variable import get_dense_and_sparse_variable

from config import Config
from dataset import generate_dataset
from model import MyModel
from optimizer import create_dense_and_sparse_optimizer
from run_mode import RunMode, UseMode

tf.compat.v1.disable_eager_execution()

_SSD_SAVE_PATH = ["ssd_data"]  # user should make sure directory exist and clean before training


class CacheModeEnum(enum.Enum):
    HBM = "HBM"
    DDR = "DDR"
    SSD = "SSD"


def make_batch_and_iterator(is_training, feature_spec_list=None,
                            use_timestamp=False, dump_graph=False, batch_number=100):
    dataset = generate_dataset(cfg, use_timestamp=use_timestamp, batch_number=batch_number)
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
        embedding = sparse_lookup(hash_table, feature, send_count, is_train=is_train,
                                  access_and_evict_config=access_and_evict_config, is_grad=is_grad,
                                  name=hash_table.table_name + "_lookup", modify_graph=modify_graph, batch=batch,
                                  serving_default_value=tf.ones(shape=(dim), dtype=tf.float32) * 2)

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
                      [cfg.user_send_cnt, cfg.item_send_cnt],
                      [True, True],
                      [cfg.user_hashtable_dim, cfg.item_hashtable_dim]]
        if use_multi_lookup:
            # add `MULTI_LOOKUP_TIMES` times
            for i, _ in enumerate(input_list):
                input_list[i].extend([input_list[i][0]] * MULTI_LOOKUP_TIMES)
        if USE_TIMESTAMP:
            tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, batch["timestamp"])
        model = model_forward(input_list, batch,
                              is_train=is_train, modify_graph=True, config_dict=config_dict)
    else:
        input_list = [feature_spec_list,
                      [hash_table_list[0], hash_table_list[1]],
                      [cfg.user_send_cnt, cfg.item_send_cnt],
                      [True, True],
                      [cfg.user_hashtable_dim, cfg.item_hashtable_dim]]
        if use_multi_lookup:
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
    if use_multi_lookup:
        # add `MULTI_LOOKUP_TIMES` times
        for _ in range(MULTI_LOOKUP_TIMES):
            feature_spec_list.append(FeatureSpec("user_ids", table_name="user_table",
                                                 access_threshold=access_threshold,
                                                 eviction_threshold=eviction_threshold,
                                                 faae_coefficient=1))
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list


def clear_saved_model():
    mode = UseMode.mapping(os.getenv("USE_MODE"))
    if mode == UseMode.TRAIN:
        logger.info("current mode is train, will delete previous saved model data if exist.")
        save_model_path = os.path.join(os.getcwd(), "saved-model")
        shutil.rmtree(save_model_path, ignore_errors=True)
    if not (os.getenv("CACHE_MODE", "") == CacheModeEnum.SSD.value and mode == UseMode.TRAIN):
        return

    # ssd not allow overwrite file, should clear it before training
    logger.info("current cache mode is SSD, will delete previous saved ssd data if exist.")
    for part_path in _SSD_SAVE_PATH:
        if "/" not in part_path and "\\" not in part_path:
            part_path = os.path.join(os.getcwd(), part_path)
        shutil.rmtree(part_path, ignore_errors=True)
        try:
            os.mkdir(part_path)
        except OSError:
            logger.warning("ssd path has exist")  # 多进程并行，忽略异常


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")

    use_mode = UseMode.mapping(os.getenv("USE_MODE"))
    # 最大数据集生成数量
    MAX_DATASET_GENERATE = 200
    # 最大训练的步数
    MAX_TRAIN_STEPS = 200
    # 训练多少步切换为评估
    TRAIN_STEPS = 100
    # 评估多少步切换为训练
    EVAL_STEPS = 10
    # 训练多少步进行保存
    SAVING_INTERVAL = 100

    # get init configuration
    try:
        use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
        use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
        use_multi_lookup = bool(int(os.getenv("USE_MULTI_LOOKUP", 1)))
        MODIFY_GRAPH_FLAG = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
        USE_TIMESTAMP = bool(int(os.getenv("USE_TIMESTAMP", 0)))
        USE_ONE_SHOT = bool(int(os.getenv("USE_ONE_SHOT", 0)))
        USE_DETERMINISTIC = bool(int(os.getenv("USE_DETERMINISTIC", 0)))
    except ValueError as err:
        raise ValueError("please correctly config USE_MPI or USE_DYNAMIC or USE_DYNAMIC_EXPANSION or "
                         "USE_MULTI_LOOKUP or USE_MODIFY_GRAPH or USE_TIMESTAMP or USE_ONE_SHOT or USE_DETERMINISTIC"
                         "only 0 or 1 is supported.") from err

    try:
        MULTI_LOOKUP_TIMES = int(os.getenv("MULTI_LOOKUP_TIMES", 2))
    except ValueError as err:
        raise ValueError("please correctly config MULTI_LOOKUP_TIMES only int is supported.") from err

    if USE_DETERMINISTIC:
        np.random.seed(128)
        tf.random.set_random_seed(128)

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
         use_dynamic=use_dynamic,
         use_dynamic_expansion=use_dynamic_expansion,
         if_load=if_load)

    cfg = Config()
    # multi lookup config, batch size: 32 * 128 = 4096
    if use_multi_lookup and MULTI_LOOKUP_TIMES > 2:
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
    ddr_test_cfg = {"device_vocabulary_size": int(cfg.user_vocab_size * 0.2),
                    "host_vocabulary_size": int(cfg.user_vocab_size * 0.8)}
    ssd_test_cfg = {
        "device_vocabulary_size": int(cfg.user_vocab_size * 0.1),
        "host_vocabulary_size": int(cfg.user_vocab_size * 0.1),
        "ssd_vocabulary_size": int(cfg.user_vocab_size * 0.8), "ssd_data_path": _SSD_SAVE_PATH
    }
    cache_mode_dict = {CacheModeEnum.HBM.value: hbm_test_cfg, CacheModeEnum.DDR.value: ddr_test_cfg,
                       CacheModeEnum.SSD.value: ssd_test_cfg}

    cache_mode = os.getenv("CACHE_MODE")
    if cache_mode not in cache_mode_dict.keys():
        raise ValueError(f"cache mode must in {list(cache_mode_dict.keys())}, get:{cache_mode}")
    if cache_mode in ["DDR", "SSD"] and not use_dynamic:
        logger.warning("when cache_mode in [DDR, SSD], suggest use_dynamic=true to avoid tuning size parameter")
    emb_initializer = tf.compat.v1.constant_initializer(0) if USE_DETERMINISTIC \
                      else tf.compat.v1.truncated_normal_initializer()
    user_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.user_hashtable_dim]),
                                  name='user_table',
                                  emb_initializer=emb_initializer,
                                  all2all_gradients_op="sum_gradients_and_div_by_ranksize",
                                  **cache_mode_dict[cache_mode])

    item_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.item_hashtable_dim]),
                                  name='item_table',
                                  emb_initializer=emb_initializer,
                                  **cache_mode_dict[cache_mode])

    # 在predict的场景下，train model不需要被执行
    train_iterator = None
    train_model = None
    train_batch = None
    table_list = [user_hashtable, item_hashtable]
    if use_mode in [UseMode.TRAIN, UseMode.LOAD_AND_TRAIN]:
        train_iterator, train_model, train_batch = build_graph(table_list, is_train=True,
                                                               feature_spec_list=train_feature_spec_list,
                                                               config_dict=ACCESS_AND_EVICT,
                                                               batch_number=MAX_DATASET_GENERATE * get_rank_size())
    eval_iterator, eval_model, eval_batch = build_graph(table_list, is_train=False,
                                                        feature_spec_list=eval_feature_spec_list,
                                                        config_dict=ACCESS_AND_EVICT,
                                                        batch_number=MAX_DATASET_GENERATE * get_rank_size())
    dense_variables, sparse_variables = get_dense_and_sparse_variable()

    params = {"train_batch": train_batch, "eval_batch": eval_batch, "use_one_shot": USE_ONE_SHOT,
              "use_deterministic": USE_DETERMINISTIC}
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

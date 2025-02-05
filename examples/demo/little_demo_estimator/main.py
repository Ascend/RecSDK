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

import argparse
import os
import shutil
from glob import glob

import tensorflow as tf
from mx_rec.util.initialize import init, terminate_config_initializer
from mx_rec.core.asc.helper import FeatureSpec
from mx_rec.graph.modifier import GraphModifierHook
from mx_rec.graph.hooks import OrphanLookupKeySlicerHook, LookupSubgraphSlicerHook
from mx_rec.core.feature_process import EvictHook

from tf_adapter import NPURunConfig, NPUEstimator, npu_hooks_append
from nn_reader import input_fn
from nn_model_input import get_model_fn
from config import (
    Config, USE_DETERMINISTIC, GLOBAL_RANDOM_SEED, USE_DYNAMIC, USE_DYNAMIC_EXPANSION,
    USE_MULTI_LOOKUP, USE_MODIFY_GRAPH, USE_TIMESTAMP, USE_DP, USE_ONE_SHOT, MULTI_LOOKUP_TIMES,
    ENABLE_SLICER_TEST, RUN_MODE, USE_EXPORT_SAVED_MODEL
)
from demo_logger import logger
from utils import FeatureSpecIns

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def set_seed():
    import random
    import numpy as np

    random.seed(GLOBAL_RANDOM_SEED)
    np.random.seed(GLOBAL_RANDOM_SEED)
    tf.compat.v2.random.set_seed(GLOBAL_RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_RANDOM_SEED)


def main(params, config: Config):
    mg_session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    run_config = NPURunConfig(
        model_dir=params.model_dir,
        save_summary_steps=1000,  # tf.summary运行周期
        save_checkpoints_steps=params.save_checkpoints_steps,
        keep_checkpoint_max=5,
        session_config=mg_session_config,
        log_step_count_steps=1000,  # tf.logging运行周期
        precision_mode='allow_mix_precision',
        enable_data_pre_proc=True,
        iterations_per_loop=1,
        op_precision_mode='./op_precision.ini',  # high performance
        op_compiler_cache_mode="enable",
        op_compiler_cache_dir="./op_cache",
        HCCL_algorithm="level0:pairwise;level1:pairwise",
        tf_random_seed=GLOBAL_RANDOM_SEED if USE_DETERMINISTIC else None,
        deterministic=1 if USE_DETERMINISTIC else 0
    )

    # access_threshold unit counts; eviction_threshold unit seconds
    access_and_evict = None

    if not ENABLE_SLICER_TEST:
        hooks_list = [GraphModifierHook(modify_graph=USE_MODIFY_GRAPH)]
    else:
        orphan_slicer_hook = OrphanLookupKeySlicerHook()
        lookup_slicer_hook = LookupSubgraphSlicerHook(op_types=["StringToNumber"])
        hooks_list = [orphan_slicer_hook, lookup_slicer_hook, GraphModifierHook(modify_graph=USE_MODIFY_GRAPH)]

    if USE_TIMESTAMP:
        config_for_user_table = dict(access_threshold=config.access_threshold,
                                     eviction_threshold=config.eviction_threshold)
        config_for_item_table = dict(access_threshold=config.access_threshold,
                                     eviction_threshold=config.eviction_threshold)
        access_and_evict = dict(user_table=config_for_user_table, item_table=config_for_item_table)
        evict_hook = EvictHook(evict_enable=True, evict_time_interval=10)
        hooks_list.append(evict_hook)

    est = NPUEstimator(
        model_fn=get_model_fn(config, access_and_evict),
        params=params,
        model_dir=params.model_dir,
        config=run_config
    )

    if RUN_MODE == 'train':
        est.train(input_fn=lambda: input_fn(params, config), max_steps=params.max_steps,
                  hooks=npu_hooks_append(hooks_list))

    elif RUN_MODE == 'train_and_evaluate':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(params, config, use_one_shot=USE_ONE_SHOT),
                                            max_steps=params.max_steps, hooks=npu_hooks_append(hooks_list))

        if not ENABLE_SLICER_TEST:
            # 在开启evict时，eval时不支持淘汰，所以无需加入evict hook
            eval_hook_list = [GraphModifierHook(modify_graph=USE_MODIFY_GRAPH)]
        else:
            orphan_slicer_hook = OrphanLookupKeySlicerHook()
            lookup_slicer_hook = LookupSubgraphSlicerHook(op_types=["StringToNumber"])
            eval_hook_list = [orphan_slicer_hook, lookup_slicer_hook,
                              GraphModifierHook(modify_graph=USE_MODIFY_GRAPH)]

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(params, config, is_eval=True,
                                                                    use_one_shot=USE_ONE_SHOT),
                                          steps=params.eval_steps, hooks=npu_hooks_append(eval_hook_list),
                                          throttle_secs=0)
        tf.estimator.train_and_evaluate(est, train_spec=train_spec, eval_spec=eval_spec)

        if USE_EXPORT_SAVED_MODEL:
            _export_model("./model_path", config, est)

    elif RUN_MODE == 'predict':
        results = est.predict(input_fn=lambda: input_fn(params, config),
                              hooks=npu_hooks_append(hooks_list=hooks_list), yield_single_examples=False)
        output_pred1 = []
        output_pred2 = []
        labels = []

        for res in results:
            output_pred1.append(res['task_1'][0])
            output_pred2.append(res['task_2'][0])
            labels.append(res['label'][0])
    else:
        raise ValueError(f"RUN_MODE not in [train, predict, train_and_evaluate]")

    terminate_config_initializer()
    logger.info("Demo done!")


def _export_model(save_path: str, config: Config, est: tf.compat.v1.estimator.Estimator):
    _del_related_dir(save_path)

    def _serving_input_fn():
        inputs = {
            "user_ids": tf.compat.v1.placeholder(shape=(None, config.user_feat_cnt), dtype=tf.int64, name="user_ids"),
            "item_ids": tf.compat.v1.placeholder(shape=(None, config.item_feat_cnt), dtype=tf.int64, name="item_ids"),
            "label_0": tf.compat.v1.placeholder(shape=(None,), dtype=tf.float32, name="label_0"),
            "label_1": tf.compat.v1.placeholder(shape=(None,), dtype=tf.float32, name="label_1"),
        }
        return tf.estimator.export.ServingInputReceiver(features=inputs, receiver_tensors=inputs)

    target_pb_path = os.path.abspath(save_path)
    export_path = est.export_saved_model(target_pb_path, _serving_input_fn).decode("utf-8")
    logger.info("The export saved model path is %s.", export_path)


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

    if not USE_EXPORT_SAVED_MODEL and not RUN_MODE.startswith("train"):
        return
    logger.warning("Current mode contains train, will delete previous saved model data if exist.")
    _del_related_dir("_rank*")
    _del_related_dir("ssd_sparse_model_rank*")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, default='_rank')
    parser.add_argument('--learning_rate', type=float, default=0.0008)
    parser.add_argument('--max_data_generate_steps', type=int, default=200)  # 生成数据最大步数
    parser.add_argument('--max_steps', type=int, default=200)  # train的最大步数
    parser.add_argument('--train_steps', type=int, default=100)  # 训练train_steps步后进行eval
    parser.add_argument('--eval_steps', type=int, default=10)  # 每次eval的步数
    # 每隔step保存一次模型, 若在train_and_evaluate模式, 还会进行eval, 注: 若设为None, NPURunConfig内部会设默认值100
    parser.add_argument('--save_checkpoints_steps', type=int, default=200)
    args, unknowns = parser.parse_known_args()

    if RUN_MODE == 'train':
        args.train_steps = -1
        args.eval_steps = -1
    elif RUN_MODE == 'predict':
        args.eval_steps = -1
    elif RUN_MODE == 'train_and_evaluate':
        args.save_checkpoints_steps = args.train_steps
    else:
        raise ValueError(f"RUN_MODE not in [train, predict, train_and_evaluate]")

    _clear_saved_model()

    # set init
    if USE_DETERMINISTIC:
        set_seed()
    init(train_steps=args.train_steps,
         eval_steps=args.eval_steps,
         save_steps=args.save_checkpoints_steps,
         max_steps=args.max_steps,
         use_dynamic=USE_DYNAMIC,
         use_dynamic_expansion=USE_DYNAMIC_EXPANSION)

    cfg = Config()
    # multi lookup config, batch size: 32 * 128 = 4096
    if USE_MULTI_LOOKUP and MULTI_LOOKUP_TIMES > 2:
        cfg.batch_size = 32
    # init FeatureSpecIns
    FeatureSpecIns.set_instance()
    main(args, cfg)

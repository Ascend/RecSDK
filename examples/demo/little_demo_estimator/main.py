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

import tensorflow as tf

from mx_rec.util.initialize import init, terminate_config_initializer
from mx_rec.util.communication.hccl_ops import get_rank_id
from mx_rec.core.asc.helper import FeatureSpec
from mx_rec.graph.modifier import GraphModifierHook
from mx_rec.graph.acg_push_ops import ACGPushOpsToDatasetHook
from mx_rec.core.feature_process import EvictHook
from mx_rec.util.log import logger

from tf_adapter import NPURunConfig, NPUEstimator, npu_hooks_append, DumpConfig
from nn_reader import input_fn
from nn_model_input import get_model_fn
from config import Config
from utils import FeatureSpecIns

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def main(params, cfg):
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
        HCCL_algorithm="level0:pairwise;level1:pairwise"
    )

    # access_threshold unit counts; eviction_threshold unit seconds
    access_and_evict = None

    if not params.enable_push_ops_test:
        hooks_list = [GraphModifierHook(modify_graph=params.modify_graph)]
    else:
        hooks_list = [ACGPushOpsToDatasetHook(dump_graph=True), GraphModifierHook(modify_graph=params.modify_graph)]

    if params.use_timestamp:
        config_for_user_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold)
        config_for_item_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold)
        access_and_evict = dict(user_table=config_for_user_table, item_table=config_for_item_table)


        evict_hook = EvictHook(evict_enable=True, evict_time_interval=10)
        hooks_list.append(evict_hook)
    create_fs_params = dict(cfg=cfg, use_timestamp=params.use_timestamp,
                            use_multi_lookup=use_multi_lookup, multi_lookup_times=MULTI_LOOKUP_TIMES)
    est = NPUEstimator(
        model_fn=get_model_fn(create_fs_params, cfg, access_and_evict),
        params=params,
        model_dir=params.model_dir,
        config=run_config
    )

    if params.run_mode == 'train':
        est.train(input_fn=lambda: input_fn(params, create_fs_params, cfg), max_steps=params.max_steps,
                  hooks=npu_hooks_append(hooks_list))

    elif params.run_mode == 'train_and_evaluate':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(params, create_fs_params, cfg,
                                                                      use_one_shot=args.use_one_shot),
                                            max_steps=params.max_steps, hooks=npu_hooks_append(hooks_list))
        # 在开启evict时，eval时不支持淘汰，所以无需加入evict hook

        if not params.enable_push_ops_test:
            eval_hook_list = [GraphModifierHook(modify_graph=params.modify_graph)]
        else:
            eval_hook_list = [ACGPushOpsToDatasetHook(dump_graph=True),
                              GraphModifierHook(modify_graph=params.modify_graph)]

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(params, create_fs_params, cfg, is_eval=True,
                                                                    use_one_shot=args.use_one_shot),
                                          steps=params.eval_steps, hooks=npu_hooks_append(eval_hook_list),
                                          throttle_secs=0)
        tf.estimator.train_and_evaluate(est, train_spec=train_spec, eval_spec=eval_spec)

    elif params.run_mode == 'predict':
        results = est.predict(input_fn=lambda: input_fn(params, create_fs_params, cfg),
                              hooks=npu_hooks_append(hooks_list=hooks_list), yield_single_examples=False)
        output_pred1 = []
        output_pred2 = []
        labels = []

        for res in results:
            output_pred1.append(res['task_1'][0])
            output_pred2.append(res['task_2'][0])
            labels.append(res['label'][0])

    terminate_config_initializer()
    logger.info("Demo done!")


def create_feature_spec_list(use_timestamp=False):
    access_threshold = cfg.access_threshold if use_timestamp else None
    eviction_threshold = cfg.eviction_threshold if use_timestamp else None
    feature_spec_list = [FeatureSpec("user_ids", table_name="user_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold),
                         FeatureSpec("item_ids", table_name="item_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold)]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_mode', type=str, default='train_and_evaluate')  # 运行模式，在run.sh中进行配置
    parser.add_argument('--model_ckpt_dir', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=0.0008)
    parser.add_argument('--use_timestamp', type=bool, default=False)  # 是否开启特征准入与淘汰
    parser.add_argument('--modify_graph', type=bool, default=False)  # 是否开启自动改图
    parser.add_argument('--use_multi_lookup', type=bool, default=True)  # 是否一表多查
    parser.add_argument('--multi_lookup_times', type=int, default=2)  # 一表多查次数
    parser.add_argument('--max_steps', type=int, default=200)  # train的最大步数
    parser.add_argument('--train_steps', type=int, default=100)  # 训练train_steps步后进行eval
    parser.add_argument('--eval_steps', type=int, default=10)  # 每次eval的步数
    # 每隔step保存一次模型, 若在train_and_evaluate模式, 还会进行eval, 注: 若设为None, NPURunConfig内部会设默认值100
    parser.add_argument('--save_checkpoints_steps', type=int, default=200)
    parser.add_argument('--use_one_shot', type=bool, default=False)  # 是否使用one shot iterator

    args, unknowns = parser.parse_known_args()
    # get init configuration
    try:
        use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
        use_hot = bool(int(os.getenv("USE_HOT", 0)))
        use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
        use_multi_lookup = bool(int(os.getenv("USE_MULTI_LOOKUP", 1)))
        MODIFY_GRAPH_FLAG = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
        USE_TIMESTAMP = bool(int(os.getenv("USE_TIMESTAMP", 0)))
        args.use_one_shot = bool(int(os.getenv("USE_ONE_SHOT", 0)))
        args.enable_push_ops_test = bool(int(os.getenv("ENABLE_PUSH_OPS_TEST", 0)))
    except ValueError as err:
        raise ValueError(f"please correctly config USE_MPI or USE_DYNAMIC or USE_HOT or USE_DYNAMIC_EXPANSION or "
                         f"USE_MULTI_LOOKUP or USE_MODIFY_GRAPH or USE_TIMESTAMP or USE_ONE_SHOT "
                         f"only 0 or 1 is supported.") from err

    try:
        MULTI_LOOKUP_TIMES = int(os.getenv("MULTI_LOOKUP_TIMES", 2))
    except ValueError as err:
        raise ValueError(f"please correctly config MULTI_LOOKUP_TIMES only int is supported.") from err

    if args.run_mode == 'train':
        args.train_steps = -1
        args.eval_steps = -1
    elif args.run_mode == 'predict':
        args.eval_steps = -1
    elif args.run_mode == 'train_and_evaluate':
        args.save_checkpoints_steps = args.train_steps

    # set init
    init(train_steps=args.train_steps,
         eval_steps=args.eval_steps,
         use_dynamic=use_dynamic,
         use_hot=use_hot,
         use_dynamic_expansion=use_dynamic_expansion)

    args.model_dir = f"{args.model_ckpt_dir}_rank"
    args.modify_graph = MODIFY_GRAPH_FLAG
    args.use_timestamp = USE_TIMESTAMP
    args.use_multi_lookup = use_multi_lookup
    args.multi_lookup_times = MULTI_LOOKUP_TIMES
    cfg = Config()
    # multi lookup config, batch size: 32 * 128 = 4096
    if use_multi_lookup and MULTI_LOOKUP_TIMES > 2:
        cfg.batch_size = 32
    # init FeatureSpecIns
    FeatureSpecIns.set_instance()
    main(args, cfg)

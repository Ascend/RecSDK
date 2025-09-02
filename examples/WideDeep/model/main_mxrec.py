# coding=utf-8
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
import collections
import time
import warnings
from glob import glob

import tensorflow as tf
from mpi4py import MPI
from npu_bridge.npu_init import *

from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.core.feature_process import EvictHook
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache, GraphModifierHook
from mx_rec.constants.constants import ASCEND_TIMESTAMP
from mx_rec.util.initialize import ConfigInitializer, init, terminate_config_initializer
import rec_sdk_common
from mx_rec.util.variable import get_dense_and_sparse_variable
import examples.model_common as cm
from examples.model_common import (
    sess_config, Config,
    create_feature_spec_list, clear_saved_model, evaluate, evaluate_fix, make_batch_and_iterator
)
from demo_logger import logger
from model import MyModel
from optimizer import get_dense_and_sparse_optimizer

npu_plugin.set_device_sat_mode(0)

dense_hashtable_seed = 128
sparse_hashtable_seed = 128
cm.MODEL_NAME = "WideDeep"
cm.logger = logger


def model_forward(model_args):
    feature_list = model_args.feature_list
    wide_hash_table_list = model_args.wide_hash_table_list
    deep_hash_table_list = model_args.deep_hash_table_list
    batch = model_args.batch
    is_train = model_args.is_train
    modify_graph = model_args.modify_graph
    is_use_faae = model_args.is_use_faae

    wide_embedding_list = []
    deep_embedding_list = []
    wide_feature_list = []
    deep_feature_list = []
    if is_use_faae:
        feature_list_copy = feature_list[:-1]
    else:
        feature_list_copy = feature_list

    for index, item in enumerate(feature_list_copy):
        if index % 2 == 0:
            wide_feature_list.append(item)
        else:
            deep_feature_list.append(item)

    logger.debug(f"In model_forward function, is_train: {is_train}, feature_list: {len(feature_list)}, "
                 f"wide_hash_table_list: {len(wide_hash_table_list)}, "
                 f"deep_hash_table_list: {len(deep_hash_table_list)}")

    # wide
    for wide_feature, wide_hash_table in zip(wide_feature_list, wide_hash_table_list):
        if cm.MODIFY_GRAPH_FLAG:
            wide_feature = batch["sparse_feature"]
        wide_embedding = sparse_lookup(wide_hash_table, wide_feature, cfg.send_count, dim=None, is_train=is_train,
                                  name="wide_embedding_lookup", modify_graph=modify_graph, batch=batch,
                                  access_and_evict_config=None)
        wide_embedding_list.append(wide_embedding)

    # deep
    for deep_feature, deep_hash_table in zip(deep_feature_list, deep_hash_table_list):
        if cm.MODIFY_GRAPH_FLAG:
            deep_feature = batch["sparse_feature"]
        deep_embedding = sparse_lookup(deep_hash_table, deep_feature, cfg.send_count, dim=None, is_train=is_train,
                                  name="deep_embedding_lookup", modify_graph=modify_graph, batch=batch,
                                  access_and_evict_config=None)
        deep_embedding_list.append(deep_embedding)

    if len(wide_embedding_list) == 1:
        wide_emb = wide_embedding_list[0]
        deep_emb = deep_embedding_list[0]
    elif len(wide_embedding_list) > 1:
        wide_emb = tf.reduce_sum(wide_embedding_list, axis=0, keepdims=False)
        deep_emb = tf.reduce_sum(deep_embedding_list, axis=0, keepdims=False)
    else:
        raise ValueError("the length of embedding_list must be greater than or equal to 1.")
    my_model = MyModel()

    BuildModel = collections.namedtuple("BuildModel", ["wide_embedding", "deep_embedding", "label", "is_training",
                                                       "seed", "dropout_rate", "batch_norm"])
    build_model_args = BuildModel(wide_emb, deep_emb, batch["label"], is_train, dense_hashtable_seed, 0.5, False)
    model_output = my_model.build_model(build_model_args)
    return model_output


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")

    comm = MPI.COMM_WORLD
    clear_saved_model()
    comm.Barrier()

    cm.max_train_steps = 1270
    cm.train_steps = 1120
    cm.eval_steps = 1080

    use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
    logger.info(f"USE_DYNAMIC:{use_dynamic}")
    init(train_steps=cm.train_steps, eval_steps=cm.eval_steps,
         use_dynamic=use_dynamic, use_dynamic_expansion=cm.use_dynamic_expansion)
    IF_LOAD = False
    rank_id = rec_sdk_common.communication.hccl.hccl_info.get_rank_id()
    filelist = glob(f"./saved-model/sparse-model-0")
    if filelist:
        IF_LOAD = True
    ConfigInitializer.get_instance().if_load = IF_LOAD

    cfg = Config()
    feature_spec_list_train = None
    feature_spec_list_eval = None
    if cm.use_faae:
        feature_spec_list_train = create_feature_spec_list(cfg, use_timestamp=True)
        feature_spec_list_eval = create_feature_spec_list(cfg, use_timestamp=True)
    else:
        feature_spec_list_train = create_feature_spec_list(cfg, use_timestamp=False)
        feature_spec_list_eval = create_feature_spec_list(cfg, use_timestamp=False)

    train_batch, train_iterator = make_batch_and_iterator(cfg, feature_spec_list_train, is_training=True,
                                                          dump_graph=True, is_use_faae=cm.use_faae)
    eval_batch, eval_iterator = make_batch_and_iterator(cfg, feature_spec_list_eval, is_training=False,
                                                        dump_graph=False, is_use_faae=cm.use_faae)
    logger.info(f"train_batch: {train_batch}")

    if cm.use_faae:
        cfg.dev_vocab_size = cfg.dev_vocab_size // 2

    # 创表操作
    wide_emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=sparse_hashtable_seed)
    deep_emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=sparse_hashtable_seed)

    sparse_hashtable_wide = create_table(
        key_dtype=cfg.key_type,
        dim=tf.TensorShape([cfg.emb_dim]),
        name="wide_embeddings",
        emb_initializer=wide_emb_initializer,
        **cfg.get_emb_table_cfg()
    )

    sparse_hashtable_deep = create_table(
        key_dtype=cfg.key_type,
        dim=tf.TensorShape([cfg.emb_dim]),
        name="deep_embeddings",
        emb_initializer=deep_emb_initializer,
        **cfg.get_emb_table_cfg()
    )

    if cm.use_faae:
        tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, train_batch["timestamp"])

    # 一表多查
    wide_hashtable_list = [sparse_hashtable_wide, sparse_hashtable_wide] if cm.use_multi_lookup else \
                          [sparse_hashtable_wide]
    deep_hashtable_list = [sparse_hashtable_deep, sparse_hashtable_deep] if cm.use_multi_lookup else \
                          [sparse_hashtable_deep]


    Forward = collections.namedtuple("Forward", ["feature_list", "wide_hash_table_list", "deep_hash_table_list",
                                                 "batch", "is_train", "modify_graph", "is_use_faae"])
    train_forward_args = Forward(feature_spec_list_train, wide_hashtable_list, deep_hashtable_list, train_batch,
                                True, cm.MODIFY_GRAPH_FLAG, cm.use_faae)
    eval_forward_args = Forward(feature_spec_list_eval, wide_hashtable_list, deep_hashtable_list, eval_batch,
                                False, cm.MODIFY_GRAPH_FLAG, cm.use_faae)
    train_model = model_forward(train_forward_args)
    eval_model = model_forward(eval_forward_args)

    train_variables, emb_variables = get_dense_and_sparse_variable()
    optimizer_list = [get_dense_and_sparse_optimizer(cfg)]

    rank_size = rec_sdk_common.communication.hccl.hccl_info.get_rank_size()
    train_ops = []
    # multi task training
    for loss, (model_optimizer, emb_optimizer) in zip([train_model.get("loss")], optimizer_list):
        # do model optimization
        grads = model_optimizer.compute_gradients(loss, var_list=train_variables)
        avg_grads = []
        for grad, var in grads:
            if rank_size > 1:
                grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
            if grad is not None:
                avg_grads.append((grad / 8.0, var))
        # apply gradients: update variables
        train_ops.append(model_optimizer.apply_gradients(avg_grads))

        if cm.use_dynamic_expansion:
            train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)
            train_emb_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)
            # do embedding optimization by addr
            sparse_grads = emb_optimizer.compute_gradients(loss, train_emb_list)  # local_embedding
            grads_and_vars = [(grad, address) for grad, address in zip(sparse_grads, train_address_list)]
            train_ops.append(emb_optimizer.apply_gradients(grads_and_vars))
        else:
            # do embedding optimization
            sparse_grads = emb_optimizer.compute_gradients(loss, emb_variables)
            print("sparse_grads_tensor:", sparse_grads)
            grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, emb_variables)]
            train_ops.append(emb_optimizer.apply_gradients(grads_and_vars))

    # 动态学习率更新
    train_ops.extend([cfg.global_step.assign(cfg.global_step + 1), cfg.learning_rate[0], cfg.learning_rate[1]])

    with tf.control_dependencies(train_ops):
        train_ops = tf.no_op()
        cfg.learning_rate = [cfg.learning_rate[0], cfg.learning_rate[1]]

    saver = tf.train.Saver()
    if cm.MODIFY_GRAPH_FLAG:
        modify_graph_and_start_emb_cache(dump_graph=True)
    else:
        start_asc_pipeline()

    hook_list = []
    if cm.use_faae:
        hook_evict = EvictHook(evict_enable=True, evict_time_interval=120)
        hook_list.append(hook_evict)
        if cm.MODIFY_GRAPH_FLAG:  # 该场景添加hook处理校验问题
            hook_list.append(GraphModifierHook(modify_graph=False))

    # Disable dumping data during session: set dump_data=False in sess_config:
    if cm.use_faae:
        sess = tf.compat.v1.train.MonitoredTrainingSession(
            hooks=hook_list,
            config=sess_config(dump_data=False)
        )
        sess.graph._unsafe_unfinalize()
        if not cm.MODIFY_GRAPH_FLAG:
            sess.run(train_iterator.initializer)
        else:
            sess.run(ConfigInitializer.get_instance().train_params_config.get_initializer(True))
    else:
        sess = tf.compat.v1.Session(config=sess_config(dump_data=False))
        sess.run(tf.compat.v1.global_variables_initializer())
        if not cm.MODIFY_GRAPH_FLAG:
            sess.run(train_iterator.initializer)
        else:
            sess.run(ConfigInitializer.get_instance().train_params_config.get_initializer(True))

    epoch = 0
    cost_sum = 0
    qps_sum = 0
    best_auc = 0
    iteration_per_loop = 10

    train_ops = util.set_iteration_per_loop(sess, train_ops, 10)

    # for i in range(1, TRAIN_STEPS):
    i = 0
    while True:
        i += 1
        logger.info(f"################    training at step {i * iteration_per_loop}    ################")
        start_time = time.time()

        try:
            grad, loss = sess.run([train_ops, train_model.get("loss")])
            lr = sess.run(cfg.learning_rate)
            global_step = sess.run(cfg.global_step)
        except tf.errors.OutOfRangeError:
            logger.info(f"Encounter the end of Sequence for training.")
            break

        end_time = time.time()
        cost_time = end_time - start_time
        qps = (1 / cost_time) * rank_size * cfg.batch_size * iteration_per_loop
        cost_sum += cost_time
        logger.info(f"step: {i * iteration_per_loop}; training loss: {loss}")
        logger.info(f"step: {i * iteration_per_loop}; grad: {grad}")
        logger.info(f"step: {i * iteration_per_loop}; lr: {lr}")
        logger.info(f"global step: {global_step}")
        logger.info(f"step: {i * iteration_per_loop}; current sess cost time: {cost_time:.10f}; current QPS: {qps}")
        logger.info(f"training at step:{i * iteration_per_loop}, "
                    f"table[{sparse_hashtable_wide.table_name}], "
                    f"table size:{sparse_hashtable_wide.size()}, table capacity:{sparse_hashtable_wide.capacity()}, "
                    f"table[{sparse_hashtable_deep.table_name}], "
                    f"table size:{sparse_hashtable_deep.size()}, table capacity:{sparse_hashtable_deep.capacity()}")

        if i % (cm.train_steps // iteration_per_loop) == 0:
            if cm.interval is not None:
                test_auc, test_mean_log_loss = evaluate_fix(i * iteration_per_loop, sess, eval_model, eval_iterator)
            else:
                test_auc, test_mean_log_loss = evaluate(sess, eval_model, eval_iterator, cfg)
            print("Test auc: {}; log_loss: {} ".format(test_auc, test_mean_log_loss))
            best_auc = max(best_auc, test_auc)
            logger.info(f"training step: {i * iteration_per_loop}, best auc: {best_auc}")

    sess.close()

    terminate_config_initializer()
    logger.info("Demo done!")

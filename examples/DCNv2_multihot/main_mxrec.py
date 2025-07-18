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
import time
import warnings
from glob import glob
from typing import List, Tuple

import tensorflow as tf
from npu_bridge.npu_init import *

from model import MyModel
from optimizer import get_dense_and_sparse_optimizer
from adacons_hooks import adacons_hooks
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.core.feature_process import EvictHook
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache, GraphModifierHook
from mx_rec.constants.constants import ASCEND_TIMESTAMP
from mx_rec.util.initialize import ConfigInitializer, init
import mx_rec.util as mxrec_util
from mx_rec.util.variable import get_dense_and_sparse_variable
from mx_rec.util.log import logger
import examples.model_common as cm
from examples.model_common import (
    sess_config, Config,
    create_feature_spec_list, clear_saved_model, evaluate, evaluate_fix, make_batch_and_iterator
)

npu_plugin.set_device_sat_mode(0)

DENSE_HASHTABLE_SEED = 128
SPARSE_HASHTABLE_SEED = 128
os.environ['CM_WORKER_IP'] = "x.x.x.x"
cm.MODEL_NAME = "DCNv2_multihot"
cm.logger = logger


def model_forward(feature_list, hash_table_list, batch, is_train, modify_graph):
    embedding_list = []
    logger.debug(f"In model_forward function, is_train: {is_train}, feature_list: {len(feature_list)}, "
                 f"hash_table_list: {len(hash_table_list)}")
    for feature, hash_table in zip(feature_list, hash_table_list):
        if cm.MODIFY_GRAPH_FLAG:
            feature = batch["sparse_feature"]
        embedding = sparse_lookup(hash_table, feature, cfg.send_count, dim=None, is_train=is_train,
                                  name="user_embedding_lookup", modify_graph=modify_graph, batch=batch,
                                  access_and_evict_config=None)
        embedding_list.append(embedding)

    if len(embedding_list) == 1:
        emb = embedding_list[0]

        pooling_sequence = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
        emb = tf.transpose(emb, perm=[1, 0, 2])
        pooling_s = tf.constant([i for i, count in enumerate(pooling_sequence) for _ in range(count)])
        emb = tf.segment_sum(emb, pooling_s)
        emb = tf.transpose(emb, perm=[1, 0, 2])
        bs = int(os.getenv("BATCH_SIZE"))
        emb = tf.reshape(emb, [bs, 26, 128])
    elif len(embedding_list) > 1:
        emb = tf.reduce_sum(embedding_list, axis=0, keepdims=False)
    else:
        raise ValueError("the length of embedding_list must be greater than or equal to 1.")
    my_model = MyModel()
    model_output = my_model.build_model(embedding=emb,
                                        dense_feature=batch["dense_feature"],
                                        label=batch["label"],
                                        is_training=is_train,
                                        seed=DENSE_HASHTABLE_SEED)
    return model_output


def cal_average_grad(grads_info: List[Tuple[tf.Tensor, tf.Variable]], device_size: int) -> List[
    Tuple[tf.Tensor, tf.Variable]]:
    """
    This function calculates the average gradients across all devices in a multi-device setup.
    It takes into account gradient synchronization (all-reduce) and then averages the gradients
    by dividing by the number of devices (device_size).

    Parameters:
    grads_info (list of tuples): A list of tuples where each tuple contains the gradient (cur_grad)
                                 and the corresponding variable (var) from each device.
    device_size (int): The total number of devices (e.g.NPUs) involved in the training.

    Returns:
    avg_grads (list of tuples): A list of tuples where each tuple contains the averaged gradient
                                and its corresponding variable.
    """
    avg_grads = []
    for cur_grad, var in grads_info:
        if rank_size > 1:
            cur_grad = hccl_ops.allreduce(cur_grad, "sum") if cur_grad is not None else None
        if cur_grad is not None:
            avg_grads.append((cur_grad / device_size, var))

    return avg_grads


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")
    clear_saved_model()

    cm.train_steps = int(os.getenv("TRAIN_STEP"))
    cm.eval_steps = int(os.getenv("TEST_STEP"))

    use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
    logger.info(f"USE_DYNAMIC: {use_dynamic}")
    init(train_steps=cm.train_steps, eval_steps=cm.eval_steps,
         use_dynamic=use_dynamic, use_dynamic_expansion=cm.use_dynamic_expansion)
    IF_LOAD = False
    rank_id = mxrec_util.communication.hccl_ops.get_rank_id()
    filelist = glob("./saved-model/sparse-model-0")
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

    optimizer_list = [get_dense_and_sparse_optimizer(cfg)]

    # note: variance_scaling_initializer only support HBM mode
    emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=SPARSE_HASHTABLE_SEED) \
        if cfg.cache_mode != "HBM" or cm.use_dynamic_expansion else \
        tf.compat.v1.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=SPARSE_HASHTABLE_SEED)
    sparse_hashtable = create_table(
        key_dtype=cfg.key_type,
        dim=tf.TensorShape([cfg.emb_dim]),
        name="sparse_embeddings",
        emb_initializer=emb_initializer,
        **cfg.get_emb_table_cfg()
    )
    if cm.use_faae:
        tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, train_batch["timestamp"])

    sparse_hashtable_list = [sparse_hashtable, sparse_hashtable] if cm.use_multi_lookup else [sparse_hashtable]
    train_model = model_forward(feature_spec_list_train, sparse_hashtable_list, train_batch,
                                is_train=True, modify_graph=cm.MODIFY_GRAPH_FLAG)
    eval_model = model_forward(feature_spec_list_eval, sparse_hashtable_list, eval_batch,
                               is_train=False, modify_graph=cm.MODIFY_GRAPH_FLAG)

    dense_variables, sparse_variables = get_dense_and_sparse_variable()

    rank_size = mxrec_util.communication.hccl_ops.get_rank_size()
    train_ops = []
    # multi task training
    for loss, (dense_optimizer, sparse_optimizer) in zip([train_model.get("loss")], optimizer_list):
        # do dense optimization
        grads = dense_optimizer.compute_gradients(loss, var_list=dense_variables)

        if cfg.use_adacons:
            grads_new = adacons_hooks(gradients=grads, rank_size=rank_size, device_id=rank_id)
        else:
            grads_new = cal_average_grad(grads, rank_size)
        # apply gradients: update variables
        train_ops.append(dense_optimizer.apply_gradients(grads_new))

        if cm.use_dynamic_expansion:
            from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET
            train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)
            train_emb_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)
            # do sparse optimization by addr
            sparse_grads = sparse_optimizer.compute_gradients(loss, train_emb_list)  # local_embedding
            grads_and_vars = [(grad, address) for grad, address in zip(sparse_grads, train_address_list)]
            train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))
        else:
            # do sparse optimization
            sparse_grads = sparse_optimizer.compute_gradients(loss, sparse_variables)
            logger.info(f"sparse_grads_tensor: {sparse_grads}")
            grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
            train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))

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
            logger.info("Encounter the end of Sequence for training.")
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
        logger.info(f"training at step:{i * iteration_per_loop}, table[{sparse_hashtable.table_name}], "
                    f"table size:{sparse_hashtable.size()}, table capacity:{sparse_hashtable.capacity()}")

        if i % (cm.train_steps // iteration_per_loop) == 0:
            if cm.interval is not None:
                test_auc, test_mean_log_loss = evaluate_fix(i * iteration_per_loop, sess, eval_model, eval_iterator)
            else:
                test_auc, test_mean_log_loss = evaluate(sess, eval_model, eval_iterator, cfg)
            logger.info("Test auc: {}; log_loss: {} ".format(test_auc, test_mean_log_loss))
            best_auc = max(best_auc, test_auc)
            logger.info(f"training step: {i * iteration_per_loop}, best auc: {best_auc}")

    sess.close()

    logger.info("Demo done!")

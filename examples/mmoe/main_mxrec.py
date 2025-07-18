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

import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
from npu_bridge.npu_init import *

from config import Config
from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.core.feature_process import EvictHook
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache, GraphModifierHook
from mx_rec.constants.constants import ASCEND_TIMESTAMP
from mx_rec.util.initialize import ConfigInitializer, init, terminate_config_initializer
import mx_rec.util as mxrec_util
from mx_rec.util.variable import get_dense_and_sparse_variable
import examples.model_common as cm
from examples.model_common import (
    sess_config, create_feature_spec_list, clear_saved_model, evaluate_fix, make_batch_and_iterator
)
from model import MyModel
from demo_logger import logger
from optimizer import get_dense_and_sparse_optimizer

npu_plugin.set_device_sat_mode(0)

DENSE_HASHTABLE_SEED = 128
SPARSE_HASHTABLE_SEED = 128
cm.MODEL_NAME = "MMOE"
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
    elif len(embedding_list) > 1:
        emb = tf.reduce_sum(embedding_list, axis=0, keepdims=False)
    else:
        raise ValueError("the length of embedding_list must be greater than or equal to 1.")
    emb = tf.reduce_sum(emb, axis=1)
    my_model = MyModel()
    model_output = my_model.build_model(embedding=emb,
                                        dense_feature=batch["dense_feature"],
                                        label=batch["label"],
                                        is_training=is_train,
                                        seed=DENSE_HASHTABLE_SEED)
    return model_output


def evaluate():
    print("read_test dataset")
    if not cm.MODIFY_GRAPH_FLAG:
        eval_label = eval_model.get("label")
        sess.run([eval_iterator.initializer])
    else:
        # In sess run mode, if the label from the original batch is still used for sess run, 
        # a getnext timeout error will occur, and a new batch from the new dataset needs to be used
        eval_label = ConfigInitializer.get_instance().train_params_config.get_target_batch(False).get("label")
        sess.run([ConfigInitializer.get_instance().train_params_config.get_initializer(False)])
    log_loss_list = []
    pred_income_list = []
    pred_mat_list = []
    label_income_list = []
    label_mat_list = []
    eval_current_steps = 0
    finished = False
    print("eval begin")

    while not finished:
        
        eval_current_steps += 1
        eval_start = time.time()
        try:
            eval_loss, pred, label = sess.run([eval_model.get("loss"), eval_model.get("pred"), eval_label])
        except tf.errors.OutOfRangeError:
            break
        eval_cost = time.time() - eval_start
        qps_eval = (1 / eval_cost) * rank_size * cfg.batch_size
        log_loss_list += list(eval_loss.reshape(-1))
        pred_income = pred[0]
        pred_mat = pred[1]
        pred_income_list += list(pred_income.reshape(-1))
        pred_mat_list += list(pred_mat.reshape(-1))
        label_income_list += list(label[:, 0].reshape(-1))
        label_mat_list += list(label[:, 1].reshape(-1))
        print(f"eval current_steps: {eval_current_steps}, qps: {qps_eval}")
        if eval_current_steps == cm.eval_steps:
            finished = True
        
    auc_income = roc_auc_score(label_income_list, pred_income_list)
    auc_mat = roc_auc_score(label_mat_list, pred_mat_list)
    mean_log_loss = np.mean(log_loss_list)
    return auc_income, auc_mat, mean_log_loss


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")
    clear_saved_model()

    cm.train_steps = 1000
    cm.eval_steps = 1500

    cfg = Config()
    use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
    logger.info(f"USE_DYNAMIC:{use_dynamic}")
    init(train_steps=cm.train_steps, eval_steps=cm.eval_steps,
         use_dynamic=use_dynamic, use_dynamic_expansion=cm.use_dynamic_expansion)
    
    rank_id = mxrec_util.communication.hccl_ops.get_rank_id()
    
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
    emb_initializer = tf.constant_initializer(value=0.1)
    sparse_hashtable = create_table(
        key_dtype=cfg.key_type,
        dim=tf.TensorShape([cfg.emb_dim]),
        name="sparse_embeddings",
        emb_initializer=emb_initializer,
        is_dp=cm.USE_DP,
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
    trainable_varibles = []
    trainable_varibles.extend(dense_variables)
    if cm.use_dynamic_expansion:
        trainable_varibles.append(tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)[0])
    else:
        trainable_varibles.extend(sparse_variables)
    rank_size = mxrec_util.communication.hccl_ops.get_rank_size()
    train_ops = []
    # multi task training
    for loss, (dense_optimizer, sparse_optimizer) in zip([train_model.get("loss")], optimizer_list):
        # do dense optimization
        grads = dense_optimizer.compute_gradients(loss, var_list=trainable_varibles)
        avg_grads = []
        for grad, var in grads[:-1]:
            if rank_size > 1:
                grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
            if grad is not None:
                avg_grads.append((grad / 8.0, var))
        # apply gradients: update variables
        train_ops.append(dense_optimizer.apply_gradients(avg_grads))

        if cm.use_dynamic_expansion:
            train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)
            # do sparse optimization by addr
            sparse_grads = list(grads[-1])  # local_embedding
            grads_and_vars = [(grad, address) for grad, address in zip(sparse_grads, train_address_list)]
            train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))
        else:
            # do sparse optimization
            sparse_grads = list(grads[-1])
            print("sparse_grads_tensor:", sparse_grads)
            grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
            train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))


    with tf.control_dependencies(train_ops):
        train_ops = tf.no_op()
        cfg.learning_rate = [cfg.learning_rate[0], cfg.learning_rate[1]]

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
    best_auc_income = 0
    best_auc_mat = 0
    iteration_per_loop = 10

    train_ops = util.set_iteration_per_loop(sess, train_ops, 10)

    i = 0
    while True:
        i += 1
        logger.info(f"################    training at step {i * iteration_per_loop}    ################")
        start_time = time.time()

        try:
            grad, loss, lr, global_step = sess.run([train_ops, train_model.get("loss"), 
                                                    cfg.learning_rate, cfg.global_step])
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
        logger.info(f"training at step:{i * iteration_per_loop}, table[{sparse_hashtable.table_name}], "
                    f"table size:{sparse_hashtable.size()}, table capacity:{sparse_hashtable.capacity()}")

        if i % (cm.train_steps // iteration_per_loop) == 0:
            if cm.interval is not None:
                test_auc_income, test_auc_mat, test_mean_log_loss = evaluate_fix(i * iteration_per_loop, sess,
                                                                                 eval_model, eval_iterator)
            else:
                test_auc_income, test_auc_mat, test_mean_log_loss = evaluate()
            print("Test auc income: {};Test auc mat: {} ;log_loss: {} ".format(test_auc_income, 
                                                                               test_auc_mat, test_mean_log_loss))
            best_auc_income = max(best_auc_income, test_auc_income)
            best_auc_mat = max(best_auc_mat, test_auc_mat)
            logger.info(f"training step: {i * iteration_per_loop}, best auc income: "
                        f"{best_auc_income} , best auc mat: {best_auc_mat}")


    sess.close()

    terminate_config_initializer()
    logger.info("Demo done!")

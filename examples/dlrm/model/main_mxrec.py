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
import shutil
import time
import warnings
import random
from glob import glob

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from npu_bridge.npu_init import *
import numpy as np

from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET
from mx_rec.core.asc.helper import FeatureSpec, get_asc_insert_func
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.core.feature_process import EvictHook
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache, GraphModifierHook
from mx_rec.constants.constants import ASCEND_TIMESTAMP
from mx_rec.util.initialize import ConfigInitializer, init, terminate_config_initializer
from mx_rec.util.ops import import_host_pipeline_ops
import mx_rec.util as mxrec_util
from mx_rec.util.variable import get_dense_and_sparse_variable

from optimizer import get_dense_and_sparse_optimizer
from config import sess_config, Config, SSD_DATA_PATH, CacheModeEnum
from logger import logger
from model import MyModel

npu_plugin.set_device_sat_mode(0)

dense_hashtable_seed = 128
sparse_hashtable_seed = 128
shuffle_seed = 128
random.seed(shuffle_seed)


def add_timestamp_func(batch):
    timestamp = import_host_pipeline_ops().return_timestamp(tf.cast(batch['label'], dtype=tf.int64))
    # tf.constant(np.random.randint(1,1688109060,1)), tf.int64))
    batch["timestamp"] = timestamp
    return batch


def make_batch_and_iterator(config, feature_spec_list, is_training, dump_graph, is_use_faae=False):
    if config.USE_PIPELINE_TEST:
        num_parallel = 1
    else:
        num_parallel = 8

    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.compat.v1.FixedLenFeature(shape=(config.line_per_sample,), dtype=tf.int64),
            'sparse_feature': tf.compat.v1.FixedLenFeature(shape=(26 * config.line_per_sample,), dtype=tf.int64),
            'dense_feature': tf.compat.v1.FixedLenFeature(shape=(13 * config.line_per_sample,), dtype=tf.float32),
        }
        sample = tf.compat.v1.parse_single_example(data_record, features)
        return sample

    def reshape_fn(batch):
        batch['label'] = tf.reshape(batch['label'], [-1, 1])
        batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 13])
        batch['dense_feature'] = tf.math.log(batch['dense_feature'] + 3.0)
        batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 26])
        return batch

    if is_training:
        files_list = glob(os.path.join(config.data_path, config.train_file_pattern) + '/*.tfrecord')
    else:
        files_list = glob(os.path.join(config.data_path, config.test_file_pattern) + '/*.tfrecord')
    dataset = tf.data.TFRecordDataset(files_list, num_parallel_reads=num_parallel)
    batch_size = config.batch_size // config.line_per_sample

    dataset = dataset.shard(config.rank_size, config.rank_id)
    if is_training:
        dataset = dataset.shuffle(batch_size * 1000, seed=shuffle_seed)
    if is_training:
        dataset = dataset.repeat(config.train_epoch)
    else:
        dataset = dataset.repeat(config.test_epoch)
    dataset = dataset.map(extract_fn, num_parallel_calls=num_parallel).batch(batch_size,
                                                                             drop_remainder=True)
    dataset = dataset.map(reshape_fn, num_parallel_calls=num_parallel)
    if is_use_faae:
        dataset = dataset.map(add_timestamp_func)

    if not MODIFY_GRAPH_FLAG:
        insert_fn = get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=is_training, dump_graph=dump_graph)
        dataset = dataset.map(insert_fn)

    dataset = dataset.prefetch(100)

    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    return batch, iterator


def model_forward(feature_list, hash_table_list, batch, is_train, modify_graph):
    embedding_list = []
    logger.debug(f"In model_forward function, is_train: {is_train}, feature_list: {len(feature_list)}, "
                 f"hash_table_list: {len(hash_table_list)}")
    for feature, hash_table in zip(feature_list, hash_table_list):
        if MODIFY_GRAPH_FLAG:
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
    my_model = MyModel()
    model_output = my_model.build_model(embedding=emb,
                                        dense_feature=batch["dense_feature"],
                                        label=batch["label"],
                                        is_training=is_train,
                                        seed=dense_hashtable_seed)
    return model_output


def evaluate():
    print("read_test dataset")
    if not MODIFY_GRAPH_FLAG:
        eval_label = eval_model.get("label")
        sess.run([eval_iterator.initializer])
    else:
        # 在sess run模式下，若还是使用原来batch中的label去sess run，则会出现getnext超时报错，需要使用新数据集中的batch
        eval_label = ConfigInitializer.get_instance().train_params_config.get_target_batch(False).get("label")
        sess.run([ConfigInitializer.get_instance().train_params_config.get_initializer(False)])
    log_loss_list = []
    pred_list = []
    label_list = []
    eval_current_steps = 0
    finished = False
    print("eval begin")

    while not finished:
        try:
            eval_current_steps += 1
            eval_start = time.time()
            eval_loss, pred, label = sess.run([eval_model.get("loss"), eval_model.get("pred"), eval_label])
            eval_cost = time.time() - eval_start
            qps_eval = (1 / eval_cost) * rank_size * cfg.batch_size
            log_loss_list += list(eval_loss.reshape(-1))
            pred_list += list(pred.reshape(-1))
            label_list += list(label.reshape(-1))
            print(f"eval current_steps: {eval_current_steps}, qps: {qps_eval}")
            if eval_current_steps == eval_steps:
                finished = True
        except tf.errors.OutOfRangeError:
            finished = True
    auc = roc_auc_score(label_list, pred_list)
    mean_log_loss = np.mean(log_loss_list)
    return auc, mean_log_loss


def evaluate_fix(step):
    print("read_test dataset evaluate_fix")
    if not MODIFY_GRAPH_FLAG:
        sess.run([eval_iterator.initializer])
    else:
        sess.run([ConfigInitializer.get_instance().train_params_config.get_initializer(False)])
    log_loss_list = []
    pred_list = []
    label_list = []
    eval_current_steps = 0
    finished = False
    print("eval begin")
    while not finished:
        try:
            eval_current_steps += 1
            eval_loss, pred, label = sess.run([eval_model.get("loss"), eval_model.get("pred"), eval_model.get("label")])
            log_loss_list += list(eval_loss.reshape(-1))
            pred_list += list(pred.reshape(-1))
            label_list += list(label.reshape(-1))
            print(f"eval current_steps: {eval_current_steps}")

            if eval_current_steps == eval_steps:
                finished = True
        except tf.errors.OutOfRangeError:
            finished = True

    label_numpy = np.array(label_list)
    pred_numpy = np.array(pred_list)
    if not os.path.exists(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}"):
        os.makedirs(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}")

    if os.path.exists(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}/label_{rank_id}.npy"):
        os.remove(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}/label_{rank_id}.npy")
    if os.path.exists(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}/pred_{rank_id}.npy"):
        os.remove(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}/pred_{rank_id}.npy")
    if os.path.exists(f"flag_{rank_id}.txt"):
        os.remove(f"flag_{rank_id}.txt")
    np.save(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}/label_{rank_id}.npy", label_numpy)
    np.save(os.path.abspath(".") + f"/interval_{interval}/numpy_{step}/pred_{rank_id}.npy", pred_numpy)
    os.mknod(f"flag_{rank_id}.txt")
    while True:
        file_exists_list = [os.path.exists(f"flag_{i}.txt") for i in range(rank_size)]
        if sum(file_exists_list) == rank_size:
            print("All saved!!!!!!!!!!")
            break
        else:
            print("Waitting for saving numpy!!!!!!!!")
            time.sleep(1)
            continue

    auc = roc_auc_score(label_list, pred_list)
    mean_log_loss = np.mean(log_loss_list)
    return auc, mean_log_loss


def create_feature_spec_list(use_timestamp=False):
    access_threshold = None
    eviction_threshold = None
    if use_timestamp:
        access_threshold = 1000
        eviction_threshold = 180

    feature_spec_list = [FeatureSpec("sparse_feature", table_name="sparse_embeddings", batch_size=cfg.batch_size,
                                     access_threshold=access_threshold, eviction_threshold=eviction_threshold)]
    if use_multi_lookup:
        feature_spec_list.append(FeatureSpec("sparse_feature", table_name="sparse_embeddings",
                                             batch_size=cfg.batch_size,
                                             access_threshold=access_threshold,
                                             eviction_threshold=eviction_threshold))
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list


def _del_related_dir(del_path: str) -> None:
    if not os.path.isabs(del_path):
        del_path = os.path.join(os.getcwd(), del_path)
    dirs = glob(del_path)
    for sub_dir in dirs:
        shutil.rmtree(sub_dir, ignore_errors=True)
        logger.info(f"Delete dir:{sub_dir}")


def _clear_saved_model() -> None:
    _del_related_dir("/root/ascend/log/*")
    _del_related_dir("kernel*")
    _del_related_dir("model_dir_rank*")
    _del_related_dir("op_cache")

    if os.getenv("CACHE_MODE", "") != CacheModeEnum.SSD.value:
        return
    logger.info("Current cache mode is SSD, and file overwrite is not allowed in SSD mode, deleting exist directory"
                " then create empty directory for this use case.")
    for sub_path in SSD_DATA_PATH:
        _del_related_dir(sub_path)
        os.makedirs(sub_path, mode=0o550, exist_ok=True)
        logger.info(f"Create dir:{sub_path}")


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")
    _clear_saved_model()

    rank_id = int(os.getenv("RANK_ID")) if os.getenv("RANK_ID") else None
    rank_size = int(os.getenv("TRAIN_RANK_SIZE")) if os.getenv("TRAIN_RANK_SIZE") else None
    interval = int(os.getenv("INTERVAL")) if os.getenv("INTERVAL") else None
    train_steps = 10000
    eval_steps = 1360

    try:
        use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
        use_multi_lookup = bool(int(os.getenv("USE_MULTI_LOOKUP", 0)))
        MODIFY_GRAPH_FLAG = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
        use_faae = bool(int(os.getenv("USE_FAAE", 0)))
    except ValueError as err:
        raise ValueError("please correctly config USE_DYNAMIC_EXPANSION or USE_MULTI_LOOKUP or USE_FAAE "
                         "or USE_MODIFY_GRAPH only 0 or 1 is supported.") from err

    use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
    logger.info(f"USE_DYNAMIC:{use_dynamic}")
    init(train_steps=train_steps, eval_steps=eval_steps,
         use_dynamic=use_dynamic, use_dynamic_expansion=use_dynamic_expansion)
    IF_LOAD = False
    rank_id = mxrec_util.communication.hccl_ops.get_rank_id()
    filelist = glob(f"./saved-model/sparse-model-0")
    if filelist:
        IF_LOAD = True
    ConfigInitializer.get_instance().if_load = IF_LOAD

    cfg = Config()
    feature_spec_list_train = None
    feature_spec_list_eval = None
    if use_faae:
        feature_spec_list_train = create_feature_spec_list(use_timestamp=True)
        feature_spec_list_eval = create_feature_spec_list(use_timestamp=True)
    else:
        feature_spec_list_train = create_feature_spec_list(use_timestamp=False)
        feature_spec_list_eval = create_feature_spec_list(use_timestamp=False)

    train_batch, train_iterator = make_batch_and_iterator(cfg, feature_spec_list_train, is_training=True,
                                                          dump_graph=True, is_use_faae=use_faae)
    eval_batch, eval_iterator = make_batch_and_iterator(cfg, feature_spec_list_eval, is_training=False,
                                                        dump_graph=False, is_use_faae=use_faae)
    logger.info(f"train_batch: {train_batch}")

    if use_faae:
        cfg.dev_vocab_size = cfg.dev_vocab_size // 2

    optimizer_list = [get_dense_and_sparse_optimizer(cfg)]

    # note: variance_scaling_initializer only support HBM mode
    emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=sparse_hashtable_seed) \
        if cfg.cache_mode != "HBM" or use_dynamic_expansion else \
        tf.compat.v1.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=sparse_hashtable_seed)
    sparse_hashtable = create_table(
        key_dtype=cfg.key_type,
        dim=tf.TensorShape([cfg.emb_dim]),
        name="sparse_embeddings",
        emb_initializer=emb_initializer,
        **cfg.get_emb_table_cfg()
    )
    if use_faae:
        tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, train_batch["timestamp"])

    sparse_hashtable_list = [sparse_hashtable, sparse_hashtable] if use_multi_lookup else [sparse_hashtable]
    train_model = model_forward(feature_spec_list_train, sparse_hashtable_list, train_batch,
                                is_train=True, modify_graph=MODIFY_GRAPH_FLAG)
    eval_model = model_forward(feature_spec_list_eval, sparse_hashtable_list, eval_batch,
                               is_train=False, modify_graph=MODIFY_GRAPH_FLAG)

    dense_variables, sparse_variables = get_dense_and_sparse_variable()
    trainable_varibles = []
    trainable_varibles.extend(dense_variables)
    if use_dynamic_expansion:
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

        if use_dynamic_expansion:
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

    # 动态学习率更新
    train_ops.extend([cfg.global_step.assign(cfg.global_step + 1), cfg.learning_rate[0], cfg.learning_rate[1]])

    with tf.control_dependencies(train_ops):
        train_ops = tf.no_op()
        cfg.learning_rate = [cfg.learning_rate[0], cfg.learning_rate[1]]

    saver = tf.train.Saver()
    if MODIFY_GRAPH_FLAG:
        modify_graph_and_start_emb_cache(dump_graph=True)
    else:
        start_asc_pipeline()

    hook_list = []
    if use_faae:
        hook_evict = EvictHook(evict_enable=True, evict_time_interval=120)
        hook_list.append(hook_evict)
        if MODIFY_GRAPH_FLAG:  # 该场景添加hook处理校验问题
            hook_list.append(GraphModifierHook(modify_graph=False))

    # with tf.compat.v1.Session(config=sess_config(dump_data=False)) as sess:
    if use_faae:
        sess = tf.compat.v1.train.MonitoredTrainingSession(
            hooks=hook_list,
            config=sess_config(dump_data=False)
        )
        sess.graph._unsafe_unfinalize()
        if not MODIFY_GRAPH_FLAG:
            sess.run(train_iterator.initializer)
        else:
            sess.run(ConfigInitializer.get_instance().train_params_config.get_initializer(True))
    else:
        sess = tf.compat.v1.Session(config=sess_config(dump_data=False))
        sess.run(tf.compat.v1.global_variables_initializer())
        if not MODIFY_GRAPH_FLAG:
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
        logger.info(f"training at step:{i * iteration_per_loop}, table[{sparse_hashtable.table_name}], "
                    f"table size:{sparse_hashtable.size()}, table capacity:{sparse_hashtable.capacity()}")

        if i % (train_steps // iteration_per_loop) == 0:
            if interval is not None:
                test_auc, test_mean_log_loss = evaluate_fix(i * iteration_per_loop)
            else:
                test_auc, test_mean_log_loss = evaluate()
            print("Test auc: {}; log_loss: {} ".format(test_auc, test_mean_log_loss))
            best_auc = max(best_auc, test_auc)
            logger.info(f"training step: {i * iteration_per_loop}, best auc: {best_auc}")

    sess.close()

    terminate_config_initializer()
    logger.info("Demo done!")

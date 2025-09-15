# coding=utf-8
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import random
import shutil
import time
from enum import Enum
from glob import glob
from sklearn.metrics import roc_auc_score

import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu.npu_config import NPURunConfig

from rec_sdk_common.communication.hccl.hccl_info import get_rank_id, get_local_rank_size
from mx_rec.constants.constants import LIBREC_EOS_OPS_SO, LIBREC_TF_REC_V1_CPU_SO
from mx_rec.core.asc.helper import FeatureSpec, get_asc_insert_func
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.util.initialize import ConfigInitializer

from examples.util.path_validator import validate_read_file

MODEL_NAME = None
SSD_DATA_PATH = ["ssd_data"]
SHUFFLE_SEED = 128
random.seed(SHUFFLE_SEED)
logger = None

train_steps = 0
eval_steps = 0
max_train_steps = 0

rank_id = int(os.getenv("RANK_ID")) if os.getenv("RANK_ID") else None
rank_size = int(os.getenv("TRAIN_RANK_SIZE")) if os.getenv("TRAIN_RANK_SIZE") else None
interval = int(os.getenv("INTERVAL")) if os.getenv("INTERVAL") else None

try:
    use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
    use_multi_lookup = bool(int(os.getenv("USE_MULTI_LOOKUP", 0)))
    MODIFY_GRAPH_FLAG = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
    USE_DP = bool(int(os.getenv("USE_DP", 0)))
    use_faae = bool(int(os.getenv("USE_FAAE", 0)))
    huge_tle_enable = bool(int(os.getenv("HUGE_TLB_ENABLE", 0)))
except ValueError as err:
    raise ValueError("please correctly config USE_DYNAMIC_EXPANSION or USE_MULTI_LOOKUP or USE_FAAE "
                     "or USE_MODIFY_GRAPH or HUGE_TLB_ENABLE only 0 or 1 is supported.") from err


class CacheModeEnum(Enum):
    HBM = "HBM"
    DDR = "DDR"
    SSD = "SSD"


class LearningRateScheduler:
    """
    LR Scheduler combining Polynomial Decay with Warmup at the beginning.
    TF-based cond operations necessary for performance in graph mode.
    """

    def __init__(self, base_lr_dense, base_lr_sparse, warmup_steps, decay_start_step, decay_steps):
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
        self.decay_steps = tf.constant(decay_steps)
        self.decay_end_step = decay_start_step + decay_steps  # 65041
        self.poly_power = 2.0
        self.base_lr_dense = base_lr_dense
        self.base_lr_sparse = base_lr_sparse

    def calc(self, global_step):
        # used for the warmup stage
        warmup_step = tf.cast(1 / self.warmup_steps, tf.float32)
        lr_factor_warmup = 1 - tf.cast(self.warmup_steps - global_step, tf.float32) * warmup_step
        lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)
        # used for the constant stage
        lr_factor_constant = tf.cast(1.0, tf.float32)

        # used for the decay stage
        lr_factor_decay = (self.decay_end_step - global_step) / self.decay_steps
        lr_factor_decay = tf.math.pow(lr_factor_decay, self.poly_power)
        lr_factor_decay = tf.cast(lr_factor_decay, tf.float32)
        sparse_after_decay = tf.cast(1 / self.decay_steps, tf.float32)

        lr_factor_decay_sparse = tf.cond(
            global_step < self.decay_end_step,
            lambda: lr_factor_decay,
            lambda: sparse_after_decay,
        )

        lr_factor_decay_dense = tf.cond(
            global_step < self.decay_end_step,
            lambda: lr_factor_decay,
            lambda: sparse_after_decay,
        )

        poly_schedule_sparse = tf.cond(
            global_step < self.decay_start_step,
            lambda: lr_factor_constant,
            lambda: lr_factor_decay_sparse,
        )

        poly_schedule_dense = tf.cond(
            global_step < self.decay_start_step,
            lambda: lr_factor_constant,
            lambda: lr_factor_decay_dense,
        )

        lr_factor_sparse = tf.cond(
            global_step < self.warmup_steps, lambda: lr_factor_warmup, lambda: poly_schedule_sparse
        )

        lr_factor_dense = tf.cond(
            global_step < self.warmup_steps, lambda: lr_factor_warmup, lambda: poly_schedule_dense
        )

        lr_sparse = self.base_lr_sparse * lr_factor_sparse
        lr_dense = self.base_lr_dense * lr_factor_dense
        if MODEL_NAME == "DCNv2":
            lr_sparse = tf.cond(lr_sparse >= 0.0, lambda: lr_sparse, lambda: tf.cast(0.0, tf.float32))
            lr_sparse = tf.math.minimum(lr_sparse, tf.cast(10.0, tf.float32))
        elif MODEL_NAME == "DLRM":
            lr_sparse = tf.math.maximum(lr_sparse, tf.cast(0.0, tf.float32))
            lr_sparse = tf.math.minimum(lr_sparse, tf.cast(10.0, tf.float32))

        return lr_dense, lr_sparse


class Config:
    def __init__(self, ):
        try:
            self.rank_id = get_rank_id()
        except RuntimeError:
            self.rank_id = None
        except ValueError as exp:
            raise ValueError(f"Config get_rank_id ValueError:{exp}") from exp

        tmp = os.getenv("TRAIN_RANK_SIZE")
        if tmp is None:
            raise ValueError("please export TRAIN_RANK_SIZE")
        self.rank_size = int(tmp)

        self.data_path = os.getenv("DLRM_CRITEO_DATA_PATH")
        self.train_file_pattern = "train"
        self.test_file_pattern = "test"

        self.batch_size = 8192
        self.line_per_sample = 1024
        self.train_epoch = 3
        self.test_epoch = 1
        self.perform_shuffle = False

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.int64

        self.feat_cnt = 26
        self.__set_emb_table_size()

        self.field_num = 26
        self.send_count = 46000 // self.rank_size

        self.emb_dim = 128
        self.hashtable_threshold = 1

        self.USE_PIPELINE_TEST = False
        if MODEL_NAME == "DLRM":
            self.use_lazy_adam_optimizer = False
            self.use_fusion_optim = False            

        # 动态学习率
        GLOBAL_BATCH_SIZE = 8192 * 8
        LR_SCHEDULE_STEPS = [
            int(2750 * 55296 / GLOBAL_BATCH_SIZE),
            int(49315 * 55296 / GLOBAL_BATCH_SIZE),
            int(27772 * 55296 / GLOBAL_BATCH_SIZE),
        ]
        self.global_step = tf.Variable(0, trainable=False)
        _lr_scheduler = LearningRateScheduler(
            28.443,
            33.71193,
            LR_SCHEDULE_STEPS[0],
            LR_SCHEDULE_STEPS[1],
            LR_SCHEDULE_STEPS[2],
        )

        if MODEL_NAME == "DCNv2_multihot":
            self.batch_size = int(os.getenv("BATCH_SIZE"))
            self.train_epoch = 1
            self.use_adacons = bool(os.getenv("USE_ADACONS"))
            self.optimizer = os.getenv("OPTIMIZER")
            self.loss_scale = int(os.getenv("LOSS_SCALE"))
            self.send_count = 680000 // self.rank_size
            GLOBAL_BATCH_SIZE = self.batch_size * self.rank_size
            LR_SCHEDULE_STEPS = [
                int(int(os.getenv("WARM_STEPS")) / GLOBAL_BATCH_SIZE),
                int(int(os.getenv("DECAY_START_STEPS")) / GLOBAL_BATCH_SIZE),
                int(int(os.getenv("DECAY_STEPS")) / GLOBAL_BATCH_SIZE),
            ]
            _lr_scheduler = LearningRateScheduler(
                int(os.getenv("DENSE_LR")),
                int(os.getenv("SPARSE_LR")),
                LR_SCHEDULE_STEPS[0],
                LR_SCHEDULE_STEPS[1],
                LR_SCHEDULE_STEPS[2],
            )

        if MODEL_NAME == "WideDeep":
            self.batch_size = 4096
            self.line_per_sample = 1
            self.train_epoch = 1
            self.test_epoch = 9
            self.emb_dim = 8
            _lr_scheduler = LearningRateScheduler(
                0.001,
                0.001,
                LR_SCHEDULE_STEPS[0],
                LR_SCHEDULE_STEPS[1],
                LR_SCHEDULE_STEPS[2],
            )

        self.learning_rate = _lr_scheduler.calc(self.global_step)

    def __set_emb_table_size(self):
        self.cache_mode = os.getenv("CACHE_MODE")
        if self.cache_mode is None:
            raise ValueError("please export CACHE_MODE environment variable, support:[HBM, DDR, SSD]")

        if self.cache_mode == CacheModeEnum.HBM.value:
            self.dev_vocab_size = 24_000_000 * self.rank_size
            if MODEL_NAME == "DCNv2_multihot":
                self.dev_vocab_size = 18_000_000 * self.rank_size
            elif MODEL_NAME == "WideDeep":
                self.dev_vocab_size = 14_000_000 * self.rank_size
            self.host_vocab_size = 0
        elif self.cache_mode == CacheModeEnum.DDR.value:
            self.dev_vocab_size = 500_000 * self.rank_size
            self.host_vocab_size = 24_000_000 * self.rank_size
        elif self.cache_mode == CacheModeEnum.SSD.value:
            self.dev_vocab_size = 100_000 * self.rank_size
            self.host_vocab_size = 2_000_000 * self.rank_size
            self.ssd_vocab_size = 24_000_000 * self.rank_size
        else:
            raise ValueError(f"get CACHE_MODE:{self.cache_mode}, expect in [HBM, DDR, SSD]")

    def get_emb_table_cfg(self):
        if self.cache_mode == CacheModeEnum.HBM.value:
            return {"device_vocabulary_size": self.dev_vocab_size}
        elif self.cache_mode == CacheModeEnum.DDR.value:
            return {"device_vocabulary_size": self.dev_vocab_size,
                    "host_vocabulary_size": self.host_vocab_size}
        elif self.cache_mode == CacheModeEnum.SSD.value:
            return {"device_vocabulary_size": self.dev_vocab_size,
                    "host_vocabulary_size": self.host_vocab_size,
                    "ssd_vocabulary_size": self.ssd_vocab_size,
                    "ssd_data_path": SSD_DATA_PATH}
        else:
            raise RuntimeError(f"get CACHE_MODE:{self.cache_mode}, check Config.__set_emb_table_size implementation")


def sess_config(dump_data=False, dump_path="./dump_output", dump_steps="0|1|2"):
    session_config = tf.ConfigProto(allow_soft_placement=False,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = False
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = 1
    # 可选配置level0:pairwise;level1:pairwise
    custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes("level0:fullmesh;level1:fullmesh")
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = 10
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_impl_mode.ini")
    custom_op.parameter_map["op_execute_timeout"].i = 2000
    custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(
        str(13 * 1024 * 1024 * 1024))  # total 31 need 13;
    custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(18 * 1024 * 1024 * 1024))  # need 25
    custom_op.parameter_map["stream_max_parallel_num"].s = tf.compat.as_bytes("DNN_VM_AICPU:3,AIcoreEngine:3")

    if dump_data:
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_steps)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return session_config


def get_npu_run_config():
    session_config = tf.ConfigProto(allow_soft_placement=False,
                                    log_device_placement=False)

    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    run_config = NPURunConfig(
        save_summary_steps=1000,
        save_checkpoints_steps=100,
        keep_checkpoint_max=5,
        session_config=session_config,
        log_step_count_steps=20,
        precision_mode='allow_mix_precision',
        enable_data_pre_proc=True,
        iterations_per_loop=1,
        jit_compile=False,
        op_compiler_cache_mode="enable",
        HCCL_algorithm="level0:fullmesh;level1:fullmesh"  # 可选配置：level0:pairwise;level1:pairwise
    )
    return run_config


def add_timestamp_func(batch):
    timestamp = import_host_pipeline_ops(LIBREC_TF_REC_V1_CPU_SO).return_timestamp(tf.cast(batch['label'],
                                                                                           dtype=tf.int64))
    # tf.constant(np.random.randint(1,1688109060,1)), tf.int64))
    batch["timestamp"] = timestamp
    return batch


def make_batch_and_iterator(config, feature_spec_list, is_training, dump_graph, is_use_faae=False):
    if config.USE_PIPELINE_TEST:
        num_parallel = 1
    else:
        num_parallel = 8

    extract_fn = get_extract_fn(config)

    batch_size = config.batch_size // config.line_per_sample
    num_devices = config.rank_size
    device_index = config.rank_id
    
    if is_training:
        files_list = glob(os.path.join(config.data_path, config.train_file_pattern) + '/*.tfrecord')
        device_files = files_list[device_index::num_devices]
    else:
        files_list = glob(os.path.join(config.data_path, config.test_file_pattern) + '/*.tfrecord')
        device_files = files_list

    for check_file in files_list:
        validate_read_file(check_file)

    dataset = tf.data.TFRecordDataset(files_list, num_parallel_reads=num_parallel)
    dataset = dataset.shard(config.rank_size, config.rank_id)
    if MODEL_NAME == "DCNv2_multihot":
        dataset = tf.data.TFRecordDataset(device_files, num_parallel_reads=num_parallel)

    if is_training:
        dataset = dataset.shuffle(batch_size * 1000, seed=SHUFFLE_SEED)
        dataset = dataset.repeat(config.train_epoch)
    else:
        dataset = dataset.repeat(config.test_epoch)
    dataset = dataset.map(extract_fn, num_parallel_calls=num_parallel).batch(batch_size, drop_remainder=True)
    dataset = dataset.map(reshape_fn, num_parallel_calls=num_parallel)
    if MODEL_NAME == "WideDeep":
        dataset = dataset.map(map_fn, num_parallel_calls=num_parallel)
    if is_use_faae:
        dataset = dataset.map(add_timestamp_func)

    if not MODIFY_GRAPH_FLAG:
        librec = import_host_pipeline_ops(LIBREC_EOS_OPS_SO)
        channel_id = 0 if is_training else 1
        if MODEL_NAME == "WideDeep":
            dataset = dataset.eos_map(librec, channel_id, max_train_steps, eval_steps)
        if MODEL_NAME == "MMOE":
            dataset = dataset.eos_map(librec, channel_id, -1, eval_steps)
        insert_fn = get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=is_training, dump_graph=dump_graph)
        dataset = dataset.map(insert_fn)

    dataset = dataset.prefetch(100)

    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    return batch, iterator


def get_extract_fn(config):
    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.compat.v1.FixedLenFeature(shape=(config.line_per_sample,), dtype=tf.int64),
            'sparse_feature': tf.compat.v1.FixedLenFeature(shape=(26 * config.line_per_sample,), dtype=tf.int64),
            'dense_feature': tf.compat.v1.FixedLenFeature(shape=(13 * config.line_per_sample,), dtype=tf.float32),
        }
        if MODEL_NAME == "DCNv2_multihot":
            features = {
                'label': tf.compat.v1.FixedLenFeature(shape=(config.line_per_sample,), dtype=tf.int64),
                'sparse_feature': tf.compat.v1.FixedLenFeature(shape=(214 * config.line_per_sample,), dtype=tf.int64),
                'dense_feature': tf.compat.v1.FixedLenFeature(shape=(13 * config.line_per_sample,), dtype=tf.float32),
            }
        if MODEL_NAME == "WideDeep":
            features = {
                'label': tf.compat.v1.FixedLenFeature(shape=(config.line_per_sample,), dtype=tf.int64),
                'sparse_feature': tf.compat.v1.FixedLenFeature(shape=(26 * config.line_per_sample,), dtype=tf.int64),
                'dense_feature': tf.compat.v1.FixedLenFeature(shape=(13 * config.line_per_sample,), dtype=tf.int64),
            }
        if MODEL_NAME == "MMOE":
            features = {
                'label': tf.compat.v1.FixedLenFeature(shape=(2 * config.line_per_sample,), dtype=tf.int64),
                'sparse_feature': tf.compat.v1.FixedLenFeature(shape=(29 * config.line_per_sample,), dtype=tf.int64),
                'dense_feature': tf.compat.v1.FixedLenFeature(shape=(11 * config.line_per_sample,), dtype=tf.float32),
            }
        sample = tf.compat.v1.parse_single_example(data_record, features)
        return sample
    
    return extract_fn


def reshape_fn(batch):
    if MODEL_NAME == "DCNv2" or MODEL_NAME == "DLRM":
        batch['label'] = tf.reshape(batch['label'], [-1, 1])
        batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 13])
        batch['dense_feature'] = tf.math.log(batch['dense_feature'] + 3.0)
        batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 26])
    if MODEL_NAME == "DCNv2_multihot":
        batch['label'] = tf.reshape(batch['label'], [-1, 1])
        batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 13])
        batch['dense_feature'] = tf.math.log(batch['dense_feature'] + 3.0)
        batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 214])
    if MODEL_NAME == "MMOE":
        batch['label'] = tf.reshape(batch['label'], [-1, 2])
        batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 11])
        batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 29])
    if MODEL_NAME == "WideDeep":
        batch['label'] = tf.reshape(batch['label'], [-1, 1])
        batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 13])
        batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 26])
    return batch


def map_fn(batch):
    new_batch = batch
    new_batch['sparse_feature'] = tf.concat([batch['dense_feature'], batch['sparse_feature']], axis=1)
    return new_batch


def evaluate(sess, eval_model, eval_iterator, cfg):
    logger.info("read_test dataset")
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
    logger.info("eval begin")

    while not finished:
        try:
            eval_current_steps += 1
            eval_start = time.time()
            eval_loss, pred, label = sess.run([eval_model.get("loss"), eval_model.get("pred"), eval_label])
            eval_cost = time.time() - eval_start
            eval_qps = (1 / eval_cost) * rank_size * cfg.batch_size
            log_loss_list += list(eval_loss.reshape(-1))
            pred_list += list(pred.reshape(-1))
            label_list += list(label.reshape(-1))
            logger.info(f"eval current_steps: {eval_current_steps}, qps: {eval_qps}")
            if eval_current_steps == eval_steps:
                finished = True
        except tf.errors.OutOfRangeError:
            finished = True
    auc = roc_auc_score(label_list, pred_list)
    mean_log_loss = np.mean(log_loss_list)
    return auc, mean_log_loss


def evaluate_fix(step, sess, eval_model, eval_iterator):
    logger.info("read_test dataset evaluate_fix")
    if not MODIFY_GRAPH_FLAG:
        sess.run([eval_iterator.initializer])
    else:
        sess.run([ConfigInitializer.get_instance().train_params_config.get_initializer(False)])
    log_loss_list = []
    pred_list = []
    label_list = []
    eval_current_steps = 0
    finished = False
    logger.info("eval begin")
    while not finished:
        try:
            eval_current_steps += 1
            eval_loss, pred, label = sess.run([eval_model.get("loss"), eval_model.get("pred"), eval_model.get("label")])
            log_loss_list += list(eval_loss.reshape(-1))
            pred_list += list(pred.reshape(-1))
            label_list += list(label.reshape(-1))
            logger.info(f"eval current_steps: {eval_current_steps}")

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
            logger.info("All saved!!!!!!!!!!")
            break
        else:
            logger.info("Waitting for saving numpy!!!!!!!!")
            time.sleep(1)
            continue

    auc = roc_auc_score(label_list, pred_list)
    mean_log_loss = np.mean(log_loss_list)
    return auc, mean_log_loss


def create_feature_spec_list(cfg, use_timestamp=False):
    access_threshold = None
    eviction_threshold = None
    if use_timestamp:
        access_threshold = 1000
        eviction_threshold = 180

    feature_spec_list = [FeatureSpec("sparse_feature", table_name="sparse_embeddings", batch_size=cfg.batch_size,
                                     access_threshold=access_threshold, eviction_threshold=eviction_threshold)]
    if MODEL_NAME == "WideDeep":
        feature_spec_list = [
            FeatureSpec("sparse_feature", table_name="wide_embeddings", batch_size=cfg.batch_size,
                        access_threshold=access_threshold, eviction_threshold=eviction_threshold),
            FeatureSpec("sparse_feature", table_name="deep_embeddings", batch_size=cfg.batch_size,
                        access_threshold=access_threshold, eviction_threshold=eviction_threshold)
        ]
    if use_multi_lookup:
        if MODEL_NAME == "WideDeep":
            feature_spec_list.extend([FeatureSpec("sparse_feature", table_name="wide_embeddings",
                                                  batch_size=cfg.batch_size,
                                                  access_threshold=access_threshold,
                                                  eviction_threshold=eviction_threshold),
                                      FeatureSpec("sparse_feature", table_name="deep_embeddings",
                                                  batch_size=cfg.batch_size,
                                                  access_threshold=access_threshold,
                                                  eviction_threshold=eviction_threshold)])
        else:
            feature_spec_list.append(FeatureSpec("sparse_feature", table_name="sparse_embeddings",
                                                 batch_size=cfg.batch_size,
                                                 access_threshold=access_threshold,
                                                 eviction_threshold=eviction_threshold))
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list


def clear_saved_model() -> None:
    def _del_related_dir(del_path: str) -> None:
        if not os.path.isabs(del_path):
            del_path = os.path.join(os.getcwd(), del_path)
        dirs = glob(del_path)
        if get_rank_id() % get_local_rank_size() == 0:
            for sub_dir in dirs:
                shutil.rmtree(sub_dir, ignore_errors=True)
                logger.info(f"delete dir:{sub_dir}")

    _del_related_dir("/root/ascend/log/*")
    if MODEL_NAME == "DLRM" or MODEL_NAME == "WideDeep" or MODEL_NAME == "MMOE":
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
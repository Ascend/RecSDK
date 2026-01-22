#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from npu_bridge.hccl import hccl_ops  # type: ignore
from npu_bridge.npu_init import util  # type: ignore
import mxrec
from rec_sdk_common.communication.hccl.hccl_info import get_rank_size

from model import DLRM, LOSS_OP_NAME, LABEL_OP_NAME, PRED_OP_NAME, SPARSE_FEATURE_LEN, DENSE_FEATURE_LEN
from config import Config, sess_config
from optimizer import get_dense_and_sparse_optimizer
from logger import logger


class TaskRunner:
    def __init__(
        self,
        config: Config,
        toml_config: Dict[str, Union[str, int, float]],
        seed: Optional[int] = None,
    ):
        self._config = config
        self._toml_config = toml_config
        self._seed = seed
        self._sess = tf.compat.v1.Session(config=sess_config(dump_data=False))

    def __del__(self):
        self._sess.close()

    def run(self):
        train_model, train_iterator = self._model_forward(is_training=True)
        eval_model, eval_iterator = self._model_forward(is_training=False)

        train_ops = self._get_train_ops(train_model)
        with tf.control_dependencies(train_ops):
            train_ops = tf.no_op()
            self._config.learning_rate = [self._config.learning_rate[0], self._config.learning_rate[1]]

        # Training task init.
        self._sess.run(mxrec.get_init_hashtable_op())
        self._sess.run(tf.compat.v1.global_variables_initializer())
        self._sess.run(train_iterator.initializer)

        cost_sum = 0
        best_auc = 0
        iteration_per_loop = 10
        train_ops = util.set_iteration_per_loop(self._sess, train_ops, iteration_per_loop)

        # Start to training and evaluate.
        start_step = 1
        for i in range(start_step, start_step + self._config.train_steps):
            logger.info("################    training at step %d    ################", i * iteration_per_loop)
            start_time = time.time()

            try:
                _, loss = self._sess.run([train_ops, train_model.get(LOSS_OP_NAME)])
                lr = self._sess.run(self._config.learning_rate)
                global_step = self._sess.run(self._config.global_step)
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for training.")
                break

            end_time = time.time()
            cost_time = end_time - start_time
            qps = (1 / cost_time) * self._config.rank_size * self._config.batch_size * iteration_per_loop
            cost_sum += cost_time
            logger.info("Training step: %d; training loss: %s.", i * iteration_per_loop, loss)
            logger.info("Training step: %d; lr: %s.", i * iteration_per_loop, lr)
            logger.info("Global step: %d", global_step)
            logger.info("Training step: %d; current QPS: %s.", i * iteration_per_loop, qps)

            if i % self._config.train_interval == 0:
                test_auc, test_mean_log_loss = self._evaluate(eval_model, eval_iterator)
                logger.info("Test auc: %s; log_loss: %s", test_auc, test_mean_log_loss)
                best_auc = max(best_auc, test_auc)
                logger.info("Training step: %d, best auc: %s", i * iteration_per_loop, best_auc)

        logger.info("Training finished, the best auc is %s.", best_auc)

    def _get_train_ops(self, train_model: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
        trainable_variables = []
        sparse_variables = mxrec.get_sparse_embedding()
        dense_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        trainable_variables.extend(dense_variables)
        trainable_variables.extend(sparse_variables)

        # Multi task training.
        train_ops = []
        dense_optimizer, sparse_optimizer = get_dense_and_sparse_optimizer(self._config)
        loss = train_model.get(LOSS_OP_NAME)
        # Do dense optimization.
        grads = tf.gradients(loss, trainable_variables)
        grads_and_vars = [(grad, variable) for grad, variable in zip(grads, trainable_variables)]
        dense_grads_and_vars = []
        for grad, var in grads_and_vars[:-len(sparse_variables)]:
            if grad is None:
                continue
            if get_rank_size() == 1:
                dense_grads_and_vars.append((grad, var))
                continue
            grad = hccl_ops.allreduce(grad, "sum")
            dense_grads_and_vars.append((grad / get_rank_size(), var))

        train_ops.append(dense_optimizer.apply_gradients(dense_grads_and_vars))

        # Do sparse optimization.
        sparse_grads_and_vars = list(grads_and_vars[-len(sparse_variables):])
        train_ops.append(sparse_optimizer.apply_gradients(sparse_grads_and_vars))

        # Dynamic learning rate update.
        train_ops.extend(
            [
                self._config.global_step.assign(self._config.global_step + 1),
                self._config.learning_rate[0],
                self._config.learning_rate[1],
            ]
        )

        return train_ops

    def _evaluate(
        self, eval_model: Dict[str, tf.Tensor], eval_iterator: tf.compat.v1.data.Iterator
    ) -> Tuple[float, np.ndarray]:
        log_loss_list = []
        pred_list = []
        label_list = []
        eval_current_steps = 0
        finished = False

        self._sess.run([eval_iterator.initializer])
        while not finished:
            try:
                eval_current_steps += 1
                eval_start = time.time()
                eval_loss, pred, label = self._sess.run(
                    [eval_model.get(LOSS_OP_NAME), eval_model.get(PRED_OP_NAME), eval_model.get(LABEL_OP_NAME)]
                )
                eval_cost = time.time() - eval_start
                qps_eval = (1 / eval_cost) * get_rank_size() * self._config.batch_size
                log_loss_list += list(eval_loss.reshape(-1))
                pred_list += list(pred.reshape(-1))
                label_list += list(label.reshape(-1))
                logger.info("Evaluation step: %d, qps: %s.", eval_current_steps, qps_eval)
                if eval_current_steps == self._config.eval_steps:
                    finished = True
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for evaluation.")
                finished = True

        auc = roc_auc_score(label_list, pred_list)
        mean_log_loss = np.mean(log_loss_list)

        return auc, mean_log_loss

    def _make_batch_and_iterator(
        self, is_training: bool = True, num_parallel: int = 8
    ) -> Tuple[Dict[str, tf.Tensor], tf.compat.v1.data.Iterator]:
        def _extract_fn(data_record: tf.Tensor):
            features = {
                # Extract features using the keys set during creation.
                "label": tf.compat.v1.FixedLenFeature(shape=(self._config.line_per_sample,), dtype=tf.int64),
                "sparse_feature": tf.compat.v1.FixedLenFeature(
                    shape=(SPARSE_FEATURE_LEN * self._config.line_per_sample,), dtype=tf.int64
                ),
                "dense_feature": tf.compat.v1.FixedLenFeature(
                    shape=(DENSE_FEATURE_LEN * self._config.line_per_sample,), dtype=tf.float32
                ),
            }
            sample = tf.compat.v1.parse_single_example(data_record, features)
            return sample

        def _reshape_fn(batch: Dict[str, tf.Tensor]):
            batch["label"] = tf.reshape(batch["label"], [-1, 1])
            batch["dense_feature"] = tf.reshape(batch["dense_feature"], [-1, DENSE_FEATURE_LEN])
            batch["dense_feature"] = tf.math.log(tf.math.add(batch["dense_feature"], 3.0))
            batch["sparse_feature"] = tf.reshape(batch["sparse_feature"], [-1, SPARSE_FEATURE_LEN])
            return batch

        if is_training:
            files_list = glob(os.path.join(self._config.data_path, self._config.train_file_pattern) + "/*.tfrecord")
        else:
            files_list = glob(os.path.join(self._config.data_path, self._config.test_file_pattern) + "/*.tfrecord")

        dataset = tf.data.TFRecordDataset(files_list, num_parallel_reads=num_parallel)
        batch_size = self._config.batch_size // self._config.line_per_sample
        dataset = dataset.shard(self._config.rank_size, self._config.rank_id)
        if is_training:
            # This 1000 is the buffer size for shuffling.
            dataset = dataset.shuffle(buffer_size=batch_size * 1000, seed=self._seed)
        if is_training:
            dataset = dataset.repeat(self._config.train_epoch)
        else:
            dataset = dataset.repeat(self._config.test_epoch)
        dataset = dataset.map(_extract_fn, num_parallel_calls=num_parallel).batch(batch_size, drop_remainder=True)
        dataset = dataset.map(_reshape_fn, num_parallel_calls=num_parallel)
        dataset = dataset.prefetch(self._config.prefetch_num)

        iterator = dataset.make_initializable_iterator()
        batch = iterator.get_next()
        return batch, iterator

    def _model_forward(self, is_training: bool = True) -> Tuple[Dict[str, tf.Tensor], tf.compat.v1.data.Iterator]:
        batch, iterator = self._make_batch_and_iterator(is_training)

        table = mxrec.get_embedding_table(
            name="sparse_embedding_table",
            dimension=self._config.emb_dim,
            device_vocabulary_size=self._config.dev_vocab_size,
            initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=self._seed
            ),
            key_dtype=self._config.key_type,
            value_dtype=self._config.value_type,
        )

        ids = batch.get("sparse_feature")
        embedding = mxrec.embedding_lookup(table, ids)

        model_output = DLRM(self._toml_config).build_model(
            embedding=embedding,
            dense_feature=batch.get("dense_feature"),
            label=batch.get("label"),
            seed=self._seed,
        )

        return model_output, iterator

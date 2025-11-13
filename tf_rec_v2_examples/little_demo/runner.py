#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import glob
import re
from typing import Dict, List, Tuple

import tensorflow as tf
import mxrec

from config import Config, sess_config
from dataset import generate_dataset
from logger import logger
from model import Model

if tf.__version__.startswith("1"):
    from npu_bridge.hccl import hccl_ops  # type: ignore
else:
    from npu_device.compat.v1.hccl import hccl_ops  # type: ignore


class TaskRunner:
    def __init__(
        self,
        train_steps: int = 200,
        train_interval: int = 100,
        eval_steps: int = 10,
        batch_number: int = 200,
        is_deterministic: bool = False,
    ):
        self._train_steps = train_steps
        self._train_interval = train_interval
        self._eval_steps = eval_steps
        self._batch_number = batch_number

        self._sess = tf.compat.v1.Session(config=sess_config(is_deterministic=is_deterministic))

    def __del__(self):
        self._sess.close()

    def run(self, mode: str = Config.train_and_evaluate, saved_path: str = "saved-model"):
        if mode == Config.train_and_evaluate:
            self._train_and_evaluate(saved_path)
        elif mode == Config.load_and_train:
            self._load_and_train(saved_path)
        elif mode == Config.predict:
            self._predict(saved_path)
        else:
            raise ValueError(f"invalid mode: {mode}")

    def _train_and_evaluate(self, saved_path: str):
        train_model, train_iterator = self._model_forward()
        eval_model, eval_iterator = self._model_forward()

        train_ops = _get_train_ops(train_model)
        init_hashtable_op = mxrec.get_init_hashtable_op()
        self._sess.run(init_hashtable_op)
        self._sess.run(train_iterator.initializer)
        self._sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        tf_save_path = os.path.join(saved_path, Config.ckpt_name)
        embedding_table_saver = mxrec.EmbeddingTableSaver(mxrec.get_existing_tables())

        for i in range(self._train_steps):
            logger.info("################    training at step %d    ################", i + 1)
            try:
                _, loss = self._sess.run([train_ops, train_model.loss])
                logger.info("Training loss: %s.", loss)
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for training.")
                break

            if (i + 1) % self._train_interval == 0:
                self._evaluate(eval_iterator, eval_model)

                saver.save(self._sess, tf_save_path, global_step=i + 1)
                embedding_table_saver.save(self._sess, saved_path, i + 1)

                logger.info("The saved path: %s.", saved_path)

    def _load_and_train(self, saved_path: str):
        train_model, train_iterator = self._model_forward()
        eval_model, eval_iterator = self._model_forward()

        train_ops = _get_train_ops(train_model)
        init_hashtable_op = mxrec.get_init_hashtable_op()
        self._sess.run(init_hashtable_op)
        self._sess.run(train_iterator.initializer)

        tf_save_path = os.path.join(saved_path, Config.ckpt_name)
        load_step = _get_load_step(tf_save_path)

        saver = tf.compat.v1.train.Saver()
        embedding_table_saver = mxrec.EmbeddingTableSaver(mxrec.get_existing_tables())

        saver.restore(self._sess, tf_save_path + f"-{load_step}")
        embedding_table_saver.load(self._sess, saved_path, load_step)

        start_step = load_step + 1
        for i in range(start_step, start_step + self._train_steps):
            logger.info("################    training at step %d    ################", i)
            try:
                _, loss = self._sess.run([train_ops, train_model.loss])
                logger.info("Training loss: %s.", loss)
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for training.")
                break

            if i % self._train_interval == 0:
                self._evaluate(eval_iterator, eval_model)

                saver.save(self._sess, tf_save_path, global_step=i)
                embedding_table_saver.save(self._sess, saved_path, i)

                logger.info("The saved path: %s.", saved_path)

    def _predict(self, saved_path: str):
        eval_model, eval_iterator = self._model_forward()

        init_hashtable_op = mxrec.get_init_hashtable_op()
        self._sess.run(init_hashtable_op)

        saver = tf.compat.v1.train.Saver()
        tf_save_path = os.path.join(saved_path, Config.ckpt_name)
        load_step = _get_load_step(tf_save_path)
        saver.restore(self._sess, tf_save_path + f"-{load_step}")

        embedding_table_saver = mxrec.EmbeddingTableSaver(mxrec.get_existing_tables())
        embedding_table_saver.load(self._sess, saved_path, load_step)

        self._evaluate(eval_iterator, eval_model)

    def _evaluate(self, eval_iterator: tf.compat.v1.data.Iterator, eval_model: Model):
        self._sess.run(eval_iterator.initializer)
        for i in range(self._eval_steps):
            logger.info("################    evaluation at step %d    ################", i + 1)
            try:
                self._sess.run(eval_model.loss)
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for evaluation.")
                break

    def _model_forward(self) -> Tuple[Model, tf.compat.v1.data.Iterator]:
        batch, iterator = _make_batch_and_iterator(self._batch_number)
        user_table = mxrec.get_embedding_table(
            name="user_table",
            dimension=Config.user_hashtable_dim,
            device_vocabulary_size=Config.user_vocab_size,
            initializer=tf.truncated_normal_initializer(-0.01, 0.01, Config.random_seed),
        )
        item_table = mxrec.get_embedding_table(
            name="item_table",
            dimension=Config.item_hashtable_dim,
            device_vocabulary_size=Config.item_vocab_size,
            initializer=tf.truncated_normal_initializer(-0.01, 0.01, Config.random_seed),
        )

        embedding_list = []
        for table, ids in zip([user_table, item_table], [batch.get(Config.user_ids), batch.get(Config.item_ids)]):
            embedding = mxrec.embedding_lookup(table, ids)
            reduced_embedding = tf.reduce_sum(embedding, axis=1, keepdims=False)
            embedding_list.append(reduced_embedding)

        model = Model()
        model(embedding_list, batch.get(Config.label_0), batch.get(Config.label_1))

        return model, iterator


def _get_train_ops(train_model: Model) -> List[tf.Tensor]:
    train_ops = []

    # Do dense optimization.
    dense_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=Config.learning_rate)
    grads = dense_optimizer.compute_gradients(train_model.loss, var_list=dense_variables)
    avg_grads = []
    for grad, var in grads:
        if Config.rank_size > 1:
            grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
        if grad is not None:
            avg_grads.append((grad, var))
    train_ops.append(dense_optimizer.apply_gradients(avg_grads))

    # Do sparse optimization.
    sparse_optimizer = mxrec.AdamWOptimizer(learning_rate=Config.learning_rate)
    sparse_embeddings = mxrec.get_sparse_embedding()
    sparse_grads = tf.gradients(train_model.loss, sparse_embeddings)
    train_ops.append(sparse_optimizer.apply_gradients(zip(sparse_grads, sparse_embeddings)))

    return train_ops


def _make_batch_and_iterator(batch_number: int = 200) -> Tuple[Dict[str, tf.Tensor], tf.compat.v1.data.Iterator]:
    dataset = generate_dataset(batch_number)
    dataset = dataset.prefetch(Config.prefetch_num)
    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    return batch, iterator


def _get_load_step(saved_path: str) -> int:
    model_file = glob.glob(saved_path + "-*")
    pattern = f".*{Config.ckpt_name}-([0-9]+).*"
    latest_step = -1
    for file_path in model_file:
        match = re.search(pattern, file_path)
        if match and match.groups():
            step = int(match.groups()[0])
            latest_step = max(latest_step, step)
    if latest_step == -1:
        raise ValueError("latest model not found")
    return latest_step

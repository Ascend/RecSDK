# coding: UTF-8

#  Copyright (C)  2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import logging

import tensorflow as tf
from config import sess_config

from mx_rec.util.initialize import get_initializer, get_rank_id, get_rank_size, clear_channel
from mx_rec.util.variable import get_dense_and_sparse_variable
from mx_rec.util.tf_version_adapter import hccl_ops
from mx_rec.constants.constants import BaseEnum
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache


class UseMode(BaseEnum):
    TRAIN = "train"
    PREDICT = "predict"


class RunMode:

    def __init__(
            self, is_modify_graph: bool, optimizer_list: list, train_model, eval_model, train_iterator, eval_iterator,
            train_steps: int, infer_steps: int):
        self.is_modify_graph = is_modify_graph
        self.session = tf.compat.v1.Session(config=sess_config(dump_data=False))
        self.train_model = train_model
        self.train_iterator = train_iterator
        self.eval_model = eval_model
        self.eval_iterator = eval_iterator
        self.saver = tf.compat.v1.train.Saver()
        self.rank_id = get_rank_id()
        self.train_ops = []
        self.optimizer_list = optimizer_list
        self.epoch = 1
        self.train_steps = train_steps
        self.infer_steps = infer_steps

    def _infer(self):
        if self.is_modify_graph:
            self.session.run(get_initializer(False))
        else:
            self.session.run(self.eval_iterator.initializer)
        clear_channel(is_train_channel=False)
        for i in range(1, self.infer_steps + 1):
            logging.info("###############    infer at step %d    ################", i)
            try:
                self.session.run(self.eval_model.loss_list)
            except tf.errors.OutOfRangeError:
                logging.info(f"Encounter the end of Sequence for eval.")
                break

    def set_train_ops(self):
        dense_variables, sparse_variables = get_dense_and_sparse_variable()

        # multi task training
        for loss, (dense_optimizer, sparse_optimizer) in zip(self.train_model.loss_list, self.optimizer_list):
            # do dense optimization
            grads = dense_optimizer.compute_gradients(loss, var_list=dense_variables)
            avg_grads = []
            for grad, var in grads:
                if get_rank_size() > 1:
                    grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
                if grad is not None:
                    avg_grads.append((grad, var))
            # apply gradients: update variables
            self.train_ops.append(dense_optimizer.apply_gradients(avg_grads))

            if bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0))):
                from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET

                train_emb_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)
                train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)
                # do sparse optimization by addr
                local_grads = tf.gradients(loss, train_emb_list)  # local_embedding
                grads_and_vars = [(grad, address) for grad, address in zip(local_grads, train_address_list)]
                self.train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))
            else:
                # do sparse optimization
                sparse_grads = tf.gradients(loss, sparse_variables)
                grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
                self.train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))

    def train(self, eval_interval: int, saving_interval: int):
        self.set_train_ops()

        if self.is_modify_graph:
            logging.info("start to modifying graph")
            modify_graph_and_start_emb_cache(dump_graph=True)

        if self.is_modify_graph:
            self.session.run(get_initializer(True))
        else:
            self.session.run(self.train_iterator.initializer)
        self.session.run(tf.compat.v1.global_variables_initializer())

        for i in range(1, self.train_steps + 1):
            logging.info("################    training at step %d    ################", i)
            try:
                self.session.run([self.train_ops, self.train_model.loss_list])
            except tf.errors.OutOfRangeError:
                logging.info(f"Encounter the end of Sequence for training.")
                break
            else:
                if i % eval_interval == 0:
                    self.evaluate()

                if i % saving_interval == 0:
                    self.saver.save(self.session, f"./saved-model/model-{self.rank_id}", global_step=i)

        self.saver.save(self.session, f"./saved-model/model-{self.rank_id}", global_step=i)
        logging.info("################    training end    ################")

    def evaluate(self):
        logging.info("###############    start evaluate, epoch:%d    ################", self.epoch)
        self._infer()
        logging.info("###############    evaluate end, epoch::%d   ################", self.epoch)
        self.epoch += 1

    def predict(self):
        logging.info(f"###############    start predict    ################")
        import glob
        import re

        model_file = glob.glob(f"./saved-model/sparse-model-{self.rank_id}-*")
        if len(model_file) == 0:
            raise ValueError("model file not exit")

        # get the latest model
        pattern = f".*sparse-model-{self.rank_id}-([0-9]+).*"
        latest_step = -1
        for file_path in model_file:
            match = re.match(pattern, file_path)
            if match and match.groups():
                step = int(match.groups()[0])

                if step > latest_step:
                    latest_step = step
        if latest_step == -1:
            raise RuntimeError("latest model not found")

        self.saver.restore(self.session, f"./saved-model/model-{self.rank_id}-{latest_step}")
        self._infer()
        logging.info(f"###############    predict end    ################")


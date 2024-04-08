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

import os
from typing import List

import tensorflow as tf
from config import sess_config

from mx_rec.util.variable import get_dense_and_sparse_variable
from mx_rec.util.tf_version_adapter import hccl_ops
from mx_rec.constants.constants import BaseEnum
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
from mx_rec.util.log import logger
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.communication.hccl_ops import get_rank_id, get_rank_size


class UseMode(BaseEnum):
    TRAIN = "train"
    PREDICT = "predict"
    LOAD_AND_TRAIN = "load_and_train"


class RunMode:

    def __init__(
            self, is_modify_graph: bool, is_faae: bool, table_list: list, optimizer_list: list, train_model,
            eval_model, train_iterator, eval_iterator, max_train_steps: int, infer_steps: int, params: dict):
        self.is_modify_graph = is_modify_graph
        self.is_faae = is_faae
        self.session = tf.compat.v1.Session(config=sess_config(dump_data=False))
        self.train_model = train_model
        self.train_iterator = train_iterator
        self.eval_model = eval_model
        self.eval_iterator = eval_iterator
        self.rank_id = get_rank_id()
        self.train_ops = []
        self.table_list = table_list
        self.optimizer_list = optimizer_list
        self.epoch = 1
        self.max_train_steps = max_train_steps
        self.infer_steps = infer_steps
        self.use_one_shot = params.get("use_one_shot")
        self.train_batch = params.get("train_batch")
        self.eval_batch = params.get("eval_batch")

    def _infer(self):
        if not self.use_one_shot:
            initializer = self.eval_iterator.initializer if not self.is_modify_graph else \
                ConfigInitializer.get_instance().train_params_config.get_initializer(False)
            self.session.run(initializer)
        else:
            logger.debug(f"use one shot iterator and modify graph is `{self.is_modify_graph}`.")
        channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(False)
        import_host_pipeline_ops().clear_channel(channel_id)

        for i in range(1, self.infer_steps + 1):
            logger.info("###############    infer at step %d    ################", i)
            try:
                self.session.run(self.eval_model.loss_list)
            except tf.errors.OutOfRangeError:
                logger.info(f"Encounter the end of Sequence for eval.")
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

    def train(self, train_interval: int, saving_interval: int, if_load: bool, model_file: List[str]):
        self.set_train_ops()
        # In train mode, graph modify needs to be performed after compute gradients
        if self.is_modify_graph:
            logger.info("start to modifying graph")
            modify_graph_and_start_emb_cache(dump_graph=True)

        if not self.use_one_shot:
            initializer = self.train_iterator.initializer if not self.is_modify_graph else \
                ConfigInitializer.get_instance().train_params_config.get_initializer(True)
            self.session.run(initializer)
        else:
            logger.debug(f"use one shot iterator and modify graph is `{self.is_modify_graph}`.")

        self.saver = tf.compat.v1.train.Saver()
        start_step = 1

        if if_load:
            latest_step = get_load_step(model_file)
            start_step = latest_step + 1
            self.saver.restore(self.session, f"./saved-model/model-{latest_step}")
        else:
            self.session.run(tf.compat.v1.global_variables_initializer())

        for i in range(start_step, start_step + self.max_train_steps):
            logger.info("################    training at step %d    ################", i)
            try:
                self.session.run([self.train_ops, self.train_model.loss_list])
            except tf.errors.OutOfRangeError:
                logger.info(f"Encounter the end of Sequence for training.")
                break
            else:
                for t in self.table_list:
                    logger.info(f"training at step:{i}, table[{t.table_name}], table size:{t.size()}, "
                                f"table capacity:{t.capacity()}")

                if i % train_interval == 0:
                    self.evaluate()

                if i % saving_interval == 0:
                    self.saver.save(self.session, f"./saved-model/model", global_step=i)

                if self.is_faae and i == train_interval // 2:
                    logger.info("###############    set_threshold at step:%d   ################", i)
                    self.change_threshold()

        # save last step without duplication
        if i % saving_interval != 0:
            self.saver.save(self.session, f"./saved-model/model", global_step=i)

        logger.info("################    training end    ################")

    def evaluate(self):
        logger.info("###############    start evaluate, epoch:%d    ################", self.epoch)
        self._infer()
        logger.info("###############    evaluate end, epoch::%d   ################", self.epoch)
        self.epoch += 1

    def predict(self, model_file: List[str]):
        logger.info(f"###############    start predict    ################")

        # get the latest model
        latest_step = get_load_step(model_file)
        self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.session, f"./saved-model/model-{latest_step}")
        self._infer()
        logger.info(f"###############    predict end    ################")

    def change_threshold(self):
        thres_tensor = tf.constant(60, dtype=tf.int32)
        set_threshold_op = import_host_pipeline_ops().set_threshold(thres_tensor,
                                                                    emb_name=self.table_list[0].table_name,
                                                                    ids_name=self.table_list[0].table_name + "_lookup")
        self.session.run([set_threshold_op])


def get_load_step(model_file: List[str]):
    import re

    # get the latest model
    pattern = f".*sparse-model-([0-9]+).*"
    latest_step = -1
    for file_path in model_file:
        match = re.search(pattern, file_path)
        if match and match.groups():
            step = int(match.groups()[0])
            if step > latest_step:
                latest_step = step
    if latest_step == -1:
        raise RuntimeError("latest model not found")
    return latest_step
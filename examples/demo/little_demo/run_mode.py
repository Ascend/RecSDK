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

import json
import os
import stat
import sys
import time
from typing import List

import numpy as np
import tensorflow as tf
from mx_rec.constants.constants import BaseEnum
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.util.tf_version_adapter import hccl_ops
from mx_rec.util.variable import get_dense_and_sparse_variable

from config import (PRECISION_CHECK, USE_DETERMINISTIC,
                    construct_npu_sess_config)
from demo_logger import logger
from utils import (GLOBAL_RANK_SIZE, LOCAL_RANK_ID, PRECISION_CHECK_PATH,
                   PRECISION_DUMP_STEP, RANK_ZERO, PrecisionDumpInfo)


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
        self.session = tf.compat.v1.Session(
            config=construct_npu_sess_config(dump_data=PRECISION_CHECK))
        self.train_model = train_model
        self.train_iterator = train_iterator
        self.eval_model = eval_model
        self.eval_iterator = eval_iterator
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

        if self.infer_steps == -1:
            self.infer_steps = sys.maxsize  # 消耗全部数据
        for i in range(1, self.infer_steps + 1):
            logger.info("###############    infer at step %d    ################", i)
            try:
                self.session.run(self.eval_model.loss_list)
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for eval.")
                break

    def set_train_ops(self):
        dense_variables, sparse_variables = get_dense_and_sparse_variable()

        # multi task training
        for loss, (dense_optimizer, sparse_optimizer) in zip(self.train_model.loss_list, self.optimizer_list):
            # do dense optimization
            grads = dense_optimizer.compute_gradients(loss, var_list=dense_variables)
            avg_grads = []
            for grad, var in grads:
                if GLOBAL_RANK_SIZE > 1:
                    grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
                if grad is not None:
                    avg_grads.append((grad, var))
            # apply gradients: update variables
            self.train_ops.append(dense_optimizer.apply_gradients(avg_grads))

            if bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0))):
                from mx_rec.constants.constants import (
                    ASCEND_SPARSE_LOOKUP_ID_OFFSET,
                    ASCEND_SPARSE_LOOKUP_LOCAL_EMB)

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

        latest_ckpt_step = 0
        start_step = 1
        if if_load:
            latest_ckpt_step = get_load_step(model_file)
            start_step = latest_ckpt_step + 1
            self.saver.restore(self.session, f"./saved-model/model-{latest_ckpt_step}")
        else:
            self.session.run(tf.compat.v1.global_variables_initializer())

        if self.max_train_steps == -1:
            self.max_train_steps = sys.maxsize  # 消耗全部数据

        if PRECISION_CHECK:
            PrecisionDumpInfo.write_dump_info()

        loss_dict = {}

        
        for i in range(start_step, start_step + self.max_train_steps):
            logger.info("################    training at step %d    ################", i)
            try:
                should_break = self._execute_train_step(i, latest_ckpt_step, train_interval, saving_interval, loss_dict)
                if should_break:
                    break
            except tf.errors.OutOfRangeError:
                logger.info("Encounter the end of Sequence for training.")
                break

        dump_precision_dataset(self.session, self.train_iterator, LOCAL_RANK_ID)
        dump_loss_dict(loss_dict)
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
        logger.info("###############    start predict    ################")

        # get the latest model
        latest_step = get_load_step(model_file)
        self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.session, f"./saved-model/model-{latest_step}")
        self._infer()
        logger.info("###############    predict end    ################")

    def change_threshold(self):
        thres_tensor = tf.constant(60, dtype=tf.int32)
        set_threshold_op = import_host_pipeline_ops().set_threshold(thres_tensor,
                                                                    emb_name=self.table_list[0].table_name,
                                                                    ids_name=self.table_list[0].table_name + "_lookup")
        self.session.run([set_threshold_op])

    def _execute_train_step(self, i: int, latest_ckpt_step: int, train_interval: int, 
                            saving_interval: int, loss_dict: dict):
        dump_precision_ckpt(self.session, self.saver, i)
        _, loss = self.session.run([self.train_ops, self.train_model.loss_list])

        if USE_DETERMINISTIC:
            logger.info(f"train_loss: {loss[0]}")

        if PRECISION_CHECK and i in PRECISION_DUMP_STEP:
            loss_dict[i] = float(loss[0])
            if i == PRECISION_DUMP_STEP[-1]:
                time.sleep(10)
                return True

        for t in self.table_list:
            logger.info(f"training at step:{i}, table[{t.table_name}], table size:{t.size()}, "
                        f"table capacity:{t.capacity()}")

        if train_interval != -1 and (i - latest_ckpt_step) % train_interval == 0:
            self.evaluate()

        if saving_interval != -1 and (i - latest_ckpt_step) % saving_interval == 0:
            self.saver.save(self.session, f"./saved-model/model", global_step=i)

        if train_interval != -1 and self.is_faae and i == train_interval // 2:
            logger.info("###############    set_threshold at step:%d   ################", i)
            self.change_threshold()

        return False


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


def dump_loss_dict(loss_dict):
    if not PRECISION_CHECK:
        return
    
    loss_path = os.path.join(PRECISION_CHECK_PATH, "03dump_loss")
    if not os.path.exists(loss_path):
        os.makedirs(loss_path, mode=0o750, exist_ok=True)

    loss_dump_file = os.path.join(loss_path, f"{LOCAL_RANK_ID}_rank_loss.json")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(loss_dump_file, flags, modes), 'w') as fout:
        json.dump(loss_dict, fout)


def dump_precision_dataset(sess, train_iterator, rank_id):
    if not PRECISION_CHECK:
        return
    
    dataset_path = os.path.join(PRECISION_CHECK_PATH, '01dump_dataset')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, mode=0o750, exist_ok=True)
    initializer = train_iterator.initializer
    data_batch = train_iterator.get_next()
    try:
        sess.run(initializer)
        batch_index = 1
        while batch_index <= max(PRECISION_DUMP_STEP):
            batch_data = sess.run(data_batch)
            if batch_index not in PRECISION_DUMP_STEP:
                continue
            for key, value in batch_data.items():
                filename = os.path.join(dataset_path, f'data_rank_{rank_id}_batch_{batch_index}_{key}.npy')
                np.save(filename, value)
            batch_index += 1
    except tf.errors.OutOfRangeError:
        logger.info("Data set interation end.")
    

def dump_precision_ckpt(sess, saver, current_step):
    if not PRECISION_CHECK:
        return
    if current_step in PRECISION_DUMP_STEP:
        dump_ckpt_path = os.path.join(PRECISION_CHECK_PATH, "02dump_model/model")
        saver.save(sess, dump_ckpt_path, global_step=current_step)
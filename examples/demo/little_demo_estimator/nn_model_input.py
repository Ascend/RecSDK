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

from functools import partial

import tensorflow as tf
from mx_rec.constants.constants import ASCEND_TIMESTAMP

from demo_logger import logger
from nn_model_build import LittleModel
from nn_optim import get_train_op
from config import RUN_MODE, USE_TUPLE_DATA_FORMAT, USE_MODIFY_GRAPH, USE_TIMESTAMP


def log_formatter(mode, tensors):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return f"train_loss:{tensors['train_loss']:.16f}"
    elif mode == tf.estimator.ModeKeys.EVAL:
        return f"eval_loss:{tensors['eval_loss']:.16f}"
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return f"predict_loss:{tensors['predict_loss']:.16f}"
    else:
        raise NotImplementedError()


def get_model_fn(cfg, access_and_evict_config_dict=None):
    def model_fn(features, labels, mode, params):
        if USE_MODIFY_GRAPH:
            if USE_TIMESTAMP:
                model = LittleModel(cfg, mode, features, access_and_evict_config_dict=access_and_evict_config_dict)
                tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, features["timestamp"])
            else:
                model = LittleModel(cfg, mode, features)
        else:
            model = LittleModel(cfg, mode, features)

        logger.info(f"features:{features}, labels:{labels}")
        if USE_TUPLE_DATA_FORMAT:
            label_0 = labels["label_0"]
            label_1 = labels["label_1"]
        else:
            label_0 = features["label_0"]
            label_1 = features["label_1"]
        loss, prediction = model.inference(label_0, label_1)

        loss_dict = {}
        if mode == tf.estimator.ModeKeys.TRAIN:
            logger.info("Use estimator train mode")
            logging_hook = tf.compat.v1.train.LoggingTensorHook(
                {"train_loss": loss}, every_n_iter=1, formatter=partial(log_formatter, mode=mode))
            loss_dict['loss'] = [['train_loss', loss]]
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=get_train_op(params, loss_dict.get('loss')),
                                              training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            logger.info("Use estimator eval mode")
            logging_hook = tf.compat.v1.train.LoggingTensorHook(
                {"eval_loss": loss}, every_n_iter=1, formatter=partial(log_formatter, mode=mode))
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              evaluation_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.PREDICT:
            logger.info("Use estimator predict mode")
            logging_hook = tf.compat.v1.train.LoggingTensorHook(
                {"predict_loss": loss}, every_n_iter=1, formatter=partial(log_formatter, mode=mode))
            loss_dict['task_1'] = prediction[0]
            loss_dict['task_2'] = prediction[1]
            if RUN_MODE != 'export_pb':
                loss_dict['label'] = features["label_0"]
            export_outputs = {
                'predictor': tf.estimator.export.PredictOutput(loss_dict)
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=loss_dict,
                                              export_outputs=export_outputs,
                                              prediction_hooks=[logging_hook])
        else:
            raise NotImplementedError()

    return model_fn

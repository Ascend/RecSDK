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

import tensorflow as tf
from mx_rec.constants.constants import ASCEND_TIMESTAMP
from mx_rec.util.log import logger

from nn_model_build import LittleModel
from nn_optim import get_train_op


def get_model_fn(create_fs_params, cfg, access_and_evict_config_dict=None):
    def model_fn(features, labels, mode, params):
        if params.modify_graph:
            if params.use_timestamp:
                model = LittleModel(params, cfg, mode, features, create_fs_params,
                                    access_and_evict_config_dict=access_and_evict_config_dict)
                tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, features["timestamp"])
            else:
                model = LittleModel(params, cfg, mode, features, create_fs_params)
        else:
            model = LittleModel(params, cfg, mode, features, create_fs_params)

        loss, prediction = model.inference(features["label_0"], features["label_1"])

        loss_dict = {}
        if mode == tf.estimator.ModeKeys.TRAIN:
            logger.info("Use estimator train mode")
            loss_dict['loss'] = [['train_loss', loss]]
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=get_train_op(params, loss_dict.get('loss')))

        if mode == tf.estimator.ModeKeys.EVAL:
            logger.info("Use estimator eval mode")
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss)

        if mode == tf.estimator.ModeKeys.PREDICT:
            logger.info("Use estimator predict mode")
            loss_dict['task_1'] = prediction[0]

            loss_dict['task_2'] = prediction[1]
            if params.run_mode != 'export_pb':
                loss_dict['label'] = features["label_0"]

            export_outputs = {
                'predictor': tf.estimator.export.PredictOutput(loss_dict)
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=loss_dict,
                                              export_outputs=export_outputs)

    return model_fn

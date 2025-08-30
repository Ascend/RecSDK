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

import time

import tensorflow as tf

from rec_sdk_common.log import logger
from rec_sdk_common.constants.constants import ValidatorParams
from rec_sdk_common.util.tf_adapter import npu_ops
from rec_sdk_common.validator.validator import para_checker_decorator, ClassValidator, \
    IntValidator, OptionalIntValidator
from mx_rec.constants.constants import DEFAULT_EVICT_TIME_INTERVAL, TRAIN_CHANNEL_ID
from mx_rec.util.initialize import ConfigInitializer


class EvictHook(tf.compat.v1.train.SessionRunHook):
    """Sets evict based on global step or time."""
    @para_checker_decorator(
        check_option_list=[
            ("evict_enable", ClassValidator, {"classes": (bool, )}),
            ("evict_time_interval", IntValidator,
             {"min_value": 1, "max_value": ValidatorParams.MAX_INT32.value}, ["check_value"]),
            ("evict_step_interval", OptionalIntValidator,
             {"min_value": 1, "max_value": ValidatorParams.MAX_INT32.value}, ["check_value"]),
            ("evict_step_interval", ClassValidator, {"classes": (int, type(None))}),
        ]
    )
    def __init__(self,
                 evict_enable=False,
                 evict_time_interval=DEFAULT_EVICT_TIME_INTERVAL,
                 evict_step_interval=None):
        self._evict_enable = evict_enable
        self._evict_time_interval = evict_time_interval
        self._evict_step_interval = evict_step_interval
        self._hash_table_instance = dict()
        self._start_time = time.time()
        self._global_step = 0
        self._evict_op = dict()
        self._global_step_tensor = None

        if evict_step_interval is None:
            logger.info("_EvictHook - > evict_time_interval: %d", self._evict_time_interval)
        else:
            logger.info("_EvictHook - > evict_time_interval: %d, evict_step_interval: %d",
                        self._evict_time_interval, self._evict_step_interval)

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use _EvictHook.")
        self.check_name_and_get_hashtable()
        for name, instance in self._hash_table_instance.items():
            if not instance.is_hbm:
                continue
            scope_name = f"{instance.table_name}//evict"
            with tf.compat.v1.variable_scope(scope_name):
                logger.debug('Channel %s_evict_%d was built for op getnext', instance.table_name, TRAIN_CHANNEL_ID)

                evict_pos, evict_len = npu_ops.gen_npu_ops.get_next(
                    output_types=[tf.int32, tf.int32],
                    output_shapes=[[None], []],
                    channel_name=f'{instance.table_name}_evict_{TRAIN_CHANNEL_ID}')

                initialized_tensor = instance.emb_initializer(
                    instance.slice_device_vocabulary_size + instance.embedding_size) * instance.init_param

                initialized_tensor = initialized_tensor[0:evict_len, :]

                logger.debug(
                    'evict_pos output shape %r, and slice_device_vocabulary_size %d, initialized_tensor shape: %r',
                    evict_pos, instance.slice_device_vocabulary_size, initialized_tensor)

                nd_evict_pos = tf.expand_dims(evict_pos, 1)
                self._evict_op[name] = tf.compat.v1.scatter_nd_update(instance.variable, nd_evict_pos,
                                                                      initialized_tensor)

    def after_create_session(self, session, coord):
        self._global_step = session.run(self._global_step_tensor)
        logger.debug("_EvictHook - > after_create_session, step: %d", self._global_step)

    def after_run(self, run_context, run_values):
        if not self._evict_enable:
            return

        self._global_step = run_context.session.run(self._global_step_tensor)
        cur_time = time.time()
        if cur_time - self._start_time > self._evict_time_interval or \
                (self._evict_step_interval is not None and self._global_step % self._evict_step_interval == 0):
            logger.info("_EvictHook - > evict switch on!!! after_run step: %d", self._global_step)
            if not ConfigInitializer.get_instance().hybrid_manager_config.trigger_evict():
                return
            self._start_time = cur_time
            for name, instance in self._hash_table_instance.items():
                if not instance.is_hbm:
                    continue
                run_context.session.run(self._evict_op.get(name))

    def check_name_and_get_hashtable(self):
        for _, feature_spec in ConfigInitializer.get_instance().feature_spec_config.feature_spec_dict.items():
            if feature_spec.eviction_threshold:
                logger.debug("_EvictHook - > check and get instance: table_names %s", feature_spec.table_name)
                self._hash_table_instance[feature_spec.table_name] = \
                    ConfigInitializer.get_instance().sparse_embed_config.get_table_instance_by_name(
                        feature_spec.table_name)

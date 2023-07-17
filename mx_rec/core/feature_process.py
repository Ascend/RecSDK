#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import time

import tensorflow as tf

from mx_rec.constants.constants import DEFAULT_EVICT_TIME_INTERVAL, TRAIN_CHANNEL_ID
from mx_rec.util.initialize import trigger_evict, get_table_instance_by_name, export_feature_spec


class _EvictHook(tf.compat.v1.train.SessionRunHook):
    """Sets evict based on global step or time."""

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

        self.check_evict_init_params()
        logging.info(f"_EvictHook - > evict_time_interval: %d, evict_step_interval: %d", self._evict_time_interval,
                     self._evict_step_interval)

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use _EvictHook.")
        self.check_name_and_get_hashtable()
        for name, instance in self._hash_table_instance.items():
            scope_name = f"{instance.table_name}//evict"
            with tf.compat.v1.variable_scope(scope_name):
                logging.debug('Channel %s_evict_%d was built for op getnext', instance.table_name, TRAIN_CHANNEL_ID)

                from mx_rec.util.tf_version_adapter import npu_ops
                evict_pos, evict_len = npu_ops.gen_npu_ops.get_next(
                    output_types=[tf.int32, tf.int32],
                    output_shapes=[[None], []],
                    channel_name=f'{instance.table_name}_evict_{TRAIN_CHANNEL_ID}')

                initialized_tensor = instance.emb_initializer(
                    instance.slice_device_vocabulary_size + instance.embedding_size) * instance.init_param

                initialized_tensor = initialized_tensor[0:evict_len, :]

                logging.debug(
                    'evict_pos output shape %r, and slice_device_vocabulary_size %d, initialized_tensor shape: %r',
                    evict_pos, instance.slice_device_vocabulary_size, initialized_tensor)

                nd_evict_pos = tf.expand_dims(evict_pos, 1)
                self._evict_op[name] = tf.compat.v1.scatter_nd_update(instance.variable, nd_evict_pos,
                                                                      initialized_tensor)

    def after_create_session(self, session, coord):
        self._global_step = session.run(self._global_step_tensor)
        logging.debug("_EvictHook - > after_create_session, step: %d", self._global_step)

    def after_run(self, run_context, run_values):
        if not self._evict_enable:
            return

        self._global_step = run_context.session.run(self._global_step_tensor)
        cur_time = time.time()
        if cur_time - self._start_time > self._evict_time_interval or \
                (self._evict_step_interval is not None and self._global_step % self._evict_step_interval == 0):
            logging.info("_EvictHook - > evict switch on!!! after_run step: %d", self._global_step)
            if not trigger_evict():
                return
            self._start_time = cur_time
            for name in self._hash_table_instance.keys():
                run_context.session.run(self._evict_op.get(name))

    def check_name_and_get_hashtable(self):
        for _, feature_spec in export_feature_spec().items():
            if feature_spec.eviction_threshold:
                logging.debug("_EvictHook - > check and get instance: table_names %s", feature_spec.table_name)
                self._hash_table_instance[feature_spec.table_name] = get_table_instance_by_name(feature_spec.table_name)

    def check_evict_init_params(self):
        def check_type(arg, n_type, param_name):
            if not isinstance(arg, n_type):
                raise TypeError(f"{param_name} should be type '{n_type}', whose value is {arg} with type "
                                f"'{type(arg)}' in fact.")
            if type(arg) == int and arg < 1:
                raise ValueError(f"{param_name} should be bigger than 0, whose value is {arg} in fact")

        check_type(self._evict_enable, bool, "evict_enable")
        if self._evict_time_interval is not None:
            check_type(self._evict_time_interval, int, "evict_time_interval")
        if self._evict_step_interval is not None:
            check_type(self._evict_step_interval, int, "evict_time_interval")

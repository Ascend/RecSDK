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
import tensorflow as tf

from rec_sdk_common.util.tf_adapter import npu_ops
from mxrec.python.binding.runtime_manager import RuntimeManager


class TimeEvictor:
    _D2H_CHANNEL_SUFFIX = "time_evictor_d2h"
    _BIN_FILE_SUFFIX = "time_evictor.bin"

    def __init__(self, table_name: str, max_cold_secs: int):
        self._table_name = table_name
        self._max_cold_secs = max_cold_secs
        self._d2h_channel_name = self._get_device_to_host_channel_name()
        self._host_runtime_manager = RuntimeManager()

        # Start a time evictor server in host.
        self._host_runtime_manager.start_time_evictor(table_name, max_cold_secs)

    def update_last_timestamp(self, keys: tf.Tensor) -> tf.Tensor:
        if not isinstance(keys, tf.Tensor):
            raise TypeError(f"expected keys to be a tf.Tensor, but got {type(keys).__name__}")

        send_op = npu_ops.outfeed_enqueue_op(
            channel_name=self._d2h_channel_name, inputs=[keys], name="{}_op".format(self._d2h_channel_name)
        )

        with tf.control_dependencies(control_inputs=[send_op]):
            keys = tf.identity(keys)

        return keys

    def save(self, save_path: str):
        file_name = "{}_{}".format(self._table_name, self._BIN_FILE_SUFFIX)
        file_path = os.path.join(save_path, file_name)
        self._host_runtime_manager.save_time_evictor(self._table_name, file_path)

    def load(self, load_path: str):
        file_name = "{}_{}".format(self._table_name, self._BIN_FILE_SUFFIX)
        file_path = os.path.join(load_path, file_name)
        self._host_runtime_manager.load_time_evictor(self._table_name, file_path)

    def get_evicted_keys(self, table_name: str) -> tf.Tensor:
        evicted_keys = self._host_runtime_manager.get_evicted_keys(table_name)
        evicted_keys = tf.constant(evicted_keys, dtype=tf.int64)

        return evicted_keys

    def _get_device_to_host_channel_name(self) -> str:
        return "{}_{}".format(self._table_name, self._D2H_CHANNEL_SUFFIX)

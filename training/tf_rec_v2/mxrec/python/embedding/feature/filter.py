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
from typing import Optional

import tensorflow as tf

from rec_sdk_common.util.tf_adapter import npu_ops
from mxrec.python.binding.runtime_manager import RuntimeManager


class CountFilter:
    _D2H_CHANNEL_SUFFIX = "count_filter_d2h"
    _H2D_CHANNEL_SUFFIX = "count_filter_h2d"
    _BIN_FILE_SUFFIX = "count_filter.bin"

    def __init__(self, table_name: str, min_used_times: int):
        self._table_name = table_name
        self._min_used_times = min_used_times
        self._d2h_channel_name = self._get_device_to_host_channel_name()
        self._h2d_channel_name = self._get_host_to_device_channel_name()
        self._host_runtime_manager = RuntimeManager()

        # Start a count filter server in host.
        self._host_runtime_manager.start_count_filter(table_name, min_used_times)

    def count_and_filter(self, keys: tf.Tensor, cnts: tf.Tensor) -> tf.Tensor:
        if not isinstance(keys, tf.Tensor):
            raise TypeError(f"expected keys to be a tf.Tensor, but got {type(keys).__name__}")
        if not isinstance(cnts, tf.Tensor):
            raise TypeError(f"expected cnts to be a tf.Tensor, but got {type(keys).__name__}")

        filtered_keys: Optional[tf.Tensor] = None

        send_op = npu_ops.outfeed_enqueue_op(
            channel_name=self._d2h_channel_name, inputs=[keys, cnts], name="{}_op".format(self._d2h_channel_name)
        )

        with tf.control_dependencies(control_inputs=[send_op]):
            filtered_keys = npu_ops.channel_get_next(
                channel_name=self._h2d_channel_name, output_types=[tf.int64], output_shapes=[None]
            )[0]

        return filtered_keys

    def save(self, save_path: str):
        file_name = "{}_{}".format(self._table_name, self._BIN_FILE_SUFFIX)
        file_path = os.path.join(save_path, file_name)
        self._host_runtime_manager.save_count_filter(self._table_name, file_path)

    def load(self, load_path: str):
        file_name = "{}_{}".format(self._table_name, self._BIN_FILE_SUFFIX)
        file_path = os.path.join(load_path, file_name)
        self._host_runtime_manager.load_count_filter(self._table_name, file_path)

    def _get_device_to_host_channel_name(self) -> str:
        return "{}_{}".format(self._table_name, self._D2H_CHANNEL_SUFFIX)

    def _get_host_to_device_channel_name(self) -> str:
        return "{}_{}".format(self._table_name, self._H2D_CHANNEL_SUFFIX)

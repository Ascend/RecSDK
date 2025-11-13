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

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


class MockRuntimeManager:
    @staticmethod
    def start_time_evictor(*args, **kwargs):
        pass

    @staticmethod
    def start_count_filter(*args, **kwargs):
        pass

    @staticmethod
    def save_count_filter(*args, **kwargs):
        pass

    @staticmethod
    def save_time_evictor(*args, **kwargs):
        pass

    @staticmethod
    def load_count_filter(*args, **kwargs):
        pass

    @staticmethod
    def load_time_evictor(*args, **kwargs):
        pass

    @staticmethod
    def get_evicted_keys(*args, **kwargs):
        return [1, 2, 3]


def mock_get_device_id():
    return 0


def npu_session_config() -> tf.compat.v1.ConfigProto:
    config = tf.compat.v1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return config


test_graph = tf.compat.v1.Graph()
sess = tf.compat.v1.Session(graph=test_graph, config=npu_session_config())

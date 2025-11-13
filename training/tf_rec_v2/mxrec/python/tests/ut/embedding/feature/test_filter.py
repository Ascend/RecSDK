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

import pytest
import tensorflow as tf

from mxrec.python.embedding.feature.filter import CountFilter
from mxrec.python.tests.ut.ut_utils import test_graph, mock_get_device_id, MockRuntimeManager


class TestCountFilter:
    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()

    @staticmethod
    def test_count_and_filter_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.embedding.feature.filter.RuntimeManager", MockRuntimeManager)
        count_filter = CountFilter(table_name="test_count_and_filter_table", min_used_times=2)

        with test_graph.as_default():
            data = {
                "keys": tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int64),
                "cnts": tf.constant([[1, 1, 0, 0, 0]], dtype=tf.int32),
            }
            dataset = tf.data.Dataset.from_tensor_slices(data)
            dataset = dataset.repeat(2)

            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch = iterator.get_next()

            filtered_keys = count_filter.count_and_filter(batch["keys"], batch["cnts"])

        assert isinstance(filtered_keys, tf.Tensor)

    @staticmethod
    def test_filter_save_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.embedding.feature.filter.RuntimeManager", MockRuntimeManager)
        count_filter = CountFilter(table_name="test_filter_save", min_used_times=2)

        try:
            count_filter.save("./test")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_filter_load_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.embedding.feature.filter.RuntimeManager", MockRuntimeManager)
        count_filter = CountFilter(table_name="test_count_and_filter_table", min_used_times=2)

        try:
            count_filter.load("./test")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

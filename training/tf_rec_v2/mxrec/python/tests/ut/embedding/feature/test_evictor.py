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

from mxrec.python.embedding.feature.evictor import TimeEvictor
from mxrec.python.tests.ut.ut_utils import test_graph, mock_get_device_id, MockRuntimeManager


class TestTimeEvictor:
    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()

    @staticmethod
    def test_update_last_timestamp_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.embedding.feature.evictor.RuntimeManager", MockRuntimeManager)
        time_evictor = TimeEvictor(table_name="test_update_last_timestamp_table", max_cold_secs=2)

        with test_graph.as_default():
            data = {
                "keys": tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int64),
            }
            dataset = tf.data.Dataset.from_tensor_slices(data)
            dataset = dataset.repeat(count=2)

            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch = iterator.get_next()

            keys = time_evictor.update_last_timestamp(batch["keys"])

        assert isinstance(keys, tf.Tensor)

    @staticmethod
    def test_evictor_save_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.embedding.feature.evictor.RuntimeManager", MockRuntimeManager)
        time_evictor = TimeEvictor(table_name="test_evictor_save", max_cold_secs=2)

        try:
            time_evictor.save("./test")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_evictor_load_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.embedding.feature.evictor.RuntimeManager", MockRuntimeManager)
        time_evictor = TimeEvictor(table_name="test_evictor_load", max_cold_secs=2)

        try:
            time_evictor.load("./test")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_get_evicted_keys_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.embedding.feature.evictor.RuntimeManager", MockRuntimeManager)
        time_evictor = TimeEvictor(table_name="test_get_evicted_keys", max_cold_secs=2)

        try:
            time_evictor.get_evicted_keys("test_get_evicted_keys")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

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

import mxrec
from mxrec.python.embedding.table.static_emb_table import StaticEmbTable
from mxrec.python.config.parser import TomlParser
from mxrec.python.tests.ut.ut_utils import test_graph, mock_get_device_id


class TestGetEmbeddingTable:
    """Test for 'mxrec.python.embedding.embedding.get_embedding_table'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        TomlParser._instance = None

    @staticmethod
    def test_get_static_table_ok():
        table = mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        assert isinstance(table, StaticEmbTable)

    @staticmethod
    def test_get_exist_table_ok():
        table1 = mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        table2 = mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        assert id(table1) == id(table2)

    @staticmethod
    def test_name_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name=111,
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_name_whitelist_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name=r"test_name\n",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "Note: It should be a string consisting of numbers, letters, and underscores" in str(excinfo.value)

    @staticmethod
    def test_dev_voc_size_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size="100",
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "is not int" in str(excinfo.value)

    @staticmethod
    def test_dev_voc_size_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=-1,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "is less than" in str(excinfo.value)

    @staticmethod
    def test_dev_voc_size_only_dev_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=0,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "currently, the embedding table only supports storage in device memory" in str(excinfo.value)

    @staticmethod
    def test_initializer_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer="tf.compat.v1.truncated_normal_initializer()",
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "Invalid parameter type" in str(excinfo.value)

    @staticmethod
    def test_key_dtype_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int32,
                value_dtype=tf.float32,
            )
        assert "only supports key dtype of tf.int64" in str(excinfo.value)

    @staticmethod
    def test_value_dtype_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float16,
            )
        assert "only supports value dtype of tf.float32" in str(excinfo.value)

    @staticmethod
    def test_dist_strategy_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                distribution_strategy=1,
            )
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_dist_strategy_whitelist_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                distribution_strategy=r"MP\n",
            )
        assert "Note: It should be a string consisting of numbers, letters, and underscores" in str(excinfo.value)

    @staticmethod
    def test_dist_strategy_invalid_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                distribution_strategy="UNKNOWN",
            )
        assert "the embedding table only support MP(model parallelism)" in str(excinfo.value)

    @staticmethod
    def test_min_used_times_type_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                min_used_times="invalid",
            )

    @staticmethod
    def test_min_used_times_upper_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                min_used_times=-1,
            )

    @staticmethod
    def test_min_used_times_lower_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                min_used_times=1 << 31,
            )

    @staticmethod
    def test_max_cold_secs_type_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                max_cold_secs="invalid",
            )

    @staticmethod
    def test_max_cold_secs_upper_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                max_cold_secs=-1,
            )

    @staticmethod
    def test_max_cold_secs_lower_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                max_cold_secs=1 << 64,
            )


class TestGetInitHashTableOp:
    """Test for 'mxrec.python.embedding.embedding.get_init_hashtable_op'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        TomlParser._instance = None

    @staticmethod
    def test_get_init_hashtable_op_ok():
        mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        init_ops = mxrec.get_init_hashtable_op()
        assert len(init_ops) != 0


class TestGetExistingTables:
    """Test for 'mxrec.python.embedding.embedding.get_existing_tables'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        TomlParser._instance = None

    @staticmethod
    def test_get_existing_tables_ok():
        mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        tables = mxrec.get_existing_tables()
        assert len(tables) != 0
        assert isinstance(tables[0], (StaticEmbTable,))

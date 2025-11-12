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
import shutil

import pytest
import tensorflow as tf

import mxrec
from mxrec.python.config.parser import TomlParser
from mxrec.python.tests.ut.ut_utils import mock_get_device_id


class TestEmbeddingTableSaver:
    """Test for 'mxrec.python.training.saver.EmbeddingTableSaver'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        TomlParser._instance = None

    @staticmethod
    def test_emb_tables_empty():
        with pytest.raises(ValueError) as excinfo:
            mxrec.EmbeddingTableSaver([])
        assert "empty `emb_tables`" in str(excinfo.value)

    @staticmethod
    def test_save_op_ok(monkeypatch):
        class _MockSession:
            @staticmethod
            def run(*args, **kwargs):
                return [100]

        monkeypatch.setattr("mxrec.python.training.saver.class_safe_check", _mock_class_safe_check)
        monkeypatch.setattr("mxrec.python.training.saver.dir_safe_check", _mock_dir_safe_check)
        mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        embedding_table_saver = mxrec.EmbeddingTableSaver(mxrec.get_existing_tables())
        save_path = "./ut_test_save_path"
        sess = _MockSession()
        try:
            embedding_table_saver.save(sess, save_path, global_step=1)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")
        _delete_path(save_path)

    @staticmethod
    def test_load_op_ok(monkeypatch):
        class _MockSession:
            @staticmethod
            def run(*args, **kwargs):
                return [100]

        monkeypatch.setattr("mxrec.python.training.saver.class_safe_check", _mock_class_safe_check)
        monkeypatch.setattr("mxrec.python.training.saver.dir_safe_check", _mock_dir_safe_check)
        mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        embedding_table_saver = mxrec.EmbeddingTableSaver(mxrec.get_existing_tables())
        save_path = "./ut_test_save_path"
        sess = _MockSession()
        try:
            embedding_table_saver.load(sess, save_path, global_step=1)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")
        _delete_path(save_path)


def _delete_path(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def _mock_class_safe_check(*args, **kwargs):
    pass


def _mock_dir_safe_check(*args, **kwargs):
    pass

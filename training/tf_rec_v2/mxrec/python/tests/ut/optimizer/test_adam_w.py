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
from mxrec.python.config.parser import TomlParser
from mxrec.python.tests.ut.ut_utils import test_graph, mock_get_device_id


class TestAdamWOptimizer:
    """Test for 'mxrec.python.optimizer.adam_w'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        TomlParser._instance = None

    @staticmethod
    def test_apply_grad_ok():
        with test_graph.as_default():
            table = mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
            ids = tf.constant([[1, 2, 4, 1], [1, 3, 10, 2]], dtype=tf.int64)
            lookup_res = mxrec.embedding_lookup(table, ids)

            dense_tensor = tf.compat.v1.get_variable(
                "test_apply_grad_ok", shape=(table.dim, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            reduced_emb = tf.reduce_sum(lookup_res, axis=1)
            logits = tf.matmul(reduced_emb, dense_tensor)
            logits = tf.reshape(logits, (-1,))

            labels = tf.constant([0, 1], dtype=tf.float32)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)

            sparse_optimizer = mxrec.AdamWOptimizer(learning_rate=0.01)
            sparse_embeddings = mxrec.get_sparse_embedding()
            sparse_grads = tf.gradients(loss, sparse_embeddings)
            train_ops = sparse_optimizer.apply_gradients(zip(sparse_grads, sparse_embeddings))

            assert isinstance(train_ops, tf.Operation) and train_ops.name == "AdamWOptimizer"

    @staticmethod
    def test_lr_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(learning_rate="xxx")
        assert "Invalid parameter type" in str(excinfo.value)

    @staticmethod
    def test_lr_max_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(learning_rate=10.1)
        assert "'learning_rate' is bigger than" in str(excinfo.value)

    @staticmethod
    def test_lr_min_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(learning_rate=-0.1)
        assert "'learning_rate' is less than" in str(excinfo.value)

    @staticmethod
    def test_weight_decay_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(weight_decay="xxx")
        assert "Invalid parameter type" in str(excinfo.value)

    @staticmethod
    def test_weight_decay_max_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(weight_decay=10.1)
        assert "'weight_decay' is bigger than" in str(excinfo.value)

    @staticmethod
    def test_weight_decay_min_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(weight_decay=-0.1)
        assert "'weight_decay' is less than" in str(excinfo.value)

    @staticmethod
    def test_beta1_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(beta_1="xxx")
        assert "Invalid parameter type" in str(excinfo.value)

    @staticmethod
    def test_beta1_max_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(beta_1=1.1)
        assert "'beta_1' is bigger than or equal" in str(excinfo.value)

    @staticmethod
    def test_beta1_min_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(beta_1=-0.1)
        assert "'beta_1' is less than" in str(excinfo.value)

    @staticmethod
    def test_beta2_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(beta_2="xxx")
        assert "Invalid parameter type" in str(excinfo.value)

    @staticmethod
    def test_beta2_max_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(beta_2=1.1)
        assert "'beta_2' is bigger than or equal" in str(excinfo.value)

    @staticmethod
    def test_beta2_min_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(beta_2=-0.1)
        assert "'beta_2' is less than" in str(excinfo.value)

    @staticmethod
    def test_epsilon_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(epsilon="xxx")
        assert "Invalid parameter type" in str(excinfo.value)

    @staticmethod
    def test_epsilon_max_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(epsilon=1.1)
        assert "'epsilon' is bigger than or equal" in str(excinfo.value)

    @staticmethod
    def test_epsilon_min_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(epsilon=-0.1)
        assert "'epsilon' is less than" in str(excinfo.value)

    @staticmethod
    def test_name_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(name=1)
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_name_whitelist_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.AdamWOptimizer(name=r"test\n")
        assert "Note: It should be a string consisting of numbers, letters, and underscores" in str(excinfo.value)

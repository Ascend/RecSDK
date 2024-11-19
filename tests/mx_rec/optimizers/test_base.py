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

import unittest
from unittest import mock

import tensorflow as tf
import numpy as np

from mx_rec.optimizers.base import CustomizedOptimizer
from tests.mx_rec.core.mock_class import MockConfigInitializer, MockSparseEmbedding


@mock.patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=mock.Mock(return_value=MockConfigInitializer()),
)
class TestProcessGradValueMaskFunc(unittest.TestCase):
    """Test for 'mx_rec.optimizers.base.CustomizedOptimizer._process_grad_value_mask'."""

    @mock.patch.multiple("mx_rec.optimizers.base", get_rank_size=mock.MagicMock(return_value=1))
    @mock.patch("mx_rec.optimizers.base.npu_ops.gen_npu_ops.get_next")
    @mock.patch("mx_rec.optimizers.base.ConfigInitializer")
    def test_padding_keys_mask_true(self, opt_base_config_initializer, mock_get_next):
        with tf.Graph().as_default():
            test_table = MockSparseEmbedding("test_table", embedding_size=5)
            mock_config_initializer = MockConfigInitializer(var=test_table)
            mock_config_initializer.use_static = True
            opt_base_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            nd_value = tf.constant(
                [
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4],
                    [5, 5, 5, 5, 5],
                ],
                dtype=tf.float32,
            )
            mock_get_next.return_value = [[1, 1, 0, 1, 0]]
            res = CustomizedOptimizer._process_grad_value_mask(test_table.variable, nd_value)
            res_true = tf.constant(
                [
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0],
                    [4, 4, 4, 4, 4],
                    [0, 0, 0, 0, 0],
                ],
                dtype=tf.float32,
            )

            with tf.compat.v1.Session() as sess:
                sess_res = sess.run(res)
                sess_res_true = sess.run(res_true)
                self.assertTrue(np.all(sess_res == sess_res_true))

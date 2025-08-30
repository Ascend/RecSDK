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

from core.generator_dataset import generate_dataset, Config
from core.mock_class import MockConfigInitializer
from data.mock_class import MockEosOpsLib


@mock.patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=mock.Mock(return_value=MockConfigInitializer()),
)
class TestEosDatasetClass(unittest.TestCase):
    """
    Test for 'mx_rec.data.dataset.EosDataset'.
    """

    def test_init(self):
        """
        case: 实例化EosDataset，使用eos_map
        """

        with tf.Graph().as_default():
            dataset_ori = generate_dataset(Config(batch_size=2, batch_number=2))
            dataset = dataset_ori.eos_map(MockEosOpsLib(dataset_ori._variant_tensor), 0)
            iterator = dataset.make_initializable_iterator()
            batch = iterator.get_next()
            with tf.Session() as sess:
                sess.run(iterator.initializer)
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(batch)
                self.assertIsNotNone(batch.get("item_ids"))


if __name__ == '__main__':
    unittest.main()

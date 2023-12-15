#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import unittest

import tensorflow as tf

from tests.mx_rec.core.generator_dataset import generate_dataset, Config
from tests.mx_rec.data.mock_class import MockEosOpsLib


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

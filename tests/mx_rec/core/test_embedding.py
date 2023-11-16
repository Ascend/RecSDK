#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.


import unittest
import tensorflow as tf
from mx_rec.core.embedding import set_specific_value_for_non_valid_key

if tf.__version__.startswith("1"):
    tf.enable_eager_execution()
else:
    tf.compat.v1.enable_eager_execution()


class TestSetSpecificValueForNonValidKey(unittest.TestCase):
    """
    Test Suite for set_specific_value_for_non_valid_key.
    """

    def setUp(self):
        """
        准备步骤
        :return:无
        """
        super().setUp()

    def tearDown(self):
        """
        销毁步骤
        :return: 无
        """
        super().tearDown()

    def test_given_admit_if_turned_off_then_return_raw_embedding(self):
        # given
        id_offsets = tf.constant([0, 1, 2, 3])
        embeddings = tf.ones(shape=(4, 10), dtype=tf.float32)
        access_threshold = None
        serving_default_value = tf.ones(shape=(4, 10), dtype=tf.float32) * 2

        # when
        modified_emb = set_specific_value_for_non_valid_key(id_offsets, embeddings,
                                                            access_threshold, serving_default_value)

        # then
        result = bool(tf.reduce_all(tf.equal(embeddings, modified_emb)))
        self.assertTrue(result)

    def test_given_no_default_value_and_all_valid_key_then_return_raw_embedding(self):
        # given
        id_offsets = tf.constant([0, 1, 2, 3])
        embeddings = tf.ones(shape=(4, 10), dtype=tf.float32)
        access_threshold = 1
        serving_default_value = None

        # when
        modified_emb = set_specific_value_for_non_valid_key(id_offsets, embeddings,
                                                            access_threshold, serving_default_value)

        # then
        result = bool(tf.reduce_all(tf.equal(embeddings, modified_emb)))
        self.assertTrue(result)

    def test_given_no_default_value_and_invalid_key_then_emb_of_invalid_key_set_to_zero(self):
        # given
        id_offsets = tf.constant([-1, 1, 2, 3])
        embeddings = tf.ones(shape=(4, 2), dtype=tf.float32)
        access_threshold = 1
        serving_default_value = None

        # when
        modified_emb = set_specific_value_for_non_valid_key(id_offsets, embeddings,
                                                            access_threshold, serving_default_value)

        # then
        result = modified_emb.numpy().tolist()[0]
        self.assertEqual([0, 0], result)

    def test_given_default_value_and_invalid_key_then_emb_of_invalid_key_set_to_default_value(self):
        # given
        id_offsets = tf.constant([-1, 1, 2, 3])
        embeddings = tf.ones(shape=(4, 2), dtype=tf.float32)
        access_threshold = 1
        serving_default_value = tf.ones(shape=(4, 2), dtype=tf.float32) * 2

        # when
        modified_emb = set_specific_value_for_non_valid_key(id_offsets, embeddings,
                                                            access_threshold, serving_default_value)

        # then
        result = modified_emb.numpy().tolist()[0]
        self.assertEqual([2, 2], result)

    def test_given_default_value_and_with_all_invalid_key_then_emb_of_invalid_key_set_to_default_value(self):
        # given
        id_offsets = tf.constant([-1, -1, -1, -1])
        embeddings = tf.ones(shape=(4, 2), dtype=tf.float32)
        access_threshold = 1
        serving_default_value = tf.ones(shape=(4, 2), dtype=tf.float32) * 2

        # when
        modified_emb = set_specific_value_for_non_valid_key(id_offsets, embeddings,
                                                            access_threshold, serving_default_value)

        # then
        result = bool(tf.reduce_all(tf.equal(serving_default_value, modified_emb)))
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()

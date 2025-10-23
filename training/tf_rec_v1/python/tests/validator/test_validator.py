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

import unittest

import tensorflow as tf
from tensorflow.python.client.session import BaseSession
from mx_rec.validator.validator import LearningRateValidator


class ParameterCheckerTest(unittest.TestCase):
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

    def test_learning_rate_validator(self):
        if hasattr(BaseSession, "old_run_method"):
            BaseSession.run = BaseSession.old_run_method

        with tf.Graph().as_default():
            self.assertTrue(LearningRateValidator(
                name="whatever",
                value=tf.constant([1.0]),
                min_value=0.0,
                max_value=10.0
            ).check_value().check().is_valid())

            try:
                self.assertTrue(LearningRateValidator(
                    name="whatever",
                    value=tf.constant([11.0]),
                    min_value=0.0,
                    max_value=10.0
                ).check_value().check().is_valid())

            except ValueError as exp:
                self.assertEqual(type(exp), ValueError)
            else:
                self.fail("ValueError not raised.")

            try:
                self.assertTrue(LearningRateValidator(
                    name="whatever",
                    value=tf.constant([1.0, 2.0]),
                    min_value=0.0,
                    max_value=10.0
                ).check_value_for_left_open_interval().check().is_valid())

            except ValueError as exp:
                self.assertEqual(type(exp), ValueError)
            else:
                self.fail("ValueError not raised.")
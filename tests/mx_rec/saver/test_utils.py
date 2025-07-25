#!/usr/bin/env python3
# coding: UTF-8
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.saver.utils import check_files_in_directories, get_optimizer_dict_by_table_name
from tests.mx_rec.core.mock_class import MockConfigInitializer


class TestCheckFilesInDirectories(unittest.TestCase):
    def test_exist(self):
        root_path = "./tmp_check_files_in_directories"
        if tf.io.gfile.isdir(root_path):
            tf.io.gfile.rmtree(root_path)
        os.mkdir(root_path)

        data_path = os.path.join(root_path, "xx.meta.xx")
        f = open(data_path, "w")
        f.write('model_checkpoint_path: "ckpt-123"')
        f.close()

        res = check_files_in_directories(root_path, ["*.meta.*"])
        self.assertTrue(res)
        tf.io.gfile.rmtree(root_path)

    def test_not_exist(self):
        root_path = "./tmp_check_files_in_directories"
        if tf.io.gfile.isdir(root_path):
            tf.io.gfile.rmtree(root_path)
        os.mkdir(root_path)

        data_path = os.path.join(root_path, "xx.xxx.xx")
        f = open(data_path, "w")
        f.write('model_checkpoint_path: "ckpt-123"')
        f.close()

        res = check_files_in_directories(root_path, ["*.meta.*"])
        self.assertFalse(res)
        tf.io.gfile.rmtree(root_path)


class TestGetOptimizerDictByTableName(unittest.TestCase):
    @mock.patch("mx_rec.saver.utils.ConfigInitializer")
    def test_experimental_mode_is_none(self, utils_config_initializer):
        mock_config_init = MockConfigInitializer()
        mock_config_init.get_instance().optimizer_config.set_optimizer_for_table(
            table_name="test_table",
            optimizer_name="xxx",
            optimizer_dict={},
            is_training=True,
        )
        utils_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)

        self.assertEqual(get_optimizer_dict_by_table_name("test_table"), {"xxx": {}})

    @mock.patch("mx_rec.saver.utils.ConfigInitializer")
    def test_experimental_mode_is_train(self, utils_config_initializer):
        mock_config_init = MockConfigInitializer(experimental_mode="train")
        mock_config_init.get_instance().optimizer_config.set_optimizer_for_table(
            table_name="test_table",
            optimizer_name="xxx",
            optimizer_dict={},
            is_training=True,
        )
        utils_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)

        self.assertEqual(get_optimizer_dict_by_table_name("test_table"), {"xxx": {}})

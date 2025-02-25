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

import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.saver.patch import (
    get_sparse_vars,
    init_check,
    save_check,
    check_characters_is_valid,
    saver_from_object_based_checkpoint,
)
from tests.mx_rec.core.mock_class import MockConfigInitializer


class TestGetSparseVars(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.reset_default_graph()

    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @mock.patch("mx_rec.saver.patch.ConfigInitializer")
    def test_var_list_is_none(self, saver_patch_config_initializer):
        mock_config_init = MockConfigInitializer(use_dynamic_expansion=False)
        saver_patch_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        res = get_sparse_vars(None)
        self.assertEqual(len(res), 0)

    @mock.patch("mx_rec.saver.patch.ConfigInitializer")
    def test_var_list_type_err(self, saver_patch_config_initializer):
        mock_config_init = MockConfigInitializer(use_dynamic_expansion=False)
        saver_patch_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        with self.assertRaises(TypeError):
            get_sparse_vars({})

    @mock.patch("mx_rec.saver.patch.ConfigInitializer")
    def test_ok(self, saver_patch_config_initializer):
        mock_config_init = MockConfigInitializer(use_dynamic_expansion=False)
        saver_patch_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        tf.compat.v1.add_to_collection("ASCEND_GLOBAL_HASHTABLE_COLLECTION", "test_table")
        res = get_sparse_vars(var_list=["test_table"])
        self.assertNotEqual(len(res), 0)


class TestInitCheck(unittest.TestCase):
    def test_defer_build_and_var_list_is_not_none(self):
        with self.assertRaises(ValueError):
            init_check(["var"], ["var"])

    @mock.patch.multiple(
        "mx_rec.saver.patch.context",
        executing_eagerly=mock.MagicMock(return_value=True),
    )
    def test_executing_eagerly_is_true_var_list_err(self):
        with self.assertRaises(RuntimeError):
            init_check(["var"], None)


class TestSaveCheck(unittest.TestCase):
    def test_latest_filename_err(self):
        latest_filename = "/path/to/file.txt"
        with self.assertRaises(ValueError):
            save_check(latest_filename, tf.compat.v1.Session())

    @mock.patch.multiple(
        "mx_rec.saver.patch.context",
        executing_eagerly=mock.MagicMock(return_value=False),
    )
    def test_sess_err(self):
        latest_filename = "file.txt"
        with self.assertRaises(TypeError):
            save_check(latest_filename, "xxx")


class TestCheckCharactersIsValid(unittest.TestCase):
    def test_valid(self):
        res = check_characters_is_valid("xxx")
        self.assertTrue(res)

    def test_invalid(self):
        res = check_characters_is_valid("xxx\n")
        self.assertFalse(res)


class TestSaverFromObjectBasedCheckpoint(unittest.TestCase):
    def test_checkpoint_path_not_exist(self):
        checkpoint_path = "tmp_xxx_saver_from_object_based_checkpoint"
        with self.assertRaises(ValueError):
            saver_from_object_based_checkpoint(checkpoint_path)

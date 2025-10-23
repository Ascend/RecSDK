#!/usr/bin/env python3
# coding: UTF-8
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
import unittest
from unittest import mock

import tensorflow as tf
from tensorflow.python.estimator.estimator import WarmStartSettings

from mx_rec.saver.warm_start import (
    WarmStartController,
    patch_estimator_init,
    get_table_name_set_by_ckpt_path,
    SparseRestoreHook,
    patch_for_func_warm_start,
    patch_for_estimator_train,
)


class TestWarmStartController(unittest.TestCase):
    def test_init_ok(self):
        ws = WarmStartController()
        ws.add_element(path="xx", table_list=["table1"])
        ws.add_element(path="xx", table_list=["table2"])
        ws.add_table_to_prev_table(table="table1", prev_table="table2")
        self.assertEqual(ws.get_elements().get("xx"), ["table1", "table2"])


class TestPatchEstimatorInit(unittest.TestCase):
    def test_patch_warm_start_from_is_none(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_estimator_init(_mock_func)
        patch_func(warm_start_from=None)

        self.assertTrue(callable(patch_func))

    @mock.patch.multiple(
        "mx_rec.saver.warm_start",
        get_table_name_set_by_ckpt_path=mock.MagicMock(return_value="test_table"),
    )
    def test_warm_start_from_is_str(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_estimator_init(_mock_func)
        mock_ws = "xxx"
        self.assertIsNotNone(patch_func(warm_start_from=mock_ws))

    @mock.patch.multiple(
        "mx_rec.saver.warm_start",
        get_table_name_set_by_ckpt_path=mock.MagicMock(return_value="test_table"),
    )
    def test_ws_settings_param_is_str(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_estimator_init(_mock_func)
        mock_ws = WarmStartSettings(
            ckpt_to_initialize_from="/tmp",
            vars_to_warm_start=".*",
            var_name_to_vocab_info={"input_layer/sc_vocab_file_embedding/embedding_weights": 1},
            var_name_to_prev_var_name={"input_layer/sc_vocab_list_embedding/embedding_weights": "old_tensor_name"},
        )
        self.assertIsNotNone(patch_func(warm_start_from=mock_ws))

    @mock.patch.multiple(
        "mx_rec.saver.warm_start",
        get_table_name_set_by_ckpt_path=mock.MagicMock(return_value="test_table"),
    )
    def test_ws_settings_not_all_list_err(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_estimator_init(_mock_func)
        mock_ws = WarmStartSettings(
            ckpt_to_initialize_from=["/tmp"],
            vars_to_warm_start=".*",
            var_name_to_vocab_info={"input_layer/sc_vocab_file_embedding/embedding_weights": 1},
            var_name_to_prev_var_name={"input_layer/sc_vocab_list_embedding/embedding_weights": "old_tensor_name"},
        )
        with self.assertRaises(ValueError) as e:
            patch_func(warm_start_from=mock_ws)

        self.assertIn("the parameter type in the warm settings should be a list", str(e.exception))

    @mock.patch.multiple(
        "mx_rec.saver.warm_start",
        get_table_name_set_by_ckpt_path=mock.MagicMock(return_value="test_table"),
    )
    def test_ws_settings_list_length_err(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_estimator_init(_mock_func)
        mock_ws = WarmStartSettings(
            ckpt_to_initialize_from=["/tmp", "/usr"],
            vars_to_warm_start=[".*"],
            var_name_to_vocab_info={"input_layer/sc_vocab_file_embedding/embedding_weights": 1},
            var_name_to_prev_var_name=[{"input_layer/sc_vocab_list_embedding/embedding_weights": "old_tensor_name"}],
        )
        with self.assertRaises(ValueError) as e:
            patch_func(warm_start_from=mock_ws)

        self.assertIn("the parameter list list should be the same length", str(e.exception))

    @mock.patch.multiple(
        "mx_rec.saver.warm_start",
        get_table_name_set_by_ckpt_path=mock.MagicMock(return_value="test_table"),
    )
    def test_ws_settings_vars_to_warm_start_list(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_estimator_init(_mock_func)
        mock_ws = WarmStartSettings(
            ckpt_to_initialize_from=["/tmp"],
            vars_to_warm_start=[".*"],
            var_name_to_vocab_info={"input_layer/sc_vocab_file_embedding/embedding_weights": 1},
            var_name_to_prev_var_name=[{"input_layer/sc_vocab_list_embedding/embedding_weights": "old_tensor_name"}],
        )
        self.assertIsNotNone(patch_func(warm_start_from=mock_ws))

    @mock.patch.multiple(
        "mx_rec.saver.warm_start",
        get_table_name_set_by_ckpt_path=mock.MagicMock(return_value="test_table"),
    )
    def test_ws_settings_vars_to_warm_start_list_with_list(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_estimator_init(_mock_func)
        mock_ws = WarmStartSettings(
            ckpt_to_initialize_from=["/tmp"],
            vars_to_warm_start=[[".*"]],
            var_name_to_vocab_info={"input_layer/sc_vocab_file_embedding/embedding_weights": 1},
            var_name_to_prev_var_name=[{"input_layer/sc_vocab_list_embedding/embedding_weights": "old_tensor_name"}],
        )
        self.assertIsNotNone(patch_func(warm_start_from=mock_ws))


class TestGetTableNameSetByCkptPath(unittest.TestCase):
    def test_warm_start_path_not_dir(self):
        self.assertEqual(get_table_name_set_by_ckpt_path("/home/user/documents/file.txt"), [])

    def test_ckpt_path_not_exist_err(self):
        warm_start_path = "./tmp_warm_start_path"
        if tf.io.gfile.isdir(warm_start_path):
            tf.io.gfile.rmtree(warm_start_path)
        os.mkdir(warm_start_path)
        with self.assertRaises(FileNotFoundError) as e:
            get_table_name_set_by_ckpt_path(warm_start_path)
        self.assertIn("Checkpoint file is missing under", str(e.exception))
        tf.io.gfile.rmtree(warm_start_path)

    def test_ok(self):
        warm_start_path = "./tmp_warm_start_path"
        if tf.io.gfile.isdir(warm_start_path):
            tf.io.gfile.rmtree(warm_start_path)
        os.mkdir(warm_start_path)
        data_path = os.path.join(warm_start_path, "checkpoint")
        f = open(data_path, "w")
        f.write('model_checkpoint_path: "ckpt-123"')
        f.close()
        self.assertEqual(len(get_table_name_set_by_ckpt_path(warm_start_path)), 0)
        tf.io.gfile.rmtree(warm_start_path)


class TestSparseRestoreHook(unittest.TestCase):
    @mock.patch.multiple(
        "mx_rec.saver.warm_start",
        Saver=mock.MagicMock(return_value="test_table"),
    )
    def test_init_ok(self):
        hook = SparseRestoreHook()
        hook.begin()
        self.assertIsNotNone(hook)


class TestPatchForFuncWarmStart(unittest.TestCase):
    def test_patch_ok(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_for_func_warm_start(_mock_func)
        patch_func(["./tmp"], ["./tmp"], ["./tmp"], ["./tmp"])
        self.assertTrue(callable(patch_func))


class TestPatchForEstimatorTrain(unittest.TestCase):
    def test_patch_ok(self):
        def _mock_func(*args, **kwargs):
            return True

        patch_func = patch_for_estimator_train(_mock_func)
        ws = WarmStartController()
        ws.add_element(path="xx", table_list=["table1"])
        patch_func()
        self.assertTrue(callable(patch_func))

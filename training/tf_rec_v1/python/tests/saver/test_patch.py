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
from core.mock_class import MockConfigInitializer


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


class TestPatchForSecondOrStepTimer(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.reset_default_graph()

    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=1, save_delta_checkpoints_secs=1),
    )
    def test_init_all_none(self):
        with self.assertRaises(ValueError) as e:
            tf.compat.v1.train.SecondOrStepTimer(every_secs=None, every_steps=None)
        self.assertIn("Either every_secs or every_steps should be provided", str(e.exception))

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=1, save_delta_checkpoints_secs=1),
    )
    def test_init_all_not_none(self):
        with self.assertRaises(ValueError) as e:
            tf.compat.v1.train.SecondOrStepTimer(every_secs=1, every_steps=1)
        self.assertIn("Can not provide both every_secs and every_steps", str(e.exception))

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=None, save_delta_checkpoints_secs=None),
    )
    def test_init_incremental_all_not_none(self):
        with self.assertRaises(ValueError) as e:
            tf.compat.v1.train.SecondOrStepTimer(every_secs=1, every_steps=None, is_incremental_checkpoint=True)
        self.assertIn("Both save_checkpoint_due_time and save_delta_checkpoints_secs", str(e.exception))

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=1, save_delta_checkpoints_secs=1),
    )
    def test_init_ok(self):
        timer = tf.compat.v1.train.SecondOrStepTimer(every_secs=1, every_steps=None, is_incremental_checkpoint=True)
        self.assertIsNotNone(timer)

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=1, save_delta_checkpoints_secs=1),
    )
    def test_last_triggered_step_is_none(self):
        timer = tf.compat.v1.train.SecondOrStepTimer(every_secs=1, every_steps=None, is_incremental_checkpoint=True)
        self.assertTrue(timer.should_trigger_for_step(1))

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=1, save_delta_checkpoints_secs=1),
    )
    def test_last_triggered_step_equal(self):
        timer = tf.compat.v1.train.SecondOrStepTimer(every_secs=1, every_steps=None, is_incremental_checkpoint=True)
        timer.update_last_triggered_step(1)
        self.assertFalse(timer.should_trigger_for_step(1))

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=1, save_delta_checkpoints_secs=1),
    )
    def test_last_triggered_step_with_incremental_is_true_ok(self):
        timer = tf.compat.v1.train.SecondOrStepTimer(every_secs=1, every_steps=None, is_incremental_checkpoint=True)
        timer.update_last_triggered_step(1)
        self.assertFalse(timer.should_trigger_for_step(2))

    @mock.patch(
        "mx_rec.saver.patch.ConfigInitializer",
        new=MockConfigInitializer(save_checkpoint_due_time=1, save_delta_checkpoints_secs=1),
    )
    def test_last_triggered_step_with_incremental_is_false_ok(self):
        timer = tf.compat.v1.train.SecondOrStepTimer(every_secs=1, every_steps=None, is_incremental_checkpoint=False)
        timer.update_last_triggered_step(1)
        self.assertFalse(timer.should_trigger_for_step(2))


class TestCheckpointSaverHook(unittest.TestCase):
    @mock.patch("mx_rec.saver.patch.ConfigInitializer", new=MockConfigInitializer())
    def test_init_ok(self):
        hook = tf.compat.v1.train.CheckpointSaverHook("tmp", save_secs=1)
        self.assertIsNotNone(hook)

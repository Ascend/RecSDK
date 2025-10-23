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

import unittest

from mx_rec.util.config_utils.train_param import TrainParamsConfig


class TestTrainParamsConfig(unittest.TestCase):
    def test_init_ok(self):
        train_params_config = TrainParamsConfig()
        self.assertEqual(train_params_config.iterator_type, "")
        self.assertFalse(train_params_config.is_last_round)
        self.assertFalse(train_params_config.is_graph_modify_hook_running)
        self.assertEqual(train_params_config.sparse_dir, "")
        self.assertEqual(train_params_config.ascend_global_hashtable_collection, "ASCEND_GLOBAL_HASHTABLE_COLLECTION")
        self.assertIsNone(train_params_config.dataset_element_spec)
        self.assertIsNone(train_params_config.experimental_mode)
        self.assertIsInstance(train_params_config.bool_gauge_set, set)

    def test_set_experimental_mode_ok(self):
        train_params_config = TrainParamsConfig()
        train_params_config.experimental_mode = "xxx"
        self.assertEqual(train_params_config.experimental_mode, "xxx")

    def test_set_iterator_type_ok(self):
        train_params_config = TrainParamsConfig()
        train_params_config.iterator_type = "xxx"
        self.assertEqual(train_params_config.iterator_type, "xxx")

    def test_set_is_graph_modify_hook_running_ok(self):
        train_params_config = TrainParamsConfig()
        train_params_config.is_graph_modify_hook_running = True
        self.assertTrue(train_params_config.is_graph_modify_hook_running)

    def test_set_sparse_dir_ok(self):
        train_params_config = TrainParamsConfig()
        train_params_config.sparse_dir = "xxx"
        self.assertEqual(train_params_config.sparse_dir, "xxx")

    def test_set_is_last_round_ok(self):
        train_params_config = TrainParamsConfig()
        train_params_config.is_last_round = True
        self.assertTrue(train_params_config.is_last_round)

    def test_set_ascend_global_hashtable_collection_ok(self):
        train_params_config = TrainParamsConfig()
        train_params_config.ascend_global_hashtable_collection = "xxx"
        self.assertEqual(train_params_config.ascend_global_hashtable_collection, "xxx")

    def test_set_dataset_element_spec_ok(self):
        train_params_config = TrainParamsConfig()
        train_params_config.dataset_element_spec = "xxx"
        self.assertEqual(train_params_config.dataset_element_spec, "xxx")

    def test_insert_train_mode(self):
        train_params_config = TrainParamsConfig()
        train_params_config.insert_training_mode_channel_id(is_training=True)
        self.assertEqual(train_params_config.get_training_mode_channel_id(is_training=True), 0)

    def test_insert_eval_mode(self):
        train_params_config = TrainParamsConfig()
        train_params_config.insert_training_mode_channel_id(is_training=False)
        self.assertEqual(train_params_config.get_training_mode_channel_id(is_training=False), 1)

    def test_insert_bool_gauge(self):
        train_params_config = TrainParamsConfig()
        train_params_config.insert_bool_gauge(name="xxx")
        self.assertEqual(train_params_config.bool_gauge_set, {"xxx"})

    def test_insert_merged_multi_lookup(self):
        train_params_config = TrainParamsConfig()
        train_params_config.insert_merged_multi_lookup(is_training=True, value=True)
        self.assertTrue(train_params_config.get_merged_multi_lookup(is_training=True))

    def test_set_target_batch(self):
        train_params_config = TrainParamsConfig()
        train_params_config.set_target_batch(is_training=True, batch="xxx")
        self.assertEqual(train_params_config.get_target_batch(is_training=True), "xxx")

    def test_set_initializer(self):
        train_params_config = TrainParamsConfig()
        train_params_config.set_initializer(is_training=True, initializer="xxx")
        self.assertEqual(train_params_config.get_initializer(is_training=True), "xxx")

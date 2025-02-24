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


from mx_rec.util.initialize import ConfigInitializer, terminate_config_initializer
from mx_rec.util.config_utils.embedding_utils import SparseEmbedConfig
from mx_rec.util.config_utils.feature_spec_utils import FeatureSpecConfig
from mx_rec.util.config_utils.hybrid_mgmt_utils import HybridManagerConfig
from mx_rec.util.config_utils.optimizer_utils import OptimizerConfig
from mx_rec.util.config_utils.train_param import TrainParamsConfig


@mock.patch.multiple(
    "mx_rec.util.perf_factory.bind_cpu",
    get_local_rank_size=mock.MagicMock(return_value=1),
    get_rank_id=mock.MagicMock(return_value=0),
)
class TestInitialize(unittest.TestCase):
    def test_init_ok(self):
        config_init = ConfigInitializer()
        self.assertFalse(config_init.use_lccl)
        self.assertIsNone(config_init.save_checkpoint_due_time)
        self.assertIsNone(config_init.save_delta_checkpoints_secs)
        self.assertFalse(config_init.is_incremental_checkpoint)
        self.assertIsNone(config_init.restore_model_version)
        self.assertFalse(config_init.modify_graph)
        self.assertEqual(config_init.max_steps, -1)
        self.assertEqual(config_init.train_steps, -1)
        self.assertEqual(config_init.eval_steps, -1)
        self.assertEqual(config_init.save_steps, -1)
        self.assertFalse(config_init.if_load)
        self.assertFalse(config_init.use_static)
        self.assertFalse(config_init.use_dynamic_expansion)
        self.assertIsInstance(config_init.sparse_embed_config, SparseEmbedConfig)
        self.assertIsInstance(config_init.feature_spec_config, FeatureSpecConfig)
        self.assertIsInstance(config_init.hybrid_manager_config, HybridManagerConfig)
        self.assertIsInstance(config_init.optimizer_config, OptimizerConfig)
        self.assertIsInstance(config_init.train_params_config, TrainParamsConfig)

    def test_set_modify_graph(self):
        config_init = ConfigInitializer()
        config_init.modify_graph = True
        self.assertTrue(config_init.modify_graph)

    def test_set_max_steps(self):
        config_init = ConfigInitializer()
        config_init.max_steps = 11
        self.assertEqual(config_init.max_steps, 11)

    def test_set_train_steps(self):
        config_init = ConfigInitializer()
        config_init.train_steps = 11
        self.assertEqual(config_init.train_steps, 11)

    def test_set_eval_steps(self):
        config_init = ConfigInitializer()
        config_init.eval_steps = 11
        self.assertEqual(config_init.eval_steps, 11)

    def test_set_save_steps(self):
        config_init = ConfigInitializer()
        config_init.save_steps = 11
        self.assertEqual(config_init.save_steps, 11)

    def test_set_if_load(self):
        config_init = ConfigInitializer()
        config_init.if_load = True
        self.assertTrue(config_init.if_load)

    def test_set_use_static(self):
        config_init = ConfigInitializer()
        config_init.use_static = False
        self.assertFalse(config_init.use_static)

    def test_set_sparse_embed_config(self):
        config_init = ConfigInitializer()
        sparse_embed_config = SparseEmbedConfig()
        config_init.sparse_embed_config = sparse_embed_config
        self.assertIsInstance(config_init.sparse_embed_config, SparseEmbedConfig)

    def test_set_feature_spec_config(self):
        config_init = ConfigInitializer()
        feature_spec_config = FeatureSpecConfig()
        config_init.feature_spec_config = feature_spec_config
        self.assertIsInstance(config_init.feature_spec_config, FeatureSpecConfig)

    def test_set_hybrid_manager_config(self):
        config_init = ConfigInitializer()
        hybrid_manager_config = HybridManagerConfig()
        config_init.hybrid_manager_config = hybrid_manager_config
        self.assertIsInstance(config_init.hybrid_manager_config, HybridManagerConfig)

    def test_set_optimizer_config(self):
        config_init = ConfigInitializer()
        optimizer_config = OptimizerConfig()
        config_init.optimizer_config = optimizer_config
        self.assertIsInstance(config_init.optimizer_config, OptimizerConfig)

    def test_set_train_params_config(self):
        config_init = ConfigInitializer()
        train_params_config = TrainParamsConfig()
        config_init.train_params_config = train_params_config
        self.assertIsInstance(config_init.train_params_config, TrainParamsConfig)

    def test_terminate_ok(self):
        terminate_config_initializer()
        self.assertTrue(callable(terminate_config_initializer))

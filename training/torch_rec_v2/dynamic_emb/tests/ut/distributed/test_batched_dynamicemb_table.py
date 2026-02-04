#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import sys
import unittest
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any

import torch

from dynamic_emb.distributed.batched_dynamicemb_table import (
    BatchedDynamicEmbeddingTablesV2,
    WeightDecayMode,
    CreateOptimizerConfig,
)
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import EmbOptimType

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class TestBatchedDynamicEmbeddingTablesV2(unittest.TestCase):
    def setUp(self):
        self.mock_table = MagicMock()
        self.patched_device_timestamp = patch(
            "dynamic_emb.distributed.batched_dynamicemb_table.device_timestamp"
        )
        self.patched_create_table = patch(
            "dynamic_emb.distributed.key_value_table.create_dynamicemb_table",
            return_value=self.mock_table
        )
        self.mocked_device_timestamp = self.patched_device_timestamp.start()
        self.mocked_create_table = self.patched_create_table.start()
        self.mocked_device_timestamp.return_value = (5)
        dynamicemb_options_list: List[Dict[str, Any]] = []
        dynamicemb_options_list.append(
            DynamicEmbTableOptions(
                index_type=torch.int64,
                embedding_dtype=torch.float32,
                optimizer_type=EmbOptimType.ADAM,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL,
                    value=0.0,),
                dim=10
            )
        )
        feature_table_map = [0]
        self.emb_module = BatchedDynamicEmbeddingTablesV2(
            table_options=dynamicemb_options_list,
            pooling_mode=DynamicEmbPoolingMode.NONE,
            feature_table_map=feature_table_map,
            table_names=["table1"],
            device=torch.device("cpu"),
        )

    def tearDown(self):
        self.patched_device_timestamp.stop()
        self.patched_create_table.stop()

        del self.mock_table
        del self.emb_module
        del self.mocked_device_timestamp
        del self.mocked_create_table

    def test_create_cache_storage(self):
        self.emb_module._create_cache_storage()
        self.assertEqual(len(self.emb_module._storages), 1)

    def test_create_initializers(self):
        self.emb_module._create_initializers()
        self.assertEqual(len(self.emb_module._initializers), 2)

    def test_create_optimizer(self):
        optimizer_type = EmbOptimType.ADAM
        optim_config = CreateOptimizerConfig(
            optimizer_type=optimizer_type,
            stochastic_rounding=False,
            gradient_clipping=False,
            max_gradient=1.0,
            max_norm=0.0,
            learning_rate=0.001,
            eps=1.0e-8,
            initial_accumulator_value=0.0,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eta=0.001,
            momentum=0.9,
            weight_decay_mode=WeightDecayMode.L2,
            counter_based_regularization=None,
            cowclip_regularization=None
        )
        self.emb_module._create_optimizer(optim_config)

        self.assertIsNotNone(self.emb_module._optimizer)
        self.assertEqual(self.emb_module._optimizer_type, EmbOptimType.ADAM)
        self.assertEqual(self.emb_module._optimizer_args.learning_rate, 0.001)

    def test_forward(self):
        self.emb_module._storages = []
        self.emb_module._caches = []
        self.emb_module._dynamicemb_options = [
            DynamicEmbTableOptions(score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED)
        ]
        self.emb_module.set_score({"table1": 10})
        self.emb_module.training = True
        self.emb_module.output_dtype = torch.float32

        mock_tensor = torch.ones(10, dtype=torch.float32)

        with patch("dynamic_emb.distributed.batched_dynamicemb_table.DynamicEmbeddingFunctionV2.apply") as mock_apply:
            mock_apply.return_value = (mock_tensor)
            indices = torch.tensor([1, 2], dtype=torch.int64)
            offsets = torch.tensor([0, 2], dtype=torch.int64)

            result = self.emb_module.forward(indices, offsets)

            self.assertEqual(result.numel(), 10)
            self.assertEqual(result.dtype, torch.float32)
            mock_apply.assert_called_once()

    def test_set_score(self):
        self.emb_module._dynamicemb_options = [
            DynamicEmbTableOptions(score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED)
        ]
        named_score = {"table1": 100}
        self.emb_module.set_score(named_score)
        self.assertEqual(self.emb_module._scores["table1"], 100)

    def test_update_score(self):
        self.emb_module._dynamicemb_options = [
            DynamicEmbTableOptions(score_strategy=DynamicEmbScoreStrategy.TIMESTAMP)
        ]
        mock_timestamp = 200
        with patch("dynamic_emb.distributed.batched_dynamicemb_table.device_timestamp", return_value=mock_timestamp):
            self.emb_module._update_score()

        new_scores = self.emb_module.get_score()
        self.assertEqual(new_scores["table1"], mock_timestamp)

        # 测试STEP策略
        self.emb_module._dynamicemb_options = [
            DynamicEmbTableOptions(score_strategy=DynamicEmbScoreStrategy.STEP)
        ]
        self.emb_module._update_score()
        new_scores = self.emb_module.get_score()
        self.assertEqual(new_scores["table1"], mock_timestamp + 1)

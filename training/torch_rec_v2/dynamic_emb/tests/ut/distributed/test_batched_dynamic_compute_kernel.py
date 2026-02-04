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
import enum
import unittest
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.distributed as dist
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig, ShardedEmbeddingTable
from torchrec.modules.embedding_configs import DataType
from torchrec.distributed.types import ShardedTensorMetadata, ShardMetadata

from dynamic_emb.distributed.dynamicemb_config import DynamicEmbTableOptions
from dynamic_emb.distributed.batched_dynamicemb_compute_kernel import BatchedDynamicEmbedding

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class MockOptimizerType(enum.Enum):
    Adam = "adam"
    Null = "null"


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:23457",
            world_size=1,
            rank=0
        )
    return dist.group.WORLD


class TestBatchedDynamicEmbedding(unittest.TestCase):
    def setUp(self):
        self.mock_table1 = Mock(spec=ShardedEmbeddingTable)
        self.mock_table1.name = "table_0"
        self.mock_table1.fused_params = {"dynamicemb_options": DynamicEmbTableOptions()}
        self.mock_table1.global_metadata = ShardedTensorMetadata(
            shards_metadata=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1, 1])])
        self.mock_table1.global_metadata.tensor_properties = Mock()
        self.mock_table1.num_features = Mock(return_value=1)
        self.mock_table1.num_embeddings = 1024
        self.mock_table1.feature_names = ["t0"]

        self.mock_table2 = Mock(spec=ShardedEmbeddingTable)
        self.mock_table2.name = "table_1"
        self.mock_table2.fused_params = {"dynamicemb_options": DynamicEmbTableOptions()}
        self.mock_table2.global_metadata = ShardedTensorMetadata(
            shards_metadata=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1, 1])])
        self.mock_table2.global_metadata.tensor_properties = Mock()
        self.mock_table2.num_features = Mock(return_value=1)
        self.mock_table2.num_embeddings = 1024
        self.mock_table2.feature_names = ["t1"]

        self.mock_config = Mock(spec=GroupedEmbeddingConfig)
        self.mock_config.embedding_tables = [self.mock_table1, self.mock_table2]
        self.mock_config.data_type = DataType.FP32

        self.mock_config.fused_params = {
            "optimizer": MockOptimizerType.Adam,
            "learning_rate": 0.01,
            "customized_compute_kernel": None,
            "dist_type": None,
            "dynamicemb_options": None,
        }
        self.mock_config.table_name_to_count = {"table_0": 1, "table_1": 1}
        self.pg = init_distributed()
        self.device = torch.device("cpu")
        self.mock_dynamicemb_options = [DynamicEmbTableOptions(), DynamicEmbTableOptions()]

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch("dynamic_emb.distributed.batched_dynamicemb_compute_kernel.BatchedDynamicEmbeddingTablesV2")
    def test_initialization(self, mock_batched_emb_tables):
        mock_emb_module = MagicMock()
        mock_emb_module.embedding_dtype = torch.float32
        mock_emb_module._dynamicemb_options = self.mock_dynamicemb_options  # 你的动态配置列表
        mock_batched_emb_tables.return_value = mock_emb_module

        # 初始化被测试类
        emb_layer = BatchedDynamicEmbedding(
            config=self.mock_config,
            pg=self.pg,
            device=self.device
        )

        # 验证 BatchedDynamicEmbeddingTablesV2 被正确初始化
        mock_batched_emb_tables.assert_called_once()
        self.assertEqual(len(emb_layer._param_per_table), 2)

    @patch("dynamic_emb.distributed.batched_dynamicemb_compute_kernel.BatchedDynamicEmbeddingTablesV2")
    def test_state_dict(self, mock_batched_emb_tables):
        mock_emb_module = MagicMock()
        mock_emb_module.embedding_dtype = torch.float32
        mock_emb_module._dynamicemb_options = self.mock_dynamicemb_options  # 你的动态配置列表
        mock_batched_emb_tables.return_value = mock_emb_module

        # 初始化被测试类
        emb_layer = BatchedDynamicEmbedding(
            config=self.mock_config,
            pg=self.pg,
            device=self.device
        )

        # mock split_embedding_weights 返回值
        mock_weight1 = torch.tensor([[1.0]], device=self.device)
        mock_weight2 = torch.tensor([[2.0]], device=self.device)
        emb_layer.split_embedding_weights = Mock(return_value=[mock_weight1, mock_weight2])
        with patch(
            "torchrec.distributed.types.ShardedTensor._init_from_local_shards_and_global_metadata",
            return_value=MagicMock(name="mocked_sharded_tensor")
        ) as mock_target_method:
            state_dict = emb_layer.state_dict()
            self.assertIn("table_0.weight", state_dict)
            self.assertIn("table_1.weight", state_dict)
            self.assertTrue(hasattr(state_dict["table_0.weight"], "_local_shards"))

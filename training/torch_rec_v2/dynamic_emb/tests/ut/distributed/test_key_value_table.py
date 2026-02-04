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

import torch

from dynamic_emb.distributed.key_value_table import (
    KeyValueTable,
    KeyValueTableFunction,
)
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbDataType,
)
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    EmbOptimType,
    OptimizerArgs,
    BaseDynamicEmbeddingOptimizerV2,
)

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class MockOptimizer(BaseDynamicEmbeddingOptimizerV2):
    def __init__(self, args):
        super().__init__(args)
        self.state_dim = 2
        self.fused_update_with_pointer_called = False
        self.fused_update_called = False


    def get_state_dim(self, emb_dim):
        return self.state_dim

    def get_opt_args(self):
        pass

    def update(self, grads, embs, states):
        pass

    def set_opt_args(self, args):
        pass

    def get_initial_optim_states(self):
        return [torch.tensor([0.0, 0.0])]

    def fused_update(self, grads, values):
        self.fused_update_called = True

    def fused_update_with_pointer(self, grads, pointers, dtype):
        # 简单的mock实现，用于验证调用
        self.fused_update_with_pointer_called = True
        # 我们可以在这里添加更多的验证逻辑，比如检查grads的形状和dtype等


class TestKeyValueTable(unittest.TestCase):
    def setUp(self):
        self.options = DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            optimizer_type=EmbOptimType.ADAM,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL,
                value=0.0,),
            dim=10,
        )
        args = OptimizerArgs()
        self.mock_optimizer = MockOptimizer(args)
        self.mock_table = MagicMock()

    def test_find_impl_partial_hit(self):
        batch_size = 4
        unique_keys = torch.tensor([101, 102, 103, 104], dtype=torch.int64)  # 测试key
        unique_embs = torch.empty(
            batch_size, self.options.dim,
            dtype=self.options.embedding_dtype
        )

        expected_founds = torch.tensor([True, False, True, False])  # 101、103命中
        expected_pointers = torch.tensor([42, 0, 105, 0])  # 命中的key对应底层存储指针
        # 预期的缺失结果
        expected_num_missing = 2
        expected_missing_keys = torch.tensor([102, 104])
        expected_missing_indices = torch.tensor([1, 3])

        with patch("dynamic_emb.distributed.key_value_table.create_dynamicemb_table", return_value=self.mock_table), \
             patch("dynamic_emb.distributed.key_value_table.EvictStrategy") as mock_evict_strategy, \
                patch("dynamic_emb.distributed.key_value_table.find_pointers") as mock_find_pointers, \
                patch("dynamic_emb.distributed.key_value_table.load_from_pointer") as mock_load_from_pointers:
            # 配置EvictStrategy mock：模拟非KLru策略（让_use_score=True）
            mock_evict_strategy.KLru = "KLru"
            mock_evict_strategy.LRU = "LRU"
            self.mock_table.get_evict_strategy.return_value = mock_evict_strategy.LRU
            self.mock_table.get_key_type.return_value = DynamicEmbDataType.Int64
            self.mock_table.get_value_type.return_value = DynamicEmbDataType.Float32

            kv_table = KeyValueTable(self.options, self.mock_optimizer)

            # 配置find_pointers mock：模拟修改传入的pointers和founds张量
            def find_pointers_side_effect(table, batch, keys, pointers, founds, *args):
                pointers.copy_(expected_pointers)  # 写入命中的指针
                founds.copy_(expected_founds)      # 写入命中标记
            mock_find_pointers.side_effect = find_pointers_side_effect
            num_missing, missing_keys, missing_indices = kv_table.find_impl(unique_keys, unique_embs)

            self.assertEqual(num_missing, expected_num_missing)
            torch.testing.assert_close(missing_keys, expected_missing_keys)  # 缺失key一致
            torch.testing.assert_close(missing_indices, expected_missing_indices)  # 缺失索引一致

    def test_insert_with_score(self):
        """测试当 _use_score 为 True 时的插入行为。"""
        # Arrange
        batch_size = 3
        unique_keys = torch.tensor([101, 102, 103], dtype=torch.long)
        unique_values = torch.randn(batch_size, self.options.dim, dtype=self.options.embedding_dtype)

        expected_score_value = 42
        self.options.score = expected_score_value # 假设 score 从 options 中获取

        with patch("dynamic_emb.distributed.key_value_table.create_dynamicemb_table", return_value=self.mock_table), \
             patch("dynamic_emb.distributed.key_value_table.EvictStrategy") as mock_evict_strategy:

            mock_evict_strategy.KLru = "KLru"
            self.mock_table.get_evict_strategy.return_value = mock_evict_strategy.LRU # 不是 KLru
            self.mock_table.get_key_type.return_value = DynamicEmbDataType.Int64
            self.mock_table.get_value_type.return_value = DynamicEmbDataType.Float32

            kv_table = KeyValueTable(self.options, self.mock_optimizer)
            kv_table.score = expected_score_value # 直接设置 score

            kv_table.insert(unique_keys, unique_values)
            # 获取实际传递给 insert_or_assign 的参数
            call_args = self.mock_table.update.call_args

            actual_keys = call_args[0][1]
            actual_values = call_args[0][2]
            actual_scores = call_args[0][3]

            torch.testing.assert_close(actual_keys, unique_keys)
            torch.testing.assert_close(actual_values, unique_values.to(kv_table.value_type())) # 确保类型匹配

            self.assertIsNotNone(actual_scores)
            self.assertEqual(actual_scores.dtype, torch.int32)
            self.assertTrue((actual_scores == expected_score_value).all())

    def test_update_with_return_missing(self):
        batch_size = 4
        keys = torch.tensor([101, 102, 103, 104], dtype=torch.long)
        grads = torch.randn(batch_size, self.options.dim, dtype=self.options.embedding_dtype) # 假设梯度只针对嵌入部分

        expected_founds = torch.tensor([True, False, True, False])
        expected_pointers = torch.tensor([42, 0, 105, 0])

        expected_num_missing = 2
        expected_missing_keys = torch.tensor([102, 104])
        expected_missing_indices = torch.tensor([1, 3])

        with patch("dynamic_emb.distributed.key_value_table.create_dynamicemb_table", return_value=self.mock_table), \
             patch("dynamic_emb.distributed.key_value_table.EvictStrategy") as mock_evict_strategy, \
                patch("dynamic_emb.distributed.key_value_table.find_pointers") as mock_find_pointers:

            mock_evict_strategy.KLru = "KLru"
            self.mock_table.get_evict_strategy.return_value = mock_evict_strategy.LRU
            self.mock_table.get_key_type.return_value = DynamicEmbDataType.Int64
            self.mock_table.get_value_type.return_value = DynamicEmbDataType.Float32

            kv_table = KeyValueTable(self.options, self.mock_optimizer)
            kv_table._score_update = False

            def find_pointers_side_effect(table, batch, keys, pointers, founds):
                pointers.copy_(expected_pointers)
                founds.copy_(expected_founds)
            mock_find_pointers.side_effect = find_pointers_side_effect

            num_missing, missing_keys, missing_indices = kv_table.update(keys, grads, return_missing=True)

            self.assertEqual(num_missing, expected_num_missing)
            torch.testing.assert_close(missing_keys, expected_missing_keys)
            torch.testing.assert_close(missing_indices, expected_missing_indices)

            self.assertTrue(self.mock_optimizer.fused_update_with_pointer_called)


class TestKeyValueTableFunction(unittest.TestCase):
    def setUp(self):
        # 创建存储模拟对象
        self.mock_storage = Mock()
        self.mock_storage.embedding_dim = MagicMock(return_value=64)
        self.mock_storage.embedding_dtype = MagicMock(return_value=torch.float32)
        self.mock_storage.value_dim = MagicMock(return_value=64)
        self.mock_storage.enable_update = MagicMock(return_value=False)

        self.mock_storage.find_embeddings = MagicMock(return_value=(
            torch.tensor(0),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long)
        ))

        def find_side_effect(keys, values, founds):
            founds[:] = torch.rand(keys.numel()) < 0.6
            founds[:5] = True
            founds[5:keys.numel()] = False
            missing_keys = keys[~founds]
            return (torch.sum(founds), missing_keys, founds)

        self.mock_storage.find = MagicMock(side_effect=find_side_effect)
        self.mock_storage.insert = MagicMock()
        self.mock_storage.update = MagicMock()
        self.mock_storage.init_optimizer_state = MagicMock(return_value=torch.randn(1, 64))

        # 创建优化器模拟对象
        self.mock_optimizer = Mock()
        self.mock_optimizer.fused_update = MagicMock()

    def test_lookup_with_no_keys(self):
        unique_keys = torch.empty((0,), dtype=torch.long)
        unique_embs = torch.empty((0, 64))

        KeyValueTableFunction.lookup(
            storage=self.mock_storage,
            unique_keys=unique_keys,
            unique_embs=unique_embs,
            initializer=Mock(),
            training=True
        )

        self.mock_storage.find_embeddings.assert_not_called()

    def test_lookup_with_all_existing_keys(self):
        batch_size = 10
        unique_keys = torch.randint(0, 100, (batch_size,), dtype=torch.long)
        unique_embs = torch.randn(batch_size, 64)

        initializer = Mock()
        KeyValueTableFunction.lookup(
            storage=self.mock_storage,
            unique_keys=unique_keys,
            unique_embs=unique_embs,
            initializer=initializer,
            training=True
        )

        initializer.assert_not_called()
        self.mock_storage.insert.assert_not_called()

    def test_update_with_storage_enabled(self):
        self.mock_storage.enable_update.return_value = True
        unique_keys = torch.randint(0, 100, (5,), dtype=torch.long)
        unique_grads = torch.randn(5, 64)

        KeyValueTableFunction.update(
            storage=self.mock_storage,
            unique_keys=unique_keys,
            unique_grads=unique_grads,
            optimizer=self.mock_optimizer
        )

        self.mock_storage.update.assert_called_once_with(
            unique_keys, unique_grads, return_missing=False
        )

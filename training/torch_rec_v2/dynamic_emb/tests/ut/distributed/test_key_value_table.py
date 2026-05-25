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
npu_device = torch.device("npu:0")

def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class MockOptimizer(BaseDynamicEmbeddingOptimizerV2):
    def __init__(self, args=None):
        if args is None:
            args = OptimizerArgs()
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

    def _run_find_impl_with_founds(self, expected_founds, expected_pointers, is_pure_hbm_mode):
        batch_size = expected_founds.numel()
        unique_keys = torch.arange(101, 101 + batch_size, dtype=torch.int64)
        unique_embs = torch.empty(
            batch_size, self.options.dim, dtype=self.options.embedding_dtype
        )

        with patch("dynamic_emb.distributed.key_value_table.create_dynamicemb_table", return_value=self.mock_table), \
             patch("dynamic_emb.distributed.key_value_table.EvictStrategy") as mock_evict_strategy, \
             patch("dynamic_emb.distributed.key_value_table.find_pointers") as mock_find_pointers, \
             patch("dynamic_emb.distributed.key_value_table.dyn_emb_is_pure_hbm_mode",
                   return_value=is_pure_hbm_mode) as mock_is_pure_hbm_mode, \
             patch("dynamic_emb.distributed.key_value_table.load_from_pointer") as mock_load_from_pointer, \
             patch("dynamic_emb.distributed.key_value_table.load_from_pointer_hybrid") as mock_load_from_pointer_hybrid:
            mock_evict_strategy.KLru = "KLru"
            mock_evict_strategy.LRU = "LRU"
            self.mock_table.get_evict_strategy.return_value = mock_evict_strategy.LRU
            self.mock_table.get_key_type.return_value = DynamicEmbDataType.Int64
            self.mock_table.get_value_type.return_value = DynamicEmbDataType.Float32

            kv_table = KeyValueTable(self.options, self.mock_optimizer)

            def find_pointers_side_effect(table, batch, keys, pointers, founds, *args):
                pointers.copy_(expected_pointers)
                founds.copy_(expected_founds)

            mock_find_pointers.side_effect = find_pointers_side_effect
            return kv_table.find_impl(unique_keys, unique_embs), {
                "mock_is_pure_hbm_mode": mock_is_pure_hbm_mode,
                "mock_load_from_pointer": mock_load_from_pointer,
                "mock_load_from_pointer_hybrid": mock_load_from_pointer_hybrid,
            }

    def test_find_impl_partial_hit(self):
        expected_founds = torch.tensor([True, False, True, False])  # 101、103命中
        expected_pointers = torch.tensor([42, 0, 105, 0])  # 命中的key对应底层存储指针
        expected_num_missing = 2
        expected_missing_keys = torch.tensor([102, 104])
        expected_missing_indices = torch.tensor([1, 3])

        (num_missing, missing_keys, missing_indices), mocks = self._run_find_impl_with_founds(
            expected_founds, expected_pointers, is_pure_hbm_mode=True
        )

        self.assertEqual(num_missing, expected_num_missing)
        torch.testing.assert_close(missing_keys, expected_missing_keys)
        torch.testing.assert_close(missing_indices, expected_missing_indices)
        mocks["mock_is_pure_hbm_mode"].assert_called_once_with(self.mock_table)
        mocks["mock_load_from_pointer"].assert_called_once()
        mocks["mock_load_from_pointer_hybrid"].assert_not_called()

    def test_find_impl_pure_hbm_mode_uses_load_from_pointer(self):
        expected_founds = torch.tensor([True, True])
        expected_pointers = torch.tensor([11, 22], dtype=torch.long)

        _, mocks = self._run_find_impl_with_founds(
            expected_founds, expected_pointers, is_pure_hbm_mode=True
        )

        mocks["mock_is_pure_hbm_mode"].assert_called_once_with(self.mock_table)
        mocks["mock_load_from_pointer"].assert_called_once()
        mocks["mock_load_from_pointer_hybrid"].assert_not_called()
        pointers_new, _ = mocks["mock_load_from_pointer"].call_args[0]
        torch.testing.assert_close(pointers_new, expected_pointers)

    def test_find_impl_non_pure_hbm_mode_uses_load_from_pointer_hybrid(self):
        expected_founds = torch.tensor([True, True])
        expected_pointers = torch.tensor([33, 44], dtype=torch.long)

        _, mocks = self._run_find_impl_with_founds(
            expected_founds, expected_pointers, is_pure_hbm_mode=False
        )

        mocks["mock_is_pure_hbm_mode"].assert_called_once_with(self.mock_table)
        mocks["mock_load_from_pointer_hybrid"].assert_called_once()
        mocks["mock_load_from_pointer"].assert_not_called()
        pointers_new, _ = mocks["mock_load_from_pointer_hybrid"].call_args[0]
        torch.testing.assert_close(pointers_new, expected_pointers)

    def test_find_impl_no_hit_skips_load(self):
        expected_founds = torch.tensor([False, False, False, False])
        expected_pointers = torch.zeros(4, dtype=torch.long)

        _, mocks = self._run_find_impl_with_founds(
            expected_founds, expected_pointers, is_pure_hbm_mode=True
        )

        mocks["mock_is_pure_hbm_mode"].assert_not_called()
        mocks["mock_load_from_pointer"].assert_not_called()
        mocks["mock_load_from_pointer_hybrid"].assert_not_called()

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
            self.assertEqual(actual_scores.dtype, torch.int64)
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


    def _create_new_test_kv_table(self):
        options = DynamicEmbTableOptions(
            embedding_dtype=torch.float32,
            index_type=torch.int64,
            dim=16,
            init_capacity=1000,
            max_capacity=10000,
            local_hbm_for_values=1024 * 1024 * 10,
            device_id=0,
            bucket_capacity=128,
            max_load_factor=0.5,
            block_size=128,
            io_block_size=1024,
            io_by_cpu=False,
            use_constant_memory=False,
            reserved_key_start_bit=0,
            num_of_buckets_per_alloc=1,
            training=True
        )
        optimizer = MockOptimizer()
        with patch("dynamic_emb.distributed.key_value_table.create_dynamicemb_table", return_value=self.mock_table):
            table = KeyValueTable(options=options, optimizer=optimizer)
        return table
        
    def test_count_matched(self):
        def mock_count_matched(table, threshold, num_matched):
            num_matched[0] = 50

        kv_table = self._create_new_test_kv_table()
        num_matched = torch.zeros(1, dtype=torch.long, device=npu_device)
        
        with patch("dynamic_emb.distributed.key_value_table.count_matched", side_effect=mock_count_matched):
            kv_table.count_matched(50, num_matched)
        
        print("√ count_matched 接口测试通过")

    def test_export_batch_matched(self):
        def mock_export_batch_matched(table, threshold, batch_size, search_offset, d_count, d_keys, d_vals):
            d_count[0] = 10

        kv_table = self._create_new_test_kv_table()
        d_count = torch.zeros(1, device=npu_device)
        d_keys = torch.empty(32, dtype=torch.long, device=npu_device)
        d_vals = torch.empty(32, 16, device=npu_device)
        
        with patch("dynamic_emb.distributed.key_value_table.export_batch_matched", side_effect=mock_export_batch_matched):
            kv_table.export_batch_matched(30, 32, 0, d_count, d_keys, d_vals)
        
        print("√ export_batch_matched 接口测试通过")

    def test_insert_and_evict(self):
        def mock_insert_and_evict(table, batch, keys, values, score, ek, ev, es, ne):
            ne[0] = 2

        kv_table = self._create_new_test_kv_table()
        keys = torch.tensor([1,2,3,4,5], device=npu_device)
        values = torch.randn(5, 32, device=npu_device)
        
        with patch("dynamic_emb.distributed.key_value_table.insert_and_evict", side_effect=mock_insert_and_evict):
            kv_table.insert_and_evict(keys, values)
        
        print("√ insert_and_evict 接口测试通过")
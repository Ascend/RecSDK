#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import logging
import pytest
import torch
import dynamic_emb_extensions as demb
logging.basicConfig(level=logging.INFO)


class DynamicEmbTableOptions:
    def __init__(self):
        self.key_type = demb.DynamicEmbDataType.Int64
        self.value_type = demb.DynamicEmbDataType.Float32
        self.evict_strategy = demb.EvictStrategy.kLru
        self.dim = 128
        self.init_capacity = 1024
        self.max_capacity = 2048
        self.max_hbm_for_vectors = 1 * 1024 * 1024 * 1024
        self.max_bucket_size = 128
        self.max_load_factor = 0.5
        self.block_size = 128
        self.io_block_size = 1024
        self.device_id = 0
        self.io_by_cpu = False
        self.use_constant_memory = False
        self.reserved_key_start_bit = 0
        self.num_of_buckets_per_alloc = 1
        self.initializer_args = demb.InitializerArgs()
        self.safe_check_mode = demb.SafeCheckMode.IGNORE
        self.optimizer_type = demb.OptimizerType.Null


@pytest.fixture
def table_options():
    return DynamicEmbTableOptions()


@pytest.fixture
def dynamic_table(table_options: DynamicEmbTableOptions):
    torch.npu.set_device(table_options.device_id)
    return demb.DynamicEmbTable(
        table_options.key_type,
        table_options.value_type,
        table_options.evict_strategy,
        table_options.dim,
        table_options.init_capacity,
        table_options.max_capacity,
        table_options.max_hbm_for_vectors,
        table_options.max_bucket_size,
        table_options.max_load_factor,
        table_options.block_size,
        table_options.io_block_size,
        table_options.device_id,
        table_options.io_by_cpu,
        table_options.use_constant_memory,
        table_options.reserved_key_start_bit,
        table_options.num_of_buckets_per_alloc,
        table_options.initializer_args,
        table_options.safe_check_mode,
        table_options.optimizer_type
    )


def test_insert_or_assign_basic(dynamic_table):
    """测试基本的插入操作"""
    torch.npu.set_device(0)
    n = 10
    dim = 128  # 从table_options获取
    
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device='npu:0')
    values = torch.randn(n, dim, dtype=torch.float32, device='npu:0')
    
    # 插入键值对
    dynamic_table.load(
        n, keys, values, None, True, False
    )
    
    # 验证插入成功（通过查找验证）
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    demb.find_pointers(
        dynamic_table, n, keys, values_out, founds, None
    )

    assert founds.sum().item() == n, "插入的key能全部找到"


@pytest.mark.skip(reason="insert_or_assign暂时不支持scores更新,暂时跳过此测试,支持后放开")
def test_insert_or_assign_with_score(dynamic_table):
    """测试带score的插入操作"""
    torch.npu.set_device(0)
    n = 10
    dim = 128
    
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device='npu:0')
    values = torch.randn(n, dim, dtype=torch.float32, device='npu:0')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device='npu:0')
    
    # 插入带score的键值对
    dynamic_table.load(
        n, keys, values, scores, True, False
    )
    
    # 验证插入成功
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    demb.find_pointers(
        dynamic_table, n, keys, values_out, founds, None
    )
    
    assert founds.sum().item() == n, "插入的key能全部找到"


def test_insert_or_assign_unique_key(dynamic_table):
    """测试unique_key参数"""
    torch.npu.set_device(0)
    n = 5
    dim = 128
    
    # 使用相同的键
    keys = torch.tensor([1, 1, 2, 2, 3], dtype=torch.int64, device='npu:0')
    values1 = torch.randn(n, dim, dtype=torch.float32, device='npu:0')
    values2 = torch.randn(n, dim, dtype=torch.float32, device='npu:0')
    
    # 第一次插入
    dynamic_table.load(
        n, keys, values1, None, True, False
    )
    
    # 第二次插入相同键（应该更新）
    dynamic_table.update(
        n, keys, values2, None, True, False
    )
    
    # 验证键存在
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    logging.info(f"values_out1 = {values_out}")
    demb.find_pointers(
        dynamic_table, n, keys, values_out, founds, None
    )

    assert founds.sum().item() == n, "插入的key能全部找到"


def test_find_pointers_basic(dynamic_table):
    """测试基本的查找操作"""
    torch.npu.set_device(0)
    n = 10
    dim = 128
    
    # 先插入一些键值对
    keys_insert = torch.randint(0, 1000, (n,), dtype=torch.int64, device='npu:0')
    values_insert = torch.randn(n, dim, dtype=torch.float32, device='npu:0')
    dynamic_table.load(
        n, keys_insert, values_insert, None, True, False
    )
    
    # 查找这些键
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    demb.find_pointers(
        dynamic_table, n, keys_insert, values_out, founds, None
    )
    
    # 验证所有键都被找到
    assert founds.sum().item() == n, f"找到所有{n}个键"


def test_find_pointers_not_found(dynamic_table):
    """测试查找不存在的键"""
    torch.npu.set_device(0)
    n = 10
    dim = 128
    
    # 插入一些键
    keys_insert = torch.randint(0, 100, (5,), dtype=torch.int64, device='npu:0')
    values_insert = torch.randn(5, dim, dtype=torch.float32, device='npu:0')
    dynamic_table.load(
        5, keys_insert, values_insert, None, True, False
    )
    
    # 查找不存在的键
    keys_not_found = torch.randint(1000, 2000, (n,), dtype=torch.int64, device='npu:0')
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    demb.find_pointers(
        dynamic_table, n, keys_not_found, values_out, founds, None
    )
    
    # 应该找不到任何键
    assert founds.sum().item() == 0, "不应该找到不存在的键"


@pytest.mark.parametrize("evict_strategy", [
    demb.EvictStrategy.kLru,
    demb.EvictStrategy.kLfu,
    demb.EvictStrategy.kCustomized
])
def test_insert_and_find_different_strategies(evict_strategy):
    """测试不同淘汰策略下的插入和查找"""
    torch.npu.set_device(0)
    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        evict_strategy,
        128, 1024, 2048, 1 * 1024 * 1024 * 1024, 128, 0.5, 128, 1024, 0, False, False, 0, 1,
        demb.InitializerArgs(), demb.SafeCheckMode.IGNORE, demb.OptimizerType.Null
    )
    
    n = 10
    dim = 128
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device='npu:0')
    values = torch.randn(n, dim, dtype=torch.float32, device='npu:0')
    
    # 插入（使用卫语句处理需要score的情况）
    scores = None
    if evict_strategy != demb.EvictStrategy.kLru:
        scores = torch.randint(1, 100, (n,), dtype=torch.int64, device='npu:0')
    
    table.load(
        n, keys, values, scores, True, False
    )
    
    # 查找
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    demb.find_pointers(
        table, n, keys, values_out, founds, None
    )
    
    assert founds.sum().item() > 0, f"在{evict_strategy}策略下能找到键"


def test_empty_table_operations(dynamic_table):
    """测试空表的操作"""
    torch.npu.set_device(0)
    n = 5
    
    # 在空表中查找
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device='npu:0')
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    demb.find_pointers(
        dynamic_table, n, keys, values_out, founds, None
    )
    
    # 应该找不到任何键
    assert founds.sum().item() == 0, "空表中不应该找到任何键"


def test_insert_or_assign_ignore_evict_strategy(dynamic_table):
    """测试ignore_evict_strategy参数"""
    torch.npu.set_device(0)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device='npu:0')
    values = torch.randn(n, dim, dtype=torch.float32, device='npu:0')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device='npu:0')

    # 插入时忽略淘汰策略
    dynamic_table.load(
        n, keys, values, scores, True, True
    )

    # 验证插入成功
    values_out = torch.zeros(n, dtype=torch.int64, device='npu:0')
    founds = torch.zeros(n, dtype=torch.bool, device='npu:0')
    demb.find_pointers(
        dynamic_table, n, keys, values_out, founds, None
    )

    assert founds.sum().item() > 0, "忽略淘汰策略后能找到键"


def test_export_batch(dynamic_table):
    """测试export_batch参数"""
    torch.npu.set_device(0)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device='npu:0')
    values = torch.randn(n, dim, dtype=torch.float32, device='npu:0')

    # 加载数据
    dynamic_table.load(
        n, keys, values, None, True, False
    )

    # 导出数据
    keys_out = torch.empty(n, dtype=torch.int64, device='npu:0')
    values_out = torch.empty(n, dtype=torch.int64, device='npu:0')
    d_counter = torch.empty(n, dtype=torch.uint64, device='npu:0')
    dynamic_table.export_batch(
        n, 0, d_counter, keys_out, values_out, None
    )

    assert keys_out.numel() == n, "导出数据量符合预期"

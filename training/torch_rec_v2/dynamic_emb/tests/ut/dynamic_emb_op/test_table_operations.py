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

# pylint: disable=too-many-lines,redefined-outer-name,logging-fstring-interpolation

import logging
import pytest
import torch
import dynamic_emb_extensions as demb

logging.basicConfig(level=logging.INFO)

# 统一控制设备ID
DEVICE_ID = 0


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
        self.device_id = DEVICE_ID
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


def create_dynamic_table(**overrides):
    table_options = DynamicEmbTableOptions()
    for name, value in overrides.items():
        setattr(table_options, name, value)
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
        table_options.optimizer_type,
    )


@pytest.fixture
def dynamic_table(table_options: DynamicEmbTableOptions):
    return create_dynamic_table(**table_options.__dict__)


DDR_INITIALIZER_CASES = ["constant", "debug", "normal", "truncated_normal", "uniform"]


def make_initializer_args(initializer):
    if initializer == "constant":
        return demb.InitializerArgs("constant", 0.0, 1.0, 0.0, 1.0, -0.25)
    if initializer == "debug":
        return demb.InitializerArgs("debug", 0.0, 1.0, 0.0, 1.0, 0.0)
    if initializer == "normal":
        return demb.InitializerArgs("normal", 0.0, 1.0, 0.0, 1.0, 0.0)
    if initializer == "truncated_normal":
        return demb.InitializerArgs("truncated_normal", 0.0, 0.876, -2.0, 2.0, 0.0)
    if initializer == "uniform":
        return demb.InitializerArgs("uniform", 0.5, 0.2887, 0.0, 1.0, 0.0)
    raise ValueError(f"unsupported initializer {initializer}")


def assert_initialized_values(initializer, values, keys):
    if initializer == "constant":
        expected_values = torch.full_like(values, -0.25)
        torch.testing.assert_close(values.cpu(), expected_values.cpu())
    elif initializer == "debug":
        expected_values = (keys % 100000).to(torch.float32).unsqueeze(1).expand_as(values)
        torch.testing.assert_close(values.cpu(), expected_values.cpu())
    elif initializer == "normal":
        values_mean = values.mean().item()
        values_std = values.std().item()
        assert abs(values_mean) < 0.3, f"normal initializer mean should be close to 0.0, got {values_mean}"
        assert abs(values_std - 1.0) < 0.3, f"normal initializer std should be close to 1.0, got {values_std}"
    elif initializer == "truncated_normal":
        assert (values >= -2.0).all(), "truncated_normal values should be >= -2.0"
        assert (values <= 2.0).all(), "truncated_normal values should be <= 2.0"
    elif initializer == "uniform":
        assert (values >= 0.0).all(), "uniform values should be >= 0.0"
        assert (values <= 1.0).all(), "uniform values should be <= 1.0"
        values_mean = values.mean().item()
        assert abs(values_mean - 0.5) < 0.3, f"uniform initializer mean should be close to 0.5, got {values_mean}"
    else:
        raise ValueError(f"unsupported initializer {initializer}")


def test_insert_or_assign_basic(dynamic_table):
    """测试基本的插入操作"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128  # 从table_options获取

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    # 插入键值对
    dynamic_table.load(n, keys, values, None, True, False)

    # 验证插入成功（通过查找验证）
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find_pointers(dynamic_table, n, keys, values_out, founds, None)

    assert founds.sum().item() == n, "插入的key能全部找到"


def test_insert_or_assign_with_score():
    """测试带score的插入操作"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    # 使用LFU策略的table，支持score参数
    table = create_dynamic_table(dim=dim, evict_strategy=demb.EvictStrategy.kLfu)

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    # 插入带score的键值对
    table.load(n, keys, values, scores, True, False)

    # 验证插入成功
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find_pointers(table, n, keys, values_out, founds, None)

    assert founds.sum().item() == n, "插入的key能全部找到"


def test_insert_or_assign_unique_key(dynamic_table):
    """测试unique_key参数"""
    torch.npu.set_device(DEVICE_ID)
    n = 5
    dim = 128

    # 使用相同的键
    keys = torch.tensor([1, 1, 2, 2, 3], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values1 = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    values2 = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    # 第一次插入
    dynamic_table.load(n, keys, values1, None, True, False)

    # 第二次插入相同键（应该更新）
    dynamic_table.update(n, keys, values2, None, True, False)

    # 验证键存在
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    logging.info(f"values_out1 = {values_out}")
    demb.find_pointers(dynamic_table, n, keys, values_out, founds, None)

    assert founds.sum().item() == n, "插入的key能全部找到"


def test_find_pointers_basic(dynamic_table):
    """测试基本的查找操作"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    # 先插入一些键值对
    keys_insert = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_insert = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n, keys_insert, values_insert, None, True, False)

    # 查找这些键
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find_pointers(dynamic_table, n, keys_insert, values_out, founds, None)

    # 验证所有键都被找到
    assert founds.sum().item() == n, f"找到所有{n}个键"


def test_find_pointers_not_found(dynamic_table):
    """测试查找不存在的键"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    # 插入一些键
    keys_insert = torch.randint(0, 100, (5,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_insert = torch.randn(5, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(5, keys_insert, values_insert, None, True, False)

    # 查找不存在的键
    keys_not_found = torch.randint(1000, 2000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find_pointers(dynamic_table, n, keys_not_found, values_out, founds, None)

    # 应该找不到任何键
    assert founds.sum().item() == 0, "不应该找到不存在的键"


def test_find_and_initialize_constant(dynamic_table):
    torch.npu.set_device(DEVICE_ID)
    n_existing = 4
    n_missing = 3
    dim = 128
    init_value = 0.5

    existing_keys = torch.arange(1, n_existing + 1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.arange(1001, 1001 + n_missing, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.cat([existing_keys, missing_keys])
    n = keys.numel()

    value_ptrs = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    initializer_args = demb.InitializerArgs("constant", 0.0, 1.0, 0.0, 1.0, init_value)

    demb.find_and_initialize(dynamic_table, n, keys, value_ptrs, values_out, founds, initializer_args)

    assert torch.all(founds[:n_existing]).item(), "existing keys should be found"
    assert not torch.any(founds[n_existing:]).item(), "missing keys should not be found"
    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    torch.testing.assert_close(
        values_out[n_existing:].cpu(),
        torch.full((n_missing, dim), init_value, dtype=torch.float32),
    )
    assert demb.dyn_emb_rows(dynamic_table) == n_existing


@pytest.mark.parametrize("initializer", DDR_INITIALIZER_CASES)
def test_find_and_initialize_ddr(initializer):
    torch.npu.set_device(DEVICE_ID)
    n_existing = 4
    dim = 204800

    table = create_dynamic_table(dim=dim, max_hbm_for_vectors=0)

    existing_keys = torch.tensor([11, 22, 33, 44], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.arange(n_existing * dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}').reshape(
        n_existing, dim
    )
    table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.tensor([1011, 1022, 1033, 1044], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.stack(
        [
            existing_keys[0],
            missing_keys[0],
            existing_keys[1],
            missing_keys[1],
            existing_keys[2],
            missing_keys[2],
            existing_keys[3],
            missing_keys[3],
        ]
    )
    n = keys.numel()

    value_ptrs = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    initializer_args = make_initializer_args(initializer)

    demb.find_and_initialize(table, n, keys, value_ptrs, values_out, founds, initializer_args)

    expected_founds = torch.tensor([True, False, True, False, True, False, True, False], device=f'npu:{DEVICE_ID}')

    torch.testing.assert_close(founds.cpu(), expected_founds.cpu())
    torch.testing.assert_close(values_out[0::2].cpu(), existing_values.cpu())
    assert_initialized_values(initializer, values_out[1::2], keys[1::2])
    assert demb.dyn_emb_rows(table) == n_existing


def test_find_and_initialize_debug(dynamic_table):
    torch.npu.set_device(DEVICE_ID)
    n = 5
    dim = 128
    keys = torch.tensor([7, 101, 100001, 200002, 99999], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    value_ptrs = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    initializer_args = demb.InitializerArgs("debug", 0.0, 1.0, 0.0, 1.0, 0.0)

    demb.find_and_initialize(dynamic_table, n, keys, value_ptrs, values_out, founds, initializer_args)

    # Debug initializer maps each key to key % 100000 for deterministic output.
    expected_values = (keys % 100000).to(torch.float32).unsqueeze(1).expand(n, dim)
    assert not torch.any(founds).item(), "empty table should not find any keys"
    torch.testing.assert_close(values_out.cpu(), expected_values.cpu())
    assert demb.dyn_emb_rows(dynamic_table) == 0


def test_find_and_initialize_normal():
    """测试find_and_initialize使用normal初始化器"""
    torch.npu.set_device(DEVICE_ID)
    dim = 204800
    dynamic_table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        dim,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        DEVICE_ID,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null,
    )
    n_existing = 4
    n_missing = 3
    mean = 0.0
    std = 1.0

    existing_keys = torch.arange(1, n_existing + 1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.arange(1001, 1001 + n_missing, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.cat([existing_keys, missing_keys])
    n = keys.numel()

    value_ptrs = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    initializer_args = demb.InitializerArgs("normal", mean, std, 0.0, 1.0, 0.0)

    demb.find_and_initialize(dynamic_table, n, keys, value_ptrs, values_out, founds, initializer_args)

    assert torch.all(founds[:n_existing]).item(), "existing keys should be found"
    assert not torch.any(founds[n_existing:]).item(), "missing keys should not be found"
    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    # Check missing keys were initialized with normal distribution (mean=0.0, std=1.0)
    # Mean should be close to 0.0, std should be close to 1.0
    missing_values_mean = values_out[n_existing:].mean().item()
    missing_values_std = values_out[n_existing:].std().item()
    assert abs(missing_values_mean) < 0.3, f"normal initializer mean should be close to 0.0, got {missing_values_mean}"
    assert abs(missing_values_std - 1.0) < 0.3, (
        f"normal initializer std should be close to 1.0, got {missing_values_std}"
    )
    assert demb.dyn_emb_rows(dynamic_table) == n_existing


def test_find_and_initialize_truncated_normal():
    """测试find_and_initialize使用truncated_normal初始化器"""
    torch.npu.set_device(DEVICE_ID)
    dim = 204800
    dynamic_table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        dim,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        DEVICE_ID,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null,
    )
    n_existing = 4
    n_missing = 3
    mean = 0.0
    std = 0.876
    min_val = -2.0
    max_val = 2.0

    existing_keys = torch.arange(1, n_existing + 1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.arange(1001, 1001 + n_missing, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.cat([existing_keys, missing_keys])
    n = keys.numel()

    value_ptrs = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    initializer_args = demb.InitializerArgs("truncated_normal", mean, std, min_val, max_val, 0.0)

    demb.find_and_initialize(dynamic_table, n, keys, value_ptrs, values_out, founds, initializer_args)

    assert torch.all(founds[:n_existing]).item(), "existing keys should be found"
    assert not torch.any(founds[n_existing:]).item(), "missing keys should not be found"
    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    # Check missing keys were initialized with truncated normal distribution
    # Values should be within [min_val, max_val] = [-2.0, 2.0]
    missing_values = values_out[n_existing:]
    assert (missing_values >= min_val).all(), f"truncated_normal values should be >= {min_val}"
    assert (missing_values <= max_val).all(), f"truncated_normal values should be <= {max_val}"
    assert demb.dyn_emb_rows(dynamic_table) == n_existing


def test_find_and_initialize_uniform():
    """测试find_and_initialize使用uniform初始化器"""
    torch.npu.set_device(DEVICE_ID)
    dim = 204800
    dynamic_table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        dim,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        DEVICE_ID,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null,
    )

    n_existing = 4
    n_missing = 3
    mean = 0.5
    std_dev = 0.2887
    min_val = 0.0
    max_val = 1.0

    existing_keys = torch.arange(1, n_existing + 1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.arange(1001, 1001 + n_missing, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.cat([existing_keys, missing_keys])
    n = keys.numel()

    value_ptrs = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    initializer_args = demb.InitializerArgs("uniform", mean, std_dev, min_val, max_val, 0.0)

    demb.find_and_initialize(dynamic_table, n, keys, value_ptrs, values_out, founds, initializer_args)

    assert torch.all(founds[:n_existing]).item(), "existing keys should be found"
    assert not torch.any(founds[n_existing:]).item(), "missing keys should not be found"
    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    # Check missing keys were initialized with uniform distribution
    # All values should be within [min_val, max_val] = [0.0, 1.0]
    missing_values = values_out[n_existing:]
    assert (missing_values >= min_val).all(), f"uniform values should be >= {min_val}"
    assert (missing_values <= max_val).all(), f"uniform values should be <= {max_val}"
    # Mean should be close to 0.5 for uniform distribution
    missing_values_mean = missing_values.mean().item()
    assert abs(missing_values_mean - 0.5) < 0.3, (
        f"uniform initializer mean should be close to 0.5, got {missing_values_mean}"
    )
    assert demb.dyn_emb_rows(dynamic_table) == n_existing


@pytest.mark.parametrize(
    "evict_strategy", [demb.EvictStrategy.kLru, demb.EvictStrategy.kLfu, demb.EvictStrategy.kCustomized]
)
def test_insert_and_find_different_strategies(evict_strategy):
    """测试不同淘汰策略下的插入和查找"""
    torch.npu.set_device(DEVICE_ID)
    table = create_dynamic_table(evict_strategy=evict_strategy)

    n = 10
    dim = 128
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    # 插入（使用卫语句处理需要score的情况）
    scores = None
    find_scores = None
    if evict_strategy != demb.EvictStrategy.kLru:
        scores = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
        find_scores = 1

    table.load(n, keys, values, scores, True, False)

    # 查找
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find_pointers(table, n, keys, values_out, founds, find_scores)

    assert founds.sum().item() > 0, f"在{evict_strategy}策略下能找到键"


def test_empty_table_operations(dynamic_table):
    """测试空表的操作"""
    torch.npu.set_device(DEVICE_ID)
    n = 5

    # 在空表中查找
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find_pointers(dynamic_table, n, keys, values_out, founds, None)

    # 应该找不到任何键
    assert founds.sum().item() == 0, "空表中不应该找到任何键"


def test_insert_or_assign_ignore_evict_strategy(dynamic_table):
    """测试ignore_evict_strategy参数"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    # 插入时忽略淘汰策略
    dynamic_table.load(n, keys, values, scores, True, True)

    # 验证插入成功
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find_pointers(dynamic_table, n, keys, values_out, founds, None)

    assert founds.sum().item() > 0, "忽略淘汰策略后能找到键"


def test_export_batch(dynamic_table):
    """测试export_batch参数"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    # 加载数据
    dynamic_table.load(n, keys, values, None, True, False)

    # 导出数据
    keys_out = torch.empty(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.empty(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    d_counter = torch.empty(n, dtype=torch.uint64, device=f'npu:{DEVICE_ID}')
    dynamic_table.export_batch(n, 0, d_counter, keys_out, values_out, None)

    assert keys_out.numel() == n, "导出数据量符合预期"


def test_dyn_emb_rows_basic(dynamic_table):
    """测试dyn_emb_rows基本功能"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == n, f"期望行数为{n}，实际为{rows}"


def test_dyn_emb_rows_empty(dynamic_table):
    """测试dyn_emb_rows空表"""
    torch.npu.set_device(DEVICE_ID)

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == 0, f"空表行数期望为0，实际为{rows}"


def test_dyn_emb_rows_after_erase(dynamic_table):
    """测试dyn_emb_rows删除后"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    erase_n = 5
    keys_to_erase = keys[:erase_n]
    demb.erase(dynamic_table, erase_n, keys_to_erase)

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == n - erase_n, f"删除后期望行数为{n - erase_n}，实际为{rows}"


def test_insert_and_evict_basic(dynamic_table):
    """测试insert_and_evict基本功能"""
    torch.npu.set_device(DEVICE_ID)
    n = 5
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    evicted_keys = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    evicted_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    evicted_score = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    d_evicted_counter = torch.zeros(1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    demb.insert_and_evict(
        dynamic_table,
        n,
        keys,
        values,
        None,
        evicted_keys,
        evicted_values,
        evicted_score,
        d_evicted_counter,
        True,
        False,
    )

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == n, f"期望行数为{n}，实际为{rows}"


def test_insert_and_evict_with_lfu():
    """测试insert_and_evict在LFU策略下需要score"""
    torch.npu.set_device(DEVICE_ID)
    table = create_dynamic_table(evict_strategy=demb.EvictStrategy.kLfu)

    n = 5
    dim = 128
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    evicted_keys = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    evicted_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    evicted_score = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    d_evicted_counter = torch.zeros(1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    demb.insert_and_evict(
        table,
        n,
        keys,
        values,
        scores[0].item(),
        evicted_keys,
        evicted_values,
        evicted_score,
        d_evicted_counter,
        True,
        False,
    )

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"LFU策略下期望行数为{n}，实际为{rows}"


def test_find_basic(dynamic_table):
    """测试find基本功能"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(dynamic_table, n, keys, values, founds, None)

    assert founds.sum().item() == n, f"期望找到所有{n}个键"


def test_find_with_score():
    """测试find带score查找"""
    torch.npu.set_device(DEVICE_ID)
    table = create_dynamic_table(evict_strategy=demb.EvictStrategy.kLfu)

    n = 5
    dim = 128
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    evicted_keys = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    evicted_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    evicted_score = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    d_evicted_counter = torch.zeros(1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    demb.insert_and_evict(
        table,
        n,
        keys,
        values,
        scores[0].item(),
        evicted_keys,
        evicted_values,
        evicted_score,
        d_evicted_counter,
        True,
        False,
    )

    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, keys, values, founds, None)

    assert founds.sum().item() == n, f"期望找到所有{n}个键"


def test_find_not_found(dynamic_table):
    """测试find查找不存在的key"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    not_exist_keys = torch.randint(10000, 20000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(dynamic_table, n, not_exist_keys, values, founds, None)

    assert founds.sum().item() == 0, "不应该找到任何不存在的键"


def test_erase_basic(dynamic_table):
    """测试erase基本功能"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    erase_n = 5
    keys_to_erase = keys[:erase_n]
    demb.erase(dynamic_table, erase_n, keys_to_erase)

    founds = torch.zeros(erase_n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    output_values = torch.zeros(erase_n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find(dynamic_table, erase_n, keys_to_erase, output_values, founds, None)

    assert founds.sum().item() == 0, "删除的键不应该被找到"


def test_erase_not_exists(dynamic_table):
    """测试erase删除不存在的key"""
    torch.npu.set_device(DEVICE_ID)
    n = 5
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    not_exist_keys = torch.randint(10000, 20000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    demb.erase(dynamic_table, n, not_exist_keys)

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == n, "删除不存在的键不应影响表内容"


def test_clear_basic(dynamic_table):
    """测试clear清空表"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    demb.clear(dynamic_table)

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == 0, f"清空后行数期望为0,实际为{rows}"


def test_reserve_basic(dynamic_table):
    """测试reserve预留容量"""
    torch.npu.set_device(DEVICE_ID)

    new_capacity = 4096
    demb.reserve(dynamic_table, new_capacity)

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == 0, "预留容量后空表行数仍为0"


def test_reserve_expand(dynamic_table):
    """测试reserve扩大容量"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    dynamic_table.load(n, keys, values, None, True, False)

    new_capacity = 4096
    demb.reserve(dynamic_table, new_capacity)

    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == n, f"扩大容量后行数应保持为{n}"


def test_find_or_insert_pointers_basic(dynamic_table):
    """测试find_or_insert_pointers基本功能 - 插入新键"""
    torch.npu.set_device(DEVICE_ID)
    n = 10

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')

    demb.find_or_insert_pointers(dynamic_table, n, keys, values, founds, None, True, False)
    # 新keys在table中找不到，直接插入新keys，验证table是否插入正确数量的keys
    assert founds.sum().item() == 0, "期望插入0个键"
    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == n, f"期望表行数为{n}，实际为{rows}"


def test_find_or_insert_pointers_found(dynamic_table):
    """测试find_or_insert_pointers查找已存在的键"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    # 先插入数据
    dynamic_table.load(n, keys, values, None, True, False)

    # 使用find_or_insert_pointers查找这些键
    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds_out = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')

    demb.find_or_insert_pointers(dynamic_table, n, keys, values_out, founds_out, None, True, False)

    assert founds_out.sum().item() == n, f"期望找到所有{n}个已存在的键"


def test_find_or_insert_pointers_mixed(dynamic_table):
    """测试find_or_insert_pointers混合场景 - 部分键存在，部分不存在"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    # 先插入5个键
    existing_keys = torch.randint(0, 500, (5,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(5, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(5, existing_keys, existing_values, None, True, False)

    # 构造10个键，其中5个已存在，5个不存在
    all_keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    # 确保前5个是已存在的
    all_keys[:5] = existing_keys[:5]

    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds_out = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')

    demb.find_or_insert_pointers(dynamic_table, n, all_keys, values_out, founds_out, None, True, False)

    # 前5个应该被找到，后5个应该被插入
    assert founds_out.sum().item() >= 5, "期望至少找到5个已存在的键"
    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows >= 10, f"期望表行数至少为10，实际为{rows}"


def test_find_or_insert_pointers_with_lfu():
    """测试find_or_insert_pointers在LFU策略下使用score"""
    torch.npu.set_device(DEVICE_ID)
    table = create_dynamic_table(evict_strategy=demb.EvictStrategy.kLfu)

    n = 10
    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds_out = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')

    demb.find_or_insert_pointers(table, n, keys, values_out, founds_out, scores[0].item(), True, False)

    # 新keys在table中找不到，直接插入新keys，验证table是否插入正确数量的keys
    assert founds_out.sum().item() == 0, "期望插入0个键"
    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"LFU策略下期望行数为{n}，实际为{rows}"


def test_find_or_insert_pointers_ignore_evict_strategy(dynamic_table):
    """测试find_or_insert_pointers的ignore_evict_strategy参数"""
    torch.npu.set_device(DEVICE_ID)
    n = 10

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    scores = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')

    values_out = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    founds_out = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')

    demb.find_or_insert_pointers(dynamic_table, n, keys, values_out, founds_out, scores[0].item(), True, True)

    # 新keys在table中找不到，直接插入新keys，验证table是否插入正确数量的keys
    assert founds_out.sum().item() == 0, "期望插入0个键"
    rows = demb.dyn_emb_rows(dynamic_table)
    assert rows == n, f"忽略淘汰策略后期望行数为{n}，实际为{rows}"


def test_find_or_insert_basic_find():
    """测试find_or_insert基本功能 - 查询已存在的键，验证返回值与插入数据一致（使用debug初始化器）"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128

    init_args = demb.InitializerArgs("debug", 0.0, 1.0, 0.0, 1.0, 0.0)
    table = create_dynamic_table(dim=dim, initializer_args=init_args)

    keys = torch.randint(0, 1000, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')

    table.load(n, keys, values, None, True, False)

    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, keys, values_out, None, True, False)

    assert torch.allclose(values_out, values), "查询到的值应与插入的值相同"
    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"期望表行数为{n}，实际为{rows}"


def test_find_or_insert_debug_initializer():
    """测试find_or_insert在debug初始化器下插值，验证值与MappingEmbeddingGenerator一致"""
    torch.npu.set_device(DEVICE_ID)
    dim = 128
    n = 10

    init_args = demb.InitializerArgs("debug", 0.0, 1.0, 0.0, 1.0, 0.0)
    table = create_dynamic_table(dim=dim, initializer_args=init_args)

    keys = torch.randint(0, 500, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, keys, values_out, None, True, False)

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"期望表行数为{n}，实际为{rows}"

    # 用find接口验证值与MappingEmbeddingGenerator一致
    found_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, keys, found_values, founds, None)

    assert founds.sum().item() == n, f"期望找到所有{n}个键"

    # MappingEmbeddingGenerator: generate(vec_id) = float(key % 100000)
    for i in range(n):
        expected_val = float(keys[i].item() % 100000)
        assert torch.allclose(found_values[i], torch.full((dim,), expected_val, device=f'npu:{DEVICE_ID}')), (
            f"键{keys[i]}的值应与MappingEmbeddingGenerator生成的值一致"
        )


def test_find_or_insert_constant_initializer():
    """测试find_or_insert在constant初始化器下插值，验证值与ConstEmbeddingGenerator一致"""
    torch.npu.set_device(DEVICE_ID)
    dim = 64
    n = 10
    const_val = 0.5

    init_args = demb.InitializerArgs("constant", 0.0, 1.0, 0.0, 1.0, const_val)
    table = create_dynamic_table(dim=dim, initializer_args=init_args)

    keys = torch.randint(0, 500, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, keys, values_out, None, True, False)

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"期望表行数为{n}，实际为{rows}"

    found_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, keys, found_values, founds, None)

    assert founds.sum().item() == n, f"期望找到所有{n}个键"

    # ConstEmbeddingGenerator: generate(vec_id) = val (constant)
    for i in range(n):
        assert torch.allclose(found_values[i], torch.full((dim,), const_val, device=f'npu:{DEVICE_ID}')), (
            f"键{keys[i]}的所有维度值应为{const_val}"
        )


def test_find_or_insert_normal_initializer():
    """测试find_or_insert使用normal初始化器"""
    torch.npu.set_device(DEVICE_ID)
    dim = 204800
    mean = 0.0
    std = 1.0
    initializer_args = demb.InitializerArgs("normal", mean, std, 0.0, 1.0, 0.0)
    dynamic_table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        dim,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        DEVICE_ID,
        False,
        False,
        0,
        1,
        initializer_args,
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null,
    )
    n_existing = 4
    n_missing = 3

    existing_keys = torch.arange(1, n_existing + 1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.arange(1001, 1001 + n_missing, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.cat([existing_keys, missing_keys])
    n = keys.numel()

    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(dynamic_table, n, keys, values_out, None, True, False)

    # Verify existing keys are found and values match
    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    # Verify missing keys were initialized with normal distribution (mean=0.0, std=1.0)
    missing_values = values_out[n_existing:]
    missing_values_mean = missing_values.mean().item()
    missing_values_std = missing_values.std().item()
    assert abs(missing_values_mean) < 0.3, f"normal initializer mean should be close to 0.0, got {missing_values_mean}"
    assert abs(missing_values_std - 1.0) < 0.3, (
        f"normal initializer std should be close to 1.0, got {missing_values_std}"
    )
    # Total rows should be n_existing + n_missing since missing keys are inserted
    assert demb.dyn_emb_rows(dynamic_table) == n


def test_find_or_insert_truncated_normal_initializer():
    """测试find_or_insert使用truncated_normal初始化器"""
    torch.npu.set_device(DEVICE_ID)
    dim = 204800
    mean = 0.0
    std = 0.876
    min_val = -2.0
    max_val = 2.0
    initializer_args = demb.InitializerArgs("truncated_normal", mean, std, min_val, max_val, 0.0)
    dynamic_table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        dim,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        DEVICE_ID,
        False,
        False,
        0,
        1,
        initializer_args,
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null,
    )
    n_existing = 4
    n_missing = 3

    existing_keys = torch.arange(1, n_existing + 1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.arange(1001, 1001 + n_missing, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.cat([existing_keys, missing_keys])
    n = keys.numel()

    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(dynamic_table, n, keys, values_out, None, True, False)

    # Verify existing keys are found and values match
    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    # Verify missing keys were initialized with truncated normal distribution
    # Values should be within [min_val, max_val] = [-2.0, 2.0]
    missing_values = values_out[n_existing:]
    assert (missing_values >= min_val).all(), f"truncated_normal values should be >= {min_val}"
    assert (missing_values <= max_val).all(), f"truncated_normal values should be <= {max_val}"
    # Total rows should be n_existing + n_missing since missing keys are inserted
    assert demb.dyn_emb_rows(dynamic_table) == n


def test_find_or_insert_uniform_initializer():
    """测试find_or_insert使用uniform初始化器"""
    torch.npu.set_device(DEVICE_ID)
    dim = 204800
    mean = 0.5
    std_dev = 0.2887
    min_val = 0.0
    max_val = 1.0
    initializer_args = demb.InitializerArgs("uniform", mean, std_dev, min_val, max_val, 0.0)
    dynamic_table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        dim,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        DEVICE_ID,
        False,
        False,
        0,
        1,
        initializer_args,
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null,
    )
    n_existing = 4
    n_missing = 3

    existing_keys = torch.arange(1, n_existing + 1, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    dynamic_table.load(n_existing, existing_keys, existing_values, None, True, False)

    missing_keys = torch.arange(1001, 1001 + n_missing, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    keys = torch.cat([existing_keys, missing_keys])
    n = keys.numel()

    values_out = torch.empty(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(dynamic_table, n, keys, values_out, None, True, False)

    # Verify existing keys are found and values match
    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    # Verify missing keys were initialized with uniform distribution
    # All values should be within [min_val, max_val] = [0.0, 1.0]
    missing_values = values_out[n_existing:]
    assert (missing_values >= min_val).all(), f"uniform values should be >= {min_val}"
    assert (missing_values <= max_val).all(), f"uniform values should be <= {max_val}"
    # Mean should be close to 0.5 for uniform distribution
    missing_values_mean = missing_values.mean().item()
    assert abs(missing_values_mean - 0.5) < 0.3, (
        f"uniform initializer mean should be close to 0.5, got {missing_values_mean}"
    )
    # Total rows should be n_existing + n_missing since missing keys are inserted
    assert demb.dyn_emb_rows(dynamic_table) == n


def test_find_or_insert_mixed():
    """测试find_or_insert混合场景 - 部分键已存在，部分键需要插入，验证查询和插值均正常（使用constant初始化器）"""
    torch.npu.set_device(DEVICE_ID)
    n = 10
    dim = 128
    const_val = 0.5

    init_args = demb.InitializerArgs("constant", 0.0, 1.0, 0.0, 1.0, const_val)
    table = create_dynamic_table(dim=dim, initializer_args=init_args)

    # 先插入前5个键
    existing_keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(5, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    table.load(5, existing_keys, existing_values, None, True, False)

    # 构造10个键：前5个已存在，后5个不存在
    new_keys = torch.tensor([101, 102, 103, 104, 105], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    all_keys = torch.cat([existing_keys, new_keys])

    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, all_keys, values_out, None, True, False)

    # 验证前5个已存在的键值
    assert torch.allclose(values_out[:5], existing_values), "前5个已存在的键值应与插入的值相同"

    # 验证后5个新插入的键值应为constant值
    for i in range(5, n):
        assert torch.allclose(values_out[i], torch.full((dim,), const_val, device=f'npu:{DEVICE_ID}')), (
            f"新插入的键值应为constant值{const_val}"
        )

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"期望表行数为{n}，实际为{rows}"

    # 用find接口再次确认所有键都存在
    found_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, all_keys, found_values, founds, None)

    assert founds.sum().item() == n, f"期望找到所有{n}个键"
    assert torch.allclose(found_values[:5], existing_values), "前5个键的查找值应与插入值相同"
    for i in range(5, n):
        assert torch.allclose(found_values[i], torch.full((dim,), const_val, device=f'npu:{DEVICE_ID}')), (
            f"新插入的键查找值应为constant值{const_val}"
        )


@pytest.mark.parametrize("initializer", DDR_INITIALIZER_CASES)
def test_find_or_insert_ddr_tiling(initializer):
    """测试find_or_insert在DDR大dim切片场景下正确处理已存在和新插入的键"""
    torch.npu.set_device(DEVICE_ID)
    dim = 32769  # 大dim触发value move切片，并覆盖最后一个非满tile
    n_existing = 2
    n_new = 2
    n = n_existing + n_new

    init_args = make_initializer_args(initializer)
    table = create_dynamic_table(dim=dim, max_hbm_for_vectors=0, initializer_args=init_args)

    existing_keys = torch.tensor([1, 2], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    existing_values = torch.randn(n_existing, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    table.load(n_existing, existing_keys, existing_values, None, True, False)

    new_keys = torch.tensor([101, 102], dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    all_keys = torch.cat([existing_keys, new_keys])

    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, all_keys, values_out, None, True, False)

    torch.testing.assert_close(values_out[:n_existing].cpu(), existing_values.cpu())
    assert_initialized_values(initializer, values_out[n_existing:], new_keys)

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"期望表行数为{n}，实际为{rows}"

    found_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, all_keys, found_values, founds, None)

    assert founds.sum().item() == n, f"期望找到所有{n}个键"
    torch.testing.assert_close(found_values[:n_existing].cpu(), existing_values.cpu())
    assert_initialized_values(initializer, found_values[n_existing:], new_keys)


def test_find_or_insert_init_optstate_vec4():
    """测试find_or_insert时optimizer state的vec4路径初始化正确，dim和optstate_dim均能被4整除"""
    torch.npu.set_device(DEVICE_ID)
    dim = 128
    n = 5
    init_optstate_val = 0.5

    init_args = demb.InitializerArgs("debug", 0.0, 1.0, 0.0, 1.0, 0.0)
    table = create_dynamic_table(dim=dim, initializer_args=init_args, optimizer_type=demb.OptimizerType.AdaGrad)

    # 设置初始optimizer state值并验证getter
    table.set_initial_optstate(init_optstate_val)
    assert abs(table.get_initial_optstate() - init_optstate_val) < 1e-6, "get_initial_optstate应返回设置的值"

    keys = torch.randint(0, 500, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, keys, values_out, None, True, False)

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"vec4路径下期望表行数为{n}，实际为{rows}"

    # 使用总维度(dim + optstate_dim)来find，获取embedding和optimizer state的完整行
    total_dim = dim + table.optstate_dim()
    found_values = torch.zeros(n, total_dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, keys, found_values, founds, None)

    assert founds.sum().item() == n, f"vec4路径下期望找到所有{n}个键"

    for i in range(n):
        expected_val = float(keys[i].item() % 100000)
        # 验证embedding部分（前dim列）
        assert torch.allclose(found_values[i, :dim], torch.full((dim,), expected_val, device=f'npu:{DEVICE_ID}')), (
            f"vec4路径下键{keys[i]}的embedding值应与debug初始化器一致"
        )
        # 验证optimizer state部分（后optstate_dim列），值应为set_initial_optstate设置的值
        assert torch.allclose(
            found_values[i, dim:], torch.full((table.optstate_dim(),), init_optstate_val, device=f'npu:{DEVICE_ID}')
        ), f"vec4路径下键{keys[i]}的optimizer state值应为{init_optstate_val}"


@pytest.mark.parametrize("dim", [10, 102401])
def test_find_or_insert_init_optstate_scalar(dim):
    """测试find_or_insert时optimizer state的标量路径初始化正确，dim不能被4整除"""
    torch.npu.set_device(DEVICE_ID)
    n = 5
    init_optstate_val = 0.75

    init_args = demb.InitializerArgs("debug", 0.0, 1.0, 0.0, 1.0, 0.0)
    table = create_dynamic_table(dim=dim, initializer_args=init_args, optimizer_type=demb.OptimizerType.AdaGrad)

    # 设置初始optimizer state值并验证getter
    table.set_initial_optstate(init_optstate_val)
    assert abs(table.get_initial_optstate() - init_optstate_val) < 1e-6, "get_initial_optstate应返回设置的值"

    keys = torch.randint(0, 500, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, keys, values_out, None, True, False)

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"标量路径下期望表行数为{n}，实际为{rows}"

    # 使用总维度(dim + optstate_dim)来find，获取embedding和optimizer state的完整行
    total_dim = dim + table.optstate_dim()
    found_values = torch.zeros(n, total_dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, keys, found_values, founds, None)

    assert founds.sum().item() == n, f"标量路径下期望找到所有{n}个键"

    for i in range(n):
        expected_val = float(keys[i].item() % 100000)
        # 验证embedding部分（前dim列）
        assert torch.allclose(found_values[i, :dim], torch.full((dim,), expected_val, device=f'npu:{DEVICE_ID}')), (
            f"标量路径下键{keys[i]}的embedding值应与debug初始化器一致"
        )
        # 验证optimizer state部分（后optstate_dim列），值应为set_initial_optstate设置的值
        assert torch.allclose(
            found_values[i, dim:], torch.full((table.optstate_dim(),), init_optstate_val, device=f'npu:{DEVICE_ID}')
        ), f"标量路径下键{keys[i]}的optimizer state值应为{init_optstate_val}"


def test_find_or_insert_with_customized_score():
    """测试find_or_insert使用kCustomized策略，插入自定义score，通过find验证score和embedding正确性"""
    torch.npu.set_device(DEVICE_ID)
    dim = 128
    n = 5

    init_args = demb.InitializerArgs("debug", 0.0, 1.0, 0.0, 1.0, 0.0)
    table = create_dynamic_table(dim=dim, evict_strategy=demb.EvictStrategy.kCustomized, initializer_args=init_args)

    custom_score = torch.randint(1, 100, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')[0].item()
    keys = torch.randint(0, 500, (n,), dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    values_out = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    demb.find_or_insert(table, n, keys, values_out, custom_score, True, False)

    rows = demb.dyn_emb_rows(table)
    assert rows == n, f"期望表行数为{n}，实际为{rows}"

    found_values = torch.zeros(n, dim, dtype=torch.float32, device=f'npu:{DEVICE_ID}')
    founds = torch.zeros(n, dtype=torch.bool, device=f'npu:{DEVICE_ID}')
    found_scores = torch.zeros(n, dtype=torch.int64, device=f'npu:{DEVICE_ID}')
    demb.find(table, n, keys, found_values, founds, found_scores)

    assert founds.sum().item() == n, f"期望找到所有{n}个键"

    for i in range(n):
        expected_val = float(keys[i].item() % 100000)
        assert torch.allclose(found_values[i], torch.full((dim,), expected_val, device=f'npu:{DEVICE_ID}')), (
            f"键{keys[i]}的embedding值应与debug初始化器一致"
        )
        assert found_scores[i].item() == custom_score, (
            f"键{keys[i]}的score应为{custom_score}，实际为{found_scores[i].item()}"
        )

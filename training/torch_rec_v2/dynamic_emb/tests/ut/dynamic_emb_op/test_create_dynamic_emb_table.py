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

import pytest
import torch

import dynamic_emb_extensions as demb


class DynamicEmbTableOptions:
    def __init__(self):
        self.key_type = demb.DynamicEmbDataType.Int64
        self.value_type = demb.DynamicEmbDataType.Float32
        self.evict_strategy = demb.EvictStrategy.kLru
        self.dim = 128
        self.init_capacity = 1024
        self.max_capacity = 2048
        self.max_hbm_for_vectors = 0
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


@pytest.mark.parametrize("optimizer_type", [
    demb.OptimizerType.Null,
    demb.OptimizerType.SGD,
    demb.OptimizerType.Adam,
    demb.OptimizerType.AdamW,
    demb.OptimizerType.AdaGrad,
    demb.OptimizerType.RowWiseAdaGrad
])


def test_different_optimizer_types(optimizer_type):
    """测试不同的优化器类型"""
    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        64,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        0,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        optimizer_type
    )
    # 验证表创建成功
    assert table is not None


@pytest.mark.parametrize("safe_check_mode", [
    demb.SafeCheckMode.ERROR,
    demb.SafeCheckMode.WARNING,
    demb.SafeCheckMode.IGNORE
])


def test_different_safe_check_modes(safe_check_mode):
    """测试不同的安全检查模式"""
    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        64,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        0,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        safe_check_mode,
        demb.OptimizerType.Null
    )
    # 验证表创建成功
    assert table is not None


@pytest.mark.parametrize("key_type, value_type, evict_strategy", [
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.Float32, demb.EvictStrategy.kCustomized),
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.Float32, demb.EvictStrategy.kLru),
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.Float32, demb.EvictStrategy.kLfu),
    
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.Float16, demb.EvictStrategy.kCustomized),
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.Float16, demb.EvictStrategy.kLru),
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.Float16, demb.EvictStrategy.kLfu),

    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.BFloat16, demb.EvictStrategy.kCustomized),
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.BFloat16, demb.EvictStrategy.kLru),
    (demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.BFloat16, demb.EvictStrategy.kLfu),

    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.Float32, demb.EvictStrategy.kCustomized),
    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.Float32, demb.EvictStrategy.kLru),
    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.Float32, demb.EvictStrategy.kLfu),

    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.Float16, demb.EvictStrategy.kCustomized),
    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.Float16, demb.EvictStrategy.kLru),
    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.Float16, demb.EvictStrategy.kLfu),

    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.BFloat16, demb.EvictStrategy.kCustomized),
    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.BFloat16, demb.EvictStrategy.kLru),
    (demb.DynamicEmbDataType.UInt64, demb.DynamicEmbDataType.BFloat16, demb.EvictStrategy.kLfu)
])


def test_all_template_combinations(key_type, value_type, evict_strategy):
    """测试所有在mock_hkv_variable.cpp中实例化的模板组合"""
    table = demb.DynamicEmbTable(
        key_type,
        value_type,
        evict_strategy,
        64,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        0,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null
    )
    assert table.get_key_type() == key_type
    assert table.get_evict_strategy() == evict_strategy
    # 1. 覆盖 get_value_type
    assert table.get_value_type() == value_type
    # 2. 覆盖 get_max_capacity 
    assert table.get_max_capacity() == 2048 
    # 3. 覆盖 get_initializer_args
    args = table.get_initializer_args()
    assert args is not None


@pytest.mark.parametrize("init_args", [
    demb.InitializerArgs(),  # 默认参数
    demb.InitializerArgs("normal", 0.0, 0.1, 0.0, 1.0, 0.0),
    demb.InitializerArgs("uniform", 0.0, 1.0, -1.0, 1.0, 0.0),
    demb.InitializerArgs("constant", 0.0, 1.0, 0.0, 1.0, 0.5),
    demb.InitializerArgs("normal", 1.0, 0.5, 0.0, 2.0, 0.0)
])


def test_different_initializer_args(init_args):
    """测试不同的初始化器参数"""
    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        64,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        0,
        False,
        False,
        0,
        1,
        init_args,
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null
    )
    # 验证初始化器参数正确设置
    table_init_args_states = table.get_initializer_args().__getstate__()
    init_args_states = init_args.__getstate__()
    assert table_init_args_states[0] == init_args_states[0]
    assert table_init_args_states[1] == init_args_states[1]
    assert table_init_args_states[2] == init_args_states[2]
    assert table_init_args_states[3] == init_args_states[3]
    assert table_init_args_states[4] == init_args_states[4]
    assert table_init_args_states[5] == init_args_states[5]


@pytest.mark.parametrize("init_capacity,max_capacity", [
    (1024, 2048),
    (512, 1024),
    (2048, 4096)
])


def test_different_capacities(init_capacity, max_capacity):
    """测试不同的容量设置"""
    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        64,
        init_capacity,
        max_capacity,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        0,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null
    )
    # 验证最大容量正确设置（注意：代码中会转换为2的幂）
    # 这里我们只检查是否创建成功
    assert table is not None


@pytest.mark.parametrize("io_by_cpu", [False])
@pytest.mark.parametrize("max_hbm_for_vectors", [1 * 1024 * 1024 * 1024])
def test_io_by_cpu_options(io_by_cpu, max_hbm_for_vectors):
    """测试不同的IO模式"""
    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        64,
        1024,
        2048,
        max_hbm_for_vectors,
        128,
        0.5,
        128,
        1024,
        0,
        io_by_cpu,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null
    )
    assert table is not None


@pytest.mark.parametrize("use_constant_memory", [True, False])

def test_constant_memory_options(use_constant_memory):
    """测试是否使用常量内存"""
    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        64,
        1024,
        2048,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        0,
        False,
        use_constant_memory,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null
    )
    assert table is not None

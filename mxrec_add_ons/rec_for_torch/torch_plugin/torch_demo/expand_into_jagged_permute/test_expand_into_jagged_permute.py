#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import itertools
import sysconfig

import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

# 定义用到的卡以及so库的位置
DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# 定义参数数据类型
PERMUTE_TYPE = [np.int32]
OFFSET_TYPE = [np.int32]
TYPE_LIST = list(itertools.product(PERMUTE_TYPE, OFFSET_TYPE))

PERMUTE_KEY = 'permute'
INPUT_OFFSETS_KEY = 'input_offsets'
OUTPUT_OFFSETS_KEY = 'output_offsets'
OUTPUT_SIZE_KEY = 'output_size'


def get_expand_into_jagged_permute_result(tensors: dict, device: str = 'cpu', is_mxrec: bool = False):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}
    # 根据device类型进行npu转换
    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if is_mxrec:
        result = torch.ops.mxrec.expand_into_jagged_permute(
            tensors[PERMUTE_KEY],
            tensors[INPUT_OFFSETS_KEY],
            tensors[OUTPUT_OFFSETS_KEY],
            tensors[OUTPUT_SIZE_KEY]
        )
    else:
        result = torch.ops.fbgemm.expand_into_jagged_permute(
            tensors[PERMUTE_KEY],
            tensors[INPUT_OFFSETS_KEY],
            tensors[OUTPUT_OFFSETS_KEY],
            tensors[OUTPUT_SIZE_KEY]
        )
    if device and device.startswith('npu'):
        torch_npu.npu.synchronize()
    return result.cpu() if isinstance(result, torch.Tensor) else result


def generate_test_data(num_features, max_batch_size):
    """生成测试数据"""
    # 生成特征级别的permute
    permute = np.random.permutation(num_features).astype(np.int32)

    # 生成输入长度和偏移量（特征级别的长度）
    input_lengths = np.random.randint(1, max_batch_size + 1, size=num_features)
    input_offsets = np.concatenate([[0], np.cumsum(input_lengths)]).astype(np.int32)

    # 生成输出长度和偏移量（特征级别的permuted长度）
    permuted_lengths = input_lengths[permute]
    output_offsets = np.concatenate([[0], np.cumsum(permuted_lengths)]).astype(np.int32)

    # 计算输出大小
    output_size = np.sum(permuted_lengths)
    return {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: output_offsets,
        OUTPUT_SIZE_KEY: output_size
    }


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_basic(types, is_mxrec):
    """测试基本功能"""
    ptype, otype = types
    # 简单测试用例
    permute = np.array([2, 0, 1], dtype=ptype)  # 3个特征
    input_offsets = np.array([0, 3, 7, 9], dtype=otype)  # 包含起始0和结束值
    # permuted后长度: [2, 3, 4] (对应特征2, 0, 1)
    output_offsets = np.array([0, 2, 5, 9], dtype=otype)  # 包含起始0和所有累加值
    output_size = 9  # 2 + 3 + 4
    params = {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: output_offsets,
        OUTPUT_SIZE_KEY: output_size
    }

    golden = get_expand_into_jagged_permute_result(params)
    result = get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)
    # 验证结果类型和形状
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == output_size
    assert result.dtype == torch.int32
    assert torch.allclose(result, golden, atol=1e-4)


@pytest.mark.parametrize("num_features", [10, 50, 100])
@pytest.mark.parametrize("max_batch_size", [5, 20, 50])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_random(num_features, max_batch_size, is_mxrec):
    """测试随机生成的测试用例"""
    test_data = generate_test_data(num_features, max_batch_size)
    params = {
        PERMUTE_KEY: test_data[PERMUTE_KEY],
        INPUT_OFFSETS_KEY: test_data[INPUT_OFFSETS_KEY],
        OUTPUT_OFFSETS_KEY: test_data[OUTPUT_OFFSETS_KEY],
        OUTPUT_SIZE_KEY: test_data[OUTPUT_SIZE_KEY]
    }

    golden = get_expand_into_jagged_permute_result(params)
    result = get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)
    # 验证结果类型和形状
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == test_data[OUTPUT_SIZE_KEY]
    # 验证结果值的一致性
    assert torch.allclose(result, golden, atol=1e-4)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_large_input(is_mxrec):
    """测试非常大的输入情况"""
    num_features = 10000
    max_batch_size = 100
    test_data = generate_test_data(num_features, max_batch_size)
    params = {
        PERMUTE_KEY: test_data[PERMUTE_KEY],
        INPUT_OFFSETS_KEY: test_data[INPUT_OFFSETS_KEY],
        OUTPUT_OFFSETS_KEY: test_data[OUTPUT_OFFSETS_KEY],
        OUTPUT_SIZE_KEY: test_data[OUTPUT_SIZE_KEY]
    }

    golden = get_expand_into_jagged_permute_result(params)
    result = get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)
    assert torch.allclose(result, golden, atol=1e-4)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_empty_input(is_mxrec):
    """测试空输入的情况"""
    # 对于空输入，permute应该为空，input_offsets应该为[0]
    params = {
        PERMUTE_KEY: np.array([], dtype=np.int32),
        INPUT_OFFSETS_KEY: np.array([0], dtype=np.int32),  # 空输入的偏移量应该是[0]
        OUTPUT_OFFSETS_KEY: np.array([0], dtype=np.int32),  # 空输出的偏移量应该是[0]
        OUTPUT_SIZE_KEY: 0
    }

    with pytest.raises(RuntimeError):
        get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_invalid_output_size(is_mxrec):
    """测试输出大小不匹配的情况"""
    permute = np.array([0, 1], dtype=np.int32)
    input_offsets = np.array([0, 2, 5], dtype=np.int32)
    # permuted后长度: [2, 3] (恒等permute)
    output_offsets = np.array([0, 2, 5], dtype=np.int32)
    output_size = 4  # 实际应该是5 (2+3)
    params = {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: output_offsets,
        OUTPUT_SIZE_KEY: output_size
    }

    with pytest.raises(RuntimeError):
        get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_mismatched_offsets(is_mxrec):
    """测试偏移量不匹配的情况"""
    permute = np.array([0, 1], dtype=np.int32)
    input_offsets = np.array([0, 2, 5], dtype=np.int32)    # 特征长度: [2, 3]
    output_offsets = np.array([0, 1, 3], dtype=np.int32)   # 错误的偏移量
    output_size = 5
    params = {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: output_offsets,
        OUTPUT_SIZE_KEY: output_size
    }

    with pytest.raises(RuntimeError):
        get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_non_monotonic_offsets(is_mxrec):
    """测试非单调递增偏移量"""
    permute = np.array([0, 1], dtype=np.int32)
    input_offsets = np.array([0, 2, 5], dtype=np.int32)
    output_offsets = np.array([0, 3, 2], dtype=np.int32)  # 非单调递增
    output_size = 5
    params = {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: output_offsets,
        OUTPUT_SIZE_KEY: output_size
    }

    with pytest.raises(RuntimeError):
        get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_size_mismatch(is_mxrec):
    """测试permute和input_offsets大小不匹配的情况"""
    permute = np.array([0, 1, 2], dtype=np.int32)  # 3个元素
    input_offsets = np.array([0, 2, 5], dtype=np.int32)  # 3个元素，但需要4个元素才能匹配
    params = {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: np.array([0, 2, 5], dtype=np.int32),
        OUTPUT_SIZE_KEY: 5
    }

    with pytest.raises(RuntimeError):
        get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_2d_input(is_mxrec):
    """测试输入为2D的情况"""
    params = {
        PERMUTE_KEY: np.array([[0, 1], [1, 0]], dtype=np.int32),  # 2D permute
        INPUT_OFFSETS_KEY: np.array([0, 2, 5], dtype=np.int32),
        OUTPUT_OFFSETS_KEY: np.array([0, 2, 5], dtype=np.int32),
        OUTPUT_SIZE_KEY: 5
    }

    with pytest.raises(RuntimeError):
        get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_dtype_mismatch(is_mxrec):
    """测试数据类型不匹配的情况"""
    params = {
        PERMUTE_KEY: np.array([0, 1], dtype=np.int32),
        INPUT_OFFSETS_KEY: np.array([0, 2, 5], dtype=np.int64),  # 数据类型不匹配
        OUTPUT_OFFSETS_KEY: np.array([0, 2, 5], dtype=np.int32),
        OUTPUT_SIZE_KEY: 5
    }

    with pytest.raises(RuntimeError):
        get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_identity_permute(is_mxrec):
    """测试恒等permute的情况"""
    num_features = 5
    permute = np.arange(num_features, dtype=np.int32)  # 恒等permute
    input_lengths = np.array([2, 3, 4, 1, 2], dtype=np.int32)
    input_offsets = np.concatenate([[0], np.cumsum(input_lengths)]).astype(np.int32)
    # 恒等permute，输出偏移量与输入相同
    output_offsets = input_offsets.copy()
    output_size = np.sum(input_lengths)
    params = {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: output_offsets,
        OUTPUT_SIZE_KEY: output_size
    }

    golden = get_expand_into_jagged_permute_result(params)
    result = get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)
    assert torch.allclose(result, golden, atol=1e-4)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_expand_into_jagged_permute_single_feature(is_mxrec):
    """测试单特征的情况"""
    permute = np.array([0], dtype=np.int32)
    input_offsets = np.array([0, 5], dtype=np.int32)  # 单个特征有5个batch
    output_offsets = np.array([0, 5], dtype=np.int32)
    output_size = 5
    params = {
        PERMUTE_KEY: permute,
        INPUT_OFFSETS_KEY: input_offsets,
        OUTPUT_OFFSETS_KEY: output_offsets,
        OUTPUT_SIZE_KEY: output_size
    }

    golden = get_expand_into_jagged_permute_result(params)
    result = get_expand_into_jagged_permute_result(params, DEVICE, is_mxrec)
    assert torch.allclose(result, golden, atol=1e-4)
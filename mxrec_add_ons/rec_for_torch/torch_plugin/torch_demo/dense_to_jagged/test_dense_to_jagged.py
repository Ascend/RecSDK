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
import logging
import sysconfig

import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

DEVICE = "npu:0"
logging.getLogger().setLevel(logging.INFO)
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DENSE_DIM0 = [128, 40] # 测试不同batch大小
DENSE_DIM1 = [210] # 固定特征维度1
DENSE_DIM2 = [1, 8] # 固定特征维度2
DIM_LIST = list(itertools.product(DENSE_DIM0, DENSE_DIM1, DENSE_DIM2))

DENSE_DATATYPE = [torch.float32, torch.int64, torch.bfloat16, torch.float16, torch.int32]  # 增加BF16、FP16和INT32支持
OFFSET_DATATYPE = [torch.int32, torch.int64]  # 偏移量数据类型
TYPE_LIST = list(itertools.product(DENSE_DATATYPE, OFFSET_DATATYPE))

# 边界测试用例
EDGE_CASE_DIMS = [
    (1, 10, 1),       # 最小batch和特征维度
    (10, 1, 16),      # 最小序列长度
    (1, 1, 1),        # 所有维度都最小
    (256, 500, 32),   # 较大的batch和特征维度
]


def get_result(device, denses, offsets, types, use_output_size):
    """获取指定设备上的算子执行结果"""
    dense_datatype, offset_datatype = types
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(device)
    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(device)

    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    # 获取输出大小（最后一个偏移量即总元素数）
    output_size = None
    if use_output_size:
        output_size = jagged_id_offset[-1]

    # 执行核心操作：稠密张量→不规则张量
    jagged_embedding = torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]
    return jagged_embedding.cpu()


def compare_results(golden_result, npu_result, tolerance=1e-4):
    """比较CPU和NPU的结果"""
    # 检查两个结果的形状是否相同
    assert golden_result.shape == npu_result.shape, \
        f"Shape mismatch: golden {golden_result.shape} vs npu {npu_result.shape}"

    # 对所有张量进行数值比较（包括空张量）
    if golden_result.numel() > 0:
        result_forward = torch.abs(golden_result - npu_result) < tolerance
        assert result_forward.all().item(), "Result values do not match within tolerance"
    else:
        # 空张量直接通过检查（形状已验证）
        assert torch.equal(golden_result, npu_result), "Empty tensors should be equal"


def generate_test_data(dense_dim0, dense_dim1, dense_dim2, dense_dtype):
    """生成测试数据"""
    if dense_dtype in [torch.bfloat16, torch.float16]:
        denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    elif dense_dtype == torch.int32:
        denses = np.random.randint(0, 1000, (dense_dim0, dense_dim1, dense_dim2)).astype(np.int32)
    else:
        denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)  # 确保 len(offsets) == dense.shape[0]
    return denses, offsets


def get_tolerance(dense_dtype):
    """根据数据类型获取相应的容差值"""
    if dense_dtype == torch.float16:
        return 1e-3  # float16: 双千分之一
    elif dense_dtype == torch.bfloat16:
        return 5e-3  # bfloat16: 双千分之五
    elif dense_dtype in [torch.int32, torch.float32, torch.int64]:
        return 1e-4  # int32、float32和int64: 双万分之一
    else:
        raise ValueError(f"Unsupported data type: {dense_dtype}")


def run_test(denses, offsets, types, use_output_size=False):
    """运行测试的核心逻辑"""
    # 获取结果
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, use_output_size)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, use_output_size)

    # 根据数据类型获取相应的容差值
    tolerance = get_tolerance(types[0])
    compare_results(golden_result, npu_result, tolerance)


@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("use_output_size", [True, False])  # 测试是否传入 output_size
def test_dense_to_jagged(dims, types, use_output_size):
    """基本功能测试"""
    dense_dim0, dense_dim1, dense_dim2 = dims
    # 1. 生成随机输入数据
    dense_datatype, _ = types
    denses, offsets = generate_test_data(dense_dim0, dense_dim1, dense_dim2, dense_datatype)

    run_test(denses, offsets, types, use_output_size)


@pytest.mark.parametrize("dims", EDGE_CASE_DIMS)
@pytest.mark.parametrize("dense_dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_edge_cases(dims, dense_dtype, offset_dtype):
    """边界情况测试：测试各种极端维度组合"""
    dense_dim0, dense_dim1, dense_dim2 = dims

    # 生成测试数据
    denses, offsets = generate_test_data(dense_dim0, dense_dim1, dense_dim2, dense_dtype)
    types = (dense_dtype, offset_dtype)

    run_test(denses, offsets, types)


def test_dense_to_jagged_empty_offsets():
    """测试空偏移量的情况"""
    # 创建空的偏移量 - 确保 len(offsets) == dense.shape[0]
    denses = np.random.randn(0, 10, 8).astype(np.float32)  # 0个batch
    offsets = np.array([])  # 空偏移量
    types = (torch.float32, torch.int64)

    assert len(denses) == 0 and len(offsets) == 0, "Expected empty tensors"
    run_test(denses, offsets, types)


def test_dense_to_jagged_large_offsets():
    """测试大偏移量的情况"""
    # 创建大的偏移量
    dense_dim0 = 5
    denses = np.random.randn(dense_dim0, 100, 16).astype(np.float32)
    # 创建较大的偏移量，但不超过dense_dim1
    offsets = np.random.randint(0, 100, dense_dim0)  # 确保 len(offsets) == dense.shape[0]
    types = (torch.float32, torch.int64)

    run_test(denses, offsets, types)


@pytest.mark.parametrize("dense_dtype", [torch.bfloat16, torch.float16])
def test_bf16_fp16_precision(dense_dtype):
    """专门测试BF16和FP16精度"""
    # 创建特定测试数据
    dense_dim0 = 10
    denses = np.random.randn(dense_dim0, 50, 8).astype(np.float32)
    offsets = np.random.randint(0, 50, dense_dim0)  # 确保 len(offsets) == dense.shape[0]
    types = (dense_dtype, torch.int64)

    run_test(denses, offsets, types)


@pytest.mark.parametrize("dense_dtype", [torch.int32])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_int32_dense_precision(dense_dtype, offset_dtype):
    """专门测试int32 dense精度"""
    # 创建特定测试数据
    dense_dim0 = 10
    denses = np.random.randint(0, 1000, (dense_dim0, 50, 8)).astype(np.int32)
    offsets = np.random.randint(0, 50, dense_dim0)  # 确保 len(offsets) == dense.shape[0]
    types = (dense_dtype, offset_dtype)

    run_test(denses, offsets, types)


def test_dense_to_jagged_forward_npu_fbgemm_call():
    """测试通过fbgemm.dense_to_jagged_forward调用dense_to_jagged_forward_npu函数"""
    # 准备测试数据
    dense_dim0, dense_dim1, dense_dim2 = 10, 20, 8
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)
    
    # 将数据移到NPU设备
    dense_torch = torch.from_numpy(denses).to(torch.float32).to(DEVICE)
    offsets_torch = torch.from_numpy(offsets).to(torch.int64).to(DEVICE)
    
    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)
    
    # 获取输出大小
    output_size = jagged_id_offset[-1]
    
    # 通过fbgemm调用dense_to_jagged_forward
    jagged_embedding = torch.ops.fbgemm.dense_to_jagged_forward(
        dense_torch, [jagged_id_offset], output_size)
    
    # 验证结果
    assert jagged_embedding is not None
    assert jagged_embedding.shape[0] == output_size
    assert jagged_embedding.shape[1] == dense_dim2


def test_dense_to_jagged_forward_npu_mxrec_call():
    """测试通过mxrec.dense_to_jagged_forward调用dense_to_jagged_forward_npu函数"""
    # 准备测试数据
    dense_dim0, dense_dim1, dense_dim2 = 10, 20, 8
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)
    
    # 将数据移到NPU设备
    dense_torch = torch.from_numpy(denses).to(torch.float32).to(DEVICE)
    offsets_torch = torch.from_numpy(offsets).to(torch.int64).to(DEVICE)
    
    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)
    
    # 获取输出大小
    output_size = jagged_id_offset[-1]
    
    # 通过mxrec调用dense_to_jagged_forward
    jagged_embedding = torch.ops.mxrec.dense_to_jagged_forward(
        dense_torch, [jagged_id_offset], output_size)
    
    # 验证结果
    assert jagged_embedding is not None
    assert jagged_embedding.shape[0] == output_size
    assert jagged_embedding.shape[1] == dense_dim2


def test_dense_to_jagged_forward_npu_int32_dense():
    """测试int32类型dense张量的处理"""
    # 准备测试数据
    dense_dim0, dense_dim1, dense_dim2 = 10, 20, 8
    denses = np.random.randint(0, 1000, (dense_dim0, dense_dim1, dense_dim2)).astype(np.int32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)
    
    # 将数据移到NPU设备
    dense_torch = torch.from_numpy(denses).to(torch.int32).to(DEVICE)
    offsets_torch = torch.from_numpy(offsets).to(torch.int64).to(DEVICE)
    
    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)
    
    # 获取输出大小
    output_size = jagged_id_offset[-1]
    
    # 通过fbgemm调用dense_to_jagged_forward处理int32类型
    jagged_embedding = torch.ops.fbgemm.dense_to_jagged_forward(
        dense_torch, [jagged_id_offset], output_size)
    
    # 验证结果
    assert jagged_embedding is not None
    assert jagged_embedding.shape[0] == output_size
    assert jagged_embedding.shape[1] == dense_dim2
    assert jagged_embedding.dtype == torch.int32


def test_dense_to_jagged_npu_fbgemm_call():
    """测试通过fbgemm.dense_to_jagged调用dense_to_jagged_npu函数"""
    # 准备测试数据
    dense_dim0, dense_dim1, dense_dim2 = 10, 20, 8
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)
    
    # 将数据移到NPU设备
    dense_torch = torch.from_numpy(denses).to(torch.float32).to(DEVICE)
    offsets_torch = torch.from_numpy(offsets).to(torch.int64).to(DEVICE)
    
    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)
    
    # 获取输出大小
    output_size = jagged_id_offset[-1]
    
    # 通过fbgemm调用dense_to_jagged
    jagged_embedding, offset_list = torch.ops.fbgemm.dense_to_jagged(
        dense_torch, [jagged_id_offset], output_size)
    
    # 验证结果
    assert jagged_embedding is not None
    assert len(offset_list) == 1
    assert jagged_embedding.shape[0] == output_size
    assert jagged_embedding.shape[1] == dense_dim2
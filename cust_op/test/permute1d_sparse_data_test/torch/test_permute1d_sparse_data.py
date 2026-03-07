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
import itertools
import random
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
LENGTHS_TYPE = [np.int64, np.int32]
VALUES_TYPE = [np.int64, np.int32, np.float32, np.float16]
WEIGHTS_TYPE = [None, np.float32]                    # weights为可选参数
TYPE_LIST = list(itertools.product(PERMUTE_TYPE, LENGTHS_TYPE, VALUES_TYPE, WEIGHTS_TYPE))
INT64_PERMUTE_TYPE = [np.int64]
FP16_WEIGHTS_TYPE = [np.float16]
TYPE_LIST_1 = list(itertools.product(INT64_PERMUTE_TYPE, LENGTHS_TYPE, VALUES_TYPE, FP16_WEIGHTS_TYPE))
# 定义参数shape
# permute shape为[BASE_T]
# lengths shape为[1 ~ (2T - 1)]
# extra_t用于测试permute和lengths不等长的情况，lengths[BASE_T + extra_T]
BASE_T = np.random.randint(2, 500, 5)       # 随机生成5个介于2到500之间的整数，代表稀疏数据的原始维度
EXTRA_T = [1, 0, -1]
SHAPE_LIST = list(itertools.product(BASE_T, EXTRA_T))


def get_result(tensors: dict, device: str = 'cpu', is_mxrec: bool = False):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    # 根据device类型进行npu转换
    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if is_mxrec:
        results = torch.ops.mxrec.permute_1D_sparse_data(**tensors)
    else:
        results = torch.ops.fbgemm.permute_1D_sparse_data(**tensors)

    if device and device.startswith('npu'):
        torch_npu.npu.synchronize()
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


# test_type 0 测试正常情况下的permute1d_sparse_data算子功能 
# test_type 1 测试permutedim小,values长度大场景
# test_type 2 测试permutedim大,values长度小场景
def init_tensor(types, shapes, enable_permuted_sum, test_type=0):
    ptype, ltype, vtype, wtype = types
    t, extra_t = shapes
    if test_type == 0:
        extra_t = random.randint(1, t - 1) * extra_t
        permute = np.random.choice(t + extra_t, t).astype(dtype=np.int32)
        lengths = np.random.randint(200, 2000, size=t + extra_t, dtype=ltype)
    elif test_type == 1:
        permute = np.random.choice(t, t).astype(dtype=np.int32)
        lengths = np.random.randint(30000, 500000, size=t, dtype=np.int64)
    else:
        permute = np.random.choice(t, t).astype(dtype=ptype)
        lengths = np.random.randint(10, 15000, size=t, dtype=np.int64)
        
    total_length = int(lengths.sum())
    values = np.arange(0, total_length, dtype=vtype)
    weights = np.arange(0, total_length, dtype=wtype) if wtype else None
    permuted_lengths_sum = lengths[permute].sum() if enable_permuted_sum else None

    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': weights,
        'permuted_lengths_sum': permuted_lengths_sum
    }
    return params


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute1d_sparse_data(types, shapes, enable_permuted_sum, is_mxrec):
    """
    测试正常情况下的permute1d_sparse_data算子功能
    Params:
        permute: (T) dtype=int32
        lengths: (T + T') dtype=ltype
                L = lengths[:T].sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
        permuted_lengths_sum: int
    """
    params = init_tensor(types, shapes, enable_permuted_sum, 0)

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_small_permuted_dim_large_values_length(is_mxrec):
    """
    测试permutedim小,values长度大场景
    """
    t = 8
    params = init_tensor((np.int32, np.int64, np.int32, np.float32), (t, t), True, 1)

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_large_permuted_dim_small_values_length(is_mxrec):
    """
    测试permutedim大,values长度小场景
    """
    t = 872
    params = init_tensor((np.int32, np.int64, np.int32, np.float32), (t, t), True, 2)

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_empty_input(is_mxrec):
    """
    测试空输入的情况
    """
    params = {
        'permute': np.array([], dtype=np.int32),
        'lengths': np.array([], dtype=np.int32),
        'values': np.array([], dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': None
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_invalid_weights_length(is_mxrec):
    """
    测试weights长度与values不匹配的情况
    """
    t = 5
    params = {
        'permute': np.arange(t, dtype=np.int32),
        'lengths': np.ones(t, dtype=np.int32),
        'values': np.arange(t, dtype=np.int32),
        'weights': np.arange(t + 1, dtype=np.float32),  # 长度不匹配
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        result = get_result(params, DEVICE, is_mxrec)
        assert result is not None


@pytest.mark.parametrize("types", TYPE_LIST_1)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute1d_sparse_data_type_list_1(types, shapes, enable_permuted_sum, is_mxrec):
    params = init_tensor(types, shapes, enable_permuted_sum)
    weights_None = None
    params_golden = {
        'permute': params['permute'].astype(dtype=np.int32),
        'lengths': params['lengths'],
        'values': params['values'],
        'weights': weights_None,
        'permuted_lengths_sum': params['permuted_lengths_sum']
    }

    golden = get_result(params_golden)
    params_golden['values'] = params['weights']
    golden_weights = get_result(params_golden)
    result = get_result(params, DEVICE, is_mxrec)
    assert torch.allclose(golden[0], result[0], atol=1e-5)
    assert torch.allclose(golden[1], result[1], atol=1e-5)
    assert torch.allclose(golden_weights[1], result[2], atol=1e-5)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_2d_input(is_mxrec):
    """
    测试输入为2D的情况
    """
    t = 5
    params = {
        'permute': np.arange(t, dtype=np.int32).reshape(1, -1),  # 2D permute
        'lengths': np.ones(t, dtype=np.int32),
        'values': np.arange(t, dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        result = get_result(params, DEVICE, is_mxrec)
        assert result is not None

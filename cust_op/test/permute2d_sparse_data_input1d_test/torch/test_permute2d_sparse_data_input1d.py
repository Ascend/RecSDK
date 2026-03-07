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

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# cpu不支持np.int64，只能支持np.int32
PTYPE = [np.int32]
LTYPE = [np.int64, np.int32]
VTYPE = [np.int64, np.int32, np.float32, np.float16]
WTYPE = [None, np.float32]
TYPE_LIST = list(itertools.product(PTYPE, LTYPE, VTYPE, WTYPE))
INT64_PTYPE = [np.int64]
FP16_WTYPE = [np.float16]
TYPE_LIST_1 = list(itertools.product(INT64_PTYPE, LTYPE, VTYPE, FP16_WTYPE))

# lengths shape为[1 ~ (2T - 1) * B]
# extra_t用于测试permute和lengths不等长的情况，lengths[(T + extra_T) * B]
T = np.random.randint(2, 500, 5)
EXTRA_T = [1, 0, -1]
B = [128, 1024, 2048, 20480]
B_FP16 = [128, 20480]
SHAPE_LIST = list(itertools.product(T, EXTRA_T, B))
SHAPE_LIST_1 = list(itertools.product(T, EXTRA_T, B_FP16))


def get_result(tensors: dict, device: str = 'cpu', is_mxrec: bool = False):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if is_mxrec:
        results = torch.ops.mxrec.permute_2D_sparse_data_input1D(**tensors)
    else:
        results = torch.ops.fbgemm.permute_2D_sparse_data_input1D(**tensors)
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


# test_type 0 测试正常情况下的permute1d_sparse_data算子功能 
# test_type 1 测试permutedim小,values长度大场景
# test_type 2 测试permutedim大,values长度小场景
def init_tensor(types, shapes, enable_permuted_sum, test_type=0):
    ptype, ltype, vtype, wtype = types
    t, extra_t, b = shapes
    if test_type == 0:
        extra_t = random.randint(1, t - 1) * extra_t if extra_t > 0 else extra_t  # Consistent randomization
        permute = np.random.choice(t + extra_t, t).astype(dtype=ptype)
        lengths_2d = np.random.randint(1, 10, size=(t + extra_t, b), dtype=ltype)
    elif test_type == 1:
        permute = np.random.choice(t, t).astype(dtype=ptype)
        lengths_2d = np.random.randint(10000, 30000, size=(t, b), dtype=ltype)
    else:
        permute = np.random.choice(t, t).astype(dtype=ptype)
        lengths_2d = np.random.randint(10, 800, size=(t, b), dtype=ltype)
    
    lengths = lengths_2d.flatten()
    total_length = int(lengths.sum())
    values = (np.arange(0, total_length, dtype=vtype) 
          if vtype != np.float16 
          else np.random.rand(total_length).astype(vtype))
    weights = np.arange(0, total_length, dtype=wtype) if wtype else None
    permuted_lengths_sum = lengths_2d[permute].sum() if enable_permuted_sum else None

    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'stride': b,
        'weights': weights,
        'permuted_lengths_sum': permuted_lengths_sum
    }
    return params


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute2d_sparse_data_input1d(types, shapes, enable_permuted_sum, is_mxrec):
    """
    Params:
        permute: (T) dtype=int32
        lengths: (T * B) dtype=ltype (1D flattened tensor)
                 L = lengths.sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
        stride: int64_t = B (batch size, used to reshape 1D lengths to 2D [T, B] for internal 2D permutation;
                must divide lengths.size(0) evenly, e.g., lengths.size(0) % stride == 0)
    """
    params = init_tensor(types, shapes, enable_permuted_sum, 0) 

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST)
def test_small_permuted_dim_large_values_length(types):
    """
        测试permutedim小,values长度大场景
    """
    t = 32
    b = 128
    params = init_tensor(types, (t, t, b), True, 1)

    golden = get_result(params)
    result = get_result(params, DEVICE)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST)
def test_large_permuted_dim_small_values_length(types):
    """
        测试permutedim大,values长度小场景
    """
    t = 872
    b = 32
    params = init_tensor(types, (t, t, b), True, 2)

    golden = get_result(params)
    result = get_result(params, DEVICE)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST_1)
@pytest.mark.parametrize("shapes", SHAPE_LIST_1)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
def test_permute2d_sparse_data_input1d_type_list_1(types, shapes, enable_permuted_sum):
    params = init_tensor(types, shapes, enable_permuted_sum, 0) 
    weights_None = None
    t, extra_t, b = shapes
    params_golden = {
        'permute': params['permute'].astype(dtype=np.int32),
        'lengths': params['lengths'],
        'values': params['values'],
        'stride': b,
        'weights': weights_None,
        'permuted_lengths_sum': params['permuted_lengths_sum']
    }

    golden = get_result(params_golden)
    # permute2d_sparse_data cpu不支持weights为fp16,weights golden 由permute2d_sparse_data cpu计算value另外计算
    params_golden['values'] = params['weights']
    golden_weights = get_result(params_golden)
    result = get_result(params, DEVICE)
    assert torch.allclose(golden[0], result[0], atol=1e-5)
    assert torch.allclose(golden[1], result[1], atol=1e-5)
    assert torch.allclose(golden_weights[1], result[2], atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST)
def test_invalid_stride(types):
    """
    Test that invalid stride (not dividing lengths.size(0)) raises RuntimeError.
    """
    ptype, ltype, vtype, wtype = types
    t = 5
    b = 4  # Valid stride
    invalid_stride = b + 1  # Will not divide evenly

    permute = np.random.choice(t, t).astype(dtype=np.int32)
    lengths_2d = np.random.randint(1, 10, size=(t, b), dtype=ltype)
    lengths = lengths_2d.flatten()[:-1]  # Make length not divisible by invalid_stride
    total_length = int(lengths.sum())
    values = np.arange(0, total_length, dtype=vtype)
    weights = np.arange(0, total_length, dtype=wtype) if wtype else None
    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'stride': invalid_stride,
        'weights': weights,
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):  # Expect view size mismatch
        result = get_result(params, DEVICE)
        assert result is not None

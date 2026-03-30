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

PTYPE = [np.int32]
LTYPE = [np.int64, np.int32]
VTYPE = [
    torch.int64,
    torch.int32,
    torch.float32,
    torch.float16,
    torch.bfloat16,
]
WTYPE = [
    None,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int32,
    torch.int64,
]
TYPE_LIST = list(itertools.product(PTYPE, LTYPE, VTYPE, WTYPE))

# lengths shape为[1 ~ (2T - 1), B]
# extra_t用于测试permute和lengths不等长的情况，lengths[T + extra_T, B]
T = np.random.randint(2, 500, 5)
EXTRA_T = [1, 0, -1]
B = [128, 1024, 2048, 20480]
SHAPE_LIST = list(itertools.product(T, EXTRA_T, B))


# Python 版 permute_2D_sparse_data
def permute_2d_sparse_data_reference(params):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in params.items()}
    permute, lengths, values = tensors["permute"], tensors["lengths"], tensors["values"]
    weights = tensors.get("weights")

    permuteConti = permute.contiguous()
    lengthsConti = lengths.contiguous()
    valuesConti = values.contiguous()
    if lengthsConti.size(0) == 0 or lengthsConti.size(1) == 0 or permuteConti.numel() == 0:
        return (
            lengthsConti.clone(),
            valuesConti.clone(),
            None if weights is None else weights.clone(),
        )

    w = None
    if weights is not None:
        w = weights.contiguous()
        if w.dim() not in (1, 2):
            raise ValueError(f"weights must be 1D [L] or 2D [L, D], but got {w.dim()}")
        if w.size(0) != valuesConti.size(0):
            raise ValueError(f"weights dim0 {w.size(0)} not match values dim0 {valuesConti.size(0)}")

    rowSum = lengthsConti.sum(dim=1, dtype=torch.int64)
    off = torch.zeros(rowSum.numel() + 1, dtype=torch.int64)
    off[1:] = rowSum.cumsum(dim=0)

    rows = permuteConti.tolist()
    outV = torch.cat([valuesConti[off[r] : off[r + 1]] for r in rows])
    if w is None:
        outW = None
    else:
        # 1D / 2D 均在 dim=0（L）上切段再拼接，输出为 [Lout] 或 [Lout, D]
        outW = torch.cat([w[off[r] : off[r + 1]] for r in rows], dim=0)

    return lengthsConti[permuteConti], outV, outW


def _make_values(total_length: int, vdtype: torch.dtype) -> torch.Tensor:
    if vdtype in (torch.float16, torch.bfloat16, torch.float32):
        if total_length == 0:
            return torch.zeros(0, dtype=vdtype)
        return torch.rand(total_length, dtype=vdtype)
    return torch.arange(total_length, dtype=vdtype)


def _make_weights(total_length: int, wdtype: torch.dtype, ncols: int) -> torch.Tensor:
    if total_length == 0:
        if ncols == 1:
            return torch.zeros(0, dtype=wdtype)
        return torch.zeros(0, ncols, dtype=wdtype)
    if ncols == 1:
        if wdtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return torch.rand(total_length, dtype=wdtype)
        return torch.arange(total_length, dtype=wdtype)
    if wdtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.stack(
            [torch.rand(total_length, dtype=wdtype) + float(c) for c in range(ncols)],
            dim=1,
        )
    flat = torch.arange(total_length * ncols, dtype=wdtype)
    return flat.reshape(total_length, ncols)


def get_result(tensors: dict, device: str = 'cpu', is_mxrec: bool = False, d2: bool = True):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if is_mxrec:
        if d2:
            results = torch.ops.mxrec.permute_2D_sparse_data(**tensors)
        else:
            results = torch.ops.mxrec.permute_sparse_data(**tensors)
    else:
        if d2:
            results = torch.ops.fbgemm.permute_2D_sparse_data(**tensors)
        else:
            results = torch.ops.fbgemm.permute_sparse_data(**tensors)
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


# test_type 0 测试正常情况下的permute1d_sparse_data算子功能 
# test_type 1 测试permutedim小,values长度大场景
# test_type 2 测试permutedim大,values长度小场景
def init_tensor(types, shapes, enable_permuted_sum, test_type=0, weight_ncols=1):
    ptype, ltype, vdtype, wdtype = types
    t, extra_t, b = shapes
    if test_type == 0:
        extra_t = random.randint(1, t - 1) * extra_t
        permute = np.random.choice(t + extra_t, t).astype(dtype=ptype)
        lengths = np.random.randint(1, 10, size=(t + extra_t, b), dtype=ltype)
    elif test_type == 1:
        permute = np.random.choice(t, t).astype(dtype=ptype)
        lengths = np.random.randint(10000, 30000, size=(t, b), dtype=ltype)
    else:
        permute = np.random.choice(t, t).astype(dtype=ptype)
        lengths = np.random.randint(10, 800, size=(t, b), dtype=ltype)

    total_length = int(lengths.sum())
    values = _make_values(total_length, vdtype)
    weights = _make_weights(total_length, wdtype, weight_ncols) if wdtype else None
    
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
@pytest.mark.parametrize("d2", [True, False])
def test_permute2d_sparse_data(types, shapes, enable_permuted_sum, is_mxrec, d2):
    """
    Params:
        permute: (T) dtype=int32
        lenghts: (T + T', B) dtype=int32/int64
                 L = lengths[:T].sum()
        values: (L) dtype=int32/int64/fp32/fp16(half)/bf16
        weights: (L) dtype=None, or fp32/fp16/bf16/double/int32/int64
    """
    params = init_tensor(types, shapes, enable_permuted_sum)

    golden = list(permute_2d_sparse_data_reference(params))
    result = get_result(params, DEVICE, is_mxrec, d2=d2)

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

    golden = list(permute_2d_sparse_data_reference(params))
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

    golden = list(permute_2d_sparse_data_reference(params))
    result = get_result(params, DEVICE)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


# 为避免测试用例过多，减少了T, B的取值，只保留了2个T, B的取值
WEIGHT_NUM_COLS = [2, 4, 8, 10]
T = np.random.randint(2, 500, 2)
EXTRA_T = [1, 0, -1]
B = [128, 1024]
SHAPE_LIST = list(itertools.product(T, EXTRA_T, B))

@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True])
@pytest.mark.parametrize("weight_ncols", WEIGHT_NUM_COLS)
def test_multicol_weight(types, shapes, enable_permuted_sum, weight_ncols):
    """
    测试weights为2D的情况, dense_dim() > 1, D > 1。
    """
    params = init_tensor(types, shapes, enable_permuted_sum, weight_ncols=weight_ncols)
    golden = list(permute_2d_sparse_data_reference(params))
    result = get_result(params, DEVICE)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_empty_input(is_mxrec):
    """
    测试空输入的情况
    """
    params = {
        'permute': np.array([], dtype=np.int32),
        'lengths': np.empty((0, 0), dtype=np.int32),
        'values': np.array([], dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': None
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


@pytest.mark.parametrize("is_mxrec", [True, False])
def test_lengths_all_zero_permuted_lengths_sum_zero(is_mxrec):
    """
    测试lengths全为0,permuted_lengths_sum为0的情况
    """
    t = 10
    b = 4

    permute = np.arange(0, t).astype(np.int32)
    lengths = np.zeros((t, b), dtype=np.int32)
    values = np.array([], dtype=np.int32)
    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': None,
        'permuted_lengths_sum': 0
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)

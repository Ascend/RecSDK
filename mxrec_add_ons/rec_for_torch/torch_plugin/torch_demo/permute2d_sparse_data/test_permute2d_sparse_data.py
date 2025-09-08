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
import random
import sysconfig

import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

DEVICE = "npu:7"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

PTYPE = [np.int32]
LTYPE = [np.int64, np.int32]
VTYPE = [np.int64, np.int32, np.float32]
WTYPE = [None, np.float32]
TYPE_LIST = list(itertools.product(PTYPE, LTYPE, VTYPE, WTYPE))

# lengths shape为[1 ~ (2T - 1), B]
# extra_t用于测试permute和lengths不等长的情况，lengths[T + extra_T, B]
T = np.random.randint(2, 30, 4)
EXTRA_T = [1, 0, -1]
B = [2048, 20480, 204800]
SHAPE_LIST = list(itertools.product(T, EXTRA_T, B))

_ZERO_PERMUTE = list(itertools.product([0], [1], B))
_ZERO_LENGTH = list(itertools.product(T, [1], [0]))
_NOT_PERFORMED_SHAPES = _ZERO_PERMUTE + _ZERO_LENGTH


def get_result(tensors: dict, device: str = 'cpu'):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    results = torch.ops.fbgemm.permute_2D_sparse_data(**tensors)
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
def test_permute2d_sparse_data(types, shapes, enable_permuted_sum):
    """
    Params:
        permute: (T) dtype=int32
        lengths: (T + T', B) dtype=ltype
                 L = lengths[:T].sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
    """
    ptype, ltype, vtype, wtype = types
    t, extra_t, b = shapes
    extra_t = random.randint(1, t - 1) * extra_t

    permute = np.random.choice(t + extra_t, t).astype(dtype=np.int32)
    lengths = np.ones((t + extra_t, b), dtype=ltype)
    values = np.arange(0, (t + extra_t) * b, dtype=vtype)
    weights = np.arange(0, (t + extra_t) * b, dtype=wtype) if wtype else None
    permuted_lengths_sum = lengths[permute].sum() if enable_permuted_sum else None
    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': weights,
        'permuted_lengths_sum': permuted_lengths_sum
    }
    golden = get_result(params)
    result = get_result(params, DEVICE)
    _check_result(golden, result)


def _check_result(golden, result):
    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", _NOT_PERFORMED_SHAPES)
@pytest.mark.parametrize("enable_permuted_sum", [True])
def test_permute2d_sparse_data_for_not_performed(types, shapes, enable_permuted_sum):
    t, extra_t, b = shapes
    extra_t = random.randint(1, max(t - 1, 1)) * extra_t

    ptype, ltype, vtype, wtype = types
    permute = np.random.choice(t + extra_t, t).astype(dtype=np.int32)
    lengths = np.ones((t + extra_t, b), dtype=ltype)

    # Not perform permutation, the difference in values size and lengths size can be ignored.
    values = np.arange(0, (t + extra_t) * 1, dtype=vtype)
    weights = np.arange(0, (t + extra_t) * 1, dtype=wtype) if wtype else None
    permuted_lengths_sum = lengths[permute].sum() if enable_permuted_sum else None
    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': weights,
        'permuted_lengths_sum': permuted_lengths_sum
    }

    # The golden need use original data, because there is a difference between cpu and gpu in fbgemm repo.
    golden = (lengths, values, weights)
    golden = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in golden]
    result = get_result(params, DEVICE)
    _check_result(golden, result)

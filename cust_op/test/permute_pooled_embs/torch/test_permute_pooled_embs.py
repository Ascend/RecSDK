#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True   # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False      # 关闭CuDNN自动优化

set_seed(10000)


def get_result(tensors: dict, device: str = 'cpu', is_mxrec: bool = False):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if is_mxrec:
        results = torch.ops.mxrec.permute_pooled_embs(**tensors)
    else:
        results = torch.ops.fbgemm.permute_pooled_embs(**tensors)

    if device and device.startswith('npu'):
        torch_npu.npu.synchronize()
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


T = np.random.randint(4, 41, 6)
B = [32, 128, 512, 1024, 2048, 4096, 8192, 16384]
SHAPE_LIST = list(itertools.product(T, B))


@pytest.mark.parametrize("types", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute_pooled_embs_aligned(types, shapes, is_mxrec):
    """
    Params:
        pooled_embs: (B, sum(embs_dims)) dtype=etype
        offset_dim: (T + 1) dtype=int64
        permute: (T) dtype=int64
        inv_permute: (T) dtype=int64
        inv_offset_dim: (T + 1) dtype=int64
    """
    etype = types[0] if isinstance(types, tuple) else types
    t, b = shapes

    # 每个特征的维度随机选择[16, 32, 64, 128, 256, 512, 1024]中的一个
    choices = torch.tensor([16, 32, 64, 128, 256, 512, 1024], dtype=torch.int64)
    embs_dims = choices[torch.randint(0, len(choices), (t,), dtype=torch.int64)]
    offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(embs_dims, dim=0)])
    permute = torch.randperm(t, dtype=torch.int64)
    inv_permute = torch.empty_like(permute)
    for i, p in enumerate(permute):
        inv_permute[p] = i
    inv_embs_dims = embs_dims[permute]
    inv_offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(inv_embs_dims, dim=0)])
    pooled_embs = torch.randn(b, embs_dims.sum().item()).to(etype)

    params = {
        'pooled_embs': pooled_embs,
        'offset_dim_list': offset_dim_list,
        'permute_list': permute,
        'inv_offset_dim_list': inv_offset_dim_list,
        'inv_permute_list': inv_permute,
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)


T = np.random.randint(1, 41, 6)
B = np.random.randint(1, 16385, 8)
SHAPE_LIST = list(itertools.product(T, B))


@pytest.mark.parametrize("types", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute_pooled_embs_unaligned(types, shapes, is_mxrec):
    """
    Params:
        pooled_embs: (B, sum(embs_dims)) dtype=etype
        offset_dim: (T + 1) dtype=int64
        permute: (T) dtype=int64
        inv_permute: (T) dtype=int64
        inv_offset_dim: (T + 1) dtype=int64
    """
    etype = types[0] if isinstance(types, tuple) else types
    t, b = shapes

    embs_dims = torch.randint(1, 1025, size=(t,), dtype=torch.int64)
    offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(embs_dims, dim=0)])
    permute = torch.randperm(t, dtype=torch.int64)
    inv_permute = torch.empty_like(permute)
    for i, p in enumerate(permute):
        inv_permute[p] = i
    inv_embs_dims = embs_dims[permute]
    inv_offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(inv_embs_dims, dim=0)])
    pooled_embs = torch.randn(b, embs_dims.sum().item()).to(etype)

    params = {
        'pooled_embs': pooled_embs,
        'offset_dim_list': offset_dim_list,
        'permute_list': permute,
        'inv_offset_dim_list': inv_offset_dim_list,
        'inv_permute_list': inv_permute,
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)


T = np.random.randint(1, 41, 6)
B = np.random.randint(1, 16385, 8)
SHAPE_LIST = list(itertools.product(T, B))


@pytest.mark.parametrize("types", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute_pooled_embs_aligned_embs(types, shapes, is_mxrec):
    """
    Params:
        pooled_embs: (B, sum(embs_dims)) dtype=etype
        offset_dim: (T + 1) dtype=int64
        permute: (T) dtype=int64
        inv_permute: (T) dtype=int64
        inv_offset_dim: (T + 1) dtype=int64
    """
    etype = types[0] if isinstance(types, tuple) else types
    t, b = shapes

    # 每个特征的维度随机选择[16, 32, 64, 128, 256, 512, 1024]中的一个
    choices = torch.tensor([16, 32, 64, 128, 256, 512, 1024], dtype=torch.int64)
    embs_dims = choices[torch.randint(0, len(choices), (t,), dtype=torch.int64)]
    offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(embs_dims, dim=0)])
    permute = torch.randperm(t, dtype=torch.int64)
    inv_permute = torch.empty_like(permute)
    for i, p in enumerate(permute):
        inv_permute[p] = i
    inv_embs_dims = embs_dims[permute]
    inv_offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(inv_embs_dims, dim=0)])
    pooled_embs = torch.randn(b, embs_dims.sum().item()).to(etype)

    params = {
        'pooled_embs': pooled_embs,
        'offset_dim_list': offset_dim_list,
        'permute_list': permute,
        'inv_offset_dim_list': inv_offset_dim_list,
        'inv_permute_list': inv_permute,
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)

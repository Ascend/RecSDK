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

import sysconfig
import pytest
import torch
import fbgemm_gpu  # noqa: F401
import torch_npu  # noqa: F401
import numpy as np

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_loss(index: np.ndarray, weight: np.ndarray):
    weight_tensor = torch.nn.Parameter(torch.from_numpy(weight)).to(torch.float64)
    weight_tensor.retain_grad()

    index_tensor = torch.from_numpy(index).to(torch.int64)

    result = torch.index_select(weight_tensor, dim=0, index=index_tensor.view(-1))

    loss = torch.mean(result)
    loss.backward()

    grad = weight_tensor.grad.cpu().clone()
    return result.cpu().detach().numpy(), grad


def get_loss_op(index: np.ndarray, weight: np.ndarray, device: str):
    torch.npu.set_device(device)

    weight_tensor = torch.nn.Parameter(torch.from_numpy(weight)).to(device)
    weight_tensor.retain_grad()

    index_tensor = torch.from_numpy(index).to(device)

    result = torch.ops.mxrec.gather_for_rank1(weight_tensor, index_tensor.view(-1))

    loss = torch.mean(result)
    loss.backward()

    grad = weight_tensor.grad.cpu().clone()
    return result.cpu().detach().numpy(), grad


@pytest.mark.parametrize("embedding_dim", [1, 7, 8, 16, 32, 64, 128, 129, 256, 512])
@pytest.mark.parametrize("index_shape", [1, 63, 64, 70, 128, 256, 512])
@pytest.mark.parametrize("device", ["npu:0", "npu:5"])
def test_gather_for_rank1(embedding_dim, index_shape, device):
    weight = np.random.randn(embedding_dim).astype(dtype=np.float32)

    index = np.random.randint(embedding_dim, size=index_shape).astype(np.int64)

    result, grad = get_loss(index, weight)
    result_op, grad_op = get_loss_op(index, weight, device)

    assert np.allclose(result, result_op, atol=1e-6)
    assert np.allclose(grad, grad_op, atol=1e-5)


# ruff: off
def test_gather_for_rank1_exceptions():
    """测试 gather_for_rank1 算子的异常校验"""
    # pylint: disable=unused-variable,line-too-long
    target_device = "npu:0"
    x = torch.randn(10, device=target_device)
    index = torch.tensor([0, 2, 4], dtype=torch.int64, device=target_device)

    # 1. 测试 x 维度非 1D (例如 2D)
    x_2d = torch.randn(10, 2, device=target_device)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.gather_for_rank1(x_2d, index)
    assert "The x should be 1D" in str(ctx.value)

    # 2. 测试 index 维度非 1D (例如 2D)
    index_2d = torch.tensor([[0, 2], [2, 4]], dtype=torch.int64, device=target_device)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.gather_for_rank1(x, index_2d)
    assert "The index should be 1D" in str(ctx.value)


# ruff: on

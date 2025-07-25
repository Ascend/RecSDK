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

import sysconfig

import pytest
import torch
import torch_npu

DEVICE = "npu:7"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_result_golden(grad: torch.Tensor, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(x, device=grad.device, dtype=grad.dtype)
    grad_x.scatter_add_(0, index, grad)
    return grad_x


def get_result_op(grad: torch.Tensor, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    grad_x, _ = torch.ops.mxrec.index_select_for_rank1_backward(grad, x, index)
    return grad_x


@pytest.mark.parametrize("embedding_dim", [129])
@pytest.mark.parametrize("index_shape", [2 ** i for i in range(15)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_gather_for_rank1(embedding_dim, index_shape, dtype):
    torch_npu.npu.set_device(DEVICE)
    grad_input = torch.randn(index_shape, dtype=torch.float32, device=DEVICE)
    x = torch.empty(embedding_dim, device=DEVICE)
    index = torch.randint(embedding_dim, size=(index_shape,), dtype=dtype, device=DEVICE)

    grad = get_result_golden(grad_input, x, index)
    grad_op = get_result_op(grad_input, x, index)
    assert torch.allclose(grad, grad_op, atol=1e-5)

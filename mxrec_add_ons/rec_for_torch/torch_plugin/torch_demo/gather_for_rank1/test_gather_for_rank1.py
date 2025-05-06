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

import pytest
import torch
import torch_npu
import numpy as np

torch.npu.config.allow_internal_format = False
torch.ops.load_library(
    "../../torch_library/2.6.0/gather_for_rank1/build/libgather_for_rank1.so"
)


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

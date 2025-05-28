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

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:7"
NUM_BUCKETS = 128


def create_rab_time_grad(num_layers: int, batchsize: int, s: int):
    return torch.randn(num_layers, batchsize, s, s) * 1e-4


def create_bucket_timestamps(batchsize: int, s: int):
    result = torch.arange(batchsize * s) % NUM_BUCKETS
    result = result.unsqueeze(-1).repeat(1, 1, s)
    return result


def rab_backward_golden(rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor):
    num_layers, b, s, _ = rab_time_grad.shape
    tsw_grad = torch.zeros(num_layers, NUM_BUCKETS, dtype=torch.float32).to(rab_time_grad.device)

    bucket_timestamps_expand = (bucket_timestamps.reshape(b, s // 2, 1, s // 2, 1)
                                .repeat(1, 1, 2, 1, 2)
                                .reshape(b, s, s)
                                .to(torch.int64))
    for n, grad in enumerate(rab_time_grad.to(torch.float32)):
        tsw_grad[n], _ = torch.ops.mxrec.index_select_for_rank1_backward(grad.view(-1),
                                                                         tsw_grad[n],
                                                                         bucket_timestamps_expand.view(-1))
    return tsw_grad


def rab_backward_op(rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor):
    return torch.ops.mxrec.relative_attn_bias_backward(rab_time_grad, bucket_timestamps, NUM_BUCKETS)


@torch.no_grad()
def rab_backward(num_layers: int, batchsize: int, s: int, dtype: torch.dtype):
    torch_npu.npu.set_device(DEVICE)

    grad = create_rab_time_grad(num_layers, batchsize, s).to(dtype).to(DEVICE)
    bucket_timestamps = create_bucket_timestamps(batchsize, s // 2).to(torch.int32).to(DEVICE)
    torch_npu.npu.synchronize()

    golden_result = rab_backward_golden(grad, bucket_timestamps).to("cpu")
    op_result = rab_backward_op(grad, bucket_timestamps).to(torch.float32).to("cpu")
    loss = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(op_result, golden_result, rtol=loss, atol=loss)


@pytest.mark.parametrize("num_layers", [1, 8])
@pytest.mark.parametrize("train_len", [500, 1000, 2000, 4000])
@pytest.mark.parametrize("candidate_len", [600])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_eval(num_layers, train_len, candidate_len, bs, dtype):
    s = 2 * train_len + candidate_len
    rab_backward(num_layers, bs, s, dtype)


@pytest.mark.parametrize("num_layers", [1, 8])
@pytest.mark.parametrize("train_len,bs", [(500, 128), (1000, 32), (1000, 64), (4000, 8)])
@pytest.mark.parametrize("candidate_len", [0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_train(num_layers, train_len, candidate_len, bs, dtype):
    s = 2 * train_len + candidate_len
    rab_backward(num_layers, bs, s, dtype)

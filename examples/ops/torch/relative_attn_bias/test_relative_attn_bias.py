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

from utils import (create_pos_w, create_past_valid_lens, init_rel_pos_bias, create_timestamps, create_rab_time_grad,
                   create_bucket_timestamps, create_timestamps_weights, NUM_BUCKETS, BUCKET_DIVISOR,
                   rab_pos_golden, rab_time_golden, rab_time_backward_golden)

DEVICE = "npu:7"


@torch.no_grad()
def rab_pos(num_layers, train_len, candidate_len, bs, dtype):
    torch_npu.npu.set_device(DEVICE)
    pos_w = create_pos_w(train_len, num_layers).to(dtype)
    past_valid_lens = create_past_valid_lens(bs, train_len).to(torch.int32)
    rel_pos_bias_list, identity_list = init_rel_pos_bias(pos_w=pos_w,
                                                         train_len=train_len,
                                                         candidate_len=candidate_len,
                                                         num_layers=num_layers)
    rel_pos_bias_list, identity_list = rel_pos_bias_list.to(dtype), identity_list.to(dtype)

    rel_pos_bias_list = rel_pos_bias_list.to(DEVICE)
    identity_list = identity_list.to(DEVICE)
    past_valid_lens = past_valid_lens.to(DEVICE)
    torch_npu.npu.synchronize()

    for rel_pos_bias, identity in zip(rel_pos_bias_list, identity_list):
        rab_pos_out = torch.ops.mxrec.relative_attn_bias_pos(rel_pos_bias=rel_pos_bias,
                                                             identity=identity,
                                                             past_valid_lens=past_valid_lens.tolist())
        rab_pos_out_golden = rab_pos_golden(rel_pos_bias=rel_pos_bias,
                                            identity=identity,
                                            past_valid_lens=past_valid_lens)
        assert torch.allclose(rab_pos_out_golden, rab_pos_out)


@torch.no_grad()
def rab_time(num_layers, train_len, candidate_len, bs, dtype):
    torch_npu.npu.set_device(DEVICE)

    past_valid_lens = create_past_valid_lens(bs, train_len).to(torch.int32)
    timestamps = create_timestamps(train_len, candidate_len, past_valid_lens).to(torch.int32)
    timestamps_weights = create_timestamps_weights(num_layers).to(dtype)

    # golden放在cpu上跑
    result_golden, index_golden = rab_time_golden(timestamps_weights=timestamps_weights.transpose(0, 1),
                                                  timestamps=timestamps,
                                                  bucket_divisor=BUCKET_DIVISOR)

    timestamps = timestamps.to(DEVICE)
    timestamps_weights = timestamps_weights.to(DEVICE)
    torch_npu.npu.synchronize()

    # 正常使用时为：result_op = torch.ops.mxrec.relative_attn_bias_time(...)
    result_op, index_op = torch.ops.mxrec.relative_attn_bias_time_with_index(timestamps_weights=timestamps_weights,
                                                                             timestamps=timestamps,
                                                                             bucket_divisor=BUCKET_DIVISOR)
    result_op, index_op = result_op.to('cpu'), index_op.to('cpu')

    result_golden, index_golden = result_golden.to('cpu'), index_golden.to('cpu')
    torch_npu.npu.synchronize()

    _, s, _ = index_op.shape
    index_op_repeat = index_op.view(bs, s, 1, s, 1).repeat(1, 1, 2, 1, 2).reshape(index_golden.shape).to(torch.int64)
    assert torch.allclose(result_golden, result_op)
    assert torch.allclose(index_golden, index_op_repeat)


@torch.no_grad()
def rab_time_backward(num_layers, train_len, candidate_len, bs, dtype):
    torch_npu.npu.set_device(DEVICE)
    s = 2 * train_len + candidate_len

    grad = create_rab_time_grad(num_layers, bs, s).to(dtype).to(DEVICE)
    bucket_timestamps = create_bucket_timestamps(bs, s // 2).to(torch.int32).to(DEVICE)
    torch_npu.npu.synchronize()

    golden_result = rab_time_backward_golden(grad, bucket_timestamps).to("cpu")
    op_result = (torch.ops.mxrec.relative_attn_bias_time_backward(grad, bucket_timestamps, NUM_BUCKETS)
                 .to(torch.float32).to("cpu"))
    loss = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(op_result, golden_result, rtol=loss, atol=loss)


@pytest.mark.parametrize("num_layers", [1, 8])
@pytest.mark.parametrize("train_len", [500, 1000, 2000, 4000])
@pytest.mark.parametrize("candidate_len", [600])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_eval(num_layers, train_len, candidate_len, bs, dtype):
    rab_time(num_layers, train_len, candidate_len, bs, dtype)
    rab_pos(num_layers, train_len, candidate_len, bs, dtype)


@pytest.mark.parametrize("num_layers", [1, 8])
@pytest.mark.parametrize("train_len,bs", [(500, 128), (1000, 32), (1000, 64), (4000, 8)])
@pytest.mark.parametrize("candidate_len", [0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_train(num_layers, train_len, candidate_len, bs, dtype):
    rab_time(num_layers, train_len, candidate_len, bs, dtype)
    rab_pos(num_layers, train_len, candidate_len, bs, dtype)


@pytest.mark.parametrize("num_layers", [1, 8])
@pytest.mark.parametrize("train_len", [500, 1000, 2000, 4000])
@pytest.mark.parametrize("candidate_len", [600])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_eval_backward(num_layers, train_len, candidate_len, bs, dtype):
    rab_time_backward(num_layers, train_len, candidate_len, bs, dtype)


@pytest.mark.parametrize("num_layers", [1, 8])
@pytest.mark.parametrize("train_len,bs", [(500, 128), (1000, 32), (1000, 64), (4000, 8)])
@pytest.mark.parametrize("candidate_len", [0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_train_backward(num_layers, train_len, candidate_len, bs, dtype):
    rab_time_backward(num_layers, train_len, candidate_len, bs, dtype)

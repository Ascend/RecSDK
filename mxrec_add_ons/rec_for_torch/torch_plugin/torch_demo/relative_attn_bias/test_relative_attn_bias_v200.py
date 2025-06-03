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

import random
import sysconfig

import pytest
import torch
import torch.nn.functional as F
import torch_npu

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:0"
NUM_BUCKETS = 128
BUCKET_DIVISOR = 0.301


def create_pos_w(train_len: int, num_layers: int) -> torch.Tensor:
    return torch.range(0, 2 * train_len).unsqueeze(1).repeat(1, num_layers)


def create_past_valid_lens(bs: int, past_len: int) -> torch.Tensor:
    return torch.randint(0, past_len, (bs,))


def create_timestamps(train_len: int, candidate_len: int, past_valid_lens: torch.Tensor) -> torch.Tensor:
    bs = past_valid_lens.size(0)
    timestamps = torch.zeros(bs, train_len + candidate_len // 2)
    for i, valid_len in enumerate(past_valid_lens):
        if valid_len > 0:
            timestamps[i, :valid_len] = torch.range(1, valid_len.int())

    if candidate_len <= 0:
        return timestamps
    timestamps[:, -candidate_len // 2:] = train_len + 1
    return timestamps


def create_timestamps_weights(num_layers: int) -> torch.Tensor:
    """
    :param num_layers:
    :return: timestamps_weights(num_layers, NUM_BUCKETS + 1)
    """
    return torch.range(0, NUM_BUCKETS).repeat(num_layers).reshape(num_layers, NUM_BUCKETS + 1)


def init_rel_pos_bias(pos_w: torch.Tensor,
                      train_len: int,
                      candidate_len: int,
                      num_layers: int) -> (torch.Tensor, torch.Tensor):
    rel_pos_bias_list, identity_list = [], []

    max_len = train_len + candidate_len // 2
    max_len_x2 = train_len * 2 + candidate_len
    for layer_num in range(num_layers):
        t = F.pad(pos_w[:2 * train_len - 1, layer_num], [0, train_len]).repeat(train_len)
        t = t[..., :-train_len].reshape(1, train_len, 3 * train_len - 2)
        r = (2 * train_len - 1) // 2

        _rel_pos_bias = t[:, :, r:-r]
        _rel_pos_bias = torch.nn.functional.pad(_rel_pos_bias,
                                                (0, candidate_len // 2, 0, candidate_len // 2),
                                                'constant',
                                                0.0)
        _rel_pos_bias = _rel_pos_bias.unsqueeze(-1).repeat(1, 1, 2, 2).reshape(1, max_len_x2, max_len_x2)

        pos_indices = torch.arange(max_len).repeat(max_len).view(max_len, max_len).to(_rel_pos_bias.device)
        pos_indices = pos_indices.unsqueeze(-1).repeat(1, 2, 2).reshape(max_len * 2, max_len * 2)
        identity = (pos_indices.t() == pos_indices).float()

        rel_pos_bias_list.append(_rel_pos_bias.squeeze(0))
        identity_list.append(identity)

    return torch.stack(rel_pos_bias_list), torch.stack(identity_list)


def rab_time_golden(ts_w: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    """
    num_buckets = 128
    num_layers = 1 - 20
    past_len = 1 - 4000
    candidate_len = 256 - 600

    :param ts_w: [num_buckets + 1][num_layers]
    :param timestamps: [bs][past_len + candidate_len // 2]
    :param bucketization_divisor: float
    :return: [num_layers][bs][1][2 * past_len + candidate_len + 1][2 * past_len + candidate_len + 2]
    """
    infer_len = timestamps.shape[1] * 2
    bs = timestamps.shape[0]
    num_layers = ts_w.shape[1]

    timestamps = timestamps.unsqueeze(-1).repeat(1, 1, 2)
    diff_timestamps = timestamps.reshape(bs, infer_len, 1) - timestamps.reshape(bs, 1, infer_len)

    clamp_max = torch.exp(torch.tensor(NUM_BUCKETS * BUCKET_DIVISOR))
    diff_timestamps = torch.log(torch.abs(diff_timestamps).clamp(1, clamp_max)) / BUCKET_DIVISOR

    bucket_timestamps = diff_timestamps.long().view(-1)
    result = torch.index_select(ts_w, dim=0, index=bucket_timestamps)
    result = result.t().view(num_layers, bs, infer_len, infer_len)
    return result


def rab_pos_golden(rel_pos_bias: torch.Tensor, identity: torch.Tensor, past_valid_lens: torch.Tensor) -> torch.Tensor:
    """
    past_len = 1 ~ 4000
    candidate_len = 256 ~ 600
    bs = 1 ~ 10

    :param rel_pos_bias: [past_len * 2 + candidate_len][past_len * 2 + candidate_len]
    :param identity: [past_len * 2 + candidate_len][past_len * 2 + candidate_len]
    :param past_valid_lens: [bs]
    :return: [bs][1][past_len * 2 + candidate_len + 2][past_len * 2 + candidate_len + 2]
    """
    bs = past_valid_lens.shape[0]
    rel_pos_bias_list = rel_pos_bias[:].unsqueeze(0).repeat(bs, 1, 1)
    for i, valid_len in enumerate(past_valid_lens):
        rel_pos_bias_list[i, valid_len:, :] = rel_pos_bias[valid_len, :]

    rel_pos_bias_list = rel_pos_bias_list * (1 - identity) + identity * rel_pos_bias_list[0, 0, 0]
    rel_pos_bias_list = rel_pos_bias_list[:, :identity.shape[0], :identity.shape[0]]
    return rel_pos_bias_list


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
        op_result = torch.ops.mxrec.relative_attn_bias_pos(rel_pos_bias=rel_pos_bias,
                                                           identity=identity,
                                                           past_valid_lens=past_valid_lens.tolist()).to('cpu')
        golden_result = rab_pos_golden(rel_pos_bias=rel_pos_bias.to('cpu'),
                                       identity=identity.to('cpu'),
                                       past_valid_lens=past_valid_lens.to('cpu'))
        assert torch.allclose(op_result, golden_result)


@torch.no_grad()
def rab_time(num_layers, train_len, candidate_len, bs, dtype):
    torch_npu.npu.set_device(DEVICE)

    past_valid_lens = create_past_valid_lens(bs, train_len).to(torch.int32)
    timestamps = create_timestamps(train_len, candidate_len, past_valid_lens).to(torch.int32)
    timestamps_weights = create_timestamps_weights(num_layers).to(dtype)

    timestamps = timestamps.to(DEVICE)
    timestamps_weights = timestamps_weights.to(DEVICE)
    torch_npu.npu.synchronize()

    rab_time_out = torch.ops.mxrec.relative_attn_bias_time(timestamps_weights=timestamps_weights,
                                                           timestamps=timestamps,
                                                           bucket_divisor=BUCKET_DIVISOR).to("cpu")
    rab_time_out_golden = rab_time_golden(ts_w=timestamps_weights.transpose(0, 1).to("cpu"),
                                          timestamps=timestamps.to("cpu"))
    torch_npu.npu.synchronize()

    assert torch.allclose(rab_time_out_golden, rab_time_out)


@pytest.mark.parametrize("num_layers", [8])
@pytest.mark.parametrize("train_len", [500, 1000, 2000, 4000])
@pytest.mark.parametrize("candidate_len", [600])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_time_eval(num_layers, train_len, candidate_len, bs, dtype):
    rab_time(num_layers, train_len, candidate_len, bs, dtype)


@pytest.mark.parametrize("num_layers", [8])
@pytest.mark.parametrize("train_len", [500, 1000, 2000, 4000])
@pytest.mark.parametrize("candidate_len", [600])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_pos_eval(num_layers, train_len, candidate_len, bs, dtype):
    rab_pos(num_layers, train_len, candidate_len, bs, dtype)

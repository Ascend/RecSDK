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

import torch
import torch.nn.functional as F
import torch_npu

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

NUM_BUCKETS = 128
BUCKET_DIVISOR = 0.301


def create_pos_w(train_len: int, num_layers: int) -> torch.Tensor:
    return torch.range(0, 2 * train_len).unsqueeze(1).repeat(1, num_layers)


def create_past_valid_lens(bs: int, past_len: int) -> torch.Tensor:
    return torch.randint(0, past_len, (bs,))


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


def create_timestamps_weights(num_layers: int):
    """
    :param num_layers:
    :return: timestamps_weights(num_layers, NUM_BUCKETS + 1)
    """
    return torch.range(0, NUM_BUCKETS).repeat(num_layers).reshape(num_layers, NUM_BUCKETS + 1)


def create_rab_time_grad(num_layers: int, batchsize: int, s: int):
    return torch.randn(num_layers, batchsize, s, s) * 1e-5


def create_bucket_timestamps(batchsize: int, s: int):
    result = torch.arange(batchsize * s) % NUM_BUCKETS
    result = result.unsqueeze(-1).repeat(1, 1, s)
    return result


def rab_time_golden(timestamps_weights: torch.Tensor,
                    timestamps: torch.Tensor,
                    bucket_divisor: float) -> (torch.Tensor, torch.Tensor):
    """
    rab time 正向仿真
    num_buckets = 128
    num_layers = 1 - 20
    past_len = 1 - 4000
    candidate_len = 256 - 600

    :param timestamps_weights: [num_buckets + 1][num_layers]
    :param timestamps: [bs][past_len + candidate_len // 2]
    :param bucket_divisor: float
    :return: [num_layers][bs][1][2 * past_len + candidate_len + 1][2 * past_len + candidate_len + 2]
    """

    infer_len = timestamps.shape[1] * 2
    bs = timestamps.shape[0]
    num_layers = timestamps_weights.shape[1]

    timestamps = timestamps.unsqueeze(-1).repeat(1, 1, 2)
    diff_timestamps = timestamps.reshape(bs, infer_len, 1) - timestamps.reshape(bs, 1, infer_len)

    clamp_max = torch.exp(torch.tensor(NUM_BUCKETS * BUCKET_DIVISOR))
    diff_timestamps = torch.log(torch.abs(diff_timestamps).clamp(1, clamp_max)) / bucket_divisor

    bucket_timestamps = diff_timestamps.long().view(-1)
    rab_time_out = torch.index_select(timestamps_weights, dim=0, index=bucket_timestamps)
    rab_time_out = rab_time_out.t().view(num_layers, bs, infer_len, infer_len)

    return rab_time_out, bucket_timestamps


def rab_time_backward_golden(rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor):
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


def rab_pos_backward_golden():
    pass

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

from dataclasses import dataclass

import torch
import torch_npu


@dataclass
class TestArgs:
    query_layer: torch.Tensor
    key_layer: torch.Tensor
    value_layer: torch.Tensor
    pos_key_layer: torch.Tensor
    pos_query_layer: torch.Tensor
    relative_pos: torch.Tensor
    atten_mask: torch.Tensor
    pos_att_type: str
    score_scale: float


@dataclass
class DataArgs:
    b: int
    n: int
    s: int
    d: int
    bucket_size: int
    max_position: int
    pos_att_type: str


def build_relative_position(
    query_size: int,
    key_size: int,
    bucket_size: int = -1,
    max_position: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    q_ids = torch.arange(0, query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(0, key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.repeat(q_ids.shape[0], 1)

    if bucket_size > 0 and max_position > 0:
        mid = bucket_size // 2
        sign = rel_pos_ids.sign()
        abs_pos = torch.where(
            (rel_pos_ids < mid) & (rel_pos_ids > -mid),
            torch.zeros_like(rel_pos_ids).fill_(mid - 1),
            rel_pos_ids.abs(),
        ).float()
        log_pos = (
            torch.ceil(
                torch.log(abs_pos / mid)
                / torch.log(torch.tensor((max_position - 1) / mid))
                * (mid - 1)

            ).long()
            + mid

        )
        bucket_pos = torch.where(abs_pos <= mid, rel_pos_ids, log_pos * sign).long()
        rel_pos_ids = bucket_pos
    rel_pos_ids = rel_pos_ids[:query_size, :]
    return rel_pos_ids


def create_binary_tensor(b: int, s: int):
    # 创建FP16类型的随机矩阵，形状为[b, 1, s, s]
    rand_tensor = torch.rand((b, 1, s, s), dtype=torch.float16)

    # 将张量二值化：大于0.5的位置设为1（后面会替换为-65504），其余设为0
    bin_tensor = torch.where(
        rand_tensor > 0.5,
        torch.tensor(1.0, dtype=torch.float16),
        torch.tensor(0.0, dtype=torch.float16),
    )

    # 将1替换为FP16的最小有限值 -65504
    min_fp16 = torch.finfo(torch.float16).min  # -65504
    result = torch.where(bin_tensor == 1, min_fp16, bin_tensor)

    return result


def create_score_scale(pos_att_type: str, d: int):
    pos_att_type_list = tuple(
        [x.strip() for x in pos_att_type.lower().split("|")] if pos_att_type else []
    )

    score_scale = d**-0.5

    scale_factor = 1 + len(pos_att_type_list)

    score_scale = (d * scale_factor) ** -0.5
    return score_scale


def create_test_data(args: DataArgs, npu_device):
    query_layer = torch.rand(args.b, args.n, args.s, args.d, dtype=torch.float16)
    key_layer = torch.rand(args.b, args.n, args.s, args.d, dtype=torch.float16)
    value_layer = torch.rand(args.b, args.n, args.s, args.d, dtype=torch.float16)

    pos_key_layer = torch.rand(2 * args.s, args.n, args.d, dtype=torch.float16)
    pos_query_layer = torch.rand(2 * args.s, args.n, args.d, dtype=torch.float16)

    relative_position = build_relative_position(
        args.s, args.s, args.bucket_size, args.max_position, query_layer.device
    )
    atten_mask = create_binary_tensor(args.b, args.s)

    score_scale = create_score_scale(args.pos_att_type, args.d)

    test_args = TestArgs(
        query_layer.to(npu_device),
        key_layer.to(npu_device),
        value_layer.to(npu_device),
        pos_key_layer.to(npu_device),
        pos_query_layer.to(npu_device),
        relative_position.to(npu_device),
        atten_mask.to(npu_device),
        args.pos_att_type,
        score_scale,
    )

    return test_args
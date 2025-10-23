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

import pytest
import torch
import torch_npu

from utils import NUM_BUCKETS, BUCKET_DIVISOR

DEVICE = "npu:0"


@pytest.mark.parametrize("ts_dim", [2, 3, 4])
@pytest.mark.parametrize("tsw_dim", [2, 3, 4])
@pytest.mark.parametrize("seq_len", [500, 5000])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_time(ts_dim, tsw_dim, seq_len, dtype):
    if ts_dim == tsw_dim == 2:
        return
    torch_npu.npu.set_device(DEVICE)

    timestamps = torch.empty(seq_len, dtype=torch.int32, device=DEVICE)  # (b, s)
    timestamps_weights = torch.empty(NUM_BUCKETS, dtype=dtype, device=DEVICE)  # (layer, bucket)

    while timestamps.dim() != ts_dim:
        timestamps = timestamps.unsqueeze(0)

    while timestamps_weights.dim() != tsw_dim:
        timestamps_weights = timestamps_weights.unsqueeze(0)

    with pytest.raises(RuntimeError):
        torch.ops.mxrec.relative_attn_bias_time(timestamps, timestamps_weights, BUCKET_DIVISOR)


@pytest.mark.parametrize("b", [5])
@pytest.mark.parametrize("s", [5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rab_pos(b, s, dtype):
    torch_npu.npu.set_device(DEVICE)
    rel_pos_bias = torch.randn(s, s, dtype=dtype, device=DEVICE)
    identity = torch.randn(s, s, dtype=dtype, device=DEVICE)
    past_valid_lens = torch.randint(1, s, (b,), dtype=dtype, device=DEVICE)

    test_cases = [
        # rel_pos_bias identity dim校验
        (torch.randn(s, s, 1), identity),
        (torch.randn(s), identity),
        (rel_pos_bias, torch.randn(s, s, 1)),
        (rel_pos_bias, torch.randn(s)),
        # rel_pos_bias identity shape校验
        (torch.randn(s, s + 1), identity),
        (torch.randn(s + 1, s), identity),
        (rel_pos_bias, torch.randn(s, s + 1)),
        (rel_pos_bias, torch.randn(s + 1, s)),
    ]

    for _rel_pos_bias, _identity in test_cases:
        with pytest.raises(RuntimeError):
            torch.ops.mxrec.relative_attn_bias_pos(
                rel_pos_bias=_rel_pos_bias,
                identity=_identity,
                past_valid_lens=past_valid_lens.tolist()
            )

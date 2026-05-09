#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import random
import numpy as np
import torch_npu


torch.set_printoptions(precision=6)
torch.npu.config.allow_internal_format = False
device_id: int = 0
torch.npu.set_device(device_id)

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libgen_position_ids_reverse_v2.so")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jagged_data_gen(batch_size, max_seq_len):
    min_seq_len = 1
    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32)

    seq_offset = torch.zeros(batch_size + 1, dtype=torch.int32)
    seq_offset[1:] = torch.cumsum(seq_lens, dim=0)

    total_len = seq_lens.sum().item()
    rspos = torch.randint(0, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    rspos = torch.min(rspos, seq_lens)

    return seq_lens, seq_offset, rspos


def gen_position_ids_reverse_v2_golden_v1(seqlen, seqlen_offsets, rspos):
    batch_size = seqlen.shape[0]
    total_len = seqlen_offsets[-1].item()
    position_ids = torch.zeros(total_len, dtype=torch.int32)

    for batch_idx in range(batch_size):
        seq_len = seqlen[batch_idx].item()
        rs_pos = rspos[batch_idx].item()

        pre_len = rs_pos
        post_len = seq_len - rs_pos
        total_output_len = pre_len + post_len

        output_start = seqlen_offsets[batch_idx].item()

        for offset in range(total_output_len):
            position_value = 0
            if offset < pre_len:
                position_value = pre_len - offset
            position_ids[output_start + offset] = position_value

    return position_ids

def gen_position_ids_reverse_v2_golden_v2(seqlen: torch.Tensor, rspos, interleaved_action: bool) -> torch.Tensor:
    """
    生成pos_ids 反向ver2 同一个i的 item action pos相同 [n-1,n-1,...,,3,2,1, 0,0,0,0,0,0]
    reverse v1: 就是单纯的 [n...0]
    reverse v2: 考虑同一个i的 item action 位置一样 但是rspos后的全都是0
    """
    position_ids = []
    batch_size = len(seqlen)
    for i in range(batch_size):
        pre_len = rspos[i].item()
        post_len = seqlen[i].item() - rspos[i].item()

        pre_arr = list(range(pre_len, 0, -1))
        if interleaved_action:
            pre_arr = [x
                       for x in pre_arr
                       for _ in (0, 1)]
            post_arr = [0] * (post_len * 2)
        else:
            post_arr = [0] * post_len
        position_ids.extend(pre_arr + post_arr)
    return torch.tensor(position_ids, dtype=seqlen.dtype, device=seqlen.device)

class TestGenPositionIdsReverseV2:
    @staticmethod
    def custom_op_exec(seqlen, seqlen_offsets, rspos, batch_size, interleaved_action=False, with_ctx=False):
        seqlen_npu = seqlen.to("npu")
        seqlen_offsets_npu = seqlen_offsets.to("npu")
        rspos_npu = rspos.to("npu")

        position_ids = torch.ops.fbgemm.gen_position_ids_reverse_v2(
            seqlen_npu,
            seqlen_offsets_npu,
            rspos_npu,
            batch_size,
            interleaved_action,
            with_ctx
        )

        torch.npu.synchronize()
        return position_ids.cpu()

    @staticmethod
    def execute(batch_size, max_seq_len):
        seqlen, seqlen_offsets, rspos = jagged_data_gen(batch_size, max_seq_len)

        position_ids_golden_v1 = gen_position_ids_reverse_v2_golden_v1(seqlen, seqlen_offsets, rspos)
        position_ids_golden_v2 = gen_position_ids_reverse_v2_golden_v2(seqlen, rspos, False)

        position_ids_custom = TestGenPositionIdsReverseV2.custom_op_exec(
            seqlen,
            seqlen_offsets,
            rspos,
            batch_size
        )


        assert torch.equal(position_ids_golden_v1, position_ids_golden_v2)
        assert torch.equal(position_ids_golden_v1, position_ids_custom)


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32, 64])
@pytest.mark.parametrize("max_seq_len", [1, 32, 128, 256, 512])
def test_gen_position_ids_reverse_v2(batch_size, max_seq_len):
    TestGenPositionIdsReverseV2.execute(batch_size, max_seq_len)

@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("max_seq_len", [64, 256])
def test_gen_position_ids_reverse_v2_large(batch_size, max_seq_len):
    TestGenPositionIdsReverseV2.execute(batch_size, max_seq_len)
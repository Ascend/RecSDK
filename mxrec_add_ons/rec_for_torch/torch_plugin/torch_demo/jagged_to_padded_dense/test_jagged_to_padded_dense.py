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
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE_ID = 0


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim):
    seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))

    seq_offset = torch.concat((torch.zeros((1, ), dtype=torch.int64), \
        torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()
    
    total_seqs = np.sum(seq_lens)

    jagged_value = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)

    return jagged_value, seq_offset, total_seqs


def jagged_to_dense(jagged_tensor, seq_lens, max_seq_len, head_num, atten_dim):
    need_pad_seq = []
    offset = 0
    for seq_len in seq_lens:
        src_tensor = torch.rand(max_seq_len, head_num, atten_dim)
        src_tensor = torch.zeros((max_seq_len, head_num, atten_dim))
        src_tensor[0:seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
        need_pad_seq.append(src_tensor)
        offset = offset + seq_len

    dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
    return dense_tensor


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("max_seq_len", [128, 256])
@pytest.mark.parametrize("num_heads", [2, 8])
@pytest.mark.parametrize("attention_dim", [32])
def test_jagged_to_padded_dense(batch_size, max_seq_len, num_heads, attention_dim):
    jagged_value, seq_offset, total_seqs = jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim)

    seq_lens = np.zeros((batch_size, )).astype(np.int64)
    for batch_id in range(batch_size):
        seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

    golden_dense = jagged_to_dense(jagged_value, seq_lens, max_seq_len, num_heads, attention_dim)

    result_dense = torch.ops.mxrec.jagged_to_padded_dense(
        jagged_value.reshape(total_seqs, num_heads * attention_dim).to(f"npu:{DEVICE_ID}"),
        [torch.from_numpy(seq_offset).to(f"npu:{DEVICE_ID}")], max_seq_len, 0.0)
    
    assert torch.allclose(golden_dense.reshape(-1), result_dense.cpu().reshape(-1), atol=1e-5)

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
import torch


def jagged_to_dense(jagged_tensor, seq_lens, max_seq_len, head_num, head_dim):
    batch_size = len(seq_lens)
    dense_tensor = torch.zeros(batch_size, max_seq_len, head_num, head_dim, dtype=jagged_tensor.dtype)

    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        dense_tensor[batch_id, :seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
        offset = offset + seq_len

    return dense_tensor


def dense_to_jagged(q, dense_tensor, seq_lens):
    dense_dim = dense_tensor.shape[3]
    tensor = torch.zeros(q.shape[0], q.shape[1], dense_dim).to(dense_tensor.dtype).cpu()

    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
        offset = offset + seq_len

    return tensor
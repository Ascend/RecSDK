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
import dataclasses
import random
import sysconfig
from enum import Enum

import numpy as np
import torch
import torch_npu

BLOCK_HEIGHT: int = 256
MAX_NUM_TARGET: int = 512

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0
torch.npu.set_device(device_id)


class MaskType(int, Enum):
    TRIL = 0  # 下三角掩码
    TRIU = 1  # 上三角掩码
    NONE = 2  # 无掩码
    CUSTOM = 3  # 自定义掩码


def get_chip():
    return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True   # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False      # 关闭CuDNN自动优化


def allclose(tensor: torch.Tensor, other: torch.Tensor, atol: float, ratio: float) -> bool:
    assert tensor.shape == other.shape
    diff = (torch.abs(tensor - other) > atol)
    diff_count = torch.sum(diff)
    return (diff_count / tensor.numel()) < ratio


def jagged_to_dense(jagged_tensor, seq_lens, head_nums, attn_dim):
    need_pad_seq = []
    offset = 0
    for seq_len in seq_lens:
        src_tensor = jagged_tensor[offset: offset + seq_len, :, :].reshape(seq_len, head_nums, attn_dim)
        need_pad_seq.append(src_tensor)
        offset = offset + seq_len

    dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
    return dense_tensor


def dense_to_jagged(q, dense_tensor, seq_lens):
    tensor = torch.zeros_like(q).cpu()

    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
        offset = offset + seq_len

    return tensor


@dataclasses.dataclass
class QKVShapeInfo:
    float_type: torch.dtype
    int_type: torch.dtype
    batch_size: int
    num_heads_q: int
    num_heads_k: int
    attention_dim: int
    max_seq_len: int
    min_seq_len: int = 1


@dataclasses.dataclass
class MaskGenInfo:
    mask_type: int | MaskType = MaskType.TRIL
    target_group_size: int = 0
    max_num_context: int = 0
    max_num_target: int = 0
    min_num_context: int = 0
    min_num_target: int = 0

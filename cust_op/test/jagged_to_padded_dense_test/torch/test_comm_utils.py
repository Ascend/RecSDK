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

import itertools
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


PRECISION_ERROR_RANGE = {
    torch.float32: 1e-4,
    torch.int64: 1e-4,
    torch.float16: 1e-3,
    torch.bfloat16: 5e-3,
    torch.int32: 1e-4,
}
VALUES_DATA_TYPES = PRECISION_ERROR_RANGE.keys()


def generate_jagged_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_types):
    """
    生成不规则(Jagged)张量测试数据
    Args:
        batch_size: 批处理大小
        max_seq_len: 单个样本最大序列长度
        num_heads: 注意力头数量
        attention_dim: 每个注意力头的维度
        data_types: tuple(values_data_type, offsets_data_type), values/offsets数据类型

    Returns:
        jagged_tensor: 不规则数据张量，形状为(total_sequences, num_heads, attention_dim)
        seq_offsets: 序列偏移量数组，表示每个样本在jagged_tensor中的起始位置
        total_sequences: 所有样本的序列总长度
    """
    # 为每个样本随机生成序列长度(1到max_seq_len之间)
    seq_lens = np.random.randint(1, max_seq_len + 1, batch_size)

    # 计算累积偏移量(前面补0)
    seq_offsets = torch.concat((
        torch.zeros((1,), dtype=data_types[1]),
        torch.cumsum(torch.from_numpy(seq_lens), dim=0),
    )).numpy()

    total_sequences = np.sum(seq_lens)

    # 生成随机数据
    values_data_type = data_types[0]
    if values_data_type in [torch.int64, torch.int32]:
        jagged_tensor = torch.randint(
            low=0, high=1000000, size=(total_sequences, num_heads, attention_dim),
            dtype=values_data_type,
        )
    else:
        jagged_tensor = torch.rand(
            total_sequences, num_heads, attention_dim,
            dtype=values_data_type,
        ).uniform_(-1, 1)

    return jagged_tensor, seq_offsets, total_sequences


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    base: dict
    sweep: dict

    def expand(self):
        if not self.sweep:
            return [self.base.copy()]
        keys = list(self.sweep.keys())
        runs = []
        for values in itertools.product(*(self.sweep[key] for key in keys)):
            run = self.base.copy()
            for key, value in zip(keys, values):
                run[key] = value
            runs.append(run)
        return runs


@dataclass
class ExecuteConfig:
    batch_size: int
    max_seq_len: int
    num_heads: int
    attention_dim: int
    use_list_max_lengths: bool
    values_data_type: Union[torch.float32, torch.int64, torch.float16, torch.bfloat16, torch.int32]
    offsets_data_type: Union[torch.int32, torch.int64]

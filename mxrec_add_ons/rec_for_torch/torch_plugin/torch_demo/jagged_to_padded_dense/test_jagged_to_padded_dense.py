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
import itertools
import sysconfig
from dataclasses import dataclass
from typing import Union

import fbgemm_gpu
import numpy as np
import pytest
import torch

# 加载NPU自定义算子库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
# 设置用的卡号
DEVICE = "npu:0"

_PRECISION_ERROR_RANGE = {
    torch.float32: 1e-4,
    torch.int64: 1e-4,
    torch.float16: 1e-3,
    torch.bfloat16: 5e-3,
    torch.int32: 1e-4
}
_VALUES_DATA_TYPES = _PRECISION_ERROR_RANGE.keys()


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
        torch.cumsum(torch.from_numpy(seq_lens), dim=0)
    )).numpy()

    total_sequences = np.sum(seq_lens)

    # 生成随机数据
    values_data_type = data_types[0]
    if values_data_type in [torch.int64, torch.int32]:
        jagged_tensor = torch.randint(
            low=0, high=1000000, size=(total_sequences, num_heads, attention_dim),
            dtype=values_data_type
        )
    else:
        jagged_tensor = torch.rand(
            total_sequences, num_heads, attention_dim,
            dtype=values_data_type
        ).uniform_(-1, 1)

    return jagged_tensor, seq_offsets, total_sequences


@dataclass
class ExecuteConfig:
    batch_size: int
    max_seq_len: int
    num_heads: int
    attention_dim: int
    use_list_max_lengths: bool
    values_data_type: Union[torch.float32, torch.int64, torch.float16, torch.bfloat16, torch.int32]
    offsets_data_type: Union[torch.int32, torch.int64]


test_params = {
    "batch_size": [2, 4],
    "max_seq_len": [128, 256],
    "num_heads": [2, 8],
    "attention_dim": [32],
    "use_list_max_lengths": [True, False],
    "values_data_type": _VALUES_DATA_TYPES,
    "offsets_data_type": [torch.int32, torch.int64],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*test_params.values())
])
def test_jagged_to_padded_dense(config: ExecuteConfig):
    """
    测试不规则张量到填充密集张量的转换算子
    测试逻辑:
    1. 生成随机测试数据
    2. 使用FBGEMM的CPU实现计算基准结果
    3. 调用NPU算子计算结果
    4. 对比两者差异(允许1e-4的误差)
    """
    batch_size = config.batch_size
    max_seq_len = config.max_seq_len
    num_heads = config.num_heads
    attention_dim = config.attention_dim
    use_list_max_lengths = config.use_list_max_lengths
    values_data_type = config.values_data_type
    offsets_data_type = config.offsets_data_type

    # 1. 生成测试数据
    data_types = (values_data_type, offsets_data_type)
    jagged_tensor, seq_offsets, total_sequences = generate_jagged_tensor(
        batch_size, max_seq_len, num_heads, attention_dim, data_types)

    # 2. 准备FBGEMM算子输入(需要展平最后两个维度)
    input_flat = jagged_tensor.reshape(total_sequences, num_heads * attention_dim)
    fbgemm_offsets = torch.from_numpy(seq_offsets)

    # 3. 调用FBGEMM CPU实现
    fbgemm_dense = torch.ops.fbgemm.jagged_to_padded_dense(
        input_flat,
        [fbgemm_offsets],
        [max_seq_len],
        0.0  # 填充值
    )

    # 4. 调用NPU算子
    npu_dense = torch.ops.mxrec.jagged_to_padded_dense(
        input_flat.to(DEVICE),
        [fbgemm_offsets.to(DEVICE)],
        [max_seq_len] if use_list_max_lengths else max_seq_len,
        0.0
    )

    # 5. 结果比对
    assert torch.allclose(
        fbgemm_dense.reshape(-1),
        npu_dense.cpu().reshape(-1),
        atol=_PRECISION_ERROR_RANGE[values_data_type],
        rtol=_PRECISION_ERROR_RANGE[values_data_type]
    ), f"NPU结果与FBGEMM CPU结果不匹配\nFBGEMM:\n{fbgemm_dense}\nNPU:\n{npu_dense.cpu()}"

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
import logging
import sysconfig

import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

DEVICE = "npu:0"
logging.getLogger().setLevel(logging.INFO)
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DENSE_DIM0 = [128, 40] # 测试不同batch大小
DENSE_DIM1 = [210] # 固定特征维度1
DENSE_DIM2 = [1, 8] # 固定特征维度2
DIM_LIST = list(itertools.product(DENSE_DIM0, DENSE_DIM1, DENSE_DIM2))

DENSE_DATATYPE = [torch.float32, torch.int64] # 测试不同数据类型
OFFSET_DATATYPE = [torch.int32, torch.int64] # 偏移量数据类型
TYPE_LIST = list(itertools.product(DENSE_DATATYPE, OFFSET_DATATYPE))


def get_result(device, denses, offsets, types, use_output_size):
    dense_datatype, offset_datatype = types
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(device)
    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(device)

    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    # 获取输出大小（最后一个偏移量即总元素数）
    output_size = None
    if use_output_size:
        output_size = jagged_id_offset[-1]

    # 执行核心操作：稠密张量→不规则张量
    jagged_embedding = torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]
    return jagged_embedding.cpu()


@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("use_output_size", [True, False])  # 测试是否传入 output_size
def test_dense_to_jagged(dims, types, use_output_size):
    dense_dim0, dense_dim1, dense_dim2 = dims
    # 1. 生成随机输入数据
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0) # 生成随机偏移量

    # 2. 分别获取CPU和NPU结果
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, use_output_size)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, use_output_size)

    # 3. 结果比对（允许1e-4的误差）
    result_forward = torch.abs(golden_result[0] - npu_result[0]) < 1e-4
    logging.info(result_forward.all().item())  # 输出是否全部通过验证
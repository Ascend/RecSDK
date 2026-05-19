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

import itertools
import random
import sysconfig

from pathlib import Path
import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

DEVICE = "npu:0"
torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


# 定义参数shape
def get_result(tensors: dict):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}
    torch.npu.set_device(DEVICE)
    tensors = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
    # also can use torch.ops.fbgemm.keyed_jagged_index_select_dim1
    results = torch.ops.mxrec.keyed_jagged_index_select_dim1(**tensors)
    torch_npu.npu.synchronize()
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


# 采用fbgemm_gpu::permute_2D_sparse_data进行cpu设置验证
def get_golden(tensors: dict, batch_num):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}
    indiceslen = tensors['indices'].size(0)
    permutelen = batch_num * indiceslen
    permute = torch.empty(permutelen, dtype=torch.int32)
    for i in range(batch_num):
        for j in range(indiceslen):
            permute[i * indiceslen + j] = tensors['batch_size'] * i + tensors['indices'][j].item()
    results = torch.ops.fbgemm.permute_1D_sparse_data(
        permute, tensors['lengths'], tensors['values'], tensors['weights'], tensors['selected_lengths_sum']
    )
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


LENGTHS_TYPE = [np.int32, np.int64]
INDICES_TYPE = [np.int64, np.int32]
VALUES_TYPE = [np.int64, np.int32, np.float32, np.float16]
WEIGHTS_TYPE = [None, np.float32]
OFFSET_TYPE = [np.int64, np.int32]
TYPE_LIST = list(itertools.product(LENGTHS_TYPE, INDICES_TYPE, VALUES_TYPE, WEIGHTS_TYPE, OFFSET_TYPE))
ENABLE_SELECTED_LENGTHS_SUM = [False, True]
INT64_LENGTHS_TYPE = [np.int64]
FP16_WEIGHTS_TYPE = [np.float16]
TYPE_LIST_1 = list(itertools.product(INT64_LENGTHS_TYPE, INDICES_TYPE, VALUES_TYPE, FP16_WEIGHTS_TYPE, OFFSET_TYPE))


# 初始化测试入参，v220 indices dtype只支持int32, 可手动调整
def init_tensor(types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum):
    length_type, indices_type, values_type, weights_type, offset_type = types

    indices = np.random.choice(batch_size, output_batch_size).astype(dtype=indices_type)
    lengths = np.random.randint(2, 500, size=batch_size * batch_num, dtype=length_type)
    total_length = int(lengths.sum())
    cumulative_lengths = np.cumsum(lengths)
    offsets = np.zeros(len(lengths) + 1, dtype=offset_type)
    offsets[1:] = cumulative_lengths

    is_float = values_type in [np.float32, np.float16]
    if is_float:
        values = np.random.rand(total_length).astype(dtype=values_type)
    else:
        values = np.random.randint(0, 2**16, (total_length,), dtype=values_type)
    weights = np.arange(0, total_length, dtype=weights_type) if weights_type else None

    permute = np.empty(batch_num * output_batch_size, dtype=indices_type)
    for i in range(batch_num):
        for j in range(output_batch_size):
            permute[i * output_batch_size + j] = batch_size * i + indices[j]
    selected_lengths_sum = lengths[permute].sum() if enable_selected_lengths_sum else None
    params = {
        'values': values,
        'lengths': lengths,
        'offsets': offsets,
        'indices': indices,
        'batch_size': batch_size,
        'weights': weights,
        'selected_lengths_sum': selected_lengths_sum,
    }
    return params


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("batch_num", [1, 8, 64, 100])
@pytest.mark.parametrize("batch_size", [1, 8, 64, 256])
@pytest.mark.parametrize("output_batch_size", [2, 8, 64, 256])
@pytest.mark.parametrize("enable_selected_lengths_sum", ENABLE_SELECTED_LENGTHS_SUM)
def test_keyed_jagged_index_select_dim1(types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum):
    """
    测试正常情况下的keyed_jagged_index_select_dim1算子功能
    Params:
        indices: (output_batch_size) dtype=int32
        lengths: (batch_size * batch_num) dtype=ltype
                L = lengths[:T].sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
        batch_size: int
        selected_lengths_sum: int
    """
    params = init_tensor(types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum)

    golden = get_golden(params, batch_num)
    result = get_result(params)
    assert torch.allclose(golden[0], result[1], atol=1e-5)
    assert torch.allclose(golden[1], result[0], atol=1e-5)
    if params['weights'] is not None:
        assert torch.allclose(golden[2], result[2], atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST_1)
@pytest.mark.parametrize("batch_num", [8, 64, 100])
@pytest.mark.parametrize("batch_size", [8, 64, 256])
@pytest.mark.parametrize("output_batch_size", [8, 64, 256])
@pytest.mark.parametrize("enable_selected_lengths_sum", ENABLE_SELECTED_LENGTHS_SUM)
def test_keyed_jagged_index_select_dim1_tpye_list_1(
    types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum
):
    params = init_tensor(types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum)
    weights_None = None
    params_golden = {
        'values': params['values'],
        'lengths': params['lengths'],
        'offsets': params['offsets'],
        'indices': params['indices'].astype(dtype=np.int32),
        'batch_size': batch_size,
        'weights': weights_None,
        'selected_lengths_sum': params['selected_lengths_sum'],
    }
    golden = get_golden(params_golden, batch_num)
    params_golden['values'] = params['weights']
    golden_weights = get_golden(params_golden, batch_num)
    result = get_result(params)
    assert torch.allclose(golden[0], result[1], atol=1e-5)
    assert torch.allclose(golden[1], result[0], atol=1e-5)
    assert torch.allclose(golden_weights[1], result[2], atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST_1)
@pytest.mark.parametrize("batch_num", [0, 2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("output_batch_size", [0, 2])
@pytest.mark.parametrize("enable_selected_lengths_sum", ENABLE_SELECTED_LENGTHS_SUM)
def test_keyed_jagged_index_select_dim1_error(
    types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum
):
    """
    测试空tensor
    """
    if batch_num != 0 and batch_size != 0 and output_batch_size != 0:
        return
    params = init_tensor(types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum)
    with pytest.raises(Exception):
        result = get_result(params)
        assert result is None

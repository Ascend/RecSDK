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
import random
import sysconfig
from pathlib import Path

import pytest
import torch
import torch_npu
import numpy as np

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent /
                           "framework/torch_plugin/torch_library/multislice_concat/build/libmultislice_concat.so"))
DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

DEFAULT_BATCH_SIZE = 32
DEFAULT_COLUMN_SIZE = 32
DEFAULT_CONCAT_NUM = 10


def golden_func(input_data_np, concat_num, concat_size_np, slice_begin_np, slice_size_np):
    input_data = torch.from_numpy(input_data_np)
    concat_size = torch.from_numpy(concat_size_np)
    slice_begin = torch.from_numpy(slice_begin_np)
    slice_length = torch.from_numpy(slice_size_np)

    offset = 0
    result_matrices = []

    for i in range(concat_num):
        curr_concat_size = concat_size[i].item()

        # 搜集当前concat的所有列数据
        all_columns = []
        for j in range(offset, offset + curr_concat_size):
            curr_begin = slice_begin[j].item()
            curr_size = slice_length[j].item()

            cur_column = input_data[:, curr_begin:curr_begin + curr_size]
            all_columns.append(cur_column)
        if all_columns:
            new_matrix = torch.cat(all_columns, dim=1)
            result_matrices.append(new_matrix)
        else:
            # 如果没有数据，添加空矩阵
            result_matrices.append(torch.tensor([]))

        offset += curr_concat_size
    
    return result_matrices


def get_result(input_data_np, concat_num, concat_size_np, slice_begin_np, slice_size_np):
    input_data = torch.from_numpy(input_data_np).to(DEVICE)
    results = torch.ops.mxrec.multislice_concat(input_data, concat_num, concat_size_np, slice_begin_np, slice_size_np)
    torch.npu.synchronize()
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


def generate_data(batch_size, column_size, concat_num, input_type=np.float16):
    input_data = np.random.rand(batch_size, column_size).astype(input_type)
    total_concat_size = np.random.randint(concat_num, 600)
    split_points = np.random.choice(np.arange(1, total_concat_size), size=concat_num - 1, replace=False)
    split_points = np.sort(split_points)
    concat_size = []
    prev = 0
    for point in split_points:
        concat_size.append(point - prev)
        prev = point
    concat_size.append(total_concat_size - prev)
    concat_size = np.array(concat_size, dtype=np.int32)

    total_slice_num = np.sum(concat_size)
    slice_begin = np.random.randint(0, column_size, size=total_slice_num, dtype=np.int32)
    slice_length_max = column_size - slice_begin
    slice_length = np.array(
        [np.random.randint(low=1, high=max_length + 1) for max_length in slice_length_max], dtype=np.int32)

    return input_data, concat_size, slice_begin, slice_length


def check_result(goldens, results, name):
    for i, (gt, pred) in enumerate(zip(goldens, results)):
        assert type(gt) is type(pred), f"类型不匹配: golden={type(gt)}, result={type(pred)}"
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert gt.shape == pred.shape, f"形状不匹配: golden={gt.shape}, result={pred.shape}"
            assert torch.allclose(gt, pred, atol=1e-4), f"第{i}个矩阵数值不匹配"


@pytest.mark.parametrize("batch_size", [8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 3072])
@pytest.mark.parametrize("column_size", [3, 64, 6400, 20000])
@pytest.mark.parametrize("concat_num", [10, 122, 256])
def test_multislice_concat_acc(batch_size, column_size, concat_num):
    np.random.seed(418)

    input_data_np, concat_size, slice_begin, slice_length = generate_data(batch_size, column_size, concat_num)

    goldens = golden_func(input_data_np, concat_num, concat_size, slice_begin, slice_length)

    results = get_result(input_data_np, concat_num, concat_size, slice_begin, slice_length)

    check_result(goldens, results,
        f"case: batch_size[{batch_size}] column_size[{column_size}] concat_num[{concat_num}]")


def test_input_tensor_dim():
    np.random.seed(88)
    _, concat_size_np, slice_begin_np, slice_size_np = generate_data(
        DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE, DEFAULT_CONCAT_NUM)

    input_data_dim_1 = np.random.rand(32,).astype(np.float16)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_dim_1, DEFAULT_CONCAT_NUM, concat_size_np, slice_begin_np, slice_size_np)

    assert "input must be 2D" in str(ctx.value)

    input_data_dim_3 = np.random.rand(32, 2, 2).astype(np.float16)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_dim_3, DEFAULT_CONCAT_NUM, concat_size_np, slice_begin_np, slice_size_np)

    assert "input must be 2D" in str(ctx.value)


def test_input_tensor_dtype():
    np.random.seed(88)
    _, concat_size_np, slice_begin_np, slice_size_np = generate_data(
        DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE, DEFAULT_CONCAT_NUM)

    input_data_np = np.random.rand(DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE).astype(np.int64)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_np, DEFAULT_CONCAT_NUM, concat_size_np, slice_begin_np, slice_size_np)
    
    assert "input tensor type" in str(ctx.value)

    input_data_np = np.random.rand(DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE).astype(np.int32)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_np, DEFAULT_CONCAT_NUM, concat_size_np, slice_begin_np, slice_size_np)
    
    assert "input tensor type" in str(ctx.value)

    input_data_np, concat_size, slice_begin, slice_length = generate_data(
        DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE, DEFAULT_CONCAT_NUM, np.float32)

    goldens = golden_func(input_data_np, DEFAULT_CONCAT_NUM, concat_size, slice_begin, slice_length)

    results = get_result(input_data_np, DEFAULT_CONCAT_NUM, concat_size, slice_begin, slice_length)

    check_result(goldens, results,
        f"case: batch_size[{DEFAULT_BATCH_SIZE}] column_size[{DEFAULT_COLUMN_SIZE}] concat_num[{DEFAULT_CONCAT_NUM}]")




def test_input_tensor_row():
    np.random.seed(88)
    _, concat_size_np, slice_begin_np, slice_size_np = generate_data(
        DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE, DEFAULT_CONCAT_NUM)

    input_data_np = np.random.rand(65536, 32).astype(np.int64)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_np, DEFAULT_CONCAT_NUM, concat_size_np, slice_begin_np, slice_size_np)

    assert "input tensor row" in str(ctx.value)


def test_input_tensor_column():
    np.random.seed(88)
    _, concat_size_np, slice_begin_np, slice_size_np = generate_data(
        DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE, DEFAULT_CONCAT_NUM)

    input_data_np = np.random.rand(32, 65536).astype(np.int64)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_np, DEFAULT_CONCAT_NUM, concat_size_np, slice_begin_np, slice_size_np)

    assert "input tensor column" in str(ctx.value)


def test_concat_size():
    np.random.seed(88)

    input_data = np.random.rand(16, 100).astype(np.float16)
    concat_size = np.array([1, 1]).astype(np.uint16)
    slice_begin = np.array([2, 2]).astype(np.uint16)
    slice_length = np.array([1, 1]).astype(np.uint16)

    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data, 10, concat_size, slice_begin, slice_length)

    assert "length of concatSize" in str(ctx.value)

    concat_size = np.array([-1, 1]).astype(np.int16)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data, 2, concat_size, slice_begin, slice_length)

    assert "concat_size" in str(ctx.value)

    concat_size = np.array([3601, 1]).astype(np.int16)
    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data, 2, concat_size, slice_begin, slice_length)

    assert "all slice num must be <= 3600" in str(ctx.value)


def test_concat_num():
    np.random.seed(12)

    input_data_np, concat_size_np, slice_begin_np, slice_size_np, = generate_data(
        DEFAULT_BATCH_SIZE, DEFAULT_COLUMN_SIZE, DEFAULT_CONCAT_NUM)

    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_np, 601, concat_size_np, slice_begin_np, slice_size_np)
    
    assert "concat_num" in str(ctx.value)

    with pytest.raises(Exception) as ctx:
        _ = get_result(input_data_np, -1, concat_size_np, slice_begin_np, slice_size_np)
    
    assert "concat_num" in str(ctx.value)


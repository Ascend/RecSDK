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

import logging
import sysconfig
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch_npu

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)


def get_expected_new_offsets_and_lengths(
    d_unique_offsets: torch.Tensor, d_table_offsets_in_feature: torch.Tensor, local_batch_size: int, dtype: torch.dtype
):
    dtype_map = {torch.int32: np.int32, torch.int64: np.int64}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}, only support torch.int32/torch.int64")
    np_dtype = dtype_map[dtype]

    d_unique_offsets_np = d_unique_offsets.cpu().numpy().astype(np_dtype)
    d_table_offsets_in_feature_np = d_table_offsets_in_feature.cpu().numpy().astype(np_dtype)

    num_table = len(d_table_offsets_in_feature_np) - 1
    num_feature_total = d_table_offsets_in_feature_np[-1]
    new_lengths_size = num_feature_total * local_batch_size

    expected_new_offsets = np.zeros(new_lengths_size + 1, dtype=np_dtype)
    expected_new_lengths = np.zeros(new_lengths_size, dtype=np_dtype)

    def binary_search_last_le(arr, num, target):
        start = 0
        end = num
        while start < end:
            middle = start + (end - start) // 2
            value = arr[middle]
            if value <= target:
                start = middle + 1
            else:
                end = middle
        return num if (start == num and arr[start - 1] != target) else start - 1

    for i in range(new_lengths_size):
        feature_id = i // local_batch_size
        table_id = binary_search_last_le(d_table_offsets_in_feature_np, num_table + 1, feature_id)

        table_feature_count = d_table_offsets_in_feature_np[table_id + 1] - d_table_offsets_in_feature_np[table_id]
        table_buckets = table_feature_count * local_batch_size

        bucket_id = i - (d_table_offsets_in_feature_np[table_id] * local_batch_size)

        unique_num = d_unique_offsets_np[table_id + 1] - d_unique_offsets_np[table_id]

        bucket_base = unique_num // table_buckets
        bucket_remainder = unique_num % table_buckets

        tmp_length = bucket_base
        if bucket_id < bucket_remainder:
            tmp_length += 1

        tmp_offset = d_unique_offsets_np[table_id]
        tmp_offset += (bucket_id * bucket_base) + (bucket_id if bucket_id < bucket_remainder else bucket_remainder)

        expected_new_lengths[i] = tmp_length
        expected_new_offsets[i] = tmp_offset

        if i == new_lengths_size - 1:
            expected_new_offsets[new_lengths_size] = tmp_offset + tmp_length

    return (torch.tensor(expected_new_offsets, dtype=dtype), torch.tensor(expected_new_lengths, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tables", [1, 3, 5])
@pytest.mark.parametrize("local_batch_size", [1, 2, 5])
@pytest.mark.parametrize("features_per_table", [2, 5])
@pytest.mark.parametrize("unique_per_table", [10, 15])
def test_get_new_length_and_offsets_npu(dtype, num_tables, local_batch_size, features_per_table, unique_per_table):
    device = "npu:0"
    device_id = int(device.split(":")[-1])

    # 构造输入的 numpy 数组
    d_table_offsets_in_feature_np = [0]
    for i in range(num_tables):
        d_table_offsets_in_feature_np.append(d_table_offsets_in_feature_np[-1] + features_per_table)
    num_feature_total = d_table_offsets_in_feature_np[-1]

    d_unique_offsets_np = [0]
    for i in range(num_tables):
        d_unique_offsets_np.append(d_unique_offsets_np[-1] + unique_per_table)

    d_unique_offsets = torch.tensor(d_unique_offsets_np, dtype=torch.int64, device=device)
    d_table_offsets_in_feature = torch.tensor(d_table_offsets_in_feature_np, dtype=torch.int64, device=device)

    new_lengths_size = num_feature_total * local_batch_size
    new_offsets = torch.empty(new_lengths_size + 1, dtype=dtype, device=device)
    new_lengths = torch.empty(new_lengths_size, dtype=dtype, device=device)

    # 调用自定义 OP
    dynamic_emb_extensions.get_new_length_and_offsets_op(
        d_unique_offsets=d_unique_offsets,
        d_table_offsets_in_feature=d_table_offsets_in_feature,
        new_offsets=new_offsets,
        new_lengths=new_lengths,
        local_batch_size=local_batch_size,
    )

    # 计算预期结果
    expected_new_offsets, expected_new_lengths = get_expected_new_offsets_and_lengths(
        d_unique_offsets=d_unique_offsets,
        d_table_offsets_in_feature=d_table_offsets_in_feature,
        local_batch_size=local_batch_size,
        dtype=dtype,
    )

    actual_new_offsets = new_offsets.cpu()
    actual_new_lengths = new_lengths.cpu()
    expected_new_offsets_cpu = expected_new_offsets.cpu()
    expected_new_lengths_cpu = expected_new_lengths.cpu()

    # 验证一致性
    offsets_match = torch.equal(actual_new_offsets, expected_new_offsets_cpu)
    lengths_match = torch.equal(actual_new_lengths, expected_new_lengths_cpu)
    total_match = offsets_match and lengths_match

    # 失败时打印调试信息
    if not total_match:
        logging.error(f"\nTest Failed!")
        logging.error(
            f"  Params: num_tables={num_tables}, local_batch_size={local_batch_size}, "
            f"features_per_table={features_per_table}, unique_per_table={unique_per_table}, dtype={dtype}"
        )
        logging.error(f"  Inputs:")
        logging.error(f"    d_unique_offsets (int64): {d_unique_offsets.cpu().numpy()}")
        logging.error(f"    d_table_offsets_in_feature (int64): {d_table_offsets_in_feature.cpu().numpy()}")
        logging.error(f"  Expected new_offsets: {expected_new_offsets_cpu.numpy()}")
        logging.error(f"  Actual new_offsets:   {actual_new_offsets.numpy()}")
        logging.error(f"  Expected new_lengths: {expected_new_lengths_cpu.numpy()}")
        logging.error(f"  Actual new_lengths:   {actual_new_lengths.numpy()}")

    # 断言
    assert offsets_match, f"new_offsets mismatch! Expected {expected_new_offsets_cpu}, got {actual_new_offsets}"
    assert lengths_match, f"new_lengths mismatch! Expected {expected_new_lengths_cpu}, got {actual_new_lengths}"
    assert total_match, "Custom op 'get_new_length_and_offsets_npu' did not produce the expected result."

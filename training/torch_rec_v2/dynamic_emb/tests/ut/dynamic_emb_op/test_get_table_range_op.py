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

import pytest
import torch
import torch_npu

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)


def get_expected_table_range(offsets_np, feature_offsets_np):
    num_table = len(feature_offsets_np) - 1
    num_feature = feature_offsets_np[-1]

    if num_feature == 0:
        batch = 0
    else:
        batch = (len(offsets_np) - 1) // num_feature

    expected_table_range = []
    for tid in range(num_table + 1):
        feature_offset = feature_offsets_np[tid]
        feature_x_batch_offset = feature_offset * batch
        if feature_x_batch_offset < len(offsets_np):
            expected_val = offsets_np[feature_x_batch_offset]
        else:
            expected_val = 0

        expected_table_range.append(expected_val)

    return torch.tensor(expected_table_range, dtype=torch.int64)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("num_tables", [1, 3, 5])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_get_table_range_op(dtype, device, num_tables, batch_size):
    """
    测试自定义 get_table_range_op 算子的正确性。

    Args:
        dtype: 输入张量的数据类型。
        device: 测试设备。
        num_tables: table 的数量。
        batch_size: 每个特征重复的 batch 次数。
    """
    device_id = int(device.split(":")[-1])

    # 1. 构造测试数据, 每个 table 有 2 个特征
    features_per_table = 2
    feature_offsets_np = [0]
    for i in range(num_tables):
        feature_offsets_np.append(feature_offsets_np[-1] + features_per_table)
    num_feature_total = feature_offsets_np[-1]

    # 构造 offsets, 步长为 10
    offsets_length = num_feature_total * batch_size + 1
    offsets_np = torch.arange(0, offsets_length * 10, 10, dtype=torch.int64).numpy()

    # 移动数据到npu设备
    feature_offsets = torch.tensor(feature_offsets_np, dtype=dtype, device=device)
    offsets = torch.tensor(offsets_np, dtype=dtype, device=device)

    # 2. 计算预期结果
    expected_table_range = get_expected_table_range(offsets_np, feature_offsets_np).to(dtype)

    # 3. 调用自定义算子计算实际结果
    torch.npu.synchronize()
    actual_table_range = dynamic_emb_extensions.get_table_range_op(offsets, feature_offsets).to(dtype)
    torch.npu.synchronize()

    # 4. 将结果移至 CPU 进行比较
    actual_table_range_cpu = actual_table_range.cpu()
    expected_table_range_cpu = expected_table_range.cpu()
    feature_offsets_cpu = feature_offsets.cpu()
    offsets_cpu = offsets.cpu()

    # 5. 打印调试信息
    match = torch.equal(actual_table_range_cpu, expected_table_range_cpu)
    if not match:
        logging.error(f"\nTest Failed for num_tables={num_tables}, batch_size={batch_size}, dtype={dtype}")
        logging.error(f"  Inputs:")
        logging.error(f"    feature_offsets: {feature_offsets_cpu}")
        logging.error(f"    offsets: {offsets_cpu[:20]}... (total length: {len(offsets_np)})")
        logging.error(f"  Expected table_range: {expected_table_range_cpu}")
        logging.error(f"  Actual table_range:   {actual_table_range_cpu}")

    # 6. 断言结果一致
    assert match, "Custom op 'get_table_range_op' did not produce the expected result."

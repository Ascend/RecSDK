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

import sysconfig
import pytest
import torch
import math
import random
import numpy as np
import torch_npu

# 默认 time_scale
DEFAULT_TIME_SCALE = 300.0
torch.set_printoptions(precision=6)
torch.npu.config.allow_internal_format = False
device_id: int = 0
torch.npu.set_device(device_id)

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libgen_position_ids_with_timestamp.so")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN自动优化


def jagged_data_gen(batch_size, max_seq_len):
    """Generate test data for gen_position_ids_with_timestamp (直接生成 tensor)"""
    min_seq_len = 1
    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32)

    seq_offset = torch.zeros(batch_size + 1, dtype=torch.int32)
    seq_offset[1:] = torch.cumsum(seq_lens, dim=0)

    # 时间戳生成：每个 batch 内的 timestamps 需要递增，确保 t_end >= timestamp
    timestamps_list = []
    for batch_idx in range(batch_size):
        seq_len = seq_lens[batch_idx].item()
        base = torch.randint(0, 5000, (1,)).item()
        timestamps_batch = torch.arange(base, base + seq_len, dtype=torch.int32)
        timestamps_list.append(timestamps_batch)

    timestamps = torch.cat(timestamps_list)

    return seq_lens, seq_offset, timestamps


class TestGenPositionIdsWithTimestamp:
    @staticmethod
    def custom_op_exec(seqlen, seqlen_offsets, timestamps, batch_size, total_seq_len, time_scale=None):
        """Execute the custom operator - time_scale 为可选参数"""
        seqlen_npu = seqlen.to("npu")
        seqlen_offsets_npu = seqlen_offsets.to("npu")
        timestamps_npu = timestamps.to("npu")

        # time_scale 为可选参数
        if time_scale is not None:
            position_ids = torch.ops.fbgemm.gen_position_ids_with_timestamp(
                seqlen_npu,
                seqlen_offsets_npu,
                timestamps_npu,
                batch_size,
                total_seq_len,
                time_scale
            )
        else:
            position_ids = torch.ops.fbgemm.gen_position_ids_with_timestamp(
                seqlen_npu,
                seqlen_offsets_npu,
                timestamps_npu,
                batch_size,
                total_seq_len
            )

        torch.npu.synchronize()
        return position_ids.cpu()

    @staticmethod
    def compute_position_ids_golden(seqlen, seqlen_offsets, timestamps, time_scale):
        """
        Compute position_ids using the reference formula:
        pos = log1p((t_end - tm) / time_scale) / log(log_base)
        where log_base = 1.1, so inv_log_base = 1 / ln(1.1) ≈ 10.4920586873
        """
        inv_log_base = 10.4920586873
        max_position_id = 1024

        batch_size = seqlen.shape[0]
        total_len = timestamps.shape[0]
        position_ids = torch.zeros(total_len, dtype=torch.int32)

        for batch_idx in range(batch_size):

            seq_len = seqlen[batch_idx].item()
            start_pos = seqlen_offsets[batch_idx].item()
            end_pos = seqlen_offsets[batch_idx + 1].item()

            actual_len = end_pos - start_pos
            if actual_len != seq_len:
                continue

            t_end = timestamps[end_pos - 1].item()

            for offset in range(seq_len):
                timestamp_idx = start_pos + offset
                timestamp = timestamps[timestamp_idx].item()

                time_diff = (t_end - timestamp) / time_scale
                log_pos = math.log(1 + time_diff) * inv_log_base
                # log_pos = math.log(1.0 + time_diff) * inv_log_base

                position_id = max(0, min(int(math.floor(log_pos)), max_position_id))
                position_ids[timestamp_idx] = position_id

        return position_ids

    @staticmethod
    def execute(batch_size, max_seq_len, time_scale=None):
        set_seed(1234)
        seqlen, seqlen_offsets, timestamps = jagged_data_gen(batch_size, max_seq_len)
        total_seq_len = timestamps.shape[0]

        # 实际使用的 time_scale
        actual_time_scale = time_scale if time_scale is not None else DEFAULT_TIME_SCALE

        position_ids_golden = TestGenPositionIdsWithTimestamp.compute_position_ids_golden(
            seqlen, seqlen_offsets, timestamps, actual_time_scale
        )

        position_ids_custom = TestGenPositionIdsWithTimestamp.custom_op_exec(
            seqlen,
            seqlen_offsets,
            timestamps,
            batch_size,
            total_seq_len,
            actual_time_scale
        )

        assert torch.equal(position_ids_golden, position_ids_custom)


# 测试不输入 time_scale（使用默认值）
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("max_seq_len", [1, 32, 256, 512, 1024])
def test_gen_position_ids_default_time_scale(batch_size, max_seq_len):
    TestGenPositionIdsWithTimestamp.execute(batch_size, max_seq_len)


# 测试输入自定义 time_scale
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [32, 128])
@pytest.mark.parametrize("time_scale", [100.0, 600.0, 3600.0])
def test_gen_position_ids_custom_time_scale(batch_size, max_seq_len, time_scale):
    TestGenPositionIdsWithTimestamp.execute(batch_size, max_seq_len, time_scale=time_scale)

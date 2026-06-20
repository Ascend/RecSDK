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

import time
import logging

import torch

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


def run_npu_performance_test(combiner, seq_len, dim, batch_size, device_id=0):
    """运行单个npu dynamic_emb_op.lookup_forward_op性能测试用例"""
    # 1. 设置npu设备
    torch.npu.set_device(device_id)

    # 2. 生成npu测试数据
    torch.manual_seed(42)  # 固定随机种子保证可复现
    src_num = 10000
    src = torch.randn(src_num, dim, dtype=torch.float32, device=f"npu:{device_id}")
    total_dims = 2 * dim
    dst = torch.zeros(size=(batch_size, total_dims), dtype=torch.float32, device=f"npu:{device_id}")
    num_vec = batch_size
    offset = seq_len * torch.arange(num_vec + 1, device=f"npu:{device_id}")
    inverse_len = num_vec * seq_len
    inverse = torch.randint(low=0, high=src_num, size=(inverse_len,), device=f"npu:{device_id}")
    accum_dims = 0

    torch.npu.synchronize(device_id)  # 确保所有npu任务执行完毕

    # 3. 测试npu性能
    start_npu = time.perf_counter()
    dynamic_emb_extensions.lookup_forward(
        src, dst, offset, inverse, combiner, total_dims, accum_dims, dim, num_vec, batch_size
    )
    torch.npu.synchronize(device_id)  # 确保所有npu任务执行完毕
    end_npu = time.perf_counter()

    avg_npu_time_us = (end_npu - start_npu) * 1_000_000
    return avg_npu_time_us


def main():
    """npu dynamic_emb_op.lookup_forward性能测试（推荐业务用例）"""
    # 测试基础配置
    device_id = 0
    test_cases = [
        (0, 30, 8, 1024),
        (0, 30, 8, 10240),
        (0, 30, 32, 1024),
        (0, 30, 32, 10240),
        (0, 30, 64, 1024),
        (0, 30, 64, 10240),
        (0, 30, 128, 1024),
        (0, 30, 128, 10240),
        (0, 100, 8, 1024),
        (0, 100, 8, 10240),
        (0, 100, 32, 1024),
        (0, 100, 32, 10240),
        (0, 100, 64, 1024),
        (0, 100, 64, 10240),
        (0, 100, 128, 1024),
        (0, 100, 128, 10240),
        (0, 200, 8, 1024),
        (0, 200, 8, 10240),
        (0, 200, 32, 1024),
        (0, 200, 32, 10240),
        (0, 200, 64, 1024),
        (0, 200, 64, 10240),
        (0, 200, 128, 1024),
        (0, 200, 128, 10240),
        (0, 500, 8, 1024),
        (0, 500, 8, 10240),
        (0, 500, 32, 1024),
        (0, 500, 32, 10240),
        (0, 500, 64, 1024),
        (0, 500, 64, 10240),
        (0, 500, 128, 1024),
        (0, 500, 128, 10240),
    ]

    # 打印标题和表头
    logging.info("=== npu dynamic_emb_op.lookup_forward_op性能测试（推荐业务用例）===")
    logging.info("%22s %22s %22s %32s %15s", "combiner", "seq_len", "dim", "batch_size", "used time(us)")
    logging.info("-" * 150)

    # 预热
    for combiner, seq_len, dim, batch_size in test_cases:
        _ = run_npu_performance_test(
            combiner=combiner, seq_len=seq_len, dim=dim, batch_size=batch_size, device_id=device_id
        )

    # 统计数据初始化
    torch_times = []

    repeats = 10
    # 遍历所有真实业务测试用例
    for combiner, seq_len, dim, batch_size in test_cases:
        one_latency = 0
        for _ in range(repeats):
            one_latency += run_npu_performance_test(
                combiner=combiner, seq_len=seq_len, dim=dim, batch_size=batch_size, device_id=device_id
            )
        avg_latency = one_latency / repeats
        torch_times.append(avg_latency)

        # 打印单例结果（格式对齐）
        logging.info("%22d %22d %22d %22d %15.2f", combiner, seq_len, dim, batch_size, avg_latency)

    # 统计汇总结果
    logging.info("-" * 150)
    # 替换为 logging.info
    logging.info("平均耗时: %.2f us", sum(torch_times) / len(torch_times))
    logging.info("耗时范围: %.2f ~ %.2f us", min(torch_times), max(torch_times))


if __name__ == "__main__":
    main()

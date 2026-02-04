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
import torch_npu

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


def run_npu_performance_test(input_length, output_length, out_dim, dtype, device_id=0):
    """运行单个npu dynamic_emb_op.gather_op性能测试用例"""
    # 1. 设置npu设备
    torch.npu.set_device(device_id)

    # 2. 生成npu测试数据
    torch.manual_seed(42)  # 固定随机种子保证可复现
    inputs = torch.randn(input_length, out_dim, dtype=torch.float32, device=f"npu:{device_id}")
    indices = torch.randint(0, input_length, (output_length, ), dtype=dtype, device=f"npu:{device_id}")
    torch.npu.synchronize(device_id)  # 确保所有npu任务执行完毕

    # 3. 测试npu原生torch.gather
    start_npu = time.perf_counter()
    _ = dynamic_emb_extensions.gather_embedding(inputs, indices)
    torch.npu.synchronize(device_id)  # 确保所有npu任务执行完毕
    end_npu = time.perf_counter()

    avg_npu_time_us = ((end_npu - start_npu) * 1_000_000)
    return avg_npu_time_us


def main():
    """npu dynamic_emb_op.gather性能测试（推荐业务用例）"""
    # 测试基础配置
    dtype = torch.int64
    device_id = 2   # 对应原测试的npu:1
    test_cases = [
        (100, 200, 8),
        (1000, 2000, 8),
        (10000, 20000, 8),
        (100000, 200000, 8),
        (1000000, 2000000, 8),
        (10000000, 20000000, 8),
        (100, 200, 128),
        (1000, 2000, 128),
        (10000, 20000, 128),
        (100000, 200000, 128),
        (1000000, 2000000, 128),
        (10000000, 20000000, 128),
        (100, 200, 512),
        (1000, 2000, 512),
        (10000, 20000, 512),
        (100000, 200000, 512),
        (1000000, 2000000, 512),
        (10000000, 20000000, 512),
    ]

    # 打印标题和表头
    logging.info("=== npu dynamic_emb_op.gather_op性能测试（推荐业务用例）===")
    logging.info("%22s %22s %32s %15s", "input length", "output length", "out dim", "used time(us)")
    logging.info("-" * 150)

    # 预热
    for input_length, output_length, out_dim in test_cases:
        _ = run_npu_performance_test(
            input_length=input_length,
            output_length=output_length,
            out_dim=out_dim,
            dtype=dtype,
            device_id=device_id
        )

    # 统计数据初始化
    torch_times = []

    repeats = 10
    # 遍历所有真实业务测试用例
    for input_length, output_length, out_dim in test_cases:
        one_latency = 0
        for _ in range(repeats):
            one_latency += run_npu_performance_test(
                input_length=input_length,
                output_length=output_length,
                out_dim=out_dim,
                dtype=dtype,
                device_id=device_id
            )
        avg_latency = one_latency / repeats
        torch_times.append(avg_latency)

        # 打印单例结果（格式对齐）
        logging.info("%22d %22d %22d %15.2f", input_length, output_length, out_dim, avg_latency)

    # 统计汇总结果
    logging.info("-" * 150)
    # 替换为 logging.info
    logging.info("平均耗时: %.2f us", sum(torch_times) / len(torch_times))
    logging.info("耗时范围: %.2f ~ %.2f us", min(torch_times), max(torch_times))

if __name__ == "__main__":
    main()
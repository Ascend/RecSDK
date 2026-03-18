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

import time
import logging
import random

import torch
import torch_npu

import dynamic_emb_extensions

logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


def get_new_length_and_offsets_npu_result(
        d_unique_offsets, d_table_offsets_in_feature, new_offsets, new_lengths, local_batch_size
):
    """npu侧算子调用封装"""
    return dynamic_emb_extensions.get_new_length_and_offsets_op(
        d_unique_offsets=d_unique_offsets,
        d_table_offsets_in_feature=d_table_offsets_in_feature,
        new_offsets=new_offsets,
        new_lengths=new_lengths,
        local_batch_size=local_batch_size
    )


def run_npu_performance_test(table_num, local_batch_size, dtype, device_id=0):
    """运行单个npu dynamic_emb_op.get_new_length_and_offsets_op性能测试用例"""
    # 1. 设置npu设备
    torch.npu.set_device(device_id)

    # 2. 生成npu测试数据
    torch.manual_seed(42)  # 固定随机种子保证可复现
    random.seed(42)

    # 2.1 构造d_table_offsets_in_feature: [0, 2, 4, ..., table_num*2]，每个table包含2个feature
    d_table_offsets_in_feature = torch.tensor(
        [i * 2 for i in range(table_num + 1)],
        dtype=dtype,
        device=f"npu:{device_id}"
    ).contiguous()

    # 2.2 构造d_unique_offsets: 模拟每个表的唯一元素偏移，前缀和
    unique_counts = [0] + [random.randint(1, 100) for _ in range(table_num)]
    d_unique_offsets = torch.cumsum(torch.tensor(unique_counts, dtype=dtype), 0).to(f"npu:{device_id}").contiguous()

    # 2.3 计算newLengthsSize
    num_feature_total = d_table_offsets_in_feature[-1].item()
    new_lengths_size = num_feature_total * local_batch_size

    # 2.4 构造输出张量
    new_offsets = torch.zeros(
        new_lengths_size + 1, dtype=dtype, device=f"npu:{device_id}"
    ).contiguous()
    new_lengths = torch.zeros(
        new_lengths_size, dtype=dtype, device=f"npu:{device_id}"
    ).contiguous()

    # 3 NPU同步避免异步计时误差
    torch.npu.synchronize()  # 确保前面数据准备完成

    # 4. 调用算子并测量时间
    start_npu = time.perf_counter()
    _ = get_new_length_and_offsets_npu_result(
        d_unique_offsets=d_unique_offsets,
        d_table_offsets_in_feature=d_table_offsets_in_feature,
        new_offsets=new_offsets,
        new_lengths=new_lengths,
        local_batch_size=local_batch_size
    )
    torch.npu.synchronize()  # 确保算子执行完成
    end_npu = time.perf_counter()

    # 5. 计算耗时并返回结果
    npu_time_us = (end_npu - start_npu) * 1e6  # 转换为微秒
    return npu_time_us


def main():
    """npu dynamic_emb_op.get_new_length_and_offsets_op性能测试"""
    # 测试基础配置
    dtype = torch.int64
    device_id = 0   # 使用的NPU设备ID
    repeats = 5  # 每个测试用例的重复次数，取平均值

    # 测试用例：table_num 和 local_batch_size 的组合
    # 基于分析的local_batch_size取值范围，为每个table_num选择代表性值
    test_cases = [
        # table_num=1: [50, 500, 5000, 50000, 500000, 2500000, 5000000, 25000000]
        (1, 50), (1, 500), (1, 5000), (1, 50000), (1, 500000), (1, 2500000), (1, 5000000), (1, 25000000),
        # table_num=10: [5, 50, 500, 5000, 50000, 250000, 500000, 2500000]
        (10, 5), (10, 50), (10, 500), (10, 5000), (10, 50000), (10, 250000), (10, 500000), (10, 2500000),
        # table_num=50: [1, 10, 100, 1000, 10000, 50000, 100000, 500000]
        (50, 1), (50, 10), (50, 100), (50, 1000), (50, 10000), (50, 50000), (50, 100000), (50, 500000),
        # table_num=100: [5, 50, 500, 5000, 25000, 50000, 250000]
        (100, 5), (100, 50), (100, 500), (100, 5000), (100, 25000), (100, 50000), (100, 250000),
    ]

    # 预热
    logging.info("算子预热")
    cust_time = []
    for table_num, local_batch_size in test_cases:
        for _ in range(repeats):
            ignore_res = run_npu_performance_test(
                table_num=table_num,
                local_batch_size=local_batch_size,
                dtype=dtype,
                device_id=device_id
            )
            cust_time.append(ignore_res)
    logging.info(f"预热完成，平均耗时: {sum(cust_time)/len(cust_time):.2f} us")

    # 打印标题和表头
    logging.info(
        "=== npu dynamic_emb_op.get_new_length_and_offsets_op性能测试 ===")
    logging.info(f"{'table_num':>12} {'local_batch_size':>18} {'平均耗时(us)':>15}")
    logging.info("-" * 50)

    total_times = []

    # 遍历所有测试用例
    for table_num, local_batch_size in test_cases:
        avg_time_us = 0.0

        # 多次重复测试平均值
        for _ in range(repeats):
            single_time = run_npu_performance_test(
                table_num=table_num,
                local_batch_size=local_batch_size,
                dtype=dtype,
                device_id=device_id
            )
            avg_time_us += single_time

        # 计算平均耗时
        avg_time_us /= repeats
        total_times.append(avg_time_us)

        # 打印单例结果（格式对齐）
        logging.info(f"{table_num:>12} {local_batch_size:>18} {avg_time_us:>15.2f}")

    # 统计汇总结果
    logging.info("-" * 50)
    logging.info("平均耗时: %.2f us", sum(total_times) / len(total_times))
    logging.info("耗时范围: %.2f ~ %.2f us", min(total_times), max(total_times))


if __name__ == "__main__":
    # 前置检查
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        logging.error("未检测到NPU，请确保在支持NPU的环境中运行此测试。")
        exit(1)
    main()
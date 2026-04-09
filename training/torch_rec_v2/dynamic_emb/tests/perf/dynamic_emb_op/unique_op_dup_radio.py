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


def run_npu_performance_test(origin_key_num, dup_ratio_target, dtype, device_id=0):
    """运行单个npu dynamic_emb_op.unique_op性能测试用例

    dup_ratio_target: 目标重复率（0.0, 0.25, 0.5, 0.75）
    返回：(internal_us, e2e_us, dup_ratio_real)
    - internal_us: C++ unique_op_with_time 返回的内部耗时（大 key 场景为 _unique2 路径）
    - e2e_us: Python 端到端计时（包含调度及同步）
    - dup_ratio_real: 实际重复率 1 - unique_num / key_size
    """
    # 1. 设置npu设备
    torch.npu.set_device(device_id)

    # 2. 在 CPU 上按目标重复率构造数据，再搬到 NPU
    # 目标唯一数 = key_size * (1 - dup_ratio_target)
    unique_num = max(1, int(round(origin_key_num * (1.0 - dup_ratio_target))))
    # 用连续整数作为唯一 key 值
    keys_unique = torch.arange(unique_num, dtype=dtype)

    keys = torch.empty(origin_key_num, dtype=dtype)
    # 先放入所有唯一值，保证 unique_num 个不重复
    keys[:unique_num] = keys_unique
    # 剩余位置用这些唯一值的随机重复填充
    if origin_key_num > unique_num:
        idx = torch.randint(0, unique_num, (origin_key_num - unique_num,), dtype=torch.long)
        keys[unique_num:] = keys_unique[idx]
    # 打乱顺序
    perm = torch.randperm(origin_key_num)
    keys = keys[perm]

    tensor_npu = keys.to(f"npu:{device_id}")
    torch.npu.synchronize(device_id)  # 确保所有npu任务执行完毕

    # 2.1 实际 key 重复率（用于打印）
    unique_real = torch.unique(keys).numel()
    dup_ratio_real = 1.0 - float(unique_real) / float(origin_key_num)

    # 3. C++ 侧内部耗时（unique_op_with_time）
    start_npu = time.perf_counter()
    _, _, _, internal_time_us = dynamic_emb_extensions.unique_op_with_time(tensor_npu)

    # 4. Python 端到端性能（调用 unique_op）
    torch.npu.synchronize(device_id)  # 确保所有npu任务执行完毕
    end_npu = time.perf_counter()
    e2e_time_us = (end_npu - start_npu) * 1_000_000

    return float(internal_time_us), float(e2e_time_us), dup_ratio_real


def main():
    """npu dynamic_emb_op.unique_op性能测试（推荐业务用例）"""
    # 测试基础配置
    dtype = torch.int64
    device_id = 2   # 对应原测试的npu:1

    # 关注的 key 大小
    key_sizes = [
        100, 1000, 10000, 100000,
        1000000, 5000000, 10000000,
        50000000, 100000000,
    ]
    # 目标重复率：0%、25%、50%、75%
    dup_ratios_target = [0.0, 0.25, 0.5, 0.75]

    # 打印标题和表头
    logging.info("=== npu dynamic_emb_op.unique_op性能测试（不同重复率）===")
    logging.info("耗时来源: C++ unique_op_with_time（internal） + Python 端到端计时（e2e）")
    logging.info("%12s %10s %18s %18s %15s",
                 "key size", "dup(%)", "internal(us)", "e2e(us)", "dup_real")
    logging.info("-" * 150)

    # 预热
    for key_size in key_sizes:
        for dup_target in dup_ratios_target:
            internal, e2e, dup = run_npu_performance_test(
                origin_key_num=key_size,
                dup_ratio_target=dup_target,
                dtype=dtype,
                device_id=device_id,
            )
            logging.info("%18.2f %18.2f %15.4f",
                internal, e2e, dup)

    repeats = 10
    # 遍历所有 key_size 和重复率组合
    for key_size in key_sizes:
        for dup_target in dup_ratios_target:
            one_internal = 0.0
            one_e2e = 0.0
            one_dup = 0.0
            for _ in range(repeats):
                internal_us, e2e_us, dup_real = run_npu_performance_test(
                    origin_key_num=key_size,
                    dup_ratio_target=dup_target,
                    dtype=dtype,
                    device_id=device_id,
                )
                one_internal += internal_us
                one_e2e += e2e_us
                one_dup += dup_real
            avg_internal = one_internal / repeats
            avg_e2e = one_e2e / repeats
            avg_dup_real = one_dup / repeats

            # 打印单例结果（格式对齐）
            logging.info("%12d %10.0f %18.2f %18.2f %15.4f",
                         key_size, dup_target * 100.0,
                         avg_internal, avg_e2e, avg_dup_real)

    logging.info("-" * 150)

if __name__ == "__main__":
    main()
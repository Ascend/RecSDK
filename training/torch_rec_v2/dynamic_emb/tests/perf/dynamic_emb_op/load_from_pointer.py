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
import pandas as pd
import torch
import dynamic_emb_extensions as demb


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    # 取出二维张量的维度
    n, m = x_2d.size()
    # 计算每个元素的字节大小
    elem_size = x_2d.element_size()
    # 计算每行的字节偏移量
    row_stride = m * elem_size
    # 计算指针数组
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


def test_load_from_pointer_perf(device, n, dim):
    """测试 load_from_pointer 算子性能"""
    torch.npu.set_device(device)

    # 数据准备
    # src: 数据源 (n, dim)
    src = torch.randn(n, dim, dtype=torch.float32, device=f'npu:{device}')
    # pointers: 计算源数据的指针
    pointers = get_dim_pointers_optimized(src)
    # dst: 目标张量，用于接收数据
    dst = torch.empty_like(src)

    # 确保准备工作完成
    torch.npu.synchronize(device)

    elapsed_time = None
    # 记录开始时间
    start_time = time.perf_counter()
    # 执行算子
    demb.load_from_pointer(pointers, dst)
    # NPU同步，确保算子执行完毕后停止计时
    torch.npu.synchronize(device)
    # 记录结束时间
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1e6 # 转换为微秒 (us)

    return elapsed_time


def main():
    """主函数"""
    cur_device = 0

    warmup_iter_num = 10 # 预热次数
    # 预热阶段
    logging.info("Starting Warm up...")
    t_warm = None
    for _ in range(warmup_iter_num):
        t_warm = test_load_from_pointer_perf(cur_device, n=128, dim=128)
    logging.info(f"Warm up finished. Time: {t_warm:.6f} us")

    # 定义测试参数范围
    n_list = [100, 1000, 10000, 100000, 1000000, 10000000]
    dim_list = [8, 128, 512]

    test_cases = []
    for dim in dim_list:
        for n in n_list:
            if n == 10000000 and dim == 512:
                continue

            test_cases.append((cur_device, n, dim))

    logging.info(f"Total test cases: {len(test_cases)}")

    # 开始测试
    results = []
    repeats = 10

    for i, (device, n, dim) in enumerate(test_cases):
        time_list = []
        for _ in range(repeats):
            time_list.append(test_load_from_pointer_perf(device, n, dim))
        results.append({
            'n': n,
            'dim': dim,
            'Time Average (us)': sum(time_list) / len(time_list),
        })

    # 导出结果
    df = pd.DataFrame(results)
    output_filename = "load_from_pointer_perf_results.txt"
    logging.info(df)
    df.to_csv(output_filename, index=False, sep='\t')

    logging.info(f"Performance test finished. Results saved to {output_filename}")


if __name__ == "__main__":
    main()

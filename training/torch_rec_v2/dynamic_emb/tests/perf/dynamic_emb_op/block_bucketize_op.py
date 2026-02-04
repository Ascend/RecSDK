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

import torch
import torch_npu

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


def run_npu_performance_test(cfg, device_id=0):
    """运行单个block_bucketsize NPU性能测试用例"""
    # 1. 设置NPU设备
    torch.npu.set_device(device_id)
    torch.manual_seed(789)  # 固定随机种子保证可复现
    
    # 2. 生成测试数据
    lengths = torch.randint(*cfg["lengths_range"], (cfg["lengths_count"],), dtype=torch.int64)
    total_indices = lengths.sum().item()
    indices = torch.randint(*cfg["indices_range"], (total_indices,), dtype=torch.int64)
    dist_type_per_feature = torch.tensor([0, 1] * (cfg["T"] // 2) + [0] * (cfg["T"] % 2), dtype=torch.int64)
    block_sizes = torch.tensor([3360] * cfg["T"], dtype=torch.int64)
    bucketize_pos = True
    sequence = True
    weights = torch.randn(total_indices, dtype=torch.float32)
    
    # 3. 数据转到NPU
    lengths_npu = lengths.to(f"npu:{device_id}", dtype=torch.int64)
    indices_npu = indices.to(f"npu:{device_id}", dtype=torch.int64)
    dist_type_npu = dist_type_per_feature.to(f"npu:{device_id}", dtype=torch.int64)
    block_sizes_npu = block_sizes.to(f"npu:{device_id}", dtype=torch.int64)
    weights_npu = weights.to(f"npu:{device_id}", dtype=torch.float32)
    
    # 4. 测试NPU自定义算子耗时（核心逻辑）
    torch.npu.synchronize(device_id)
    start_npu = time.perf_counter()
    _ = dynamic_emb_extensions.block_bucketize_sparse_features(
        lengths_npu, indices_npu, bucketize_pos, sequence,
        dist_type_npu, block_sizes_npu, cfg["my_size"], weights_npu
    )
    torch.npu.synchronize(device_id)  # 确保所有NPU任务执行完毕
    end_npu = time.perf_counter()
    
    npu_time_us = (end_npu - start_npu) * 1_000_000
    return total_indices, npu_time_us


def main():
    """NPU侧block_bucketsize性能测试主函数"""
    # 测试配置
    device_id = 4  # 测试用NPU设备ID
    test_configs = [
        {
            "name": "307200_indices",
            "T": 10, "B": 1024, "my_size": 8,
            "lengths_count": 10240, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "1536000_indices",
            "T": 50, "B": 1024, "my_size": 8,
            "lengths_count": 51200, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "3072000_indices",
            "T": 100, "B": 1024, "my_size": 8,
            "lengths_count": 102400, "lengths_range": (30, 31),
            "indices_range": (0, 10), 
            "iterations": 3
        },
        {
            "name": "6144000_indices",
            "T": 200, "B": 1024, "my_size": 8,
            "lengths_count": 203800, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "9216000_indices",
            "T": 300, "B": 1024, "my_size": 8,
            "lengths_count": 307200, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "15360000_indices",
            "T": 500, "B": 1024, "my_size": 8,
            "lengths_count": 512000, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "3072000_indices",
            "T": 10, "B": 10240, "my_size": 8,
            "lengths_count": 102400, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "15360000_indices",
            "T": 50, "B": 10240, "my_size": 8,
            "lengths_count": 512000, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "30720000_indices",
            "T": 100, "B": 10240, "my_size": 8,
            "lengths_count": 1024000, "lengths_range": (30, 31),
            "indices_range": (0, 10), 
            "iterations": 3
        },
        {
            "name": "61440000_indices",
            "T": 200, "B": 10240, "my_size": 8,
            "lengths_count": 2038000, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "92160000_indices",
            "T": 300, "B": 10240, "my_size": 8,
            "lengths_count": 3072000, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
        {
            "name": "153600000_indices",
            "T": 500, "B": 10240, "my_size": 8,
            "lengths_count": 5120000, "lengths_range": (30, 31),
            "indices_range": (0, 10),  
            "iterations": 3
        },
    ]
    
    # 打印标题和表头（新增T/B/length_count列）
    logging.info("=== npu dynamic_emb_op.block_bucketize_op performance testing ===")
    logging.info("%14s %4s %6s %16s %12s %12s", "case", "T", "B", "lengths_count", "indices_count", "NPU(us)")
    logging.info("-" * 70)

    # 预热
    for cfg in test_configs:
        total_indices, npu_us = run_npu_performance_test(cfg, device_id)

    # 遍历所有测试用例（打印T/B/length_count）
    npu_times = []
    for cfg in test_configs:
        costs = []
        total_indices = 0
        for _ in range(cfg["iterations"]):
            total_indices, npu_us = run_npu_performance_test(cfg, device_id)
            costs.append(npu_us)
        avg_npu_time_us = sum(costs) / cfg["iterations"]

        npu_times.append(avg_npu_time_us)
        logging.info("%18s %4D %6D %16d %12d %12.2f", cfg['name'], cfg['T'], cfg['B'], cfg['lengths_count'],
                     total_indices, avg_npu_time_us)
    
    # 统计汇总
    logging.info("-" * 70)
    logging.info("NPU Average Response Time: %.2f us", sum(npu_times) / len(npu_times))
    logging.info("  NPU Response Time Range: %.2f ~ %.2f us", min(npu_times), max(npu_times))

if __name__ == "__main__":
    # 检查NPU可用性
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logging.info("Error: NPU device not detected or torch_npu not installed！")
        exit(1)
    main()
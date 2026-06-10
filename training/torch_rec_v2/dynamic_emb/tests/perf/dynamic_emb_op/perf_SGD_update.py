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
import csv

import torch

import dynamic_emb_extensions as demb

logging.basicConfig(level=logging.NOTSET)


class OptimizerParams:
    """优化器参数
    lr: 学习率
    """

    def __init__(self, lr):
        self.lr = lr


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n, m = x_2d.size()
    elem_size = x_2d.element_size()
    row_stride = m * elem_size
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


def test_dynamic_emb_SGD_with_pointer(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 SGD pointer 更新算子"""
    lr = optimizer_params.lr

    torch.npu.set_device(device)
    values = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}").contiguous()
    torch.npu.synchronize(device)

    val_pointers = get_dim_pointers_optimized(values)

    elapsed_time = []
    for _ in range(iter_num_max):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.dynamic_emb_sgd_with_pointer(
            grads,
            val_pointers,
            demb.DynamicEmbDataType.Float32,
            lr,
        )
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "dynamic_emb_sgd_with_pointer",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "Time Average (us)": time_average,
        "Time Max (us)": time_max,
        "Time Min (us)": time_min,
    }


def test_dynamic_emb_SGD_with_table(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 SGD table 更新算子"""
    lr = optimizer_params.lr

    torch.npu.set_device(device)
    values = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}").contiguous()
    keys = torch.arange(1, batch_size + 1, dtype=torch.int64, device=f"npu:{device}")

    # 保障向量池落在HBM，避免回落到host memory触发Unsupport报错。
    vector_capacity = max(2048, batch_size * 2)
    bytes_per_vector = embedding_dim * 4  # sgd 仅 params, float32
    max_hbm_for_vectors = max(1 * 1024 * 1024 * 1024, int(vector_capacity * bytes_per_vector * 1.1))

    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        embedding_dim,
        1024,
        vector_capacity,
        max_hbm_for_vectors,
        128,
        0.5,
        128,
        1024,
        device,
        False,
        False,
        0,
        1,
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.Null,
    )
    table.load(batch_size, keys, values, None, True, False)
    torch.npu.synchronize(device)

    elapsed_time = []
    for _ in range(iter_num_max):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.dynamic_emb_sgd_with_table(
            table,
            batch_size,
            keys,
            grads,
            lr,
            demb.DynamicEmbDataType.Float32,
        )
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "dynamic_emb_sgd_with_table",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "Time Average (us)": time_average,
        "Time Max (us)": time_max,
        "Time Min (us)": time_min,
    }


def test_dynamic_emb_SGD_fused(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 SGD fused 更新算子"""
    lr = optimizer_params.lr

    torch.npu.set_device(device)
    values = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}").contiguous()
    torch.npu.synchronize(device)

    elapsed_time = []
    for _ in range(iter_num_max):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.dynamic_emb_sgd_fused(grads, values, lr)
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "dynamic_emb_sgd_fused",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "Time Average (us)": time_average,
        "Time Max (us)": time_max,
        "Time Min (us)": time_min,
    }


def main():
    """主函数"""
    cur_device = 0
    iter_num_max = 10

    logging.info("warm up")
    warmup_iter_num = 10
    for _ in range(warmup_iter_num):
        test_dynamic_emb_SGD_with_pointer(cur_device, 1, 128, OptimizerParams(0.001), iter_num_max)
        test_dynamic_emb_SGD_with_table(cur_device, 1, 128, OptimizerParams(0.001), iter_num_max)
        test_dynamic_emb_SGD_fused(cur_device, 1, 128, OptimizerParams(0.001), iter_num_max)

    batch_size_list = [32, 128, 512, 1024, 4096, 8192, 10240, 102400, 1024000]
    embedding_dim_list = [16, 32, 128, 512, 1024, 4096, 8192]
    lr = 0.001

    test_cases = []
    for batch_size in batch_size_list:
        for embedding_dim in embedding_dim_list:
            if batch_size * embedding_dim >= 1024000 * 4096:
                continue
            test_cases.append((cur_device, batch_size, embedding_dim, OptimizerParams(lr)))

    logging.info("start test")
    repeats = 20
    results = []
    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        result = None
        for _ in range(repeats):
            result = test_dynamic_emb_SGD_with_pointer(
                device, batch_size, embedding_dim, optimizer_params, iter_num_max
            )
        results.append(result)

    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        if batch_size in (1024000, 102400):
            continue
        if batch_size * embedding_dim >= 10240 * 4096:
            continue
        result = None
        for _ in range(repeats):
            result = test_dynamic_emb_SGD_with_table(device, batch_size, embedding_dim, optimizer_params, iter_num_max)
        results.append(result)

    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        result = None
        for _ in range(repeats):
            result = test_dynamic_emb_SGD_fused(device, batch_size, embedding_dim, optimizer_params, iter_num_max)
        results.append(result)

    excel_filename = "SGD_update_time_results.csv"
    if not results:
        logging.warning("results is empty, skip writing csv.")
        return
    fields = list(results[0].keys())
    with open(excel_filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for result in results:
            writer.writerow([result.get(field, "") for field in fields])
    logging.info("测试结果已导出到CSV文件: %s", excel_filename)


if __name__ == "__main__":
    main()

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
import os
import csv

import torch

import dynamic_emb_extensions as demb

logging.basicConfig(level=logging.NOTSET)


def write_results_to_excel_row_by_row(results, excel_filename):
    if not results:
        logging.warning("results is empty, skip writing excel.")
        return

    fields = list(results[0].keys())
    if os.path.exists(excel_filename):
        os.remove(excel_filename)

    with open(excel_filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for result in results:
            row = [result.get(field, "") for field in fields]
            writer.writerow(row)


class OptimizerParams:
    '''优化器参数
    lr: 学习率
    beta1: 一阶衰减系数
    beta2: 二阶衰减系数
    eps: 小常数，防止除零
    weight_decay: 权重衰减系数
    '''

    def __init__(self, lr, beta1, beta2, eps, weight_decay):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n, m = x_2d.size()
    elem_size = x_2d.element_size()
    row_stride = m * elem_size
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


def test_dynamic_emb_AdamW_with_pointer(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 AdamW pointer 更新算子"""
    lr = optimizer_params.lr
    beta1 = optimizer_params.beta1
    beta2 = optimizer_params.beta2
    eps = optimizer_params.eps
    weight_decay = optimizer_params.weight_decay

    torch.npu.set_device(device)
    params = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
    m = torch.zeros_like(params, dtype=torch.float32)
    v = torch.zeros_like(params, dtype=torch.float32)
    values = torch.cat([params, m, v], dim=1).contiguous()
    torch.npu.synchronize(device)

    val_pointers = get_dim_pointers_optimized(values)

    elapsed_time = []
    for _ in range(iter_num_max):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.dynamic_emb_adamW_with_pointer(
            grads,
            val_pointers,
            demb.DynamicEmbDataType.Float32,
            embedding_dim * 2,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            iter_num_max,
        )
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "dynamic_emb_adamW_with_pointer",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
        "Time Average (us)": time_average,
        "Time Max (us)": time_max,
        "Time Min (us)": time_min,
    }


def test_dynamic_emb_AdamW_with_table(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 AdamW table 更新算子"""
    lr = optimizer_params.lr
    beta1 = optimizer_params.beta1
    beta2 = optimizer_params.beta2
    eps = optimizer_params.eps
    weight_decay = optimizer_params.weight_decay

    torch.npu.set_device(device)
    params = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
    m = torch.zeros_like(params, dtype=torch.float32)
    v = torch.zeros_like(params, dtype=torch.float32)
    values = torch.cat([params, m, v], dim=1).contiguous()
    keys = torch.arange(batch_size, dtype=torch.int64, device=f"npu:{device}")

    vector_capacity = max(2048, batch_size * 2)
    bytes_per_vector = embedding_dim * 3 * 4  # params + m + v, float32
    max_hbm_for_vectors = max(1 * 1024 * 1024 * 1024, int(vector_capacity * bytes_per_vector * 1.1))

    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        embedding_dim * 3,
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
        demb.dynamic_emb_adamW_with_table(
            table,
            batch_size,
            keys,
            grads,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            iter_num_max,
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
        "Operator": "dynamic_emb_adamW_with_table",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
        "Time Average (us)": time_average,
        "Time Max (us)": time_max,
        "Time Min (us)": time_min,
    }


def test_dynamic_emb_AdamW_fused(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 AdamW fused 更新算子"""
    lr = optimizer_params.lr
    beta1 = optimizer_params.beta1
    beta2 = optimizer_params.beta2
    eps = optimizer_params.eps
    weight_decay = optimizer_params.weight_decay

    torch.npu.set_device(device)
    params = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
    m = torch.zeros_like(params, dtype=torch.float32)
    v = torch.zeros_like(params, dtype=torch.float32)
    values = torch.cat([params, m, v], dim=1).contiguous()
    torch.npu.synchronize(device)

    elapsed_time = []
    for _ in range(iter_num_max):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.dynamic_emb_adamW_fused(grads, values, lr, beta1, beta2, eps, weight_decay, iter_num_max)
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "dynamic_emb_adamW_fused",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
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
        test_dynamic_emb_AdamW_with_pointer(
            cur_device, 1, 128, OptimizerParams(0.001, 0.9, 0.999, 1e-8, 0.0), iter_num_max
        )
        test_dynamic_emb_AdamW_with_table(
            cur_device, 1, 128, OptimizerParams(0.001, 0.9, 0.999, 1e-8, 0.0), iter_num_max
        )
        test_dynamic_emb_AdamW_fused(cur_device, 1, 128, OptimizerParams(0.001, 0.9, 0.999, 1e-8, 0.0), iter_num_max)

    batch_size_list = [32, 128, 512, 1024, 4096, 8192, 10240, 102400, 1024000]
    embedding_dim_list = [16, 32, 128, 512, 1024, 4096, 8192]
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.0

    test_cases = []
    for batch_size in batch_size_list:
        for embedding_dim in embedding_dim_list:
            if batch_size * embedding_dim >= 1024000 * 4096:
                continue
            test_cases.append(
                (cur_device, batch_size, embedding_dim, OptimizerParams(lr, beta1, beta2, eps, weight_decay))
            )

    logging.info("start test")
    repeats = 20
    results = []
    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        result = None
        for _ in range(repeats):
            result = test_dynamic_emb_AdamW_with_pointer(
                device, batch_size, embedding_dim, optimizer_params, iter_num_max
            )
        results.append(result)

    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        if batch_size in (1024000, 102400):
            continue
        if batch_size * embedding_dim >= 10240 * 4096:
            continue
        for _ in range(repeats):
            result = test_dynamic_emb_AdamW_with_table(
                device, batch_size, embedding_dim, optimizer_params, iter_num_max
            )
        results.append(result)

    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        for _ in range(repeats):
            result = test_dynamic_emb_AdamW_fused(device, batch_size, embedding_dim, optimizer_params, iter_num_max)
        results.append(result)

    excel_filename = "AdamW_update_time_results.csv"
    write_results_to_excel_row_by_row(results, excel_filename)
    logging.info("测试结果已导出到CSV文件: %s", excel_filename)


if __name__ == "__main__":
    main()

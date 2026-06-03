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

logging.basicConfig(level=logging.NOTSET)


class OptimizerParams:
    """优化器参数
    lr: 学习率
    eps: 小常数，防止除零
    """

    def __init__(self, lr, eps):
        self.lr = lr
        self.eps = eps


def get_rowwise_state_dim(dtype: torch.dtype = torch.float32) -> int:
    """与 csrc/dynamic_variable_base.h 中 RowWiseAdaGrad 定义一致：16 / sizeof(T)"""
    return 16 // torch.tensor([], dtype=dtype).element_size()


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n, m = x_2d.size()
    elem_size = x_2d.element_size()
    row_stride = m * elem_size
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


def test_dynamic_emb_rowwise_adagrad_with_pointer(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 Rowwise Adagrad pointer 更新算子"""
    lr = optimizer_params.lr
    eps = optimizer_params.eps
    state_dim = get_rowwise_state_dim()

    torch.npu.set_device(device)
    params = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=torch.float32, device=f"npu:{device}")
    values = torch.cat([params, state], dim=1).contiguous()
    torch.npu.synchronize(device)

    val_pointers = get_dim_pointers_optimized(values)

    elapsed_time = []
    for _ in range(iter_num_max):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.dynamic_emb_rowwise_adagrad_with_pointer(
            grads,
            val_pointers,
            demb.DynamicEmbDataType.Float32,
            state_dim,
            lr,
            eps,
        )
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "dynamic_emb_rowwise_adagrad_with_pointer",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "state_dim": state_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "eps": eps,
        "Average Time (us)": time_average,
        "Max Time (us)": time_max,
        "Min Time (us)": time_min,
    }


def test_dynamic_emb_rowwise_adagrad_with_table(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 Rowwise Adagrad table 更新算子"""
    lr = optimizer_params.lr
    eps = optimizer_params.eps
    state_dim = get_rowwise_state_dim()
    vector_dim = embedding_dim + state_dim

    torch.npu.set_device(device)
    params = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=torch.float32, device=f"npu:{device}")
    values = torch.cat([params, state], dim=1).contiguous()
    keys = torch.arange(batch_size, dtype=torch.int64, device=f"npu:{device}")

    vector_capacity = max(2048, batch_size * 2)
    bytes_per_vector = vector_dim * 4  # float32
    max_hbm_for_vectors = max(1 * 1024 * 1024 * 1024, int(vector_capacity * bytes_per_vector * 2))

    table = demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        vector_dim,
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
        demb.dynamic_emb_rowwise_adagrad_with_table(
            table,
            batch_size,
            keys,
            grads,
            lr,
            eps,
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
        "Operator": "dynamic_emb_rowwise_adagrad_with_table",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "state_dim": state_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "eps": eps,
        "Average Time (us)": time_average,
        "Max Time (us)": time_max,
        "Min Time (us)": time_min,
    }


def test_dynamic_emb_rowwise_adagrad_fused(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 Rowwise Adagrad fused 更新算子"""
    lr = optimizer_params.lr
    eps = optimizer_params.eps
    state_dim = get_rowwise_state_dim()

    torch.npu.set_device(device)
    params = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=torch.float32, device=f"npu:{device}")
    values = torch.cat([params, state], dim=1).contiguous()
    torch.npu.synchronize(device)

    elapsed_time = []
    for _ in range(iter_num_max):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.dynamic_emb_rowwise_adagrad_fused(grads, values, lr, eps)
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "dynamic_emb_rowwise_adagrad_fused",
        "Batch Size": batch_size,
        "embedding_dim": embedding_dim,
        "state_dim": state_dim,
        "grads_num": batch_size * embedding_dim,
        "lr": lr,
        "eps": eps,
        "Average Time (us)": time_average,
        "Max Time (us)": time_max,
        "Min Time (us)": time_min,
    }


def main():
    """主函数"""
    cur_device = 1

    logging.info("warm up")
    warmup_iter_num = 10
    warmup_repeats = 1
    warmup_optimizer_params = OptimizerParams(0.001, 1e-8)
    warmup_batch_size = 1
    warmup_embedding_dim = 128
    pointer_warmup_result = None
    table_warmup_result = None
    fused_warmup_result = None
    for warmup_idx in range(warmup_iter_num):
        pointer_warmup_result = test_dynamic_emb_rowwise_adagrad_with_pointer(
            cur_device, warmup_batch_size, warmup_embedding_dim, warmup_optimizer_params, warmup_repeats
        )
        table_warmup_result = test_dynamic_emb_rowwise_adagrad_with_table(
            cur_device, warmup_batch_size, warmup_embedding_dim, warmup_optimizer_params, warmup_repeats
        )
        fused_warmup_result = test_dynamic_emb_rowwise_adagrad_fused(
            cur_device, warmup_batch_size, warmup_embedding_dim, warmup_optimizer_params, warmup_repeats
        )
        if warmup_idx == warmup_iter_num - 1:
            logging.info(
                "Warm up finished. pointer: %.2f us, table: %.2f us, fused: %.2f us",
                pointer_warmup_result["Average Time (us)"],
                table_warmup_result["Average Time (us)"],
                fused_warmup_result["Average Time (us)"],
            )

    batch_size_list = [32, 128, 512, 1024, 4096, 8192, 10240, 102400, 1024000]
    embedding_dim_list = [16, 32, 128, 512, 1024, 4096, 8192]
    lr = 0.001
    eps = 1e-8

    test_cases = []
    for batch_size in batch_size_list:
        for embedding_dim in embedding_dim_list:
            if batch_size * embedding_dim >= 1024000 * 4096:
                continue
            test_cases.append((cur_device, batch_size, embedding_dim, OptimizerParams(lr, eps)))

    logging.info("start test")
    repeats = 20
    results = []
    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        results.append(
            test_dynamic_emb_rowwise_adagrad_with_pointer(device, batch_size, embedding_dim, optimizer_params, repeats)
        )
        results.append(
            test_dynamic_emb_rowwise_adagrad_with_table(device, batch_size, embedding_dim, optimizer_params, repeats)
        )
        results.append(
            test_dynamic_emb_rowwise_adagrad_fused(device, batch_size, embedding_dim, optimizer_params, repeats)
        )

    df = pd.DataFrame(results)
    output_filename = "Rowwise_Adagrad_update_time_results.txt"
    df.to_csv(output_filename, index=False, sep="\t")
    logging.info("测试结果已导出到文本文件: %s", output_filename)


if __name__ == "__main__":
    main()

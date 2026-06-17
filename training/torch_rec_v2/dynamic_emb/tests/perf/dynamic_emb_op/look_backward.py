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

import logging
import os
import csv
import time

import numpy as np
import torch

import dynamic_emb_extensions as demb

logging.basicConfig(level=logging.NOTSET)

BATCH_SIZES = [1024, 10240]
NUM_KEYS = [100, 1000, 10000]
EMBED_DIMS = [8, 128]
REPEAT_RATES = [0.0, 0.5, 1.0]


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


def generate_biased_offsets(total_features, batch_size, index_dtype, device):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than 0, got {batch_size}")
    if total_features < 0:
        raise ValueError(f"total_features must be non-negative, got {total_features}")

    base = total_features // batch_size
    remainder = total_features % batch_size
    offsets = [0]
    current = 0
    for i in range(batch_size):
        current += base + (1 if i < remainder else 0)
        offsets.append(current)
    if offsets[-1] != total_features:
        raise ValueError(f"offset mismatch: last={offsets[-1]}, total={total_features}")
    return torch.tensor(offsets, dtype=index_dtype, device=device)


def generate_test_data(device, batch_size, embed_dim, num_key, repeat_rate, combiner=1):
    grad_dtype = torch.float32
    index_dtype = torch.int64
    feature_num = 1
    np.random.seed(42)

    unique_count = max(1, int(num_key / (1.0 + repeat_rate)))
    inverse_indices = torch.tensor(
        np.random.randint(0, unique_count, num_key),
        dtype=index_dtype,
        device=f"npu:{device}",
    )
    grad_for_table = torch.randn((batch_size, embed_dim), dtype=grad_dtype, device=f"npu:{device}")
    unique_indices = torch.arange(unique_count, dtype=index_dtype, device=f"npu:{device}")

    biased_offsets = generate_biased_offsets(num_key, batch_size, index_dtype, f"npu:{device}")

    return {
        "grad_for_table": grad_for_table,
        "inverse_indices": inverse_indices,
        "unique_indices": unique_indices,
        "offsets_per_table": biased_offsets.clone(),
        "embed_dim": embed_dim,
        "table_num": 1,
        "batch_size": batch_size,
        "feature_num": feature_num,
        "unique_count": unique_count,
        "combiner": combiner,
        "repeat_rate": repeat_rate,
    }


def test_lookup_backward(device, batch_size, embed_dim, num_key, repeat_rate, combiner, iter_num_max):
    data = generate_test_data(device, batch_size, embed_dim, num_key, repeat_rate, combiner)
    torch.npu.set_device(device)
    unique_backward_grads = torch.zeros(
        (data["unique_count"], data["embed_dim"]),
        dtype=torch.float32,
        device=f"npu:{device}",
    )
    torch.npu.synchronize(device)

    elapsed_time = []
    for _ in range(iter_num_max):
        unique_backward_grads.zero_()
        torch.npu.synchronize(device)
        start_time = time.perf_counter()
        demb.lookup_backward(
            data["grad_for_table"],
            unique_backward_grads,
            data["unique_indices"],
            data["inverse_indices"],
            data["offsets_per_table"],
            data["embed_dim"],
            data["table_num"],
            data["batch_size"],
            data["feature_num"],
            data["offsets_per_table"][-1].item(),
            combiner=combiner,
        )
        torch.npu.synchronize(device)
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6)

    time_average = sum(elapsed_time) / iter_num_max
    time_max = max(elapsed_time)
    time_min = min(elapsed_time)

    return {
        "Device": device,
        "Operator": "lookup_backward",
        "Batch Size": batch_size,
        "embedding_dim": embed_dim,
        "num_key": num_key,
        "unique_count": data["unique_count"],
        "repeat_rate": repeat_rate,
        "combiner": combiner,
        "Time Average (us)": time_average,
        "Time Max (us)": time_max,
        "Time Min (us)": time_min,
    }


def main():
    cur_device = 0
    iter_num_max = 10
    repeats = 20

    logging.info("warm up")
    warmup_iter_num = 10
    for _ in range(warmup_iter_num):
        test_lookup_backward(cur_device, 1024, 128, 100, 0.5, 1, iter_num_max)

    test_plan = []
    for batch_size in BATCH_SIZES:
        for num_key in NUM_KEYS:
            for embed_dim in EMBED_DIMS:
                for repeat_rate in REPEAT_RATES:
                    test_plan.append((batch_size, embed_dim, num_key, repeat_rate))

    logging.info("start test")
    results = []
    for batch_size, embed_dim, num_key, repeat_rate in test_plan:
        result = None
        for _ in range(repeats):
            result = test_lookup_backward(
                cur_device, batch_size, embed_dim, num_key, repeat_rate, combiner=1, iter_num_max=iter_num_max
            )
        if result is not None:
            results.append(result)
            logging.info(
                "batch_size=%d, embed_dim=%d, num_key=%d, repeat_rate=%.2f, avg=%.2f us",
                batch_size,
                embed_dim,
                num_key,
                repeat_rate,
                result["Time Average (us)"],
            )

    excel_filename = "look_backward_time_results.csv"
    write_results_to_excel_row_by_row(results, excel_filename)
    logging.info("测试结果已导出到CSV文件: %s", excel_filename)


if __name__ == "__main__":
    main()

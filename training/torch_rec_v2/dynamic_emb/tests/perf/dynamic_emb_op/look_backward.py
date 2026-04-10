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

import torch
import numpy as np
import logging
import dynamic_emb_extensions

logging.basicConfig(level=logging.INFO)
test_cases = [
    # (batch_size, num_key)
    (8, 100),
    (8, 1000),
    (8, 10000),
    (8, 100000),
    (8, 1000000),
    (8, 5000000),
    (8, 10000000),
    (128, 100),
    (128, 1000),
    (128, 10000),
    (128, 100000),
    (128, 1000000),
    (128, 5000000),
    (128, 10000000),
]
embed_dims = [8, 128]
repeat_rates = [0.0, 0.25, 0.5, 0.75, 1.0]


def generate_test_data(batch_size, embed_dim, num_key, repeat_rate):
    """生成指定配置的测试数据"""
    GRAD_DTYPE = torch.float32
    INDEX_DTYPE = torch.int64
    feature_num = 1
    combiner = 1
    device = 0
    np.random.seed(42)
    unique_count = max(1, int(num_key / (1.0 + repeat_rate)))
    inverse_indices = torch.tensor(
        np.random.randint(0, unique_count, num_key),
        dtype=INDEX_DTYPE,
        device=device,
    )
    grad_for_table = torch.randn((batch_size, embed_dim), dtype=GRAD_DTYPE, device=device)
    unique_indices = torch.arange(unique_count, dtype=INDEX_DTYPE, device=device)

    def generate_biased_offsets(total_features, batch_size, index_dtype, device):
        if batch_size <= 0:
            raise ValueError(f"batch_size必须大于0，当前值：{batch_size}")
        if total_features < batch_size:
            return None

        base = total_features // batch_size  # 每个样本基础特征数
        remainder = total_features % batch_size  # 剩余特征数

        offsets = [0]
        current = 0
        for i in range(batch_size):
            current += base + (1 if i < remainder else 0)
            offsets.append(current)
        if offsets[-1] != total_features:
            raise ValueError(f"偏移量计算错误：最后一个偏移量{offsets[-1]} != 总特征数{total_features}")
        biased_offsets = torch.tensor(offsets, dtype=index_dtype, device=device)
        return biased_offsets

    biased_offsets_list = generate_biased_offsets(num_key, batch_size, INDEX_DTYPE, device)
    if biased_offsets_list is None:
        return None
    offsets_list_per_table = biased_offsets_list.clone()

    return {
        "grad_for_table": grad_for_table,
        "inverse_indices": inverse_indices,
        "biased_offsets": biased_offsets_list,
        "unique_indices": unique_indices,
        "offsets_per_table": offsets_list_per_table,
        "embed_dim": embed_dim,
        "table_num": 1,
        "batch_size": batch_size,
        "feature_num": feature_num,
        "unique_count": unique_count,
        "combiner": combiner,
        "device": device,
        "grad_dtype": GRAD_DTYPE,
        "index_dtype": INDEX_DTYPE,
        "repeat_rate": repeat_rate,
    }


def run_perfomance_test(data, combiner=1, device=0):
    unique_backward_grads_native = torch.zeros(
        (data["unique_count"], data["embed_dim"]), dtype=data["grad_dtype"], device=device
    )
    import time

    start_npu = time.perf_counter()
    dynamic_emb_extensions.lookup_backward(
        data["grad_for_table"],
        unique_backward_grads_native,
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
    end_npu = time.perf_counter()
    cost_time = end_npu - start_npu
    return cost_time * 100000


def main():
    # 打印标题和表头
    logging.info("=== npu dynamic_emb_op.unique_op性能测试（推荐业务用例）===")
    logging.info("%12s %10s %12s %12s %15s", "batch_size", "embed_dim", "num_key", "repeat_rate", "used time(us)")
    logging.info("-" * 150)
    repeat = 3
    # 预热
    for batch_size, num_key in test_cases:
        for embed_dim in embed_dims:
            for repeat_rate in repeat_rates:
                data = generate_test_data(batch_size, embed_dim, num_key, repeat_rate)
                if data is None:
                    continue
                _ = run_perfomance_test(data)
    npu_times = []
    for batch_size, num_key in test_cases:
        for embed_dim in embed_dims:
            for repeat_rate in repeat_rates:
                costs = []
                data = generate_test_data(batch_size, embed_dim, num_key, repeat_rate)
                if data is None:
                    continue

                for _ in range(repeat):
                    cost_time = run_perfomance_test(data)
                    costs.append(cost_time)
                avg = sum(costs) / len(costs)
                logging.info(
                    "batch_size:%4d, embed_dim:%4d, num_key:%8d, repeat_rate:%.2f",
                    batch_size,
                    embed_dim,
                    num_key,
                    repeat_rate,
                )
                logging.info("avg:%8f", avg)
                logging.info("-" * 70)
                npu_times.append(avg)

    logging.info("-" * 90)
    logging.info("look_backward time range: %2f (us)~ %2f (us) ", min(npu_times), max(npu_times))
    logging.info("avgerage time: %2f (us)", sum(npu_times) / len(npu_times))
    logging.info("-" * 90)


if __name__ == "__main__":
    main()
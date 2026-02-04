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

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


class ReduceGradPerf:
    def __init__(self, origin_key_num, range_min, range_max, dtype, device_id, dim):
        self.origin_key_num = origin_key_num
        self.range_min = range_min
        self.range_max = range_max
        self.dtype = dtype
        self.device_id = device_id
        self.dim = dim

    def test(self):
        indices = torch.randint(self.range_min, self.range_max, (self.origin_key_num,), dtype=self.dtype)
        grads = torch.randn(indices.size(0), self.dim, dtype=torch.float32)

        unique_indices, inverse = torch.unique(indices, sorted=False, return_inverse=True)

        dev = f'npu:{self.device_id}'
        dev_grads = grads.to(dev)
        dev_unique_indices = unique_indices.to(dev)
        dev_inverse = inverse.to(dev)

        # warm up
        for _ in range(10):
            dynamic_emb_extensions.reduce_grads(dev_grads, dev_unique_indices, dev_inverse)

        # test
        total_time = 0
        repeat = 10
        for _ in range(repeat):
            torch.npu.synchronize()
            start = time.perf_counter()
            dynamic_emb_extensions.reduce_grads(dev_grads, dev_unique_indices, dev_inverse)
            torch.npu.synchronize()
            end = time.perf_counter()
            total_time += end - start

        return total_time * 1000 * 1000 / repeat


def main():
    """npu dynamic_emb_op.reduce_grad性能测试（推荐业务用例）"""
    # 测试基础配置
    dtype = torch.int64
    device_id = 0
    test_cases = [
        (100, 1, 1001),
        (100, 1, 100001),
        (100, 1, 1000000001),
        (1000, 1, 1001),
        (1000, 1, 100001),
        (1000, 1, 1000000001),
        (10000, 1, 1001),
        (10000, 1, 100001),
        (10000, 1, 1000000001),
        (100000, 1, 1001),
        (100000, 1, 100001),
        (100000, 1, 1000000001),
        (1000000, 1, 1001),
        (1000000, 1, 100001),
        (1000000, 1, 1000000001),
        (5000000, 1, 1001),
        (5000000, 1, 100001),
        (5000000, 1, 1000000001),
        (10000000, 1, 1001),
        (10000000, 1, 100001),
        (10000000, 1, 1000000001),
    ]

    # 打印标题和表头
    logging.info("=== npu dynamic_emb_op.reduce_grad性能测试（推荐业务用例）===")
    logging.info(f"{'key_num':>15}|{'range_min':>15}|{'range_max':>15}|{'dim':>8}|{'avg_latency':>8}")
    logging.info("-" * 150)

    times = []
    for origin_key_num, range_min, range_max in test_cases:
        for dim in [8, 128]:
            perf = ReduceGradPerf(origin_key_num, range_min, range_max, dtype, device_id, dim)
            t = perf.test()
            times.append(t)

            logging.info(f"{origin_key_num:>15}|{range_min:>15}|{range_max:>15}|{dim:>8}|{t:>8.2f}")

    # 统计汇总结果
    logging.info("-" * 150)
    logging.info("平均耗时: %.2f us", sum(times) / len(times))
    logging.info("耗时范围: %.2f ~ %.2f us", min(times), max(times))

if __name__ == "__main__":
    main()
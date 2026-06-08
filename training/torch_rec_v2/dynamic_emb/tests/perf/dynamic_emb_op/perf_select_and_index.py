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

import argparse
import csv
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

logging.basicConfig(level=logging.WARNING)
try:
    torch.npu.init()
except Exception as e:
    logging.error("Failed to initialize NPU: %s", e)
    sys.exit(1)

try:
    from dynamic_emb_extensions import select, select_index
except Exception as e:
    logging.error("Failed to import dynamic_emb_extensions: %s", e)
    sys.exit(1)

device_id = 0
device = torch.device(f"npu:{device_id}")
torch.npu.set_device(device_id)


@dataclass
class BenchRow:
    op: str
    num_items: int
    true_ratio: float
    dtype: str
    time_us_median: float
    time_us_mean: float
    warmup_iters: int
    bench_iters: int


def parse_dtype(name: str) -> torch.dtype:
    if name == "uint64":
        return torch.uint64
    if name == "int64":
        return torch.int64
    raise ValueError(f"Unsupported dtype '{name}'; select ops only support int64 and uint64")


def build_tensors(
    num_items: int,
    true_ratio: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k = int(num_items * true_ratio)
    flags = torch.zeros(num_items, device=device, dtype=torch.bool)
    if k > 0:
        perm = torch.randperm(num_items, device=device, dtype=torch.int64)[:k]
        flags[perm] = True

    inputs = torch.arange(num_items, device=device, dtype=torch.int64)
    if dtype == torch.uint64:
        inputs = inputs.to(torch.uint64)

    outputs = torch.empty_like(inputs)
    output_indices = torch.empty(num_items, device=device, dtype=dtype)
    num_selected = torch.empty(1, device=device, dtype=torch.int64)
    return flags, inputs, outputs, output_indices, num_selected


def time_npu_kernel(fn: Callable[[], None], warmup_iters: int, iters: int) -> List[float]:
    times_us: List[float] = []

    for _ in range(warmup_iters):
        fn()

    torch.npu.synchronize()

    for _ in range(iters):
        torch.npu.synchronize()
        ts_start = time.time_ns()
        fn()
        torch.npu.synchronize()
        ts_end = time.time_ns()
        times_us.append((ts_end - ts_start) / 1000)  # ns -> us
    return times_us


def bench_select(
    flags: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    num_selected: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
) -> Tuple[float, float]:
    def run():
        select(flags, inputs, outputs, num_selected)

    t = time_npu_kernel(run, warmup_iters, bench_iters)
    return statistics.median(t), statistics.mean(t)


def bench_select_index(
    flags: torch.Tensor,
    output_indices: torch.Tensor,
    num_selected: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
) -> Tuple[float, float]:
    def run():
        select_index(flags, output_indices, num_selected)

    t = time_npu_kernel(run, warmup_iters, bench_iters)
    return statistics.median(t), statistics.mean(t)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def write_results_csv(path: str, rows: List[BenchRow]) -> None:
    fieldnames = [
        "op",
        "num_items",
        "true_ratio",
        "dtype",
        "time_us_median",
        "time_us_mean",
        "warmup_iters",
        "bench_iters",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "op": r.op,
                    "num_items": r.num_items,
                    "true_ratio": r.true_ratio,
                    "dtype": r.dtype,
                    "time_us_median": r.time_us_median,
                    "time_us_mean": r.time_us_mean,
                    "warmup_iters": r.warmup_iters,
                    "bench_iters": r.bench_iters,
                }
            )


def main():
    p = argparse.ArgumentParser(description="Performance test for select and select_index operations")
    p.add_argument("--size", type=str, default="1024,10240,102400", help="Comma-separated list of tensor sizes")
    p.add_argument("--true_ratio", type=str, default="0.01,0.1,0.5,0.9", help="Comma-separated list of true ratios")
    p.add_argument("--warmup_iters", type=int, default=20, help="Number of warmup iterations")
    p.add_argument("--bench_iters", type=int, default=50, help="Number of benchmark iterations")
    p.add_argument(
        "--dtype", type=str, default="int64", choices=["int64", "uint64"], help="Data type for inputs/indices"
    )
    p.add_argument("--csv", type=str, default="select_and_index_results.csv", metavar="PATH", help="Output CSV path")
    p.add_argument("--no-csv", action="store_true", help="Do not write CSV file")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    sizes = parse_int_list(args.size)
    true_ratios = parse_float_list(args.true_ratio)
    rows: List[BenchRow] = []

    torch.manual_seed(args.seed)

    for n in sizes:
        for tr in true_ratios:
            flags, inputs, outputs, output_indices, num_selected = build_tensors(n, tr, dtype)

            med, mean = bench_select(flags, inputs, outputs, num_selected, args.warmup_iters, args.bench_iters)
            rows.append(
                BenchRow(
                    op="select",
                    num_items=n,
                    true_ratio=tr,
                    dtype=args.dtype,
                    time_us_median=med,
                    time_us_mean=mean,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
            )

            med, mean = bench_select_index(flags, output_indices, num_selected, args.warmup_iters, args.bench_iters)
            rows.append(
                BenchRow(
                    op="select_index",
                    num_items=n,
                    true_ratio=tr,
                    dtype=args.dtype,
                    time_us_median=med,
                    time_us_mean=mean,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
            )

    if not args.no_csv:
        write_results_csv(args.csv, rows)
        print(f"Results written to {args.csv}", file=sys.stderr)

    header = f"{'op':<14}{'N':>12}{'true_ratio':>12}{'median_us':>14}{'mean_us':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.op:<14}{r.num_items:>12}{r.true_ratio:>12.2f}{r.time_us_median:>14.2f}{r.time_us_mean:>14.2f}")


if __name__ == "__main__":
    main()

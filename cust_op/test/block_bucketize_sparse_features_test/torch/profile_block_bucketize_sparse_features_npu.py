#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies
# Licensed under the Apache License, Version 2.0.

"""
NPU/GPU 侧性能 profiling 脚本：
- 预热 2 次，测量 32 次，取活跃阶段后半段 (active/2) 的平均值；
- 直接用 python 运行，不依赖 pytest：
    python profile_block_bucketize_sparse_features_npu.py --device npu
    python profile_block_bucketize_sparse_features_npu.py --device cuda
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sysconfig
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fbgemm_gpu
import torch
import torch.profiler as profiler

from block_bucketize_sparse_features_perf_cases import (
    GenTotalNumsBlocksType,
    PERF_CASES,
    _generate_batch_size_per_feature_and_max_B,
    _generate_block_bucketize_pos,
    _generate_case_tensors,
    _generate_total_num_blocks_tensors,
    _op_kwargs,
)

logging.basicConfig(level=logging.INFO)

PROFILE_WARMUP_STEPS = 18
PROFILE_MEASURE_STEPS = 32
PROFILE_TAKE = PROFILE_MEASURE_STEPS // 2  # 取活跃阶段后半段平均


@dataclass(frozen=True)
class ProfileResult:
    device_tag: str
    device_index: int
    mode: str
    measurement: str
    case_name: str
    my_size: int
    rows: int
    num_indices: int
    avg_time_us: Optional[float]

    def to_csv_row(self) -> dict:
        return {
            "case": self.case_name,
            "device": self.device_tag,
            "index": self.device_index,
            "mode": self.mode,
            "measurement": self.measurement,
            "rows": self.rows,
            "indices": self.num_indices,
            "my_size": self.my_size,
            "avg_time_us": self.avg_time_us,
        }


def _prepare_inputs(case, device, include_optionals: bool):
    lengths, indices, block_sizes, weights = _generate_case_tensors(case, include_optionals)
    tgt = device
    if not include_optionals:
        return _op_kwargs(
            lengths=lengths.to(tgt),
            indices=indices.to(tgt),
            block_sizes=block_sizes.to(tgt),
            my_size=case.my_size,
        )

    batch_size_per_feature, max_B = _generate_batch_size_per_feature_and_max_B(block_sizes, case.batch_size)
    total_num_blocks = _generate_total_num_blocks_tensors(block_sizes, case.my_size, GenTotalNumsBlocksType.RAND_TYPE)
    _, block_bucketize_pos_list = _generate_block_bucketize_pos(block_sizes, case.my_size, device)
    kwargs = _op_kwargs(
        lengths=lengths.to(tgt),
        indices=indices.to(tgt),
        block_sizes=block_sizes.to(tgt),
        my_size=case.my_size,
        weights=weights.to(tgt),
        bucketize_pos=True,
        sequence=True,
        keep_orig_idx=True,
        batch_size_per_feature=batch_size_per_feature.to(tgt),
        max_B=max_B,
        total_num_blocks=total_num_blocks.to(tgt),
        block_bucketize_pos=block_bucketize_pos_list,
    )
    return kwargs


def _emit_result(result: ProfileResult) -> ProfileResult:
    logging.info(
        "[perf-%s::%s::%s::%s] warmup=%d, active=%d, take=%d, avg_time_us=%s, "
        "rows=%d, indices=%d, my_size=%d",
        result.device_tag, result.case_name, result.mode, result.measurement,
        PROFILE_WARMUP_STEPS, PROFILE_MEASURE_STEPS, PROFILE_TAKE,
        result.avg_time_us if result.avg_time_us is not None else "N/A",
        result.rows, result.num_indices, result.my_size,
    )
    return result


def _profile_gpu(case, device, device_index, mode, include_optionals):
    dev = torch.device(device)
    if not torch.cuda.is_available():
        logging.warning("CUDA device is not available, skip profiling")
        return None
    kwargs = _prepare_inputs(case, device, include_optionals)
    pid = os.getpid()
    profiler_dir = Path.cwd() / "profiling" / f"{pid}"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    trace_file = profiler_dir / "trace.json"

    average_latency_us = 0.
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(
            wait=0, warmup=PROFILE_WARMUP_STEPS, active=PROFILE_MEASURE_STEPS, repeat=1
        ),
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_file)),
    ) as prof:
        for _ in range(PROFILE_WARMUP_STEPS + PROFILE_MEASURE_STEPS):
            torch.ops.fbgemm.block_bucketize_sparse_features(**kwargs)
            torch.cuda.synchronize(dev)
            prof.step()
        prof.stop()
        prof_events = prof.key_averages()
        evt = next(filter(lambda e: e.key == "fbgemm::block_bucketize_sparse_features", prof_events), None)

        if evt is None or evt.count != PROFILE_MEASURE_STEPS:
            raise ValueError(
                f"Expected {PROFILE_MEASURE_STEPS} iterations of block_bucketize_sparse_features, "
                f"but got {evt.count if evt else 0}"
            )
        average_latency_us = evt.device_time_total / PROFILE_MEASURE_STEPS
    
    shutil.rmtree(profiler_dir, ignore_errors=True)

    return _emit_result(ProfileResult(
        device_tag="cuda", device_index=device_index, mode=mode, measurement="trace",
        case_name=case.name, my_size=case.my_size,
        rows=kwargs["lengths"].numel(), num_indices=kwargs["indices"].numel(),
        avg_time_us=average_latency_us,
    ))


def _profile_gpu_e2e(case, device, device_index, mode, include_optionals):
    dev = torch.device(device)
    if not torch.cuda.is_available():
        logging.warning("CUDA device is not available, skip profiling")
        return None
    kwargs = _prepare_inputs(case, device, include_optionals)
    warmup_iters = PROFILE_WARMUP_STEPS + PROFILE_MEASURE_STEPS // 2
    for _ in range(warmup_iters):
        torch.ops.fbgemm.block_bucketize_sparse_features(**kwargs)
        torch.cuda.synchronize(dev)
    start = time.perf_counter()
    for _ in range(PROFILE_TAKE):
        torch.ops.fbgemm.block_bucketize_sparse_features(**kwargs)
        torch.cuda.synchronize(dev)
    elapsed_us = (time.perf_counter() - start) / PROFILE_TAKE * 1e6
    return _emit_result(ProfileResult(
        device_tag="cuda", device_index=device_index, mode=mode, measurement="e2e",
        case_name=case.name, my_size=case.my_size,
        rows=kwargs["lengths"].numel(), num_indices=kwargs["indices"].numel(),
        avg_time_us=elapsed_us,
    ))


def _profile_npu(case, device, device_index, mode, include_optionals):
    import torch_npu

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logging.warning("NPU device is not available, skip profiling")
        return None
    kwargs = _prepare_inputs(case, device, include_optionals)

    pid = os.getpid()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    profiler_dir = Path.cwd() / "profiling" / f"{pid}-{timestamp}"
    profiler_dir.mkdir(parents=True, exist_ok=True)

    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=PROFILE_WARMUP_STEPS, active=PROFILE_MEASURE_STEPS, repeat=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profiler_dir)),
    )
    prof.start()
    for _ in range(PROFILE_WARMUP_STEPS + PROFILE_MEASURE_STEPS):
        torch.ops.mxrec.block_bucketize_sparse_features(**kwargs)
        torch.npu.synchronize()
        prof.step()
    prof.stop()

    kernel_latency = {}
    csv_files = list(profiler_dir.rglob("kernel_details.csv"))
    if csv_files:
        csv_path = csv_files[0]
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kname = row.get("Name", "")
                dur = float(row.get("Duration(us)", 0.0))
                kernel_latency.setdefault(kname, []).append(dur)
    take_iters = PROFILE_TAKE
    total = 0.0
    valid = False
    for k, lst in kernel_latency.items():
        if len(lst) >= PROFILE_MEASURE_STEPS:
            total += sum(lst[-take_iters:])
            valid = True
    avg_us = total / take_iters if valid and take_iters > 0 else None

    result = _emit_result(ProfileResult(
        device_tag="npu", device_index=device_index, mode=mode, measurement="trace",
        case_name=case.name, my_size=case.my_size,
        rows=kwargs["lengths"].numel(), num_indices=kwargs["indices"].numel(),
        avg_time_us=avg_us,
    ))
    shutil.rmtree(profiler_dir, ignore_errors=True)
    return result


def _profile_npu_e2e(case, device, device_index, mode, include_optionals):
    import torch_npu

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logging.warning("NPU device is not available, skip profiling")
        return None
    kwargs = _prepare_inputs(case, device, include_optionals)
    warmup_iters = PROFILE_WARMUP_STEPS + PROFILE_MEASURE_STEPS // 2
    for _ in range(warmup_iters):
        torch.ops.mxrec.block_bucketize_sparse_features(**kwargs)
        torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(PROFILE_TAKE):
        torch.ops.mxrec.block_bucketize_sparse_features(**kwargs)
        torch.npu.synchronize()
    elapsed_us = (time.perf_counter() - start) / PROFILE_TAKE * 1e6
    return _emit_result(ProfileResult(
        device_tag="npu", device_index=device_index, mode=mode, measurement="e2e",
        case_name=case.name, my_size=case.my_size,
        rows=kwargs["lengths"].numel(), num_indices=kwargs["indices"].numel(),
        avg_time_us=elapsed_us,
    ))


def main():
    parser = argparse.ArgumentParser(description="Profile block_bucketize_sparse_features on NPU/GPU")
    parser.add_argument("--device", choices=["npu", "cuda"], default="npu", help="device type to profile")
    parser.add_argument("--index", type=int, default=0, help="device index, default 0")
    parser.add_argument(
        "--measurement", choices=["trace", "e2e"], default="trace", help="trace: kernel时间; e2e: 端到端"
    )
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA device is not available, skip profiling")
            return
    else:
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            logging.warning("NPU device is not available, skip profiling")
            return
        import torch_npu
        torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

    device_str = f"{args.device}:{args.index}"
    results = []
    for mode, include_optionals in (("basic", False), ("full", True)):
        for case in PERF_CASES:
            if args.device == "cuda":
                if args.measurement == "trace":
                    res = _profile_gpu(case, device_str, args.index, mode, include_optionals)
                else:
                    res = _profile_gpu_e2e(case, device_str, args.index, mode, include_optionals)
            else:
                if args.measurement == "trace":
                    res = _profile_npu(case, device_str, args.index, mode, include_optionals)
                else:
                    res = _profile_npu_e2e(case, device_str, args.index, mode, include_optionals)
            if res is not None:
                results.append(res)

    if results:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M")
        csv_name = f"{args.device}_{args.index}_{ts}_profile.csv"
        csv_path = Path.cwd() / csv_name
        fieldnames = ["case", "device", "index", "mode", "measurement", "rows", "indices", "my_size", "avg_time_us"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(r.to_csv_row() for r in results)
        logging.info("perf results saved to %s", csv_path)


if __name__ == "__main__":
    main()

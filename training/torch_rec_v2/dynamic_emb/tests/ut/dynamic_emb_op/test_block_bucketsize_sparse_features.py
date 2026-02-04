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
# 系统库
import logging
import sysconfig
import time
from typing import Optional, Tuple, Dict, List

# 三方库
import pytest
import torch
import torch_npu
import numpy as np

# 本项目库
import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)


def block_bucketize_sparse_features_cpu(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    bucketize_pos: bool,
    sequence: bool,
    dist_type_per_feature: torch.Tensor,
    block_sizes: torch.Tensor,
    my_size: int,
    weights: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.int64,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

    lengths = lengths.cpu().to(dtype=dtype, non_blocking=True).contiguous()
    indices = indices.cpu().to(dtype=dtype, non_blocking=True).contiguous()
    dist_type_per_feature = dist_type_per_feature.cpu().to(dtype=dtype, non_blocking=True).contiguous()
    block_sizes = block_sizes.cpu().to(dtype=dtype, non_blocking=True).contiguous()
    weights_contig = (
        weights.cpu().to(dtype=torch.float32, non_blocking=True).contiguous() if weights is not None else None
    )

    lengths_size = lengths.size(0)
    indices_size = indices.size(0)
    T = block_sizes.size(0)
    B = lengths_size // T

    offsets = torch.cumsum(lengths, dim=0, dtype=dtype).contiguous()
    int_options = dict(dtype=dtype, device=torch.device("cpu"))
    new_lengths = torch.zeros((lengths_size * my_size,), **int_options)
    new_indices = torch.empty((indices_size,), **int_options)

    # 权重
    new_weights: Optional[torch.Tensor] = None
    if weights_contig is not None:
        new_weights = torch.empty_like(weights_contig, dtype=torch.float32)

    # 分桶位置
    new_pos: Optional[torch.Tensor] = None
    if bucketize_pos:
        new_pos = torch.empty((indices_size,), **int_options)

    # 反桶化
    unbucketize_permute: Optional[torch.Tensor] = None
    if sequence:
        unbucketize_permute = torch.empty((indices_size,), **int_options)

    for b_t in range(lengths_size):
        t = b_t // B
        use_roundrobin = bool(dist_type_per_feature[t].item())
        blk_size = block_sizes[t].item()
        rowstart = 0 if b_t == 0 else offsets[b_t - 1].item()
        rowend = offsets[b_t].item()

        for i in range(rowstart, rowend):
            idx = indices[i].item()

            if use_roundrobin:
                p = idx % my_size
            else:
                if idx < blk_size * my_size:
                    p = idx // blk_size
                else:
                    p = idx % my_size

            new_lengths[p * lengths_size + b_t] += 1

    new_lengths_total_size = lengths_size * my_size
    current_offsets = torch.zeros_like(new_lengths, dtype=dtype)
    if new_lengths_total_size > 1:
        current_offsets[1:] = torch.cumsum(new_lengths[:-1], dim=0, dtype=dtype)

    for b_t in range(lengths_size):
        t = b_t // B
        use_roundrobin = bool(dist_type_per_feature[t].item())
        blk_size = block_sizes[t].item()

        rowstart = 0 if b_t == 0 else offsets[b_t - 1].item()
        rowend = offsets[b_t].item()

        for i in range(rowstart, rowend):
            idx = indices[i].item()

            if use_roundrobin:
                p = idx % my_size
                new_idx = idx
            else:
                if idx < blk_size * my_size:
                    p = idx // blk_size
                    new_idx = idx % blk_size
                else:
                    p = idx % my_size
                    new_idx = idx // my_size

            pos = current_offsets[p * lengths_size + b_t].item()
            current_offsets[p * lengths_size + b_t] += 1

            new_indices[pos] = new_idx

            if sequence:
                unbucketize_permute[i] = pos

            if weights_contig is not None:
                new_weights[pos] = weights_contig[i]

            if bucketize_pos:
                new_pos[pos] = i - rowstart

    return new_lengths, new_indices, new_weights, new_pos, unbucketize_permute


def compare_tensors(
    name: str,
    cpu_tensor: Optional[torch.Tensor],
    npu_tensor: Optional[torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """Compare two tensors, return (match status, error message)"""
    if cpu_tensor is None and npu_tensor is None:
        return True, f"{name}: both None (correct)"

    if cpu_tensor is None or npu_tensor is None:
        return False, f"{name}: one is None while the other is not"

    npu_cpu = npu_tensor.cpu()
    if cpu_tensor.dtype in (torch.int64, torch.int32):
        npu_cpu = npu_cpu.to(dtype=cpu_tensor.dtype)
    elif cpu_tensor.dtype == torch.float32:
        npu_cpu = npu_cpu.to(dtype=torch.float32)

    if cpu_tensor.shape != npu_cpu.shape:
        return False, f"{name}: shape mismatch CPU={cpu_tensor.shape} vs NPU={npu_cpu.shape}"

    if torch.allclose(cpu_tensor, npu_cpu, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(cpu_tensor - npu_cpu)).item()
        return True, f"{name}: matched (max difference: {max_diff:.2e})"
    else:
        diff = torch.abs(cpu_tensor - npu_cpu)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff.float()).item()
        mismatch_count = torch.sum(diff > atol).item()
        mismatch_ratio = mismatch_count / cpu_tensor.numel()

        error_msg = (
            f"{name}: not matched\n"
            f"  Max difference: {max_diff:.2e}\n"
            f"  Mean difference: {mean_diff:.2e}\n"
            f"  Mismatched elements: {mismatch_count}/{cpu_tensor.numel()} ({mismatch_ratio*100:.2f}%)"
        )
        return False, error_msg


def print_performance_comparison(test_name: str, npu_time_ms: float, total_indices: int, T: int, B: int, my_size: int):
    """Print performance comparison information"""
    logging.info(f"\n=== Performance Analysis [{test_name}] ===")
    logging.info(f"  NPU average execution time: {npu_time_ms:.3f} ms")
    logging.info(f"  Throughput: {total_indices / (npu_time_ms / 1000) / 1e6:.2f} M indices/sec")
    logging.info(f"  Data scale: T={T}, B={B}, my_size={my_size}, total_indices={total_indices}")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize(
    "data_scale",
    [
        # 100 total indices (20 lengths × 5 = 100)
        {
            "name": "100_indices",
            "T": 2,
            "B": 10,
            "my_size": 4,  # T×B=20
            "lengths_count": 20,
            "lengths_range": (5, 6),
            "indices_range": (0, 10),
            "iterations": 100,
        },
        # 1000 total indices
        {
            "name": "1000_indices",
            "T": 4,
            "B": 50,
            "my_size": 4,
            "lengths_count": 200,
            "lengths_range": (5, 6),  # T×B=200
            "indices_range": (0, 10),
            "iterations": 50,
        },
        # 10000 total indices
        {
            "name": "10000_indices",
            "T": 8,
            "B": 250,
            "my_size": 8,  # T×B=2000
            "lengths_count": 2000,
            "lengths_range": (5, 6),
            "indices_range": (0, 10),
            "iterations": 10,
        },
        # 100000 total indices
        {
            "name": "100000_indices",
            "T": 16,
            "B": 1250,
            "my_size": 8,  # T×B=20000
            "lengths_count": 20000,
            "lengths_range": (5, 6),
            "indices_range": (0, 10),
            "iterations": 5,
        },
    ],
)
def test_block_bucketsize_consistency_by_scale(device, data_scale, dtype):
    """
    Functional consistency test: cover 100/1000/10000 total indices scenarios
    Verify CPU and NPU calculation results match exactly (support int32/int64)
    """
    torch.manual_seed(789)
    cfg = data_scale
    logging.info(f"\n=== Functional Test - {cfg['name']} (dtype={dtype}) ===")

    lengths = torch.randint(*cfg["lengths_range"], (cfg["lengths_count"],), dtype=dtype)
    total_indices = lengths.sum().item()
    logging.info(f"  Generated total indices: {total_indices} (target: {cfg['name'].split('_')[0]})")

    indices = torch.randint(*cfg["indices_range"], (total_indices,), dtype=dtype)
    dist_type_per_feature = torch.tensor([0, 1] * (cfg["T"] // 2) + [0] * (cfg["T"] % 2), dtype=dtype)
    block_sizes = torch.tensor([3360] * cfg["T"], dtype=dtype)
    bucketize_pos = True
    sequence = True
    weights = torch.randn(total_indices, dtype=torch.float32)

    # CPU实现
    cpu_result = block_bucketize_sparse_features_cpu(
        lengths, indices, bucketize_pos, sequence, dist_type_per_feature, block_sizes, cfg["my_size"], weights, dtype
    )
    new_lengths_cpu, new_indices_cpu, new_weights_cpu, new_pos_cpu, unbucketize_permute_cpu = cpu_result

    # NPU实现
    torch.npu.set_device(device)
    lengths_npu = lengths.to(device, dtype=dtype)
    indices_npu = indices.to(device, dtype=dtype)
    dist_type_npu = dist_type_per_feature.to(device, dtype=dtype)
    block_sizes_npu = block_sizes.to(device, dtype=dtype)
    weights_npu = weights.to(device, dtype=torch.float32)

    npu_result = dynamic_emb_extensions.block_bucketize_sparse_features(
        lengths_npu, indices_npu, bucketize_pos, sequence, dist_type_npu, block_sizes_npu, cfg["my_size"], weights_npu
    )
    new_lengths_npu, new_indices_npu, new_weights_npu, new_pos_npu, unbucketize_permute_npu = npu_result

    # 结果校验
    checks = [
        ("new_lengths", new_lengths_cpu, new_lengths_npu),
        ("new_indices", new_indices_cpu, new_indices_npu),
        ("new_weights", new_weights_cpu, new_weights_npu, 1e-4, 1e-4),
        ("new_pos", new_pos_cpu, new_pos_npu),
        ("unbucketize_permute", unbucketize_permute_cpu, unbucketize_permute_npu),
    ]

    all_match = True
    error_messages = []
    for check in checks:
        name, cpu_t, npu_t = check[:3]
        rtol = check[3] if len(check) > 3 else 1e-5
        atol = check[4] if len(check) > 4 else 1e-5
        match, msg = compare_tensors(name, cpu_t, npu_t, rtol, atol)
        logging.info(msg)
        if not match:
            all_match = False
            error_messages.append(msg)

    # 断言结果一致
    assert all_match, f"{cfg['name']} (dtype={dtype}) functional test failed\n{chr(10).join(error_messages)}"


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("T", [2, 8])  # 特征数
@pytest.mark.parametrize("B", [10, 250])  # Batch数
@pytest.mark.parametrize("my_size", [4, 8])  # 分片数
@pytest.mark.parametrize("lengths_range", [(5, 6), (10, 11)])
@pytest.mark.parametrize("indices_range", [(0, 10)])
@pytest.mark.parametrize("bucketize_pos", [True, False])
@pytest.mark.parametrize("sequence", [True, False])
@pytest.mark.parametrize("has_weights", [True, False])
def test_block_bucketsize_performance(
    device, T, B, my_size, lengths_range, indices_range, bucketize_pos, sequence, has_weights, dtype
):
    torch.manual_seed(789)
    torch.npu.set_device(device)
    device_str = f"npu:{device}"
    lengths_count = T * B  # lengths总数 = 特征数 × Batch数
    lengths = torch.randint(*lengths_range, (lengths_count,), dtype=dtype)
    total_indices = lengths.sum().item()  # 总索引数
    indices = torch.randint(*indices_range, (total_indices,), dtype=dtype)
    dist_type_per_feature = torch.tensor([0, 1] * (T // 2) + [0] * (T % 2), dtype=dtype)
    block_sizes = torch.tensor([3360] * T, dtype=dtype)
    weights = torch.randn(total_indices, dtype=torch.float32) if has_weights else None

    lengths_npu = lengths.to(device_str, dtype=dtype)
    indices_npu = indices.to(device_str, dtype=dtype)
    dist_type_npu = dist_type_per_feature.to(device_str, dtype=dtype)
    block_sizes_npu = block_sizes.to(device_str, dtype=dtype)
    weights_npu = weights.to(device_str, dtype=torch.float32) if has_weights else None

    # 预热
    warmup_iter = 10
    for _ in range(warmup_iter):
        _ = block_bucketize_sparse_features_cpu(
            lengths, indices, bucketize_pos, sequence, dist_type_per_feature, block_sizes, my_size, weights, dtype
        )

    for _ in range(warmup_iter):
        _ = dynamic_emb_extensions.block_bucketize_sparse_features(
            lengths_npu, indices_npu, bucketize_pos, sequence, dist_type_npu, block_sizes_npu, my_size, weights_npu
        )
    torch.npu.synchronize(device)

    # 性能测试迭代次数
    if total_indices < 1000:
        test_iter = 2
    elif total_indices < 10000:
        test_iter = 1
    else:
        test_iter = 1

    # 测试CPU性能
    start_time = time.time()
    for _ in range(test_iter):
        _ = block_bucketize_sparse_features_cpu(
            lengths, indices, bucketize_pos, sequence, dist_type_per_feature, block_sizes, my_size, weights, dtype
        )
    cpu_total_time = time.time() - start_time
    cpu_avg_time_ms = (cpu_total_time / test_iter) * 1000  # 平均耗时(ms)
    cpu_throughput = total_indices / (cpu_total_time / test_iter) / 1e6  # 吞吐量(M indices/sec)

    # 测试NPU性能
    start_time = time.time()
    for _ in range(test_iter):
        npu_res = dynamic_emb_extensions.block_bucketize_sparse_features(
            lengths_npu, indices_npu, bucketize_pos, sequence, dist_type_npu, block_sizes_npu, my_size, weights_npu
        )
    npu_total_time = time.time() - start_time
    npu_avg_time_ms = (npu_total_time / test_iter) * 1000
    npu_throughput = total_indices / (npu_total_time / test_iter) / 1e6

    # 计算加速比
    speedup = cpu_avg_time_ms / npu_avg_time_ms if npu_avg_time_ms > 0 else float("inf")

    # 打印性能结果
    logging.info("\n" + "=" * 80)
    logging.info(f"性能测试场景（dtype={dtype}）：")
    logging.info(f"  T={T}, B={B}, my_size={my_size}, lengths_range={lengths_range}")
    logging.info(
        f"  indices_range={indices_range}, bucketize_pos={bucketize_pos}, \
            sequence={sequence}, has_weights={has_weights}"
    )
    logging.info(f"  总索引数: {total_indices}, 测试迭代次数: {test_iter}")
    logging.info("-" * 80)
    logging.info(f"CPU 性能:")
    logging.info(f"  总耗时: {cpu_total_time:.4f}s, 平均耗时: {cpu_avg_time_ms:.3f}ms/次")
    logging.info(f"  吞吐量: {cpu_throughput:.2f} M indices/sec")
    logging.info(f"NPU 性能:")
    logging.info(f"  总耗时: {npu_total_time:.4f}s, 平均耗时: {npu_avg_time_ms:.3f}ms/次")
    logging.info(f"  吞吐量: {npu_throughput:.2f} M indices/sec")
    logging.info("-" * 80)
    logging.info(f"加速比 (CPU/NPU): {speedup:.2f}x")
    logging.info("=" * 80)


@pytest.mark.parametrize("device", ["npu:0"])
def test_fixed_case_new_lengths_new_indices(device):

    logging.info("\n=== Fixed Case Test - Verify new_lengths and new_indices ===")
    lengths_npu = torch.tensor(
        [2, 2, 1, 1, 1, 1, 1, 1, 91, 91, 91, 91, 91, 91, 91, 91, 1, 1, 1, 1, 1, 1, 1, 1],
        device=device,
        dtype=torch.int64,
    )

    indices_npu = torch.tensor(
        [
            4,
            5,
            3,
            7,
            6,
            8,
            2,
            9,
            1,
            0,
            3108,
            4571,
            1373,
            1831,
            1175,
            2313,
            2096,
            2951,
            1125,
            2046,
            707,
            3168,
            2991,
            3683,
            3686,
            1199,
            1580,
            5952,
            47,
            1721,
            2762,
            260,
            589,
            4306,
            648,
            2571,
            2716,
            592,
            1196,
            1,
            1210,
            1198,
            4995,
            4973,
            3578,
            593,
            2997,
            858,
            50,
            27002,
            318,
            4993,
            72998,
            293,
            1136,
            1213,
            4226,
            7153,
            6539,
            5445,
            2918,
            2858,
            2628,
            1923,
            1240,
            1101,
            1089,
            778,
            527,
            480,
            356,
            231,
            95,
            33493,
            8961,
            8622,
            7361,
            6807,
            6377,
            5378,
            3471,
            2701,
            2542,
            4886,
            2028,
            1968,
            1784,
            736,
            296,
            76093,
            1233,
            34,
            70286,
            1097,
            3996,
            608,
            1291,
            377,
            5349,
            1214,
            46578,
            7438,
            32,
            1527,
            150,
            2396,
            3793,
            78499,
            1221,
            2959,
            3681,
            750,
            6016,
            26587,
            79132,
            2291,
            2115,
            1393,
            1246,
            2,
            6934,
            6711,
            6502,
            6218,
            5669,
            4848,
            4361,
            3147,
            2890,
            82459,
            86190,
            42351,
            69088,
            904,
            8656,
            111,
            3967,
            7076,
            74458,
            6552,
            64614,
            85414,
            76251,
            4776,
            1961,
            410,
            2683,
            1356,
            19,
            442,
            1610,
            508,
            3114,
            33660,
            2329,
            41997,
            2019,
            6874,
            32587,
            8970,
            4878,
            5690,
            8340,
            3030,
            5995,
            36535,
            3949,
            39183,
            1466,
            475,
            4720,
            5379,
            16,
            8370,
            7064,
            7371,
            7318,
            7022,
            5944,
            5419,
            5072,
            4388,
            4248,
            3248,
            2961,
            2822,
            2793,
            2380,
            2014,
            1971,
            691,
            457,
            357,
            7701,
            7366,
            7362,
            6986,
            6793,
            6333,
            6166,
            6058,
            4699,
            4220,
            4011,
            3992,
            3977,
            3799,
            110,
            780,
            588,
            344,
            165,
            153,
            595,
            316,
            1270,
            364,
            733,
            597,
            329,
            500,
            367,
            539,
            253,
            161,
            208,
            185,
            339,
            25,
            17,
            1036,
            586,
            288,
            1704,
            924,
            474,
            788,
            1387,
            1517,
            1584,
            786,
            368,
            2706,
            594,
            2355,
            1206,
            1917,
            266,
            350,
            2916,
            551,
            337,
            317,
            2710,
            590,
            4308,
            2985,
            2470,
            2294,
            2273,
            2081,
            2006,
            2001,
            1997,
            1954,
            1876,
            1645,
            1333,
            2791,
            2012,
            2174,
            235,
            1080,
            1304,
            923,
            920,
            1374,
            1653,
            908,
            2700,
            2640,
            7156,
            38038,
            1262,
            5618,
            7256,
            37733,
            34405,
            1288,
            903,
            1201,
            1267,
            1228,
            1208,
            1225,
            1148,
            3508,
            3060,
            1203,
            1252,
            1219,
            913,
            1250,
            1204,
            1284,
            541,
            1234,
            1276,
            2176,
            1266,
            4327,
            910,
            965,
            1258,
            1247,
            3468,
            912,
            33794,
            3198,
            1222,
            7090,
            1200,
            4034,
            1293,
            1227,
            4262,
            8798,
            1079,
            902,
            3477,
            2394,
            5343,
            6025,
            1197,
            1980,
            3272,
            720,
            2692,
            83,
            428,
            1265,
            1617,
            29,
            1245,
            1280,
            5339,
            246,
            1966,
            3160,
            1243,
            2336,
            2930,
            162,
            1719,
            2324,
            2580,
            994,
            1635,
            2912,
            190,
            319,
            2391,
            2908,
            1912,
            2333,
            123,
            300,
            2501,
            6291,
            6951,
            6946,
            6945,
            6909,
            6867,
            6863,
            58,
            1464,
            4235,
            4223,
            4210,
            4214,
            4179,
            4166,
            4167,
            4164,
            4161,
            4144,
            4139,
            3070,
            2280,
            3265,
            1218,
            2804,
            3462,
            1370,
            180,
            3052,
            3785,
            3176,
            2599,
            2395,
            333,
            2144,
            2393,
            441,
            1186,
            2018,
            2087,
            1296,
            2616,
            1230,
            3911,
            1673,
            4641,
            3481,
            1394,
            1449,
            44555,
            1193,
            922,
            1212,
            5291,
            1209,
            6669,
            1237,
            2068,
            307,
            308,
            306,
            112290,
            8638,
            100714,
            215,
            27904,
            94896,
            40819,
            6773,
            56367,
            31658,
            55442,
            26662,
            65261,
            5971,
            46976,
            8949,
            5377,
            30810,
            101,
            4979,
            109374,
            94959,
            72226,
            55269,
            2131,
            7936,
            7937,
            7939,
            7327,
            4422,
            7941,
            7938,
            7396,
            7820,
            5147,
            7577,
            8405,
            7940,
            34155,
            61937,
            7942,
            7328,
            7935,
            7331,
            5225,
            6611,
            8014,
            6666,
            3083,
            5878,
            2730,
            68157,
            99114,
            1729,
            2729,
            2728,
            2712,
            1732,
            2663,
            471,
            1375,
            2989,
            2289,
            2296,
            3046,
            514,
            1641,
            1060,
            2302,
            3175,
            39,
            1500,
            1914,
            417,
            1440,
            1747,
            1916,
            2572,
            224,
            1614,
            1665,
            372,
            708,
            52,
            550,
            3261,
            1184,
            1279,
            2581,
            1482,
            3255,
            440,
            2124,
            543,
            1569,
            1367,
            70,
            585,
            1513,
            1806,
            2252,
            2699,
            2723,
            198,
            1306,
            1676,
            1573,
            2600,
            512,
            1690,
            172,
            1391,
            1544,
            1779,
            24,
            1037,
            196,
            849,
            3697,
            1172,
            2020,
            1307,
            342,
            969,
            2108,
            2885,
            2248,
            2872,
            3699,
            3044,
            2406,
            555,
            2356,
            3725,
            951,
            919,
            1162,
            3730,
            1483,
            953,
            373,
            1257,
            1171,
            3521,
            81,
            2946,
            2863,
            1173,
            1953,
            2944,
            2947,
            3384,
            1376,
            2948,
            2949,
            2529,
            1220,
            3104,
            42543,
            1287,
            948,
            1231,
            7360,
            1104,
            431,
            74510,
            73323,
            4007,
            98124,
            108583,
            4705,
            2098,
            435,
            2054,
            784,
            1380,
            370,
            151,
            466,
            3623,
            1278,
            48,
            8796,
            6358,
            7107,
            5444,
            33437,
            1967,
            8039,
            26375,
            3964,
            4085,
            941,
            2205,
            7164,
            7075,
            4795,
            26386,
            3639,
            7569,
            6528,
            8636,
            4799,
            7386,
            25937,
            3635,
            8167,
            2795,
            3507,
            3421,
            3028,
            4499,
            5292,
            8651,
            3552,
            934,
            3675,
            3984,
            10,
            5267,
            2167,
            2150,
            2011,
            1562,
            1371,
            1275,
            653,
            277,
            256,
            7373,
            5507,
            5218,
            4447,
            4321,
            3752,
            3702,
            3591,
            3534,
            3524,
            3396,
            3264,
            2953,
            2915,
            2622,
            2471,
            2367,
            2269,
            2125,
            2082,
            1960,
            1032,
            1030,
            1010,
            986,
            830,
            15,
            5309,
            5247,
            4700,
            4558,
            4545,
            3877,
            3784,
            3754,
            3701,
            3483,
            2922,
            2142,
            2135,
            2111,
            2089,
            2052,
            1681,
            1453,
            1011,
            3213,
            3479,
            3959,
            3740,
            45722,
            8644,
            3022,
            1017,
            1907,
            1663,
            1722,
            3963,
            8623,
            3086,
            4016,
            41566,
            3624,
            3039,
            5882,
            6428,
            2083,
            6239,
            2268,
            587,
            454,
            141,
            5254,
            6754,
            42738,
            5128,
            5418,
            7380,
            26479,
            31193,
            57555,
            20675,
            36266,
            65169,
            30028,
            70221,
            88773,
            63888,
        ],
        device=device,
        dtype=torch.int64,
    )

    bucketize_pos = False
    sequence = True
    dist_type_npu = torch.tensor([1, 1, 1], device=device, dtype=torch.int64)
    block_sizes_npu = torch.tensor([16, 131072, 131072], device=device, dtype=torch.int64)
    cfg = {"my_size": 1}
    weights_npu = None

    lengths_cpu = lengths_npu.cpu()
    indices_cpu = indices_npu.cpu()
    dist_type_cpu = dist_type_npu.cpu()
    block_sizes_cpu = block_sizes_npu.cpu()

    cpu_result = block_bucketize_sparse_features_cpu(
        lengths_cpu, indices_cpu, bucketize_pos, sequence, dist_type_cpu, block_sizes_cpu, cfg["my_size"], weights_npu
    )
    new_lengths_cpu, new_indices_cpu, _, _, _ = cpu_result

    npu_result = dynamic_emb_extensions.block_bucketize_sparse_features(
        lengths_npu, indices_npu, bucketize_pos, sequence, dist_type_npu, block_sizes_npu, cfg["my_size"], weights_npu
    )
    new_lengths_npu, new_indices_npu, _, _, _ = npu_result

    checks = [("new_lengths", new_lengths_cpu, new_lengths_npu), ("new_indices", new_indices_cpu, new_indices_npu)]

    all_match = True
    error_messages = []
    for check in checks:
        name, cpu_t, npu_t = check[:3]
        rtol = check[3] if len(check) > 3 else 1e-5
        atol = check[4] if len(check) > 4 else 1e-5
        match, msg = compare_tensors(name, cpu_t, npu_t, rtol, atol)
        logging.info(msg)
        if not match:
            all_match = False
            error_messages.append(msg)

    logging.info(f"  lengths_size: {lengths_cpu.size(0)}")
    logging.info(f"  indices_size: {indices_cpu.size(0)}")
    logging.info(f"  new_lengths shape: CPU={new_lengths_cpu.shape}, NPU={new_lengths_npu.shape}")
    logging.info(f"  new_indices shape: CPU={new_indices_cpu.shape}, NPU={new_indices_npu.shape}")

    assert all_match, f"Fixed case test failed\n{chr(10).join(error_messages)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

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
from enum import IntEnum
import numpy as np
import torch
import dynamic_emb_extensions
import pytest

logging.basicConfig(level=logging.INFO)


class Combiner(IntEnum):
    SUM = 0
    MEAN = 1


# -------------------------- 测试数据生成 --------------------------
test_cases = [
    # (FEATURE_NUM, COMBINER, DEVICE, EMBED_DIM, BATCH_SIZE, UNIQUE_COUNT)
    (1, Combiner.SUM, 0, 8, 2, 5),  # sum聚合
    (1, Combiner.MEAN, 0, 8, 2, 5),  # mean聚合
    (1, Combiner.SUM, 0, 8, 3, 30),  # 不同embed_dim和batch_size，sum聚合
    (1, Combiner.MEAN, 0, 8, 4, 1000),  # 不同embed_dim和batch_size，mean聚合
    (2, Combiner.MEAN, 0, 128, 4, 1000),  # 不同embed_dim和batch_size，mean聚合
    (2, Combiner.SUM, 0, 8, 128, 100000),  # 不同feature_num，sum聚合
]
repeat_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
grad_dtypes = [torch.float32, torch.float16, torch.bfloat16]
index_dtypes = [torch.int32, torch.int64, torch.uint64]

# PTA/torch_npu 限制（与 dynamic_emb C++ 算子能力无关）：
# 1. scatter/gather 等 indexing 的 index 仅支持 int32/int64（不支持 uint64 index）。
# 2. device 上的 uint64 tensor 不支持多数 PyTorch 算子（索引、减法、赋值等），
#    报错示例：Tensor self not implemented for DT_UINT64 / index not implemented for DT_UINT64。
# lookup_backward NPU 内核已支持 uint64 index；含 uint64 golden 的 UT 在当前 PTA 下跳过。
PTA_SCATTER_INDEX_DTYPES = (torch.int32, torch.int64)
PTA_UINT64_UT_SKIP_REASON = (
    "PTA/torch_npu 暂不支持 NPU 上 uint64 tensor 的 PyTorch 参考算子 "
    "(索引/减法/scatter_add_ 等，报错 Tensor self not implemented for DT_UINT64)。"
    "lookup_backward C++ 算子已支持 uint64 index；待 PTA 支持后再启用 golden 比对 UT。"
)


def scatter_index_for_pta(indices: torch.Tensor) -> torch.Tensor:
    """Cast index tensor to a dtype supported by PTA scatter/gather ops."""
    if indices.dtype not in PTA_SCATTER_INDEX_DTYPES:
        return indices.to(torch.int64)
    return indices


def _grad_tol(grad_dtype):
    if grad_dtype == torch.float16:
        return 1e-3, 1e-3
    if grad_dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 1e-5, 1e-6


def generate_test_data(
    feature_num, combiner, device, embed_dim, batch_size, unique_count, repeat_rate, grad_dtype, index_dtype=torch.int64
):
    """生成指定配置的测试数据"""
    INDEX_DTYPE = index_dtype

    grad_for_table = torch.ones((batch_size, embed_dim), dtype=grad_dtype, device=device)

    np.random.seed(42)
    # 重复率定义：total_features = unique_count * (1 + repeat_rate)
    total_features = int(unique_count * (1.0 + repeat_rate))
    inverse_indices = torch.tensor(np.random.randint(0, unique_count, total_features), dtype=INDEX_DTYPE, device=device)

    def generate_biased_offsets(total_features, batch_size, index_dtype, device):
        if batch_size <= 0:
            raise ValueError(f"batch_size必须大于0，当前值：{batch_size}")
        if total_features < batch_size:
            total_features = batch_size
            logging.warning("总特征数%s小于批次大小%s，已自动扩容为%s", total_features, batch_size, batch_size)

        base = total_features // batch_size  # 每个样本基础特征数
        remainder = total_features % batch_size  # 剩余特征数
        offsets = [0]
        current = 0
        for i in range(batch_size):
            current += base + (1 if i < remainder else 0)
            offsets.append(current)

        assert offsets[-1] == total_features, f"偏移量计算错误：最后一个偏移量{offsets[-1]} != 总特征数{total_features}"

        biased_offsets = torch.tensor(offsets, dtype=index_dtype, device=device)
        return biased_offsets

    biased_offsets_list = generate_biased_offsets(total_features, batch_size, INDEX_DTYPE, device)
    unique_indices = torch.tensor(np.random.randint(1, 200, unique_count), dtype=INDEX_DTYPE, device=device)

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
        "grad_dtype": grad_dtype,
        "index_dtype": INDEX_DTYPE,
    }


index_dtype_cases = [
    (1, Combiner.SUM, 0, 8, 2, 5),
    (1, Combiner.MEAN, 0, 8, 4, 1000),
]


@pytest.mark.parametrize("index_dtype", index_dtypes, ids=["i32", "i64", "u64"])
@pytest.mark.parametrize("test_params", index_dtype_cases)
def test_lookup_backward_index_dtype(test_params, index_dtype):
    """NPU lookup_backward 支持 int32/int64/uint64 index；uint64 golden UT 见 skip 说明。"""
    if index_dtype == torch.uint64:
        pytest.skip(PTA_UINT64_UT_SKIP_REASON)

    feature_num, combiner, device, embed_dim, batch_size, unique_count = test_params
    data = generate_test_data(
        feature_num=feature_num,
        combiner=combiner,
        device=device,
        embed_dim=embed_dim,
        batch_size=batch_size,
        unique_count=unique_count,
        repeat_rate=0.25,
        grad_dtype=torch.float32,
        index_dtype=index_dtype,
    )

    unique_backward_grads_native = torch.zeros((unique_count, embed_dim), dtype=data["grad_dtype"], device=device)
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
        combiner=int(combiner),
    )

    unique_backward_grads_torch = torch_simulate_lookup_backward(
        grad_for_table=data["grad_for_table"],
        inverse_indices=data["inverse_indices"],
        biased_offsets=data["biased_offsets"],
        unique_count=unique_count,
        batch_size=data["batch_size"],
        feature_num=data["feature_num"],
        dim=embed_dim,
        combiner=combiner,
    )

    native_result = unique_backward_grads_native.reshape(unique_count, embed_dim)
    torch_result = unique_backward_grads_torch.reshape(unique_count, embed_dim)
    rtol, atol = _grad_tol(torch.float32)
    assert torch.allclose(native_result, torch_result, rtol=rtol, atol=atol)


# -------------------------- Torch算子模拟lookup_backward --------------------------
def find_idx_by_binary_search(biased_offsets, num, feature_idx):
    start = 0
    end = num
    while start < end:
        mid = start + (end - start) // 2
        value = biased_offsets[mid]
        if value <= feature_idx:
            start = mid + 1
        else:
            end = mid
    return num if (start == num and biased_offsets[start - 1] != feature_idx) else start - 1


def torch_simulate_lookup_backward(
    grad_for_table,  # 切分后的梯度 [batchsize, embed_dim]，如[1,3]
    inverse_indices,  # 逆索引 [total_indices]，如[0,1,1,2]
    biased_offsets,  # 样本偏移 ，如[0,2,4]
    unique_count,  # 唯一索引数量，如3
    batch_size,
    feature_num,
    dim,  # 嵌入维度，如3
    combiner,  # 聚合方式：sum/mean
):
    device = grad_for_table.device
    inverse_indices = inverse_indices.to(device)
    biased_offsets = biased_offsets.to(device)
    unique_grads = torch.zeros(unique_count, dim, dtype=grad_for_table.dtype, device=grad_for_table.device)
    num_total_features = len(inverse_indices)
    assert grad_for_table.shape[0] == len(biased_offsets) - 1, (
        f"grad_for_table行数{grad_for_table.shape[0]}需匹配样本数{len(biased_offsets) - 1}"
    )
    grad_broadcast = []

    for sample_idx in range(len(biased_offsets) - 1):
        start = biased_offsets[sample_idx].item()
        end = biased_offsets[sample_idx + 1].item()
        sample_feature_num = end - start
        sample_grad = grad_for_table[sample_idx : sample_idx + 1].repeat(sample_feature_num, 1)
        grad_broadcast.append(sample_grad)
    grad_broadcast = torch.cat(grad_broadcast, dim=0)
    assert grad_broadcast.shape[0] == num_total_features, (
        f"广播后梯度长度{grad_broadcast.shape[0]}需匹配总特征数{num_total_features}"
    )

    if combiner == Combiner.SUM:
        scatter_index = scatter_index_for_pta(inverse_indices)
        unique_grads.scatter_add_(dim=0, index=scatter_index.unsqueeze(1).expand(-1, dim), src=grad_broadcast)

    elif combiner == Combiner.MEAN:
        # MEAN 路径需对 biased_offsets 做索引与减法；uint64 在 NPU 上需 skip（见 PTA_UINT64_UT_SKIP_REASON）
        offsets_for_ref = biased_offsets.to(torch.int64) if biased_offsets.dtype == torch.uint64 else biased_offsets
        pooling_factors = torch.zeros(num_total_features, device=device, dtype=torch.float32)
        for feature_idx in range(num_total_features):
            sample_idx = find_idx_by_binary_search(offsets_for_ref, batch_size * feature_num + 1, feature_idx)
            pooling_factors[feature_idx] = offsets_for_ref[sample_idx + 1] - offsets_for_ref[sample_idx]
        # 与 NPU kernel 一致：在 float 域做 MEAN 除法，再 cast 回 grad dtype
        grad_mean = (grad_broadcast.float() / pooling_factors.unsqueeze(1)).to(grad_for_table.dtype)
        scatter_index = scatter_index_for_pta(inverse_indices)
        index = scatter_index.unsqueeze(1).expand(-1, dim)
        unique_grads.scatter_add_(dim=0, index=index, src=grad_mean)

    else:
        raise ValueError(f"不支持的聚合方式{combiner}，仅支持sum/mean")

    return unique_grads.reshape(-1)


# -------------------------- pytest测试用例 --------------------------
@pytest.mark.parametrize("grad_dtype", grad_dtypes, ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize(
    "repeat_rate",
    repeat_rates,
    ids=["repeat_0", "repeat_0.25", "repeat_0.5", "repeat_0.75", "repeat_1.0"],
)
@pytest.mark.parametrize("test_params", test_cases)
def test_lookup_backward(test_params, repeat_rate, grad_dtype):
    # 解析测试参数
    feature_num, combiner, device, embed_dim, batch_size, unique_count = test_params

    # 生成测试数据
    data = generate_test_data(
        feature_num=feature_num,
        combiner=combiner,
        device=device,
        embed_dim=embed_dim,
        batch_size=batch_size,
        unique_count=unique_count,
        repeat_rate=repeat_rate,
        grad_dtype=grad_dtype,
    )
    # -------------------------- 原生算子执行 --------------------------
    unique_backward_grads_native = torch.zeros((unique_count, embed_dim), dtype=data["grad_dtype"], device=device)

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
        combiner=int(combiner),
    )

    # -------------------------- Torch模拟实现执行 --------------------------
    unique_backward_grads_torch = torch_simulate_lookup_backward(
        grad_for_table=data["grad_for_table"],
        inverse_indices=data["inverse_indices"],
        biased_offsets=data["biased_offsets"],
        unique_count=unique_count,
        batch_size=data["batch_size"],
        feature_num=data["feature_num"],
        dim=embed_dim,
        combiner=combiner,
    )

    # -------------------------- 结果验证 --------------------------
    native_result = unique_backward_grads_native.reshape(unique_count, embed_dim)
    torch_result = unique_backward_grads_torch.reshape(unique_count, embed_dim)

    # 断言结果是否一致
    rtol, atol = _grad_tol(grad_dtype)
    assert torch.allclose(native_result, torch_result, rtol=rtol, atol=atol), f"""
        测试参数： feature_num={feature_num}, combiner={combiner},
        embed_dim={embed_dim}, batch_size={batch_size}, unique_count={unique_count}, repeat_rate={repeat_rate},
        grad_dtype={grad_dtype}
        原生实现和Torch模拟实现结果不一致！
        原生结果前2行：{native_result[:2]}
        Torch结果前2行：{torch_result[:2]}
        """

    logging.info(
        "测试通过 - 配置：feature_num=%s | combiner=%s | embed_dim=%s | batch_size=%s",
        feature_num,
        combiner,
        embed_dim,
        batch_size,
    )


# -------------------------- 自定义数据测试数据 --------------------------
def test_custom_case():
    TABLE_NUM = 1
    FEATURE_NUM = 1
    COMBINER = Combiner.MEAN
    DEVICE = 0
    GRAD_DTYPE = torch.float32
    INDEX_DTYPE = torch.int64
    EMBED_DIM = 3
    BATCH_SIZE = 2
    UNIQUE_COUNT = 27
    grad_for_table = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=GRAD_DTYPE, device=DEVICE)
    inverse_indices = torch.tensor(
        [7, 20, 4, 19, 23, 17, 15, 9, 25, 1, 10, 26, 21, 22, 2, 3, 14, 5, 12, 13, 11, 7, 24, 0, 6, 18, 16, 8],
        dtype=INDEX_DTYPE,
        device=DEVICE,
    )
    biased_offsets = torch.tensor([0, 14, 28], dtype=INDEX_DTYPE, device=DEVICE)
    unique_indices = torch.tensor(
        [
            16503,
            19547,
            29016,
            34020,
            44195,
            48711,
            51352,
            52512,
            56702,
            61945,
            63804,
            92332,
            130217,
            131913,
            138949,
            141079,
            142106,
            158739,
            158814,
            159223,
            159939,
            160705,
            162145,
            174663,
            180685,
            183450,
            193119,
        ],
        dtype=INDEX_DTYPE,
        device=DEVICE,
    )
    offsets_per_table = torch.tensor([0, 14, 28], dtype=INDEX_DTYPE, device=DEVICE)
    unique_backward_grads_native = torch.zeros((UNIQUE_COUNT, EMBED_DIM), dtype=GRAD_DTYPE, device=DEVICE)
    start_time_native = time.time()

    def check_tensor_type(tensor, name, expected_dtype):
        if tensor.dtype != expected_dtype:
            raise TypeError(f"张量{name}类型错误：期望{expected_dtype}，实际{tensor.dtype}")

    check_tensor_type(inverse_indices, "inverse_indices", INDEX_DTYPE)
    check_tensor_type(biased_offsets, "biased_offsets", INDEX_DTYPE)
    dynamic_emb_extensions.lookup_backward(
        grad_for_table,
        unique_backward_grads_native,
        unique_indices,
        inverse_indices,
        offsets_per_table,
        EMBED_DIM,
        TABLE_NUM,
        BATCH_SIZE,
        FEATURE_NUM,
        offsets_per_table[-1].item(),
        combiner=int(COMBINER),
    )
    time_native = (time.time() - start_time_native) * 1000

    # Torch模拟实现执行
    start_time_torch = time.time()
    unique_backward_grads_torch = torch_simulate_lookup_backward(
        grad_for_table=grad_for_table,
        inverse_indices=inverse_indices,
        unique_count=UNIQUE_COUNT,
        batch_size=BATCH_SIZE,
        feature_num=FEATURE_NUM,
        dim=EMBED_DIM,
        combiner=COMBINER,
        biased_offsets=biased_offsets,
    )
    time_torch = (time.time() - start_time_torch) * 1000  # 转毫秒

    # -------------------------- 结果验证与输出 --------------------------
    native_result = unique_backward_grads_native.reshape(UNIQUE_COUNT, EMBED_DIM)
    torch_result = unique_backward_grads_torch.reshape(UNIQUE_COUNT, EMBED_DIM)

    rtol, atol = _grad_tol(GRAD_DTYPE)
    is_equal = torch.allclose(native_result, torch_result, rtol=rtol, atol=atol)

    logging.info("=" * 80)
    logging.info(
        "测试配置：单表 | feature_num=%s | embed_dim=%s | batch_size=%s | combiner=%s",
        FEATURE_NUM,
        EMBED_DIM,
        BATCH_SIZE,
        COMBINER,
    )
    logging.info("=" * 80)
    logging.info("=======梯度是否一致：%s==========", "是" if is_equal else "否")
    logging.info("-" * 80)
    logging.info("原生lookup_backward耗时：%.6f 毫秒", time_native)
    logging.info("Torch模拟实现耗时：%.6f 毫秒", time_torch)
    logging.info("-" * 80)
    logging.info("原生实现聚合梯度（前2行）：\n%s", native_result[:2])
    logging.info("Torch模拟实现聚合梯度（前2行）：\n%s", torch_result[:2])
    logging.info("=" * 80)

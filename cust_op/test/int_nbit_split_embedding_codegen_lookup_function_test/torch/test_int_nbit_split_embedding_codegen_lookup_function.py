#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
import sysconfig
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pytest
import torch
import torch_npu
import numpy as np

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation, PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import generate_requests, round_up

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:0"

# 配置日志
logging.getLogger().setLevel(logging.INFO)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, 'npu'):
        torch.npu.manual_seed_all(seed)


set_seed(10000)


def _tensor_stats(tensor: torch.Tensor) -> str:
    flat = tensor.detach().cpu()
    if flat.numel() == 0:
        return "shape={}, empty tensor".format(tuple(flat.shape))
    return (
        f"shape={tuple(flat.shape)}, min={flat.min().item():.6f}, "
        f"max={flat.max().item():.6f}, mean={flat.float().mean().item():.6f}, "
        f"std={flat.float().std(unbiased=False).item():.6f}"
    )


def _print_debug(golden: torch.Tensor, test: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    golden_flat = golden.detach().cpu().flatten()
    test_flat = test.detach().cpu().flatten()
    diff = test_flat - golden_flat

    # 使用容差判断真正不匹配的元素（而不是简单的 != 比较）
    # 使用与 torch.testing.assert_close 相同的逻辑
    abs_diff = diff.abs()
    rel_diff = abs_diff / (golden_flat.abs() + rtol)
    mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)
    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).squeeze(-1)

    if mismatch_indices.numel() > 0:
        num_mismatches = mismatch_indices.numel()
        logging.info(f"Found {num_mismatches} mismatched elements "
                     f"(total: {golden_flat.numel()}, rtol={rtol}, atol={atol})")

        # 按顺序打印前20个错误
        max_print = min(20, num_mismatches)
        print_indices = mismatch_indices[:max_print]

        logging.info("Mismatched elements (position, golden_value, test_value, abs_diff, rel_diff):")
        for idx in print_indices:
            idx_val = idx.item()
            golden_val = golden_flat[idx_val].item()
            test_val = test_flat[idx_val].item()
            abs_diff_val = abs_diff[idx_val].item()
            rel_diff_val = rel_diff[idx_val].item()
            logging.info(f"  [{idx_val}]: golden={golden_val:.6f}, test={test_val:.6f}, "
                        f"abs_diff={abs_diff_val:.6e}, rel_diff={rel_diff_val:.6e}")

        if num_mismatches > max_print:
            logging.info(f"... and {num_mismatches - max_print} more mismatches (not shown)")
    else:
        logging.info("No mismatches found (within tolerance)")


def call_operator(
        op: IntNBitTableBatchedEmbeddingBagsCodegen,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
        dev_weights=op.weights_dev,
        uvm_weights=op.weights_uvm,
        weights_placements=op.weights_placements,
        weights_offsets=op.weights_offsets,
        weights_tys=op.weights_tys,
        D_offsets=op.D_offsets,
        total_D=op.total_D,
        max_int2_D=op.max_int2_D,
        max_int4_D=op.max_int4_D,
        max_int8_D=op.max_int8_D,
        max_float16_D=op.max_float16_D,
        max_float32_D=op.max_float32_D,
        indices=indices,
        offsets=offsets,
        pooling_mode=int(op.pooling_mode),
        indice_weights=per_sample_weights,
        output_dtype=op.output_dtype,
        lxu_cache_weights=None,
        lxu_cache_locations=None,
        row_alignment=op.row_alignment,
        max_float8_D=op.max_float8_D,
        fp8_exponent_bits=op.fp8_exponent_bits,
        fp8_exponent_bias=op.fp8_exponent_bias,
    )


def _build_module(
        *,
        embedding_specs: List[Tuple[int, int, SparseType]],
        pooling_mode: PoolingMode,
        output_dtype: SparseType,
        indices_dtype: torch.dtype,
        device: str,
) -> IntNBitTableBatchedEmbeddingBagsCodegen:
    is_cpu = device.startswith("cpu")
    location = EmbeddingLocation.HOST if is_cpu else EmbeddingLocation.DEVICE
    specs = [
        ("table", E, D, wty, location)
        for (E, D, wty) in embedding_specs
    ]
    return IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=specs,
        device=device,
        pooling_mode=pooling_mode,
        output_dtype=output_dtype,
        indices_dtype=indices_dtype,
    )


def _sync_random_weights(
        ref_module: IntNBitTableBatchedEmbeddingBagsCodegen,
        test_module: IntNBitTableBatchedEmbeddingBagsCodegen,
        weight_types: List[SparseType],
) -> None:
    ref_module.fill_random_weights()
    test_module.fill_random_weights()
    ref_split = ref_module.split_embedding_weights()
    test_split = test_module.split_embedding_weights()
    for (ref_w, ref_scale), (test_w, test_scale), _ in zip(ref_split, test_split, weight_types):
        test_w.copy_(ref_w)
        if ref_scale is not None and test_scale is not None:
            test_scale.copy_(ref_scale)


def _generate_workload(
        *,
        B: int,
        T: int,
        L: List[int],  # 每张表的bag长度: [L1, L2, ...]
        tables: List[Tuple[int, int]],  # 每张表的shape: [(E1, D1), (E2, D2), ...]
        weighted: bool,
        emulate_pruning: bool = False,
        indices_dtype: torch.dtype,
):
    """
    生成workload数据，支持每张表使用不同的bag长度

    Args:
        B: batch size
        T: 表的数量
        L: 每张表的bag长度列表，长度必须等于T
        tables: 每张表的shape列表
        weighted: 是否使用权重
        emulate_pruning: 是否模拟剪枝
        indices_dtype: indices的数据类型
    """
    assert len(L) == T, f"L的长度({len(L)})必须等于表的数量({T})"
    assert len(tables) == T, f"tables的长度({len(tables)})必须等于表的数量({T})"

    # 为每张表分别生成请求，然后合并
    all_indices_list = []
    all_weights_list = [] if weighted else None

    # 累积的indices数量，用于调整offsets
    current_indices_offset = 0

    for t in range(T):
        E_t, D_t = tables[t]
        L_t = L[t]

        # 为当前表生成请求
        requests = generate_requests(
            1, B, 1, L_t, E_t, reuse=0.1, weighted=weighted,
            emulate_pruning=emulate_pruning, use_cpu=True,
            deterministic_output=True,  # 确保生成可重复的测试数据
        )

        for req in requests:
            if weighted:
                indices_t, offsets_t, per_sample_weights_t = req.unpack_3()
            else:
                indices_t, offsets_t = req.unpack_2()
                per_sample_weights_t = None

            all_indices_list.append(indices_t)

            if weighted and per_sample_weights_t is not None:
                all_weights_list.append(per_sample_weights_t)

            # 更新累积的indices数量
            current_indices_offset += len(indices_t)

    # 合并所有表的数据
    all_indices = torch.cat(all_indices_list).to(dtype=indices_dtype)

    # 合并offsets：TBE格式是 [0, L1, 2*L1, ..., B*L1, B*L1+L2, B*L1+2*L2, ..., B*L1+B*L2, ...]
    # 即：每张表有B+1个offsets（包括起始的0），最后一张表的最后一个offset是总长度
    all_offsets = [0]
    for t in range(T):
        L_t = L[t]
        # 每张表的offsets（除了第一个0）
        for b in range(1, B + 1):
            all_offsets.append(all_offsets[-1] + L_t)
    all_offsets = torch.tensor(all_offsets, dtype=indices_dtype)

    if weighted and all_weights_list is not None:
        all_weights = torch.cat(all_weights_list)
    else:
        all_weights = None

    yield (all_indices, all_offsets, all_weights)


def _generate_table_config(
        table_num: int,
        L: int,
        is_nobag: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    生成随机表配置

    Args:
        table_num: 表的数量
        L: bag长度的取值范围
        is_nobag: 是否为nobag模式（nobag模式下所有表的D维度必须相同）

    Returns:
        (tables, L_list): tables是(E, D)的列表，L_list是每张表的bag长度列表
    """
    tables = []
    L_list = []

    # nobag模式下，所有表的D维度必须相同，在循环外生成
    if is_nobag:
        D_base = random.randint(1, 1024)
        D = D_base * 4

    for _ in range(table_num):
        # 行数：1-20000之间随机选择
        E = random.randint(1, 20000)

        # 列数：1-1024之间随机选择一个数乘以4（确保是4的倍数，符合FP8对齐要求）
        if not is_nobag:
            D_base = random.randint(1, 1024)
            D = D_base * 4

        tables.append((E, D))

        # bag长度：从1到L之间随机选择
        L_t = random.randint(1, L)
        L_list.append(L_t)

    return tables, L_list


@dataclass
class TestConfig:
    """测试配置参数"""
    pooling_mode: PoolingMode
    weighted: bool
    weights_ty: SparseType
    indices_dtype: torch.dtype
    output_dtype: SparseType
    B: int
    table_num: Optional[int] = None
    L: Optional[int] = None
    tables: Optional[List[Tuple[int, int]]] = None
    L_list: Optional[List[int]] = None


def _run_multi_table_test(config: TestConfig) -> None:
    """
    执行多表测试的公共逻辑

    Args:
        config: 测试配置参数
    """
    # 如果提供了预定义的表配置，使用它们；否则随机生成
    if config.tables is not None and config.L_list is not None:
        T = len(config.tables)
        assert len(config.L_list) == T, f"L_list的长度({len(config.L_list)})必须等于表的数量({T})"
        tables = config.tables
        L_list = config.L_list
    else:
        assert config.table_num is not None and config.L is not None, \
            "必须提供table_num和L（随机生成）或tables和L_list（预定义）"
        is_nobag = (config.pooling_mode == PoolingMode.NONE)
        tables, L_list = _generate_table_config(config.table_num, config.L, is_nobag=is_nobag)
        T = len(tables)

    # 生成embedding_specs，对D进行对齐处理
    embedding_specs = []
    for E, D in tables:
        D_aligned = round_up(D, max(config.weights_ty.align_size(), config.output_dtype.align_size()))
        embedding_specs.append((E, D_aligned, config.weights_ty))

    # 构建模块
    ref_module = _build_module(
        embedding_specs=embedding_specs,
        pooling_mode=config.pooling_mode,
        output_dtype=config.output_dtype,
        indices_dtype=config.indices_dtype,
        device="cpu",
    )
    test_module = _build_module(
        embedding_specs=embedding_specs,
        pooling_mode=config.pooling_mode,
        output_dtype=config.output_dtype,
        indices_dtype=config.indices_dtype,
        device=DEVICE,
    )

    # 同步权重
    _sync_random_weights(ref_module, test_module, [config.weights_ty] * T)

    # 执行测试
    for indices, offsets, per_sample_weights in _generate_workload(
            B=config.B, T=T, L=L_list, tables=tables, weighted=config.weighted, indices_dtype=config.indices_dtype
    ):
        psw = per_sample_weights
        golden_out = ref_module(
            indices=indices,
            offsets=offsets,
            per_sample_weights=psw,
        )
        indices_npu = indices.to(device=DEVICE)
        offsets_npu = offsets.to(device=DEVICE)
        psw_npu = psw.to(device=DEVICE) if psw is not None else None
        test_out = call_operator(test_module, indices_npu, offsets_npu, psw_npu)

        assert golden_out.shape == test_out.shape

        # 计算容差
        tol = (
            1e-3 if config.output_dtype == SparseType.FP16
            else 2 ** -7 if config.output_dtype == SparseType.BF16
            else 1e-4 if config.output_dtype == SparseType.FP32
            else 0
        )

        try:
            torch.testing.assert_close(test_out.cpu(), golden_out.cpu(), rtol=tol, atol=tol)
        except AssertionError as err:
            _print_debug(golden_out, test_out, rtol=tol, atol=tol)
            raise err


@pytest.mark.parametrize("pooling_mode", [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("output_dtype", [SparseType.FP16, SparseType.FP32, SparseType.BF16])
@pytest.mark.parametrize("table_num", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_random_multi_table_forward(
        pooling_mode: PoolingMode,
        weighted: bool,
        indices_dtype: torch.dtype,
        output_dtype: SparseType,
        table_num: int,
) -> None:
    """
    统一的多表测试用例，支持bag和nobag模式

    - 表的个数由table_num指定
    - 每张表的shape随机生成：行数在1-20000之间，列数是1-1024之间随机数乘以4（确保是4的倍数）
    - 每张表的bag长度从1-L之间随机选择
    - nobag模式下，所有表的D维度必须相同
    - nobag模式下，weighted必须为False，indices_dtype必须为int32（通过参数化限制）
    """
    # nobag模式的限制检查
    default_B = 64
    default_L = 100
    weights_ty = SparseType.FP8

    if pooling_mode == PoolingMode.NONE:
        if weighted:
            pytest.skip("nobag mode does not support weighted")
        if indices_dtype != torch.int32:
            pytest.skip("nobag mode only supports int32 indices")

    config = TestConfig(
        pooling_mode=pooling_mode,
        weighted=weighted,
        weights_ty=weights_ty,
        indices_dtype=indices_dtype,
        output_dtype=output_dtype,
        B=default_B,
        table_num=table_num,
        L=default_L,
    )
    _run_multi_table_test(config)


# Bag模式的表配置（支持不同D维度）
BAG_TABLES = [
    [(1000, 16), (2000, 16)], [(2000, 256), (1000, 256)],
    [(2000, 512), (1000, 512)], [(2000, 1024), (1000, 1024)],
    [(2000, 2048), (1000, 2048)], [(2000, 4096), (1000, 4096)],
    [(2000, 16), (1000, 256)], [(2000, 16), (1000, 512)],
    [(2000, 16), (1000, 1024)], [(2000, 16), (1000, 2048)],
    [(2000, 16), (1000, 4096)],
]

# Nobag模式的表配置（所有表的D维度相同）
NOBAG_TABLES = [
    [(1000, 16), (2000, 16)], [(1000, 32), (2000, 32)], [(1000, 64), (2000, 64)],
    [(1000, 128), (2000, 128)], [(2000, 256), (1000, 256)], [(2000, 512), (1000, 512)],
    [(2000, 1024), (1000, 1024)], [(2000, 2048), (1000, 2048)],
    [(2000, 4096), (1000, 4096)],
]


@pytest.mark.parametrize("pooling_mode", [PoolingMode.SUM, PoolingMode.MEAN])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("output_dtype", [SparseType.FP16, SparseType.FP32, SparseType.BF16])
@pytest.mark.parametrize("tables", BAG_TABLES)
def test_double_table_forward_bag(
        pooling_mode: PoolingMode,
        weighted: bool,
        indices_dtype: torch.dtype,
        output_dtype: SparseType,
        tables: List[Tuple[int, int]],
) -> None:
    """双表测试用例（Bag模式），使用预定义的表配置"""
    weights_ty = SparseType.FP8
    config = TestConfig(
        pooling_mode=pooling_mode,
        weighted=weighted,
        weights_ty=weights_ty,
        indices_dtype=indices_dtype,
        output_dtype=output_dtype,
        B=64,
        tables=tables,
        L_list=[40, 50],
    )
    _run_multi_table_test(config)


@pytest.mark.parametrize("pooling_mode", [PoolingMode.NONE])
@pytest.mark.parametrize("weighted", [False])
@pytest.mark.parametrize("indices_dtype", [torch.int32])
@pytest.mark.parametrize("output_dtype", [SparseType.FP16, SparseType.FP32, SparseType.BF16])
@pytest.mark.parametrize("tables", NOBAG_TABLES)
def test_double_table_forward_nobag(
        pooling_mode: PoolingMode,
        weighted: bool,
        indices_dtype: torch.dtype,
        output_dtype: SparseType,
        tables: List[Tuple[int, int]],
) -> None:
    """双表测试用例（Nobag模式），使用预定义的表配置"""
    weights_ty = SparseType.FP8
    config = TestConfig(
        pooling_mode=pooling_mode,
        weighted=weighted,
        weights_ty=weights_ty,
        indices_dtype=indices_dtype,
        output_dtype=output_dtype,
        B=64,
        tables=tables,
        L_list=[40, 50],
    )
    _run_multi_table_test(config)

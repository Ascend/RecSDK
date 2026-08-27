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
# pylint: disable=redefined-outer-name, duplicate-code
import sysconfig
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import pytest
import torch

from utils import (
    create_data_generator,
    ensure_hstu_custom_opp_path,
    BenchmarkRecord,
    SeqStats,
    create_batch_arbitrary_mask,
    create_k2q_sparse_info,
    get_block_q_kv,
)
from backend import KernelBackend, create_hstu_atten_backend

ensure_hstu_custom_opp_path()


@dataclass
class BenchmarkConfig:
    """Benchmark 测试配置参数"""

    test_name: str
    seed: int
    seq_all_equal: bool
    seq_max_ratio: float
    batch_size: int
    head_num: int
    head_dim_qk: int
    head_dim_v: int
    max_seqlen_q: int
    max_seqlen_k: int
    has_rab: bool
    groups: int
    data_type: torch.dtype


# 常量定义
ASCEND_DEVICE_ID = 0
ACTIVE = 20


class TestRunner:
    """测试运行器，封装测试逻辑"""

    def __init__(self, backends):
        self.backend = backends

    @staticmethod
    def _create_kernel(backend, alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k):
        """创建内核的统一方法"""
        return backend.kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k)

    def run_case(
        self,
        generator,
        batch_size: int,
        head_num: int,
        head_dim_qk: int,
        head_dim_v: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        has_rab: bool,
        data_type: torch.dtype,
        groups: int,
        is_metadata: bool,
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """运行单个测试用例

        Returns:
            (passed, detail, seq_stats): passed 为总体是否通过，detail 为详细精度数据，seq_stats 为序列长度统计
        """
        # 生成测试数据（arbitrary mask 路径下 window_size 取 (-1, -1)，num_context/num_target 置 None）
        grad, q, k, v, rab, _, seq_offset_q, seq_offset_k = generator.gen_data(
            batch_size,
            head_num,
            max_seqlen_q,
            max_seqlen_k,
            head_dim_qk,
            head_dim_v,
            has_rab,
            data_type,
            (-1, -1),
            None,
            None,
            None,
        )

        # 构造 arbitrary mask 与稀疏索引信息
        mask, arbitrary_func = create_batch_arbitrary_mask(
            batch_size, head_num, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k, groups, data_type
        )
        BLOCK_Q, BLOCK_KV = get_block_q_kv(head_dim_qk, head_dim_v, "bwd")
        k2q_sparse_info = create_k2q_sparse_info(mask, seq_offset_q, seq_offset_k, BLOCK_Q, BLOCK_KV)

        # 计算序列长度统计
        seq_stats = SeqStats.compute_seq_stats(seq_offset_q, seq_offset_k, max_seqlen_q, max_seqlen_k)

        # 配置参数
        scale = 1 / 1024
        alpha = 0.5

        # 创建内核
        kernel = self._create_kernel(
            self.backend, alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k
        )

        metadata = kernel.create_backward_metadata(q, v) if is_metadata else None
        for _ in range(ACTIVE):
            kernel.backward(
                grad, q, k, v, rab, arbitrary_func=arbitrary_func, sparse_info=k2q_sparse_info, metadata=metadata
            )

        # 验证结果，返回 (passed, detail, seq_stats)
        return seq_stats


# session 级别的 BenchmarkRecord
benchmark_record_instance = None


@pytest.fixture(scope="session")
def benchmark_record():
    """提供benchmark记录器的fixture（session级别，所有测试共享）"""
    global benchmark_record_instance
    if benchmark_record_instance is None:
        benchmark_record_instance = BenchmarkRecord("tmp_benchmark.csv")
    return benchmark_record_instance


def _run_benchmark(test_backend, test_record, config: BenchmarkConfig):
    runner = TestRunner(test_backend)
    for is_metadata in (False, True):
        # 两个分支使用相同 seed 和 shape，保证性能数据可直接对比。
        data_generator = create_data_generator(
            config.seed, seq_all_equal=config.seq_all_equal, seq_max_ratio=config.seq_max_ratio
        )

        seq_stats = runner.run_case(
            data_generator,
            config.batch_size,
            config.head_num,
            config.head_dim_qk,
            config.head_dim_v,
            config.max_seqlen_q,
            config.max_seqlen_k,
            config.has_rab,
            config.data_type,
            config.groups,
            is_metadata,
        )

        params = {
            "batch_size": config.batch_size,
            "head_num": config.head_num,
            "head_dim_qk": config.head_dim_qk,
            "head_dim_v": config.head_dim_v,
            "max_seqlen_q": config.max_seqlen_q,
            "max_seqlen_k": config.max_seqlen_k,
            "has_rab": config.has_rab,
            "data_type": str(config.data_type),
            "groups": config.groups,
            "seed": config.seed,
            "is_metadata": is_metadata,
        }
        test_record.record(params, seq_stats)


@pytest.fixture(scope="function")
def test_backend():
    """提供测试后端实例的fixture"""
    return create_hstu_atten_backend(
        KernelBackend.ASCEND_FUSE,
        device=ASCEND_DEVICE_ID,
        ops_library_dir=f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so",
    )


@pytest.mark.parametrize(
    "batch_size, head_num, seq_lens",
    [(32, 8, (512, 512)), (32, 8, (1024, 1024)), (32, 8, (2048, 2048)), (32, 8, (4096, 4096))],
)
@pytest.mark.parametrize("head_dims", [(64, 64), (128, 128)])
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("has_rab", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [123])
def test_user_case_1(
    test_backend,
    benchmark_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    groups,
    has_rab,
    data_type,
    seed,
):
    """测试用户用例"""
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    config = BenchmarkConfig(
        test_name="test_user_case_1",
        seed=seed,
        seq_all_equal=True,
        seq_max_ratio=0.9,
        batch_size=batch_size,
        head_num=head_num,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_rab=has_rab,
        groups=groups,
        data_type=data_type,
    )
    _run_benchmark(test_backend, benchmark_record, config)


@pytest.mark.parametrize(
    "batch_size, head_num, seq_lens",
    [(128, 4, (8186, 8186))],
)
@pytest.mark.parametrize("head_dims", [(128, 128)])
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("has_rab", [False])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
@pytest.mark.parametrize("seed", [123])
def test_user_case_2(
    test_backend,
    benchmark_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    groups,
    has_rab,
    data_type,
    seed,
):
    """测试用户用例"""
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    config = BenchmarkConfig(
        test_name="test_user_case_2",
        seed=seed,
        seq_all_equal=True,
        seq_max_ratio=0.9,
        batch_size=batch_size,
        head_num=head_num,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_rab=has_rab,
        groups=groups,
        data_type=data_type,
    )
    _run_benchmark(test_backend, benchmark_record, config)


if __name__ == "__main__":
    pytest.main([__file__])

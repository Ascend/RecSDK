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
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import sysconfig

import pytest
import torch

from utils import create_data_generator, ensure_hstu_custom_opp_path, Record, SeqStats
from backend import KernelBackend, create_hstu_atten_backend

# 常量定义
ASCEND_DEVICE_ID = 0

ensure_hstu_custom_opp_path()


@dataclass
class TestCaseParams:
    """测试用例参数封装"""

    test_name: str
    test_backends: Dict[str, Any]
    test_record: Any
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
    data_type: torch.dtype
    window_size: [Tuple[int, int]]
    num_context: Optional[int] = None
    num_target: Optional[int] = None
    target_group_size: Optional[int] = 1


class BackendFactory:
    """后端工厂类，统一管理后端创建"""

    @staticmethod
    def create_ascend_fuse_backend():
        """创建 Ascend 融合后端"""
        return create_hstu_atten_backend(
            KernelBackend.ASCEND_FUSE,
            device=ASCEND_DEVICE_ID,
            ops_library_dir=f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so",
        )

    @staticmethod
    def create_ascend_native_backend():
        """创建 Ascend 原生后端"""
        return create_hstu_atten_backend(KernelBackend.ASCEND_NATIVE)

    @staticmethod
    def create_pytorch_native_backend():
        """创建 pytorch cpu原生后端 双标杆"""
        return create_hstu_atten_backend(KernelBackend.PYTORCH_NATIVE)

    @staticmethod
    def get_ascend_backends() -> Dict[str, Any]:
        """获取所有 Ascend 后端"""
        return {
            "backend": BackendFactory.create_ascend_fuse_backend(),
            "ref_backend": BackendFactory.create_pytorch_native_backend(),
        }


class TestRunner:
    """测试运行器，封装测试逻辑"""

    def __init__(self, backends: Dict[str, Any]):
        self.backend = backends["backend"]
        self.ref_backend = backends["ref_backend"]
        self.validator = self.ref_backend.validator()

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
        window_size: [Tuple[int, int]],
        is_metadata: bool,
        num_context: Optional[int] = None,
        num_target: Optional[int] = None,
        target_group_size: Optional[int] = 1,
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """运行单个测试用例

        Returns:
            (passed, detail, seq_stats): passed 为总体是否通过，detail 为详细精度数据，seq_stats 为序列长度统计
        """
        # 生成测试数据
        grad, q, k, v, rab, mask, seq_offset_q, seq_offset_k = generator.gen_data(
            batch_size,
            head_num,
            max_seqlen_q,
            max_seqlen_k,
            head_dim_qk,
            head_dim_v,
            has_rab,
            data_type,
            window_size,
            num_context,
            num_target,
            target_group_size,
        )

        # 计算序列长度统计
        seq_stats = SeqStats.compute_seq_stats(seq_offset_q, seq_offset_k, max_seqlen_q, max_seqlen_k)

        # 配置参数
        scale = 1 / 1024
        alpha = 0.5

        # 创建内核
        # ascend_fuse_backend
        kernel = self._create_kernel(
            self.backend, alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k
        )
        # pytorch_native_backend
        ref_kernel = self._create_kernel(
            self.ref_backend, alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k
        )

        # 运行计算
        metadata = kernel.create_backward_metadata(q, v) if is_metadata else None
        actual = kernel.backward(
            grad, q, k, v, rab, mask, window_size, num_context, num_target, target_group_size, metadata
        )
        # expected包括原生精度golden,高精度golden结果
        expected = ref_kernel.backward(grad, q, k, v, rab, mask)

        # 验证结果，返回 (passed, detail, seq_stats)
        return self.validator.backward_verify(actual, expected), seq_stats


def _run_test_case(params: TestCaseParams):
    runner = TestRunner(params.test_backends)
    for is_metadata in (False, True):
        # 两个分支使用相同 seed 和 shape，确保输入数据完全一致。
        data_generator = create_data_generator(
            params.seed, seq_all_equal=params.seq_all_equal, seq_max_ratio=params.seq_max_ratio
        )

        (passed, detail), seq_stats = runner.run_case(
            data_generator,
            params.batch_size,
            params.head_num,
            params.head_dim_qk,
            params.head_dim_v,
            params.max_seqlen_q,
            params.max_seqlen_k,
            params.has_rab,
            params.data_type,
            params.window_size,
            is_metadata,
            params.num_context,
            params.num_target,
            params.target_group_size,
        )

        record_params = {
            "batch_size": params.batch_size,
            "head_num": params.head_num,
            "head_dim_qk": params.head_dim_qk,
            "head_dim_v": params.head_dim_v,
            "max_seqlen_q": params.max_seqlen_q,
            "max_seqlen_k": params.max_seqlen_k,
            "has_rab": params.has_rab,
            "data_type": str(params.data_type),
            "seed": params.seed,
            "is_metadata": is_metadata,
        }
        params.test_record.record(params.test_name, record_params, detail, seq_stats)

        assert passed, f"Test case failed: is_metadata={is_metadata}, detail={detail}"


@pytest.fixture(scope="function")
def test_backends():
    """提供测试后端实例的fixture"""
    return BackendFactory.get_ascend_backends()


# 使用 session 级别的 Record，确保所有测试结果写入同一个文件
test_record_instance = None


@pytest.fixture(scope="session")
def test_record():
    """提供测试结果记录器的fixture（session级别，所有测试共享）"""
    global test_record_instance
    if test_record_instance is None:
        test_record_instance = Record("test_results.xlsx")
    return test_record_instance


@pytest.fixture(scope="session", autouse=True)
def save_test_record(test_record):
    """在所有测试完成后自动保存结果"""
    yield
    # 测试会话结束后保存 Excel
    test_record.save()


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("head_num", [1])
@pytest.mark.parametrize("seq_lens", [(256, 256)])
@pytest.mark.parametrize("head_dims", [(32, 32)])
@pytest.mark.parametrize(
    "window_size,num_context,num_target,target_group_size",
    [
        ((-1, -1), None, None, None),
    ],
)
@pytest.mark.parametrize("has_rab", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [123])
def test_generic_batch_size(
    test_backends,
    test_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    window_size,
    num_context,
    num_target,
    target_group_size,
    has_rab,
    data_type,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    params = TestCaseParams(
        test_name="test_generic_batch_size",
        test_backends=test_backends,
        test_record=test_record,
        seed=seed,
        seq_all_equal=False,
        seq_max_ratio=0.9,
        batch_size=batch_size,
        head_num=head_num,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_rab=has_rab,
        data_type=data_type,
        window_size=window_size,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
    )
    _run_test_case(params)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("head_num", [1])
@pytest.mark.parametrize("seq_lens", [(256, 256)])
@pytest.mark.parametrize("head_dims", [(32, 32), (64, 64), (128, 128), (256, 256)])
@pytest.mark.parametrize(
    "window_size,num_context,num_target,target_group_size",
    [
        ((-1, -1), None, None, None),
    ],
)
@pytest.mark.parametrize("has_rab", [False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [123])
def test_generic_head_dim(
    test_backends,
    test_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    window_size,
    num_context,
    num_target,
    target_group_size,
    has_rab,
    data_type,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    params = TestCaseParams(
        test_name="test_generic_head_dim",
        test_backends=test_backends,
        test_record=test_record,
        seed=seed,
        seq_all_equal=False,
        seq_max_ratio=0.9,
        batch_size=batch_size,
        head_num=head_num,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_rab=has_rab,
        data_type=data_type,
        window_size=window_size,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
    )
    _run_test_case(params)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("head_num", [1, 4, 8, 16])
@pytest.mark.parametrize("seq_lens", [(256, 256)])
@pytest.mark.parametrize("head_dims", [(32, 32)])
@pytest.mark.parametrize(
    "window_size,num_context,num_target,target_group_size",
    [
        ((-1, -1), None, None, None),
    ],
)
@pytest.mark.parametrize("has_rab", [False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [123])
def test_generic_head_num(
    test_backends,
    test_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    window_size,
    num_context,
    num_target,
    target_group_size,
    has_rab,
    data_type,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    params = TestCaseParams(
        test_name="test_generic_head_num",
        test_backends=test_backends,
        test_record=test_record,
        seed=seed,
        seq_all_equal=False,
        seq_max_ratio=0.9,
        batch_size=batch_size,
        head_num=head_num,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_rab=has_rab,
        data_type=data_type,
        window_size=window_size,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
    )
    _run_test_case(params)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("head_num", [1, 4])
@pytest.mark.parametrize("seq_lens", [(256, 256)])
@pytest.mark.parametrize("head_dims", [(32, 32)])
@pytest.mark.parametrize(
    "window_size,num_context,num_target,target_group_size",
    [
        ((-1, -1), None, None, None),
    ],
)
@pytest.mark.parametrize("has_rab", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [123])
def test_generic_rab(
    test_backends,
    test_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    window_size,
    num_context,
    num_target,
    target_group_size,
    has_rab,
    data_type,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    params = TestCaseParams(
        test_name="test_generic_rab",
        test_backends=test_backends,
        test_record=test_record,
        seed=seed,
        seq_all_equal=False,
        seq_max_ratio=0.9,
        batch_size=batch_size,
        head_num=head_num,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_rab=has_rab,
        data_type=data_type,
        window_size=window_size,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
    )
    _run_test_case(params)


@pytest.mark.parametrize("batch_size, head_num", [(8, 1), (8, 4)])
@pytest.mark.parametrize("seq_lens", [(512, 512), (1024, 1024), (2048, 2048)])
@pytest.mark.parametrize("head_dims", [(32, 32), (128, 128)])
@pytest.mark.parametrize(
    "window_size,num_context,num_target,target_group_size",
    [
        ((-1, -1), None, None, None),
    ],
)
@pytest.mark.parametrize("has_rab", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [*range(10)])
def test_generic_sequence(
    test_backends,
    test_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    window_size,
    num_context,
    num_target,
    target_group_size,
    has_rab,
    data_type,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    params = TestCaseParams(
        test_name="test_generic_sequence",
        test_backends=test_backends,
        test_record=test_record,
        seed=seed,
        seq_all_equal=False,
        seq_max_ratio=0.9,
        batch_size=batch_size,
        head_num=head_num,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_rab=has_rab,
        data_type=data_type,
        window_size=window_size,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
    )
    _run_test_case(params)


@pytest.mark.parametrize(
    "batch_size, head_num, head_dims, seq_lens, has_rab, data_type",
    [
        (2048, 4, (64, 64), (52, 1000), False, torch.float16),
        (16, 4, (64, 64), (501, 1000), False, torch.float16),
        (2048, 2, (256, 256), (32, 32), False, torch.float16),
        (96, 2, (256, 256), (512, 3072), False, torch.float16),
        (16, 4, (64, 32), (32, 499), False, torch.float16),
        (4, 4, (128, 48), (1001, 901), False, torch.float16),
        (96, 3, (256, 256), (674, 674), False, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "window_size,num_context,num_target,target_group_size",
    [
        ((-1, -1), None, None, None),
    ],
)
@pytest.mark.parametrize("seed", [123])
def test_user_case_1(
    test_backends,
    test_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    window_size,
    num_context,
    num_target,
    target_group_size,
    has_rab,
    data_type,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    params = TestCaseParams(
        test_name="test_user_case_1",
        test_backends=test_backends,
        test_record=test_record,
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
        data_type=data_type,
        window_size=window_size,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
    )
    _run_test_case(params)


@pytest.mark.parametrize(
    "batch_size, head_num, seq_lens", [(32, 8, (512, 512)), (32, 8, (1024, 1024)), (32, 8, (2048, 2048))]
)
@pytest.mark.parametrize("head_dims", [(32, 32), (64, 64), (128, 128), (256, 256)])
@pytest.mark.parametrize(
    "window_size,num_context,num_target,target_group_size",
    [
        ((-1, -1), None, None, None),
    ],
)
@pytest.mark.parametrize("has_rab", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [123])
def test_user_case_2(
    test_backends,
    test_record,
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    window_size,
    num_context,
    num_target,
    target_group_size,
    has_rab,
    data_type,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    params = TestCaseParams(
        test_name="test_user_case_2",
        test_backends=test_backends,
        test_record=test_record,
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
        data_type=data_type,
        window_size=window_size,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
    )
    _run_test_case(params)

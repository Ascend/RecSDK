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

from pathlib import Path
import sysconfig

import numpy as np
import pytest
import torch
import torch_npu
import fbgemm_gpu

torch.npu.config.allow_internal_format = False
DEVICE = "npu:0"
REPEAT_TIMES = 3
EMB_DTYPES = [torch.int64]

CURR_DIR = Path(__file__).resolve().parent
_SINGLE_OP_SO = (
    CURR_DIR.parent.parent.parent
    / "framework/torch_plugin/torch_library/init_address_lookup/build/libinit_address_lookup.so"
)
_COMMON_SO = Path(sysconfig.get_path("purelib")) / "libfbgemm_npu_api.so"
for _so in [_SINGLE_OP_SO, _COMMON_SO]:
    if _so.exists():
        torch.ops.load_library(str(_so))
        break


def init_address_lookup_cpu(
    buffer_offsets: np.ndarray, emb_sizes: np.ndarray, output_dtype=None
) -> np.ndarray:
    """
    CPU 参考实现：初始化地址查找表

    Args:
        buffer_offsets: 形状 [num_tables + 1]，CSR格式的行偏移
        emb_sizes: 形状 [num_tables]，每个表的逻辑行数
        output_dtype: 输出类型，默认与emb_sizes一致

    Returns:
        address_lookups: 形状 [buffer_offsets[-1]]，初始化后的地址查找表
    """
    if output_dtype is None:
        output_dtype = emb_sizes.dtype if len(emb_sizes) > 0 else np.int64
    total_rows = buffer_offsets[-1] if len(buffer_offsets) > 0 else 0
    num_tables = len(emb_sizes)
    address_lookups = np.zeros(total_rows, dtype=output_dtype)

    for t in range(num_tables):
        start = buffer_offsets[t]
        end = buffer_offsets[t + 1]
        emb_size = emb_sizes[t]

        for r in range(end - start):
            idx = start + r
            if r < emb_size:
                address_lookups[idx] = r
            else:
                address_lookups[idx] = 0

    return address_lookups


def init_address_lookup_npu(
    buffer_offsets: np.ndarray,
    emb_sizes: np.ndarray,
    device: str,
    emb_dtype=torch.int64,
) -> np.ndarray:
    """NPU 算子调用"""
    torch.npu.set_device(device)

    buffer_offsets_tensor = torch.from_numpy(buffer_offsets).to(torch.int64).to(device)
    emb_sizes_tensor = (
        torch.from_numpy(
            emb_sizes.astype(np.int32 if emb_dtype == torch.int32 else np.int64)
        )
        .to(emb_dtype)
        .to(device)
    )

    total_rows = int(buffer_offsets[-1]) if len(buffer_offsets) > 0 else 0
    torch.npu.synchronize()
    for _ in range(REPEAT_TIMES):
        address_lookup_tensor = torch.empty(total_rows, dtype=emb_dtype, device=device)
        torch.ops.mxrec.init_address_lookup(
            address_lookup_tensor, buffer_offsets_tensor, emb_sizes_tensor
        )
        torch.npu.synchronize()

    return address_lookup_tensor.cpu().numpy()


class TestInitAddressLookup:
    """init_address_lookup 算子测试类"""

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_basic_case(self, device, emb_dtype):
        """基础功能测试：文档中的示例"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        buffer_offsets = np.array([0, 5, 9], dtype=np.int64)
        emb_sizes = np.array([3, 4], dtype=np_dtype)

        expected = np.array([0, 1, 2, 0, 0, 0, 1, 2, 3], dtype=np_dtype)
        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        assert np.array_equal(result_cpu, expected), (
            f"CPU result mismatch: {result_cpu}"
        )
        assert np.array_equal(result_npu, expected), (
            f"NPU result mismatch: {result_npu}"
        )

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_single_table(self, device, emb_dtype):
        """单表测试"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        buffer_offsets = np.array([0, 10], dtype=np.int64)
        emb_sizes = np.array([7], dtype=np_dtype)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        expected = np.array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0], dtype=np_dtype)
        assert np.array_equal(result_cpu, expected)
        assert np.array_equal(result_npu, expected)

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_full_table(self, device, emb_dtype):
        """测试emb_size等于buffer大小的情况（无空闲行）"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        buffer_offsets = np.array([0, 5, 13], dtype=np.int64)
        emb_sizes = np.array([5, 8], dtype=np_dtype)  # 完全匹配

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        # 所有位置都映射到自身
        expected = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np_dtype)
        assert np.array_equal(result_cpu, expected)
        assert np.array_equal(result_npu, expected)

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_empty_table(self, device, emb_dtype):
        """测试emb_size为0的情况（全部映射到0）"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        buffer_offsets = np.array([0, 5], dtype=np.int64)
        emb_sizes = np.array([0], dtype=np_dtype)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        expected = np.zeros(5, dtype=np_dtype)
        assert np.array_equal(result_cpu, expected)
        assert np.array_equal(result_npu, expected)

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("num_tables", [1, 3, 5, 10, 50])
    @pytest.mark.parametrize("device", [DEVICE])
    def test_multiple_tables(self, num_tables, device, emb_dtype):
        """多表测试"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        np.random.seed(42)
        buffer_sizes = np.random.randint(10, 100, size=num_tables)
        emb_sizes = np.array(
            [np.random.randint(0, size + 1) for size in buffer_sizes], dtype=np_dtype
        )
        buffer_offsets = np.concatenate([[0], np.cumsum(buffer_sizes)]).astype(np.int64)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        assert np.array_equal(result_cpu, result_npu), (
            f"Mismatch for {num_tables} tables, dtype={np_dtype}"
        )

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_boundary_values(self, device, emb_dtype):
        """边界值测试（0行表和1行表）"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        buffer_offsets = np.array([0, 0, 1, 2, 3], dtype=np.int64)
        emb_sizes = np.array([0, 1, 0, 1], dtype=np_dtype)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        expected = np.array([0, 0, 0], dtype=np_dtype)
        assert np.array_equal(result_cpu, expected)
        assert np.array_equal(result_npu, expected)

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_zeros(self, device, emb_dtype):
        """所有表物理行数为0测试"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        buffer_offsets = np.array([0, 0, 0, 0], dtype=np.int64)
        emb_sizes = np.array([0, 0, 0], dtype=np_dtype)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        expected = np.array([], dtype=np_dtype)
        assert np.array_equal(result_cpu, expected)
        assert np.array_equal(result_npu, expected)

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_empty(self, device, emb_dtype):
        """传入空表测试"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        buffer_offsets = np.array([], dtype=np.int64)
        emb_sizes = np.array([], dtype=np_dtype)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        expected = np.array([], dtype=np_dtype)
        assert np.array_equal(result_cpu, expected)
        assert np.array_equal(result_npu, expected)

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_small_table(self, device, emb_dtype):
        """小规模测试"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        num_tables = 100
        np.random.seed(123)
        buffer_sizes = np.random.randint(1000, 5000, size=num_tables)
        emb_sizes = np.array(
            [np.random.randint(0, size + 1) for size in buffer_sizes], dtype=np_dtype
        )
        buffer_offsets = np.concatenate([[0], np.cumsum(buffer_sizes)]).astype(np.int64)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        assert np.array_equal(result_cpu, result_npu), "Small table test failed"

    @pytest.mark.parametrize("emb_dtype", EMB_DTYPES)
    @pytest.mark.parametrize("device", [DEVICE])
    def test_medium_table(self, device, emb_dtype):
        """中规模测试"""
        np_dtype = np.int32 if emb_dtype == torch.int32 else np.int64
        num_tables = 500
        np.random.seed(2026)
        buffer_sizes = np.random.randint(5000, 20000, size=num_tables)
        emb_sizes = np.array(
            [np.random.randint(0, size + 1) for size in buffer_sizes], dtype=np_dtype
        )
        buffer_offsets = np.concatenate([[0], np.cumsum(buffer_sizes)]).astype(np.int64)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        assert np.array_equal(result_cpu, result_npu), "Medium table test failed"

    @pytest.mark.parametrize("emb_dtype", [torch.int64])
    @pytest.mark.parametrize("device", [DEVICE])
    def test_extra_large_table(self, device, emb_dtype):
        """超大表测试"""
        np_dtype = np.int64
        num_tables = 2
        np.random.seed(9)
        buffer_sizes = np.random.randint(1, 50_000_000, size=num_tables)
        emb_sizes = np.array(
            [np.random.randint(0, size + 1) for size in buffer_sizes], dtype=np_dtype
        )
        buffer_offsets = np.concatenate([[0], np.cumsum(buffer_sizes)]).astype(np.int64)

        result_cpu = init_address_lookup_cpu(buffer_offsets, emb_sizes)
        result_npu = init_address_lookup_npu(
            buffer_offsets, emb_sizes, device, emb_dtype
        )

        assert np.array_equal(result_cpu, result_npu), "Extra large table test failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

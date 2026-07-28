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
"""hstu_attn_metadata (AI CPU 自定义算子) 的 PyTorch 绑定测试。

依赖：
  1. 先编译 AICPU kernel / aclnn opapi 并安装到 vendor：
         cd cust_op/ascendc_op/ai_core_op/hstu_attn_metadata && bash run.sh --stage=install
     源码树直接运行时，需指向真正包含 op_impl/ 的 vendor 子目录：
         export ASCEND_CUSTOM_OPP_PATH=$PWD/build/vendor/hstu_attn_metadata_transformer
  2. 编译 PTA（common/build_ops.sh 会把 hstu_attn_metadata 打进 libfbgemm_npu_api.so）：
         cd cust_op/framework/torch_plugin/torch_library/common && bash build_ops.sh
"""

import sysconfig

import pytest
import torch

torch.npu.config.allow_internal_format = False

# metadata 布局常量，与 op_kernel_aicpu/hstu_attn_metadata.h 一致
AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
METADATA_STRIDE = 16
METADATA_ALIGN = 4096
HEAD_SECTION_NUM_INDEX = 0
HEAD_IS_FD_INDEX = 1
HEAD_M_BASE_SIZE_INDEX = 2
HEAD_S2_BASE_SIZE_INDEX = 3
FA_BN2_START_INDEX = 0
FA_M_START_INDEX = 1
FA_S2_START_INDEX = 2
FA_BN2_END_INDEX = 3
FA_M_END_INDEX = 4
FA_S2_END_INDEX = 5

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def expected_size(batch, num_kv_heads):
    elems = ((AIC_CORE_NUM + AIV_CORE_NUM) * batch * num_kv_heads + 1) * METADATA_STRIDE
    return ((elems + METADATA_ALIGN - 1) // METADATA_ALIGN) * METADATA_ALIGN


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def flatten_row_block(bn2, m_idx, row_blocks_per_batch, num_heads):
    """与 MetadataRowBlockScheduler::FlattenRowBlock 保持一致。"""
    total_bn2 = len(row_blocks_per_batch) * num_heads
    assert 0 <= bn2 <= total_bn2, f"bn2 越界: {bn2}, total={total_bn2}"
    if bn2 == total_bn2:
        assert m_idx == 0, f"末尾坐标必须是 ({total_bn2}, 0)，实际 m={m_idx}"
        return num_heads * sum(row_blocks_per_batch)

    batch_idx, head_idx = divmod(bn2, num_heads)
    row_blocks = row_blocks_per_batch[batch_idx]
    assert 0 <= m_idx <= row_blocks, f"mIdx 越界: bn2={bn2}, m={m_idx}, batch row blocks={row_blocks}"
    return num_heads * sum(row_blocks_per_batch[:batch_idx]) + head_idx * row_blocks + m_idx


def assert_all_row_blocks_covered(host_metadata, kv_seq_lens, num_heads):
    """验证 FA metadata 恰好覆盖 HSTU backward 的全部 K 行块，且没有重叠或遗漏。"""
    section_num = int(host_metadata[HEAD_SECTION_NUM_INDEX])
    is_fd = int(host_metadata[HEAD_IS_FD_INDEX])
    m_base = int(host_metadata[HEAD_M_BASE_SIZE_INDEX])
    assert section_num >= 1
    assert is_fd == 0, f"HSTU metadata 不应产生 FD，实际 isFd={is_fd}"
    assert m_base > 0

    row_blocks_per_batch = [ceil_div(seq_len, m_base) for seq_len in kv_seq_lens]
    expected_block_count = num_heads * sum(row_blocks_per_batch)
    intervals = []

    for section_idx in range(section_num):
        for core_idx in range(AIC_CORE_NUM):
            offset = METADATA_STRIDE + (section_idx * AIC_CORE_NUM + core_idx) * METADATA_STRIDE
            record = [int(v) for v in host_metadata[offset : offset + 6]]
            bn2_start = record[FA_BN2_START_INDEX]
            m_start = record[FA_M_START_INDEX]
            bn2_end = record[FA_BN2_END_INDEX]
            m_end = record[FA_M_END_INDEX]

            start = flatten_row_block(bn2_start, m_start, row_blocks_per_batch, num_heads)
            end = flatten_row_block(bn2_end, m_end, row_blocks_per_batch, num_heads)
            assert end >= start, (
                f"FA区间倒序: section={section_idx}, core={core_idx}, record={record}, flat=[{start}, {end})"
            )
            if end == start:
                continue

            intervals.append((start, end, section_idx, core_idx))

    intervals.sort()
    covered_blocks = []
    expected_start = 0
    for start, end, section_idx, core_idx in intervals:
        assert start == expected_start, (
            f"block覆盖不连续: 期望从{expected_start}开始，实际区间[{start}, {end})，"
            f"section={section_idx}, core={core_idx}"
        )
        covered_blocks.extend(range(start, end))
        expected_start = end

    assert covered_blocks == list(range(expected_block_count)), (
        f"block覆盖错误: expected={expected_block_count}, covered={len(covered_blocks)}, intervals={intervals}"
    )


def run_hstu_attn_metadata(
    device,
    *,
    batch_size,
    max_seqlen_q,
    max_seqlen_kv,
    num_heads_q,
    num_heads_kv,
    head_dim,
    mask_mode,
    win_left,
    win_right,
    layout,
    seqused_q=None,
    seqused_kv=None,
):
    torch.npu.set_device(device)
    return torch.ops.mxrec.hstu_attn_metadata(
        None,
        None,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        mask_mode,
        win_left,
        win_right,
        layout,
        layout,
        layout,
    )


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("kv_s", [1024, 8192])
@pytest.mark.parametrize("device", ["npu:0"])
def test_hstu_attn_metadata_attrs_only(batch_size, num_heads, kv_s, device):
    """仅通过属性(batch_size/max_seqlen)驱动，不传 seqused_* 张量，对应 C++ example 场景。"""
    metadata = run_hstu_attn_metadata(
        device,
        batch_size=batch_size,
        max_seqlen_q=1,
        max_seqlen_kv=kv_s,
        num_heads_q=num_heads,
        num_heads_kv=num_heads,
        head_dim=128,
        mask_mode=3,
        win_left=-1,
        win_right=-1,
        layout="BSND",
    )

    assert metadata.dtype == torch.int32
    assert metadata.device.type == "npu"
    assert metadata.numel() == expected_size(batch_size, num_heads)

    host = metadata.cpu()
    section_num = int(host[HEAD_SECTION_NUM_INDEX])
    m_base = int(host[HEAD_M_BASE_SIZE_INDEX])
    s2_base = int(host[HEAD_S2_BASE_SIZE_INDEX])
    assert section_num >= 1, f"sectionNum 非法: {section_num}"
    assert m_base > 0, f"mBaseSize 非法: {m_base}"
    assert s2_base > 0, f"s2BaseSize 非法: {s2_base}"
    assert_all_row_blocks_covered(host, [kv_s] * batch_size, num_heads)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("device", ["npu:0"])
def test_hstu_attn_metadata_with_seqused(batch_size, device):
    """通过 seqused_q / seqused_kv 张量提供实际序列长度。"""
    torch.npu.set_device(device)
    num_heads = 16
    query_seq_lens = [1, 127, 128, 385]
    kv_seq_lens = [257, 1024, 4096, 8192]
    seqused_q = torch.tensor(query_seq_lens, dtype=torch.int32, device=device)
    seqused_kv = torch.tensor(kv_seq_lens, dtype=torch.int32, device=device)

    metadata = run_hstu_attn_metadata(
        device,
        batch_size=-1,
        max_seqlen_q=max(query_seq_lens),
        max_seqlen_kv=max(kv_seq_lens),
        num_heads_q=num_heads,
        num_heads_kv=num_heads,
        head_dim=128,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout="BSND",
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
    )

    assert metadata.dtype == torch.int32
    assert metadata.numel() == expected_size(batch_size, num_heads)
    host = metadata.cpu()
    assert int(host[HEAD_SECTION_NUM_INDEX]) >= 1
    assert_all_row_blocks_covered(host, kv_seq_lens, num_heads)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))

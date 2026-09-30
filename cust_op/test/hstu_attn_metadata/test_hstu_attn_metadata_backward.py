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
"""hstu_attn_metadata_backward (AI CPU 自定义算子) 的 PyTorch 绑定测试。

算子语义：由 q/k 的 offsets（cu_seqlens）出发，为 **HSTU 反向的 K 轴行调度**产出 metadata。
调度单元是 K 行块（行块大小 BLOCK_M 由 head_dim 决定）：

    BN2 = batch * num_heads 条，第 b 个 batch 的每条 BN2 有 ceil(seqLenK[b] / BLOCK_M) 个行块

算子把这些行块切成若干 section（按 L2 预算），再在 section 内做 persist 分核，
把每个核负责的半开区间 [start, end) 写进 FA 槽位。设备侧读法见
`hstu_v2/c310/op_kernel/catlass_hstu/gemm/block/metadata_row_block_scheduler.hpp`
的 `LoadSection` / `FlattenRowBlock`。

验收的核心是一条 **覆盖性**：全部 section / 全部核槽位的有效区间并集，
恰好等于整张 K 行块网格且互不重叠 —— 一个块都不能漏，也不能被两个核算两遍。
（块数少于核数时切出空核是合法的：设备侧读到 blockCnt == 0 即空转。）

依赖：
  1. 编译并安装 AICPU kernel + aclnn vendor：
         cd cust_op/ascendc_op/ai_core_op/hstu_attn_metadata_backward && bash run.sh --stage=install
     源码树直接运行时，需指向真正包含 op_impl/ 的 vendor 子目录：
         export ASCEND_CUSTOM_OPP_PATH=$PWD/build/vendor/hstu_attn_metadata_backward_transformer
  2. 编译 PTA（common/build_ops.sh 会把本算子打进 libfbgemm_npu_api.so）：
         cd cust_op/framework/torch_plugin/torch_library/common && bash build_ops.sh

运行：
    pytest -sv cust_op/test/hstu_attn_metadata/test_hstu_attn_metadata_backward.py
"""

import os
import sysconfig

import pytest
import torch

# =============================================================================
# 布局常量：**独立声明**，与算子头 op_kernel_aicpu/hstu_attn_metadata_backward.h 各持一份。
#
# 刻意不复用算子侧的任何常量：写入侧（算子）与读取侧（本测试）各有一份独立声明，
# 任何一侧改了 stride / 下标就会立刻对不上。这套「按消费方读法独立回读」的做法
# 与 host_ut 里 test_build_metadata.cpp 的思路一致。
# =============================================================================
AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
HEAD_STRIDE = 16
FA_STRIDE = 16
FD_STRIDE = 16
METADATA_ALIGN = 4096  # metadata 张量按 4096 个 int32 对齐分配

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
FA_FIRST_FD_WS_INDEX = 6

OP_NAME = "hstu_attn_metadata_backward"
DEVICE_ID = 0
DEVICE = f"npu:{DEVICE_ID}"
torch.npu.set_device(DEVICE)

# 块档位真值表：与 hstu_v2 反向的编译期 L1TileShape 一一对应（TILE_K = head_dim > 128 ? 256 : 128）。
#   TILE_K=128 → L1TileShape<256,128,128> → BLOCK_M=128, BLOCK_N=256
#   TILE_K=256 → L1TileShape<128, 64,256> → BLOCK_M= 64, BLOCK_N=128
BLOCK_SHAPE_TABLE = ((128, 256), (64, 128))  # (BLOCK_M, BLOCK_N)，按 head_dim > 128 选择


# ---------------------------------------------------------------------------
# 环境探测：算子未注册 / 无 NPU 时整文件 skip，而不是报一堆 ImportError / RuntimeError
# ---------------------------------------------------------------------------
try:
    import torch_npu  # noqa: F401  # 注册 torch.npu

    _HAS_TORCH_NPU = True
except Exception:  # pragma: no cover - 纯 CPU 环境
    _HAS_TORCH_NPU = False

if _HAS_TORCH_NPU:
    # 自定义算子按 ACL_FORMAT_ND 走，关掉内部格式可避免框架自动做 NZ 转换
    torch.npu.config.allow_internal_format = False

_LIB_PATH = os.path.join(sysconfig.get_path("purelib"), "libfbgemm_npu_api.so")


def _is_op_available():
    if not _HAS_TORCH_NPU or not torch.npu.is_available():
        return False
    if not os.path.exists(_LIB_PATH):
        return False
    try:
        torch.ops.load_library(_LIB_PATH)
    except Exception:  # pragma: no cover - 已加载过或环境不匹配
        return False
    return hasattr(torch.ops.mxrec, OP_NAME)


pytestmark = pytest.mark.skipif(
    not _is_op_available(),
    reason=(
        "hstu_attn_metadata_backward 未注册：请先 bash run.sh --stage=install 安装 vendor，"
        "并用 cust_op/framework/torch_plugin/torch_library/common/build_ops.sh 编译出 libfbgemm_npu_api.so"
    ),
)


# =============================================================================
# 参考模型：镜像设备侧的读法（FlattenRowBlock / LoadSection）
# =============================================================================
def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def expected_block_shape(head_dim):
    """由 head_dim 推 (BLOCK_M, BLOCK_N)，与算子内 DeriveBlockShape 同口径但独立实现。"""
    return BLOCK_SHAPE_TABLE[1] if head_dim > 128 else BLOCK_SHAPE_TABLE[0]


def expected_metadata_size(batch_size, num_heads):
    """绑定侧的输出预分配量：sectionNum 取上界 batch*num_heads，再按 4096 元素对齐。"""
    section_num_max = batch_size * num_heads
    elems = HEAD_STRIDE + section_num_max * (AIC_CORE_NUM * FA_STRIDE + AIV_CORE_NUM * FD_STRIDE)
    return ceil_div(elems, METADATA_ALIGN) * METADATA_ALIGN


def flat_row_block(bn2, m_idx, row_blocks_per_batch, num_heads):
    """镜像 MetadataRowBlockScheduler::FlattenRowBlock。

        flatten(bn2, m) = Σ_{x<bn2} ceil(seqLenK(x / num_heads) / BLOCK_M) + m

    同一 batch 内各 head 的序列长度相同，故上式等价于
        num_heads * Σ_{b<batch_idx} blocks[b] + head_idx * blocks[batch_idx] + m
    """
    total_bn2 = len(row_blocks_per_batch) * num_heads
    assert 0 <= bn2 <= total_bn2, f"bn2 越界: {bn2}, total={total_bn2}"
    if bn2 == total_bn2:
        assert m_idx == 0, f"末尾坐标必须是 ({total_bn2}, 0)，实际 m={m_idx}"
        return num_heads * sum(row_blocks_per_batch)

    batch_idx, head_idx = divmod(bn2, num_heads)
    blocks = row_blocks_per_batch[batch_idx]
    assert 0 <= m_idx <= blocks, f"m 越界: bn2={bn2}, m={m_idx}, batch row blocks={blocks}"
    return num_heads * sum(row_blocks_per_batch[:batch_idx]) + head_idx * blocks + m_idx


def fa_record(host_metadata, section_idx, core_idx):
    """取 FA[section][core] 的整条记录（16 个 int32）。"""
    base = HEAD_STRIDE + (section_idx * AIC_CORE_NUM + core_idx) * FA_STRIDE
    return [int(v) for v in host_metadata[base : base + FA_STRIDE]]


def fd_region(host_metadata, section_num):
    """FD 区的下标范围 [base, end)。"""
    base = HEAD_STRIDE + section_num * AIC_CORE_NUM * FA_STRIDE
    end = base + section_num * AIV_CORE_NUM * FD_STRIDE
    return base, end


def used_metadata_len(metadata):
    section_num = int(metadata[HEAD_SECTION_NUM_INDEX])
    return HEAD_STRIDE + section_num * (AIC_CORE_NUM * FA_STRIDE + AIV_CORE_NUM * FD_STRIDE)


def assert_metadata_wellformed(host_metadata, seqlen_k, num_heads, head_dim):
    """按设备侧读法回读并校验 metadata。返回 (sectionNum, 用到的核数, 区间列表)。

    硬性不变量只有一条 —— 覆盖性；其余（空槽位全 0、保留位为 0、FD 区全 0）
    属于「与 ClearMetadata 一致的实现选择」，作为形状钉一并钉住。
    """
    block_m, block_n = expected_block_shape(head_dim)
    section_num = int(host_metadata[HEAD_SECTION_NUM_INDEX])

    # ---- HEAD 段 ----
    assert section_num >= 1, f"sectionNum 必须 >= 1，实际 {section_num}"
    assert section_num <= len(seqlen_k) * num_heads, (
        f"sectionNum 超过上界 batch*num_heads: {section_num} > {len(seqlen_k) * num_heads}"
    )
    assert int(host_metadata[HEAD_IS_FD_INDEX]) == 0, "反向不使用 FD，isFd 必须为 0"
    assert int(host_metadata[HEAD_M_BASE_SIZE_INDEX]) == block_m, (
        f"mBaseSize 与 head_dim={head_dim} 不符：期望 {block_m}，实际 {int(host_metadata[HEAD_M_BASE_SIZE_INDEX])}"
    )
    assert int(host_metadata[HEAD_S2_BASE_SIZE_INDEX]) == block_n, (
        f"s2BaseSize 与 head_dim={head_dim} 不符：期望 {block_n}，实际 {int(host_metadata[HEAD_S2_BASE_SIZE_INDEX])}"
    )

    row_blocks = [ceil_div(s, block_m) for s in seqlen_k]
    total_blocks = num_heads * sum(row_blocks)

    # ---- FA 段：逐 section 逐核解析 ----
    intervals = []
    used_cores_per_section = []
    for sec in range(section_num):
        first_empty = AIC_CORE_NUM  # 空槽位必须从某个位置起一整段连续到尾
        used = 0
        for core in range(AIC_CORE_NUM):
            rec = fa_record(host_metadata, sec, core)

            # 保留位：反向不用 FD，s2 与首个 FD workspace 下标恒 0（有效槽与空槽都一样）
            assert (rec[FA_S2_START_INDEX], rec[FA_S2_END_INDEX], rec[FA_FIRST_FD_WS_INDEX]) == (0, 0, 0), (
                f"反向的 s2 / fdWsIdx 必须是保留位 0: section={sec}, core={core}, record={rec[:7]}"
            )

            start = flat_row_block(rec[FA_BN2_START_INDEX], rec[FA_M_START_INDEX], row_blocks, num_heads)
            end = flat_row_block(rec[FA_BN2_END_INDEX], rec[FA_M_END_INDEX], row_blocks, num_heads)
            assert end >= start, f"FA 区间倒序: section={sec}, core={core}, record={rec[:6]}, flat=[{start}, {end})"
            assert end <= total_blocks, f"FA 区间越界: section={sec}, core={core}, end={end} > {total_blocks}"

            if all(v == 0 for v in rec):
                # 未用槽位：④ 不显式写，靠 ClearMetadata 归零
                first_empty = min(first_empty, core)
                assert end == start, f"未用槽位展平后长度必须为 0: section={sec}, core={core}"
            else:
                # 有效槽位必须从 core 0 开始连续排布，中间不能有空档
                assert core < first_empty, f"section={sec} 的有效槽位出现在空槽之后（core={core}），核槽没有连续前缀"
                used += 1
                if end > start:
                    intervals.append((start, end, sec, core))
        used_cores_per_section.append(used)
        assert used <= AIC_CORE_NUM

    # ---- 覆盖性：有效区间按展平位置排序后，必须无缝铺满 [0, total_blocks) ----
    intervals.sort()
    cursor = 0
    for start, end, sec, core in intervals:
        assert start == cursor, (
            f"K 行块覆盖不连续或重叠：期望从 {cursor} 开始，实际区间 [{start}, {end}) "
            f"(section={sec}, core={core})，全部区间={intervals}"
        )
        cursor = end
    assert cursor == total_blocks, (
        f"K 行块覆盖不完整：期望覆盖 {total_blocks} 块（num_heads={num_heads}, row_blocks={row_blocks}），"
        f"实际 {cursor}，区间={intervals}"
    )

    # ---- FD 区：整段恒 0 ----
    fd_base, fd_end = fd_region(host_metadata, section_num)
    assert fd_end <= host_metadata.numel(), f"FD 区越出 metadata 缓冲: fd_end={fd_end} > numel={host_metadata.numel()}"
    assert all(int(v) == 0 for v in host_metadata[fd_base:fd_end]), "反向不使用 FD，FD 区必须整段为 0"

    return section_num, sum(used_cores_per_section), intervals


# =============================================================================
# 调用辅助
# =============================================================================
def to_offsets(seq_lens):
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    return offsets


def run_backward_metadata(
    cu_seqlens_q,
    cu_seqlens_kv,
    num_heads,
    head_dim,
    dtype=torch.int32,
    mask_mode=None,
    num_contexts=None,
    num_targets=None,
    target_group_size=0,
    full_cnt=None,
    mask_cnt=None,
):
    offsets_q = torch.tensor(cu_seqlens_q, dtype=dtype, device=DEVICE)
    offsets_kv = torch.tensor(cu_seqlens_kv, dtype=dtype, device=DEVICE)
    if mask_mode is None:
        # 省略 mask_mode，走 schema 默认值 0（no mask）
        return torch.ops.mxrec.hstu_attn_metadata_backward(offsets_q, offsets_kv, num_heads, head_dim)
    # causal 专属段：num_contexts / num_targets（None = 未提供，列表转张量）；
    # arbitrary 专属段：full_cnt / mask_cnt（测试直接传张量，None = 未提供）
    ctx = None if num_contexts is None else torch.tensor(num_contexts, dtype=dtype, device=DEVICE)
    tgt = None if num_targets is None else torch.tensor(num_targets, dtype=dtype, device=DEVICE)
    return torch.ops.mxrec.hstu_attn_metadata_backward(
        offsets_q, offsets_kv, num_heads, head_dim, mask_mode, ctx, tgt, target_group_size, full_cnt, mask_cnt
    )


# =============================================================================
# 用例
# =============================================================================
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [1024, 8192])
def test_metadata_layout_and_coverage(batch_size, num_heads, kv_len):
    """主用例：形状/dtype/尺寸 + HEAD 字段 + 全覆盖。"""
    seqlen_q = [128] * batch_size
    seqlen_k = [kv_len] * batch_size

    metadata = run_backward_metadata(to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, head_dim=128)

    assert metadata.dtype == torch.int32
    assert metadata.device.type == "npu"
    assert metadata.numel() == expected_metadata_size(batch_size, num_heads)

    section_num, used, _ = assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=128)
    assert section_num >= 1
    assert 1 <= used <= section_num * AIC_CORE_NUM


def test_heavy_shape_splits_into_multiple_sections():
    """重负载必须真正切出多个 section —— 否则上一条用例只覆盖了单 section 的写法。"""
    batch_size, num_heads, kv_len = 4, 32, 8192
    seqlen_k = [kv_len] * batch_size

    metadata = run_backward_metadata(to_offsets([128] * batch_size), to_offsets(seqlen_k), num_heads, head_dim=128)

    section_num, _, _ = assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=128)
    # 单头驻留 ≈ 128*768 + 8192*512 ≈ 4.29MB，全部核同时铺开超过 96MB 共享 L2，
    # 故不该命中「单头即可独占每核 L2」这条短路。
    assert section_num >= 2, f"重负载应切出多个 section，实际 {section_num}"


@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_block_shape_follows_head_dim(head_dim):
    """块档位由 head_dim 决定；m 下标是按 BLOCK_M 编的，故覆盖性同时也校验了档位真值。"""
    num_heads = 8
    seqlen_k = [4096, 2048]

    metadata = run_backward_metadata(to_offsets([512, 256]), to_offsets(seqlen_k), num_heads, head_dim=head_dim)

    block_m, block_n = expected_block_shape(head_dim)
    host = metadata.cpu()
    assert int(host[HEAD_M_BASE_SIZE_INDEX]) == block_m
    assert int(host[HEAD_S2_BASE_SIZE_INDEX]) == block_n
    assert_metadata_wellformed(host, seqlen_k, num_heads, head_dim=head_dim)


def test_ragged_and_zero_length_sequences():
    """变长 + 空序列：seqLenK=0 的 batch 没有行块，但不能让后面的块被漏掉。"""
    num_heads = 4
    seqlen_q = [0, 100, 256, 300]
    seqlen_k = [0, 0, 512, 384]

    metadata = run_backward_metadata(to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, head_dim=128)

    host = metadata.cpu()
    # 只有 batch2 的 4 块 + batch3 的 3 块
    assert_metadata_wellformed(host, seqlen_k, num_heads, head_dim=128)
    total_expected = num_heads * (ceil_div(512, 128) + ceil_div(384, 128))
    assert total_expected == num_heads * 7


def test_all_sequences_empty_yields_single_empty_section():
    """K 全空的退化输入：仍要产出一个合法 section，且一个行块都不覆盖。"""
    num_heads, batch_size = 2, 3
    seqlen_k = [0] * batch_size

    metadata = run_backward_metadata(to_offsets([1] * batch_size), to_offsets(seqlen_k), num_heads, head_dim=128)

    section_num, _, intervals = assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=128)
    assert section_num == 1
    assert not intervals


def test_repeated_calls_are_deterministic():
    """同一份入参重复调用必须得到逐元素相同的结果。

    每次调用都是新分配的输出缓冲，这条能抓到「某段区间忘了清 / 状态跨调用泄漏」。
    """
    num_heads = 8
    seqlen_k = [1024, 2048, 512]

    first = run_backward_metadata(to_offsets([128, 256, 64]), to_offsets(seqlen_k), num_heads, head_dim=128).cpu()
    second = run_backward_metadata(to_offsets([128, 256, 64]), to_offsets(seqlen_k), num_heads, head_dim=128).cpu()

    assert_metadata_wellformed(first, seqlen_k, num_heads, head_dim=128)
    used = used_metadata_len(first)
    assert used == used_metadata_len(second), "两次调用的 section_num 应一致"
    assert torch.equal(first[:used], second[:used]), "两次调用结果不一致，说明输出缓存未被完整初始化或被脏数据污染"


def test_int64_offsets_are_accepted():
    """offsets 的 dtype 白名单是 int32/int64，两条路径应给出一致的结果。"""
    num_heads = 4
    seqlen_q, seqlen_k = [128, 256], [1024, 512]

    as_int32 = run_backward_metadata(
        to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, 128, dtype=torch.int32
    ).cpu()
    as_int64 = run_backward_metadata(
        to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, 128, dtype=torch.int64
    ).cpu()

    assert_metadata_wellformed(as_int32, seqlen_k, num_heads, head_dim=128)
    assert torch.equal(as_int32, as_int64), "int32 与 int64 的 offsets 必须产出相同的 metadata"


def test_covers_every_block_for_many_shapes():
    """小规模但多组几何量各跑一遍，防止覆盖性只在单一 shape 上巧合成立。"""
    cases = [
        ([1], [1], 1, 128),
        ([127], [129], 1, 128),
        ([128], [128], 3, 128),
        ([300, 700], [400, 900], 2, 128),
        ([1024, 33, 7], [33, 1024, 1], 5, 256),
        ([4096], [4096], 16, 256),
    ]
    for seqlen_q, seqlen_k, num_heads, head_dim in cases:
        metadata = run_backward_metadata(to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, head_dim)
        assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=head_dim)


def test_invalid_num_heads_is_rejected():
    offsets_q = torch.tensor([0, 128], dtype=torch.int32, device=DEVICE)
    offsets_kv = torch.tensor([0, 128], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops.mxrec.hstu_attn_metadata_backward(offsets_q, offsets_kv, 0, 128)


@pytest.mark.parametrize("bad_head_dim", [16, 100, 512])
def test_invalid_head_dim_is_rejected(bad_head_dim):
    offsets_q = torch.tensor([0, 128], dtype=torch.int32, device=DEVICE)
    offsets_kv = torch.tensor([0, 128], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops.mxrec.hstu_attn_metadata_backward(offsets_q, offsets_kv, 4, bad_head_dim)


def test_mask_mode_defaults_to_no_mask():
    """mask_mode 带默认值 0：省略与显式传 0 必须逐元素一致。"""
    seqlen_q, seqlen_k = [256, 512], [256, 512]
    num_heads, head_dim = 2, 128
    default_meta = run_backward_metadata(to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, head_dim)
    explicit_meta = run_backward_metadata(to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, head_dim, mask_mode=0)
    used = used_metadata_len(default_meta)
    assert used == used_metadata_len(explicit_meta), "两次调用的 section_num 应一致"
    assert torch.equal(default_meta[:used], explicit_meta[:used]), "mask_mode 默认值应与显式 0 等价"


@pytest.mark.parametrize("bad_mask_mode", [-1, 3])
def test_unsupported_mask_mode_is_rejected(bad_mask_mode):
    """-1/3 越界：必须在绑定或 host 校验处被拒（1=causal、2=arbitrary 已实现放行）。"""
    with pytest.raises(RuntimeError):
        run_backward_metadata([0, 128], [0, 128], 2, 128, mask_mode=bad_mask_mode)


def test_arbitrary_mask_mode_is_unlocked():
    """af（mask_mode=2）端到端：cnt 合法时跑通，且覆盖性不受 mask 权重影响。

    行块网格不因 mask 改变 —— 权重 = full_cnt + mask_cnt 只影响各核区间的**切点**，
    不影响「全覆盖 + 不重叠」这条不变式（见 kernel 侧 mask/mask_predictor.h 文件头）。
    """
    seqlen_q, seqlen_k = [128, 64], [300, 128]
    num_heads, head_dim = 2, 128
    # head_dim=128 → BLOCK_M=128 → kBlkCnt=[ceil(300/128)=3, 1] → maxBlkCntK=3 → cnt numel = batch*3 = 6
    full_cnt = torch.tensor([2, 2, 1, 0, 0, 0], dtype=torch.int32, device=DEVICE)
    mask_cnt = torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.int32, device=DEVICE)

    metadata = run_backward_metadata(
        to_offsets(seqlen_q),
        to_offsets(seqlen_k),
        num_heads,
        head_dim,
        mask_mode=2,
        full_cnt=full_cnt,
        mask_cnt=mask_cnt,
    )

    assert metadata.dtype == torch.int32
    assert metadata.numel() == expected_metadata_size(len(seqlen_q), num_heads)
    section_num, _, _ = assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=head_dim)
    assert section_num >= 1


def test_arbitrary_mask_mode_requires_cnt():
    """af 下 cnt 双双缺席 / 缺席其一：绑定层拒绝（配对规则）。"""
    with pytest.raises(RuntimeError):
        run_backward_metadata([0, 128], [0, 128], 2, 128, mask_mode=2)
    full_cnt = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        run_backward_metadata([0, 128], [0, 128], 2, 128, mask_mode=2, full_cnt=full_cnt)


def test_cnt_rejected_for_other_mask_modes():
    """full_cnt / mask_cnt 是 af 专属输入：no-mask 下出现即拒（槽位分段固定）。"""
    full_cnt = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    mask_cnt = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        run_backward_metadata([0, 128], [0, 128], 2, 128, mask_mode=0, full_cnt=full_cnt, mask_cnt=mask_cnt)


def test_arbitrary_cnt_bad_dtype_or_numel_is_rejected():
    """af 下 cnt dtype 非 int32（host 拦）/ numel != batch*maxBlkCntK（kernel 拦）。"""
    # head_dim=128 → BLOCK_M=128；kv_len=128 → maxBlkCntK=1 → batch=1 ⇒ 期望 cnt numel=1
    good = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    bad_dtype = torch.tensor([1], dtype=torch.int64, device=DEVICE)
    with pytest.raises(RuntimeError):
        run_backward_metadata([0, 128], [0, 128], 2, 128, mask_mode=2, full_cnt=bad_dtype, mask_cnt=good)


def test_arbitrary_cnt_bad_numel_is_rejected():
    """af 下 cnt numel != batch*maxBlkCntK（kernel 拦）。"""
    bad_numel = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        run_backward_metadata([0, 128], [0, 128], 2, 128, mask_mode=2, full_cnt=bad_numel, mask_cnt=bad_numel)
        torch.npu.synchronize()
    torch.npu.stop_device(DEVICE_ID)
    torch.npu.restart_device(DEVICE_ID)


# =============================================================================
# causal（mask_mode = 1）端到端（用例吸收自 PR 3202，适配本文件的无 device 参数风格）
#
# causal 只影响 SplitCoresPersist 的权重分布（掩码跳块使有效权重变小），
# metadata 的布局契约与「全覆盖、不重叠」不变量与 no_mask 完全相同，
# 故直接复用 assert_metadata_wellformed；权重闭式的口径由 kernel 侧设计文档的
# 离线对拍（verify_device_semantics）背书。
# =============================================================================
def test_causal_mask_layout_and_coverage():
    """causal 主用例：history + context + target 全量配置下的布局 / 覆盖性 / 确定性。"""
    batch_size, num_heads, head_dim = 4, 8, 128
    # causal 前置：seqLenQ <= seqLenK（deltaQK >= 0，当前 predictor 的约束域）
    seqlen_q = [512, 256, 1024, 128]
    seqlen_k = [512, 512, 1024, 512]
    num_contexts = [64, 0, 128, 32]
    num_targets = [32, 16, 0, 64]
    target_group_size = 16

    metadata = run_backward_metadata(
        to_offsets(seqlen_q),
        to_offsets(seqlen_k),
        num_heads,
        head_dim,
        mask_mode=1,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
    )

    assert metadata.dtype == torch.int32
    assert metadata.numel() == expected_metadata_size(batch_size, num_heads)
    section_num, used, _ = assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=head_dim)
    assert section_num >= 1
    assert 1 <= used <= section_num * AIC_CORE_NUM

    again = run_backward_metadata(
        to_offsets(seqlen_q),
        to_offsets(seqlen_k),
        num_heads,
        head_dim,
        mask_mode=1,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
    )
    used = used_metadata_len(metadata.cpu())
    assert used == used_metadata_len(again.cpu()), "两次调用的 section_num 应一致"
    assert torch.equal(metadata.cpu()[:used], again.cpu()[:used]), "causal 重复调用应逐元素一致"


def test_causal_heavy_shape_splits_into_multiple_sections():
    """causal 全量（history+context+target）大负载必须真正切出多个 section。

    与 no-mask 版 test_heavy_shape_splits_into_multiple_sections 对应，验证 causal 路径在
    多 section + 多核下同样成立。causal 只影响 ③ SplitCoresPersist 的块权重（target τ
    封顶使 K 尾部行块权重变小，各核区间长短因此不均），② SplitSections 的 L2 驻留模型
    与 mask 无关，section 断点与同 seqLen 的 no-mask 完全一致。

    触发多 section：单 BN2 驻留 = seqLenQ*headDim*2*3 + seqLenK*headDim*2*2 超过
    l2Byte/aicCoreNum = 96MB/36 ≈ 2.67MB 时短路 C（maxHeadBytes <= l2/aicCoreNum）不命中，
    贪心 first-fit 沿 BN2 累计、驻留跨过 96MB 时在「当前 BN2 之前」断开，本 case 得 2 个 section。
    """
    num_heads, head_dim = 8, 128
    # causal 前置 seqLenQ <= seqLenK；K=8192 → 每 batch 64 个行块
    seqlen_q = [512, 1024, 2048, 4096]
    seqlen_k = [8192, 8192, 8192, 8192]
    num_contexts = [128, 256, 512, 64]
    num_targets = [32, 64, 256, 16]
    target_group_size = 16

    metadata = run_backward_metadata(
        to_offsets(seqlen_q),
        to_offsets(seqlen_k),
        num_heads,
        head_dim,
        mask_mode=1,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
    )

    section_num, used, _ = assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=head_dim)
    assert section_num >= 2, f"causal 重负载应切出多个 section，实际 {section_num}"
    assert 1 <= used <= section_num * AIC_CORE_NUM
    assert used >= AIC_CORE_NUM, f"causal 多 section 用例应多核并行，实际用核 {used}"

    again = run_backward_metadata(
        to_offsets(seqlen_q),
        to_offsets(seqlen_k),
        num_heads,
        head_dim,
        mask_mode=1,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
    )
    used = used_metadata_len(metadata.cpu())
    assert used == used_metadata_len(again.cpu()), "两次调用的 section_num 应一致"
    assert torch.equal(metadata.cpu()[:used], again.cpu()[:used]), "causal 重复调用应逐元素一致"


@pytest.mark.parametrize(
    "num_contexts,num_targets,target_group_size",
    [
        (None, None, 0),  # 两个可选张量都不传：纯 history-only
        ([0, 0], [0, 0], 0),  # 显式全零 + tgs=0：等价 history-only 退化
        ([64, 128], None, 0),  # 仅 context
        (None, [32, 64], 16),  # 仅 target
    ],
)
def test_causal_mask_degenerate_configs(num_contexts, num_targets, target_group_size):
    """causal 各退化配置都必须产出合法 metadata（覆盖性不变量成立）。"""
    num_heads, head_dim = 4, 128
    seqlen_q, seqlen_k = [256, 512], [256, 512]

    metadata = run_backward_metadata(
        to_offsets(seqlen_q),
        to_offsets(seqlen_k),
        num_heads,
        head_dim,
        mask_mode=1,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
    )
    assert_metadata_wellformed(metadata.cpu(), seqlen_k, num_heads, head_dim=head_dim)


def test_causal_history_only_configs_are_equivalent():
    """不传可选张量 vs 显式全零 + tgs=0：同为 history-only（numContext=numTarget=0、无 τ cap），
    分核输入完全相同，metadata 必须逐元素一致。
    """
    num_heads, head_dim = 4, 128
    seqlen_q, seqlen_k = [256, 512], [256, 512]

    implicit_meta = run_backward_metadata(to_offsets(seqlen_q), to_offsets(seqlen_k), num_heads, head_dim, mask_mode=1)
    explicit_meta = run_backward_metadata(
        to_offsets(seqlen_q),
        to_offsets(seqlen_k),
        num_heads,
        head_dim,
        mask_mode=1,
        num_contexts=[0, 0],
        num_targets=[0, 0],
        target_group_size=0,
    )
    used = used_metadata_len(implicit_meta.cpu())
    assert used == used_metadata_len(explicit_meta.cpu()), "两次调用的 section_num 应一致"
    assert torch.equal(implicit_meta.cpu()[:used], explicit_meta.cpu()[:used]), (
        "history-only 的隐式（不传）与显式（全零）配置应逐元素一致"
    )


def test_causal_optional_tensor_length_mismatch_is_rejected():
    """num_contexts / num_targets 非空时长度必须等于 batch（绑定侧先拦一道）。"""
    with pytest.raises(RuntimeError):
        run_backward_metadata(
            [0, 256, 768],
            [0, 256, 768],
            2,
            128,
            mask_mode=1,
            num_contexts=[64],  # 长度 1 != batch 2
        )
    with pytest.raises(RuntimeError):
        run_backward_metadata(
            [0, 256, 768],
            [0, 256, 768],
            2,
            128,
            mask_mode=1,
            num_targets=[32, 64, 128],  # 长度 3 != batch 2
        )


def test_causal_rejected_with_arbitrary_inputs():
    """槽位分段固定：causal 下出现 full_cnt / mask_cnt 即拒（与 af 的对偶方向）。"""
    full_cnt = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    mask_cnt = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        run_backward_metadata(
            [0, 128],
            [0, 128],
            2,
            128,
            mask_mode=1,
            full_cnt=full_cnt,
            mask_cnt=mask_cnt,
        )


def test_mismatched_offsets_length_is_rejected():
    offsets_q = torch.tensor([0, 128, 256], dtype=torch.int32, device=DEVICE)
    offsets_kv = torch.tensor([0, 128], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops.mxrec.hstu_attn_metadata_backward(offsets_q, offsets_kv, 4, 128)


def test_invalid_offsets_dtype_is_rejected():
    offsets_q = torch.tensor([0.0, 128.0], dtype=torch.float32, device=DEVICE)
    offsets_kv = torch.tensor([0.0, 128.0], dtype=torch.float32, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops.mxrec.hstu_attn_metadata_backward(offsets_q, offsets_kv, 4, 128)


def test_offsets_not_starting_at_zero_is_rejected():
    """offsets 首元素必须为 0、且单调不减 —— 这两条只有看到数据才能判定，由 kernel 侧拦下。"""
    num_heads = 2
    bad_kv = [1, 128]  # 首元素非 0
    with pytest.raises(RuntimeError):
        run_backward_metadata([0, 128], bad_kv, num_heads, 128)
        torch.npu.synchronize()
    torch.npu.stop_device(DEVICE_ID)
    torch.npu.restart_device(DEVICE_ID)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))

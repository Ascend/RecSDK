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
"""对 test_hstu_attn_metadata_backward.py 里的**断言逻辑本身**做变异测试。

正式用例需要 NPU 才能跑，但断言逻辑（尤其是展平 / 覆盖性那几十行下标运算）是最容易写错、
也最难靠眼睛看出来的部分。本脚本用「按 ④ 的写盘规则合成的良构 metadata」离线验证两件事：

  1. 良构输入必须全部通过 —— 不能有假阳性（否则上板会误报）；
  2. 每一类缺陷都必须被抓住 —— 不能有假阴性（否则测试等于没写）。

改完断言后先跑这个，再上板跑正式用例。

运行（不需要 NPU，只需要 torch 可导入）：
    python3 cust_op/test/hstu_attn_metadata/selfcheck_metadata_assertions.py
"""

import importlib.util
import os
import sys

import torch

TEST_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_hstu_attn_metadata_backward.py")

spec = importlib.util.spec_from_file_location("t_hstu_bwd", TEST_PY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ceil_div = mod.ceil_div
expected_block_shape = mod.expected_block_shape
expected_metadata_size = mod.expected_metadata_size
flat_row_block = mod.flat_row_block
assert_metadata_wellformed = mod.assert_metadata_wellformed

print("[OK] 模块导入成功（本地无 NPU，pytestmark 会整文件 skip，属正常）")


# ---------------------------------------------------------------------------
# 合成良构 metadata
# ---------------------------------------------------------------------------
def flat_starts(row_blocks, num_heads):
    """每个 BN2 起点对应的展平位置；末尾多一个哨兵，长度 = total_bn2 + 1。"""
    out = []
    pos = 0
    total_bn2 = len(row_blocks) * num_heads
    for bn2 in range(total_bn2):
        out.append(pos)
        pos += row_blocks[bn2 // num_heads]
    out.append(pos)
    return out


def coord_of(pos, starts, total_bn2):
    """展平位置 → 规范坐标。优先取「进位」形式 (bn2+1, 0)，与算子 MakeBlockEndPoint 一致。"""
    for bn2 in range(total_bn2 + 1):
        if starts[bn2] == pos:
            return (bn2, 0)
        if bn2 < total_bn2 and starts[bn2] < pos < starts[bn2 + 1]:
            return (bn2, pos - starts[bn2])
    raise AssertionError(f"展平位置 {pos} 无对应坐标")


def make_good(section_plan, seqlen_k, num_heads, head_dim):
    """按 ④ 的写盘规则合成一份良构 metadata。

    section_plan: [(bn2_begin, bn2_end, [每核负责的块数, ...]), ...]
    """
    block_m, block_n = expected_block_shape(head_dim)
    row_blocks = [ceil_div(s, block_m) for s in seqlen_k]
    total_bn2 = len(seqlen_k) * num_heads
    starts = flat_starts(row_blocks, num_heads)
    section_num = len(section_plan)

    buf = [0] * expected_metadata_size(len(seqlen_k), num_heads)
    buf[mod.HEAD_SECTION_NUM_INDEX] = section_num
    buf[mod.HEAD_IS_FD_INDEX] = 0
    buf[mod.HEAD_M_BASE_SIZE_INDEX] = block_m
    buf[mod.HEAD_S2_BASE_SIZE_INDEX] = block_n

    prev_end = 0
    for sec, (b0, b1, core_blocks) in enumerate(section_plan):
        assert b0 == prev_end, f"section {sec} 起点 {b0} 未与前一段 {prev_end} 接上"
        prev_end = b1
        f0, f1 = starts[b0], starts[b1]
        assert f1 - f0 == sum(core_blocks), f"section={sec} 块数不符：区间内 {f1 - f0} 块，计划 {sum(core_blocks)}"

        cur = f0
        for core, cnt in enumerate(core_blocks):
            nxt = cur + cnt
            base = mod.HEAD_STRIDE + (sec * mod.AIC_CORE_NUM + core) * mod.FA_STRIDE
            cs = coord_of(cur, starts, total_bn2)
            ce = coord_of(nxt, starts, total_bn2)
            buf[base + mod.FA_BN2_START_INDEX] = cs[0]
            buf[base + mod.FA_M_START_INDEX] = cs[1]
            buf[base + mod.FA_BN2_END_INDEX] = ce[0]
            buf[base + mod.FA_M_END_INDEX] = ce[1]
            cur = nxt
        assert prev_end <= total_bn2
    assert prev_end == total_bn2, f"section 未铺满 BN2：{prev_end} != {total_bn2}"
    return buf


def as_tensor(buf):
    return torch.tensor(buf, dtype=torch.int32)


results = []


def expect_pass(name, buf, seqlen_k, num_heads, head_dim):
    try:
        assert_metadata_wellformed(as_tensor(buf), seqlen_k, num_heads, head_dim)
    except AssertionError as exc:
        print(f"[FAIL] {name}: 良构输入被误判 -> {exc}")
        results.append(False)
        return
    print(f"[ OK ] {name}: 良构输入通过")
    results.append(True)


def expect_caught(name, buf, seqlen_k, num_heads, head_dim):
    try:
        assert_metadata_wellformed(as_tensor(buf), seqlen_k, num_heads, head_dim)
    except AssertionError as exc:
        print(f"[ OK ] {name}: 已抓住 -> {str(exc).splitlines()[0][:100]}")
        results.append(True)
        return
    print(f"[FAIL] {name}: 缺陷未被抓住！")
    results.append(False)


# ---------------------------------------------------------------------------
# 一、良构输入必须通过
# ---------------------------------------------------------------------------
CASES = [
    ("单 section / 单核", [256], 1, 128, [(0, 1, [2])]),
    ("单 section / 3 核", [1024], 1, 128, [(0, 1, [3, 3, 2])]),
    ("多 section / 多核", [4096, 2048], 2, 128, [(0, 2, [22, 21, 21]), (2, 4, [16, 16])]),
    ("小块档 head_dim=256", [4096], 2, 256, [(0, 2, [64, 64])]),
    ("含空序列", [0, 512], 2, 128, [(0, 4, [3, 3, 2])]),
    ("K 全空", [0, 0], 2, 128, [(0, 4, [0])]),
]
for case_name, case_seqlen_k, case_num_heads, case_head_dim, plan in CASES:
    expect_pass(
        f"良构 {case_name}",
        make_good(plan, case_seqlen_k, case_num_heads, case_head_dim),
        case_seqlen_k,
        case_num_heads,
        case_head_dim,
    )

# ---------------------------------------------------------------------------
# 二、每类缺陷都必须被抓住
# ---------------------------------------------------------------------------
print("\n--- 变异：每类缺陷都应被抓住 ---")
base_seqlen_k, base_num_heads, base_head_dim = [1024], 1, 128
good = make_good([(0, 1, [3, 3, 2])], base_seqlen_k, base_num_heads, base_head_dim)


def mutate(fn, name):
    buf = list(good)
    fn(buf)
    expect_caught(name, buf, base_seqlen_k, base_num_heads, base_head_dim)


S = mod.HEAD_STRIDE
F = mod.FA_STRIDE


def CORE(core):
    return S + core * F


mutate(lambda b: b.__setitem__(mod.HEAD_M_BASE_SIZE_INDEX, 256), "mBaseSize 与 head_dim 不符")
mutate(lambda b: b.__setitem__(mod.HEAD_IS_FD_INDEX, 1), "isFd 非 0")
mutate(lambda b: b.__setitem__(mod.HEAD_SECTION_NUM_INDEX, 0), "sectionNum 为 0")
mutate(lambda b: b.__setitem__(CORE(10) + mod.FA_BN2_END_INDEX, 1), "未用核槽位非 0")
mutate(lambda b: b.__setitem__(S + mod.AIC_CORE_NUM * F, 1), "FD 区非 0")
mutate(lambda b: b.__setitem__(CORE(0) + mod.FA_S2_START_INDEX, 1), "FA s2Start 保留位非 0")
mutate(lambda b: b.__setitem__(CORE(0) + mod.FA_FIRST_FD_WS_INDEX, 9), "FA fdWsIdx 保留位非 0")
mutate(
    lambda b: b.__setitem__(CORE(2) + mod.FA_BN2_END_INDEX, b[CORE(2) + mod.FA_BN2_END_INDEX] - 1),
    "覆盖不完整（末尾少一块）",
)
mutate(lambda b: b.__setitem__(CORE(2) + mod.FA_M_END_INDEX, 7), "尾部缺口（末核未推到最后一块）")
mutate(lambda b: b.__setitem__(CORE(1) + mod.FA_M_START_INDEX, 0), "覆盖重叠（核1 起点未接力）")


def blank_core1(buf):
    for idx in range(F):
        buf[CORE(1) + idx] = 0


mutate(blank_core1, "有效槽位出现在空槽之后")
mutate(lambda b: b.__setitem__(CORE(0) + mod.FA_BN2_START_INDEX, 1), "FA 区间倒序")

print(f"\n=== 结论：{sum(results)}/{len(results)} 项符合预期 ===")
sys.exit(0 if all(results) else 1)

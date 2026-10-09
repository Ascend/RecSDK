#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
# pylint: disable=redefined-outer-name
# Pytest fixtures (kvcache_mgr / g_keys / g_values) are intentionally reused
# as test-function parameter names; that is how pytest injects fixtures, not
# an accidental shadowing of an outer-scope variable.
import time

import pytest
import torch
import torch_npu  # noqa: F401

from recsys_kvcache_manager_npu.host_kvstorage_manager import HostKVTaskStatus
from recsys_kvcache_manager_npu.kvcache_config import get_kvcache_config
from recsys_kvcache_manager_npu.kvcache_manager import KVCacheManager

NPU_AVAILABLE = False
try:
    torch.npu.current_device()
    NPU_AVAILABLE = True
except Exception:
    NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")

MAX_SEQUENCE_LENGTHS = [
    2500,
    1024,
    1280,
    2377,
    1854,
    1365,
    2729,
    2049,
    689,
    1417,
    1174,
    1987,
    596,
    520,
    1538,
    1189,
]


def create_testing_kvcache_manager():
    """创建测试用 KVCacheManager 实例，配置较小参数以便快速验证"""
    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=1024 * 2 * 32 * 4 * 128 * 2,
        max_batch_size=8,
        # Must cover the largest sequence length actually exercised below
        # (phase 5 grows uid 6 to 2729 tokens): NPUKVCacheManagerImpl::
        # alloc_single_sequence() has a hard
        # TORCH_CHECK(new_total_length <= max_sequence_length), so a smaller
        # bound here would make phase 5's allocate_kvcache() raise. Using the
        # design doc's documented typical value (4096, already page-aligned)
        # rather than an exact match to 2729, for headroom.
        max_seq_len=4096,
        dtype=torch.bfloat16,
        device=torch.npu.current_device(),
        host_kvstorage_backend="native_host",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
    )
    npu_mem = (
        kvcache_config.num_layers
        * kvcache_config.num_primary_cache_pages
        * kvcache_config.page_size
        * 2
        * kvcache_config.num_heads
        * kvcache_config.head_dim
        * 2
    ) / (1024.0**3)
    print(f"[TEST] KVCache NPU Memory Usage: {npu_mem} GiB.")

    host_mem = (kvcache_config.num_layers * kvcache_config.host_capacity_per_layer) / (1024.0**3)
    print(f"[TEST] KVCache Host Memory Usage: {host_mem} GiB.")

    mgr = KVCacheManager.from_config(kvcache_config)
    print("[TEST] Created KVCache Manager")
    return mgr


@pytest.fixture
def kvcache_mgr():
    # Function-scoped: each test gets its own fresh manager. Tests that need
    # a prior phase's cumulative state explicitly replay it via the shared
    # _phase_xx() helpers below, instead of relying on pytest's default
    # top-to-bottom run order over a single shared instance.
    return create_testing_kvcache_manager()


@pytest.fixture(scope="module")
def g_keys():
    torch.manual_seed(0)
    return [
        torch.randn((3, MAX_SEQUENCE_LENGTHS[i], 4, 128), dtype=torch.bfloat16).npu()
        for i in range(len(MAX_SEQUENCE_LENGTHS))
    ]


@pytest.fixture(scope="module")
def g_values():
    # Distinct seed from g_keys so key/value tensors don't end up numerically
    # identical, which would silently hide a K/V-swap bug behind these tests.
    torch.manual_seed(1)
    return [
        torch.randn((3, MAX_SEQUENCE_LENGTHS[i], 4, 128), dtype=torch.bfloat16).npu()
        for i in range(len(MAX_SEQUENCE_LENGTHS))
    ]


def wait_offload_complete(kvcache_mgr, retries=3000, interval_s=0.01):
    """Poll offload_try_wait() until ongoing_offload_tasks drains.

    Returns True once drained, False if the retry budget is exhausted
    (callers must assert on the return value instead of trusting an
    unbounded loop that could hang CI forever).
    """
    for _ in range(retries):
        kvcache_mgr.offload_try_wait()
        if len(kvcache_mgr.ongoing_offload_tasks) == 0:
            return True
        time.sleep(interval_s)
    return False


def _phase_1(kvcache_mgr, g_keys, g_values):
    """阶段A: 首次分配 + 全量offload
    - 8 个 user 首次分配到 NPU，验证页索引和元数据正确
    - put 写入 KV 数据，验证 paged cache 读写一致性
    - offload 到 Host，验证 D2H 搬运数据完整性
    """
    seqlen = [700, 128, 336, 624, 486, 358, 716, 537]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    # 验证：首次查询，NPU 和 Host 均无缓存，起止位置和长度全零
    assert torch.allclose(lookup_res.cached_start_indices.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.cached_lengths.cpu(), torch.zeros((8,), dtype=torch.int32))

    # 分配页并验证元数据：页索引连续、indptr 正确、last_page_len 对齐
    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    assert torch.allclose(
        kvcache_metadata.kv_indices.cpu(),
        torch.tensor(list(range(0, 125)), dtype=torch.int32),
    )
    assert torch.allclose(
        kvcache_metadata.kv_indptr.cpu(),
        torch.tensor([0, 22, 26, 37, 57, 73, 85, 108, 125], dtype=torch.int32),
    )
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len.cpu(),
        torch.tensor([i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32),
    )
    assert torch.allclose(kvcache_metadata.total_history_lengths.cpu(), sequence_lengths)
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:].cpu() - kvcache_metadata.total_history_offsets[:-1].cpu(),
        sequence_lengths,
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:].cpu() - kvcache_metadata.new_history_offsets[:-1].cpu(),
        sequence_lengths,
    )

    assert torch.allclose(
        kvcache_metadata.new_history_nnz_npu.cpu(),
        torch.tensor(
            [
                sum(seqlen),
            ],
            dtype=torch.int32,
        ),
    )
    assert kvcache_metadata.new_history_nnz == sum(seqlen)

    for layer_idx in range(3):
        kvcache_mgr.npu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in keys], dim=0),
            torch.cat([v[layer_idx] for v in values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    # 验证 NPU paged cache 读写一致性：get 出来的数据应与 put 写入的完全一致
    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.npu_kvcache_mgr.get(page_ids, last_page_lens, layer_idx)
            assert torch.allclose(cached_k.cpu(), k[layer_idx].cpu()), (
                f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            )
            assert torch.allclose(cached_v.cpu(), v[layer_idx].cpu()), (
                f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"
            )

    # Offload：将 NPU 上的 KV cache 异步搬运到 Host pinned memory
    task_handle = kvcache_mgr.offload_launch(index_meta)
    assert task_handle is not None, "offload_launch returned None: nothing was offloaded"
    assert wait_offload_complete(kvcache_mgr), "offload did not complete within the retry budget"

    # 验证 Host 侧 offload 数据完整性（向下对齐到 page_size=32）
    host_len = [seqlen[i] // 32 * 32 for i in range(8)]
    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

        kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
            [
                uid,
            ],
            False,
        )[0]
        cached_k, cached_v = kvdata.unbind(dim=2)
        cached_k = cached_k.reshape(cached_k.size(0), -1, cached_k.size(3), cached_k.size(4))
        cached_v = cached_v.reshape(cached_v.size(0), -1, cached_v.size(3), cached_v.size(4))
        assert torch.allclose(cached_k, k.cpu()), f"Key mismatch for uid {user_ids[i].item()}"
        assert torch.allclose(cached_v, v.cpu()), f"Value mismatch for uid {user_ids[i].item()}"

    torch.npu.synchronize()


def _phase_2(kvcache_mgr, g_keys, g_values):
    """阶段B: 增量分配 + onboard 跳过（NPU已有完整数据）+ 部分offload
    - 序列长度增长，新增部分页
    - onboard 被跳过（NPU 侧已有比 Host 更新的数据）
    - 追加写入新 token 的 KV 数据
    - offload 更长的序列到 Host
    """
    cachedlen = [700, 128, 336, 624, 486, 358, 716, 537]
    seqlen = [1400, 256, 672, 1248, 973, 716, 1432, 1075]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.allclose(lookup_res.cached_start_indices.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.cached_lengths.cpu(), torch.tensor(cachedlen, dtype=torch.int32))
    assert torch.allclose(lookup_res.npu_cached_lengths.cpu(), torch.tensor(cachedlen, dtype=torch.int32))
    assert torch.allclose(
        lookup_res.host_cached_lengths.cpu(),
        torch.tensor([(i // 32) * 32 for i in cachedlen], dtype=torch.int32),
    )

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    assert kvcache_metadata.kv_indices.size(0) == 245
    assert torch.allclose(
        kvcache_metadata.kv_indptr.cpu(),
        torch.tensor([0, 44, 52, 73, 112, 143, 166, 211, 245], dtype=torch.int32),
    )
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len.cpu(),
        torch.tensor([i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32),
    )
    assert torch.allclose(kvcache_metadata.total_history_lengths.cpu(), sequence_lengths)
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:].cpu() - kvcache_metadata.total_history_offsets[:-1].cpu(),
        sequence_lengths,
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:].cpu() - kvcache_metadata.new_history_offsets[:-1].cpu(),
        torch.tensor(deltalen, dtype=torch.int32),
    )

    assert torch.allclose(
        kvcache_metadata.new_history_nnz_npu.cpu(),
        torch.tensor(
            [
                sum(deltalen),
            ],
            dtype=torch.int32,
        ),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    # 验证：onboard 被跳过，因为 NPU 上已有比 Host 更新的完整数据
    assert kvcache_metadata.kv_onload_handle.status == HostKVTaskStatus.SKIPPED

    for layer_idx in range(3):
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)
    assert kvcache_metadata.kv_onload_handle.handle is None

    for layer_idx in range(3):
        kvcache_mgr.npu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.npu_kvcache_mgr.get(page_ids, last_page_lens, layer_idx)
            assert torch.allclose(cached_k.cpu(), k[layer_idx].cpu()), (
                f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            )
            assert torch.allclose(cached_v.cpu(), v[layer_idx].cpu()), (
                f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"
            )

    task_handle = kvcache_mgr.offload_launch(index_meta)
    assert task_handle is not None, "offload_launch returned None: nothing was offloaded"
    assert wait_offload_complete(kvcache_mgr), "offload did not complete within the retry budget"

    host_len = [seqlen[i] // 32 * 32 for i in range(8)]
    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

        kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
            [
                uid,
            ],
            False,
        )[0]
        cached_k, cached_v = kvdata.unbind(dim=2)
        cached_k = cached_k.reshape(cached_k.size(0), -1, cached_k.size(3), cached_k.size(4))
        cached_v = cached_v.reshape(cached_v.size(0), -1, cached_v.size(3), cached_v.size(4))
        assert torch.allclose(cached_k, k.cpu()), f"Key mismatch for uid {user_ids[i].item()}"
        assert torch.allclose(cached_v, v.cpu()), f"Value mismatch for uid {user_ids[i].item()}"

    torch.npu.synchronize()


def _phase_3(kvcache_mgr, g_keys, g_values):
    """阶段C: 驱逐 NPU 后重新分配 + 从 Host onboard + 部分offload
    - 先驱逐 NPU 上的 user，使其数据仅在 Host
    - 重新分配时需要从 Host onboard（触发 H2D + scatter 流水线）
    - 验证 onboard 后 NPU 上的数据与原始数据一致
    """
    cachedlen = [i // 32 * 32 for i in [1400, 256, 672, 1248, 973, 716, 1432, 1075]]
    seqlen = [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    # 阶段C: 先驱逐 NPU 上的 user，使其数据仅留在 Host 侧
    kvcache_mgr.evict(user_ids, for_npu=True)

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    # 验证：驱逐后 NPU 无缓存，Host 有缓存（之前 offload 的数据）
    assert torch.allclose(lookup_res.cached_start_indices.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.cached_lengths.cpu(), torch.tensor(cachedlen, dtype=torch.int32))
    assert torch.allclose(lookup_res.npu_cached_lengths.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.host_cached_lengths.cpu(), torch.tensor(cachedlen, dtype=torch.int32))

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    assert kvcache_metadata.kv_indices.size(0) == 354
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len.cpu(),
        torch.tensor([i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32),
    )
    assert torch.allclose(kvcache_metadata.total_history_lengths.cpu(), sequence_lengths)
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:].cpu() - kvcache_metadata.total_history_offsets[:-1].cpu(),
        sequence_lengths,
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:].cpu() - kvcache_metadata.new_history_offsets[:-1].cpu(),
        torch.tensor(deltalen, dtype=torch.int32),
    )

    assert torch.allclose(
        kvcache_metadata.new_history_nnz_npu.cpu(),
        torch.tensor(
            [
                sum(deltalen),
            ],
            dtype=torch.int32,
        ),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    # 启动 onboard：将 Host 侧数据搬回 NPU（H2D + scatter 流水线）
    kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    for layer_idx in range(3):
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

    for layer_idx in range(3):
        kvcache_mgr.npu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    # 验证 onboard 后 NPU 上的数据与原始数据一致
    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.npu_kvcache_mgr.get(page_ids, last_page_lens, layer_idx)
            assert torch.allclose(cached_k.cpu(), k[layer_idx].cpu()), (
                f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            )
            assert torch.allclose(cached_v.cpu(), v[layer_idx].cpu()), (
                f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"
            )

    task_handle = kvcache_mgr.offload_launch(index_meta)
    assert task_handle is not None, "offload_launch returned None: nothing was offloaded"
    assert wait_offload_complete(kvcache_mgr), "offload did not complete within the retry budget"

    # 验证 Host 侧 offload 数据完整性
    host_len = [seqlen[i] // 32 * 32 for i in range(8)]
    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

        kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
            [
                uid,
            ],
            False,
        )[0]
        cached_k, cached_v = kvdata.unbind(dim=2)
        cached_k = cached_k.reshape(cached_k.size(0), -1, cached_k.size(3), cached_k.size(4))
        cached_v = cached_v.reshape(cached_v.size(0), -1, cached_v.size(3), cached_v.size(4))
        assert torch.allclose(cached_k, k.cpu()), f"Key mismatch for uid {user_ids[i].item()}"
        assert torch.allclose(cached_v, v.cpu()), f"Value mismatch for uid {user_ids[i].item()}"

    torch.npu.synchronize()


def _phase_4(kvcache_mgr, g_keys, g_values):
    """阶段D: 新 user (u8~u15) 分配 + 全量offload

    分配所需页数（290）加上阶段 3 结束时 u0~u7 仍占用的页数（354）合计
    644，超过 num_primary_cache_pages=512：allocate_kvcache() 内部会因页
    池不足触发 LRU，把 u0~u7 部分页驱逐回 host 以腾出空间——这才是
    docstring 里"驱逐旧 user"的真正来源，也是 _phase_5 期望值的前提，
    见 _phase_5 的说明。本函数开头的 evict(user_ids=u8~u15, for_npu=True)
    只是对本阶段刚构造、从未分配过的新用户做防御性清空，是幂等空操作，
    不是那次跨用户驱逐。
    - 验证驱逐后新 user 的分配和读写正确
    - 验证新 user 的 offload 数据完整
    """
    cachedlen = [0 for _ in range(8)]
    seqlen = [689, 1417, 1174, 1987, 596, 520, 1538, 1189]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

    user_ids = torch.tensor(list(range(8, 16)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(8, 16))]
    values = [g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(8, 16))]

    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    kvcache_mgr.evict(user_ids, for_npu=True)

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.allclose(lookup_res.cached_start_indices.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.cached_lengths.cpu(), torch.tensor(cachedlen, dtype=torch.int32))
    assert torch.allclose(lookup_res.npu_cached_lengths.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.host_cached_lengths.cpu(), torch.tensor(cachedlen, dtype=torch.int32))

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    assert kvcache_metadata.kv_indices.size(0) == 290
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len.cpu(),
        torch.tensor([i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32),
    )
    assert torch.allclose(kvcache_metadata.total_history_lengths.cpu(), sequence_lengths)
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:].cpu() - kvcache_metadata.total_history_offsets[:-1].cpu(),
        sequence_lengths,
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:].cpu() - kvcache_metadata.new_history_offsets[:-1].cpu(),
        torch.tensor(deltalen, dtype=torch.int32),
    )

    assert torch.allclose(
        kvcache_metadata.new_history_nnz_npu.cpu(),
        torch.tensor(
            [
                sum(deltalen),
            ],
            dtype=torch.int32,
        ),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    for layer_idx in range(3):
        kvcache_mgr.npu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    for i, uid in enumerate(range(8, 16)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.npu_kvcache_mgr.get(page_ids, last_page_lens, layer_idx)
            assert torch.allclose(cached_k.cpu(), k[layer_idx].cpu()), (
                f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            )
            assert torch.allclose(cached_v.cpu(), v[layer_idx].cpu()), (
                f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"
            )

    task_handle = kvcache_mgr.offload_launch(index_meta)
    assert task_handle is not None, "offload_launch returned None: nothing was offloaded"
    assert wait_offload_complete(kvcache_mgr), "offload did not complete within the retry budget"

    host_len = [seqlen[i] // 32 * 32 for i in range(8)]
    for i, uid in enumerate(range(8, 16)):
        k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

        kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
            [
                uid,
            ],
            False,
        )[0]
        cached_k, cached_v = kvdata.unbind(dim=2)
        cached_k = cached_k.reshape(cached_k.size(0), -1, cached_k.size(3), cached_k.size(4))
        cached_v = cached_v.reshape(cached_v.size(0), -1, cached_v.size(3), cached_v.size(4))
        assert torch.allclose(cached_k, k.cpu()), f"Key mismatch for uid {user_ids[i].item()}"
        assert torch.allclose(cached_v, v.cpu()), f"Value mismatch for uid {user_ids[i].item()}"

    torch.npu.synchronize()


def _phase_5(kvcache_mgr, g_keys, g_values):
    """阶段E: 旧 user (u0~u7) 重新分配 + 部分onboard + offload + 驱逐清理

    必须在 _phase_1..4 之后依次重放：期望的 npu_cached_* 快照（部分驱逐，
    如 uid0 只剩最后 16 个 token 在 NPU）是 _phase_4 给 u8~u15 分配 290 页
    时，与 _phase_3 结束时 u0~u7 已占用的 354 页相加（354+290>512）触发
    LRU 反向驱逐 u0~u7 的直接结果，不只是重放到 _phase_3 就够。
    - 旧 user 重新分配，部分数据从 Host onboard，部分在 NPU 残留
    - 验证 NPU+Host 混合 lookup 结果正确
    - onboard 后验证数据一致性
    - 最后驱逐 Host 和 NPU 上的 user，验证清理完整
    """
    cachedlen = [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]
    seqlen = [2500, 1024, 1280, 2377, 1854, 1365, 2729, 2049]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.allclose(lookup_res.cached_start_indices.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.cached_lengths.cpu(), torch.tensor(cachedlen, dtype=torch.int32))
    assert torch.allclose(
        lookup_res.npu_cached_start_indices.cpu(),
        torch.tensor([1984, 0, 0, 1760, 0, 0, 0, 0], dtype=torch.int32),
    )
    assert torch.allclose(
        lookup_res.npu_cached_lengths.cpu(),
        torch.tensor([16, 0, 0, 23, 1391, 1024, 2047, 1537], dtype=torch.int32),
    )
    assert torch.allclose(
        lookup_res.host_cached_lengths.cpu(),
        torch.tensor([1984, 512, 960, 1760, 1376, 1024, 2016, 1536], dtype=torch.int32),
    )

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    assert kvcache_metadata.kv_indices.size(0) == 478
    assert torch.allclose(
        kvcache_metadata.kv_last_page_len.cpu(),
        torch.tensor([i % 32 if i % 32 > 0 else 32 for i in seqlen], dtype=torch.int32),
    )
    assert torch.allclose(kvcache_metadata.total_history_lengths.cpu(), sequence_lengths)
    assert torch.allclose(
        kvcache_metadata.total_history_offsets[1:].cpu() - kvcache_metadata.total_history_offsets[:-1].cpu(),
        sequence_lengths,
    )
    assert torch.allclose(
        kvcache_metadata.new_history_offsets[1:].cpu() - kvcache_metadata.new_history_offsets[:-1].cpu(),
        torch.tensor(deltalen, dtype=torch.int32),
    )

    assert torch.allclose(
        kvcache_metadata.new_history_nnz_npu.cpu(),
        torch.tensor(
            [
                sum(deltalen),
            ],
            dtype=torch.int32,
        ),
    )
    assert kvcache_metadata.new_history_nnz == sum(deltalen)

    kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    for layer_idx in range(3):
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

    for layer_idx in range(3):
        kvcache_mgr.npu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in new_keys], dim=0),
            torch.cat([v[layer_idx] for v in new_values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i], values[i]
        for layer_idx in range(3):
            page_ids = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[i] : kvcache_metadata.kv_indptr[i + 1]]
            last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
            cached_k, cached_v = kvcache_mgr.npu_kvcache_mgr.get(page_ids, last_page_lens, layer_idx)
            assert torch.allclose(cached_k.cpu(), k[layer_idx].cpu()), (
                f"Layer {layer_idx} key mismatch for uid {user_ids[i].item()}"
            )
            assert torch.allclose(cached_v.cpu(), v[layer_idx].cpu()), (
                f"Layer {layer_idx} value mismatch for uid {user_ids[i].item()}"
            )

    task_handle = kvcache_mgr.offload_launch(index_meta)
    assert task_handle is not None, "offload_launch returned None: nothing was offloaded"
    assert wait_offload_complete(kvcache_mgr), "offload did not complete within the retry budget"

    host_len = [seqlen[i] // 32 * 32 for i in range(8)]
    for i, uid in enumerate(range(0, 8)):
        k, v = keys[i][:, : host_len[i], ...], values[i][:, : host_len[i], ...]

        kvdata = kvcache_mgr.host_kvstorage_manager.impl_.get_kvdata_tensor(
            [
                uid,
            ],
            False,
        )[0]
        cached_k, cached_v = kvdata.unbind(dim=2)
        cached_k = cached_k.reshape(cached_k.size(0), -1, cached_k.size(3), cached_k.size(4))
        cached_v = cached_v.reshape(cached_v.size(0), -1, cached_v.size(3), cached_v.size(4))
        assert torch.allclose(cached_k, k.cpu()), f"Key mismatch for uid {user_ids[i].item()}"
        assert torch.allclose(cached_v, v.cpu()), f"Value mismatch for uid {user_ids[i].item()}"

    # 验证驱逐清理：先驱逐 Host，再驱逐 NPU，确认两侧缓存均已清空
    kvcache_mgr.evict(user_ids, for_host=True)
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.equal(lookup_res.host_cached_lengths.cpu(), torch.zeros(8, dtype=torch.int32))

    kvcache_mgr.evict(user_ids, for_npu=True, for_host=True)
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.equal(lookup_res.npu_cached_lengths.cpu(), torch.zeros(8, dtype=torch.int32))

    torch.npu.synchronize()


def test_phase_1_allocate_total_and_offload_total(kvcache_mgr, g_keys, g_values):
    _phase_1(kvcache_mgr, g_keys, g_values)


def test_phase_2_allocate_partial_onboard_skipped_offload_partial(kvcache_mgr, g_keys, g_values):
    _phase_1(kvcache_mgr, g_keys, g_values)
    _phase_2(kvcache_mgr, g_keys, g_values)


def test_phase_3_allocate_partial_onboard_total_offload_partial(kvcache_mgr, g_keys, g_values):
    _phase_1(kvcache_mgr, g_keys, g_values)
    _phase_2(kvcache_mgr, g_keys, g_values)
    _phase_3(kvcache_mgr, g_keys, g_values)


def test_phase_4_allocate_evict_and_offload_total(kvcache_mgr, g_keys, g_values):
    _phase_4(kvcache_mgr, g_keys, g_values)


def test_phase_5_allocate_evict_onboard_partial_offload_total(kvcache_mgr, g_keys, g_values):
    _phase_1(kvcache_mgr, g_keys, g_values)
    _phase_2(kvcache_mgr, g_keys, g_values)
    _phase_3(kvcache_mgr, g_keys, g_values)
    _phase_4(kvcache_mgr, g_keys, g_values)
    _phase_5(kvcache_mgr, g_keys, g_values)


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))

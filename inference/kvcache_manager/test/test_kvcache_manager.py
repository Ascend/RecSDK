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


@pytest.fixture
def kvcache_mgr():
    # Function-scoped: each test gets its own fresh manager instead of sharing
    # one module-level instance. Tests that need a prior test's cumulative
    # state now explicitly replay the relevant _step_xx helper(s) below,
    # rather than depending on pytest's default top-to-bottom run order.
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
        # (_step_05 grows uid 6 to 2729 tokens): NPUKVCacheManagerImpl::
        # alloc_single_sequence() has a hard
        # TORCH_CHECK(new_total_length <= max_sequence_length), so a smaller
        # bound here would make _step_05's allocate_kvcache() raise. Using
        # the design doc's documented typical value (4096, already
        # page-aligned) rather than an exact match to 2729, for headroom.
        max_seq_len=4096,
        dtype=torch.bfloat16,
        device=torch.npu.current_device(),
        host_kvstorage_backend="native_host",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
    )
    return KVCacheManager.from_config(kvcache_config)


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


def _step_01(kvcache_mgr, g_keys, g_values):
    """Body of test_01_allocate_total_and_offload_total."""
    seqlen = [700, 128, 336, 624, 486, 358, 716, 537]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.allclose(lookup_res.cached_start_indices.cpu(), torch.zeros((8,), dtype=torch.int32))
    assert torch.allclose(lookup_res.cached_lengths.cpu(), torch.zeros((8,), dtype=torch.int32))

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


def _step_02(kvcache_mgr, g_keys, g_values):
    """Body of test_02_allocate_partial_onboard_skipped_offload_partial."""
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
        _start, _end = (
            kvcache_metadata.total_history_offsets[i],
            kvcache_metadata.total_history_offsets[i + 1],
        )
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


def _step_03(kvcache_mgr, g_keys, g_values):
    """Body of test_03_allocate_partial_onboard_total_offload_partial."""
    cachedlen = [i // 32 * 32 for i in [1400, 256, 672, 1248, 973, 716, 1432, 1075]]
    seqlen = [2000, 512, 960, 1783, 1391, 1024, 2047, 1537]
    deltalen = [seqlen[i] - cachedlen[i] for i in range(8)]

    user_ids = torch.tensor(list(range(0, 8)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)

    keys = [g_keys[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]
    values = [g_values[uid][:, : seqlen[i], ...] for i, uid in enumerate(range(0, 8))]

    new_keys = [k[:, cachedlen[i] : seqlen[i], ...] for i, k in enumerate(keys)]
    new_values = [v[:, cachedlen[i] : seqlen[i], ...] for i, v in enumerate(values)]

    kvcache_mgr.evict(user_ids, for_npu=True)

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
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
        _start, _end = (
            kvcache_metadata.total_history_offsets[i],
            kvcache_metadata.total_history_offsets[i + 1],
        )
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


def _step_04(kvcache_mgr, g_keys, g_values):
    """Body of test_04_allocate_evict_and_offload_total.

    Allocates a disjoint uid range (8-15) against the same shared
    num_primary_cache_pages=512 pool. When replayed after _step_01..03 (see
    test_05), the pool no longer has enough free pages for both uid ranges
    (354 pages already held by uids 0-7, 290 more needed here), so this step
    also exercises LRU eviction pressure back onto uids 0-7 -- which is a
    load-bearing precondition for test_05's expected npu_cached_* values,
    not just an independent scenario.
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
        _start, _end = (
            kvcache_metadata.total_history_offsets[i],
            kvcache_metadata.total_history_offsets[i + 1],
        )
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


def _step_05(kvcache_mgr, g_keys, g_values):
    """Body of test_05_allocate_evict_onboard_partial_offload_total.

    Must run after _step_01..04 in that order: its expected npu_cached_*
    values reflect the LRU eviction that _step_04's uid 8-15 allocation
    forces back onto uids 0-7 once the shared 512-page pool is oversubscribed
    (354 pages held by uids 0-7 + 290 needed for uids 8-15 > 512 capacity).
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
        _start, _end = (
            kvcache_metadata.total_history_offsets[i],
            kvcache_metadata.total_history_offsets[i + 1],
        )
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

    kvcache_mgr.evict(user_ids, for_host=True)
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.equal(lookup_res.host_cached_lengths.cpu(), torch.zeros(8, dtype=torch.int32))

    kvcache_mgr.evict(user_ids, for_npu=True, for_host=True)
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    assert torch.equal(lookup_res.npu_cached_lengths.cpu(), torch.zeros(8, dtype=torch.int32))

    torch.npu.synchronize()


def test_01_allocate_total_and_offload_total(kvcache_mgr, g_keys, g_values):
    _step_01(kvcache_mgr, g_keys, g_values)


def test_02_allocate_partial_onboard_skipped_offload_partial(kvcache_mgr, g_keys, g_values):
    _step_01(kvcache_mgr, g_keys, g_values)
    _step_02(kvcache_mgr, g_keys, g_values)


def test_03_allocate_partial_onboard_total_offload_partial(kvcache_mgr, g_keys, g_values):
    _step_01(kvcache_mgr, g_keys, g_values)
    _step_02(kvcache_mgr, g_keys, g_values)
    _step_03(kvcache_mgr, g_keys, g_values)


def test_04_allocate_evict_and_offload_total(kvcache_mgr, g_keys, g_values):
    _step_04(kvcache_mgr, g_keys, g_values)


def test_05_allocate_evict_onboard_partial_offload_total(kvcache_mgr, g_keys, g_values):
    _step_01(kvcache_mgr, g_keys, g_values)
    _step_02(kvcache_mgr, g_keys, g_values)
    _step_03(kvcache_mgr, g_keys, g_values)
    _step_04(kvcache_mgr, g_keys, g_values)
    _step_05(kvcache_mgr, g_keys, g_values)


_INVALID_CONFIG_CASES = [
    ({"offload_chunksize": 100}, ValueError, "multiple of"),
    ({"host_kvstorage_backend": "bogus_backend"}, NotImplementedError, "Unknown host kvcache backend"),
    ({"offload_mode": "bogus_mode"}, ValueError, "Unknown offload_mode"),
    ({"host_kvstorage_fail_policy": "bogus_policy"}, ValueError, "Unknown host_kvstorage_fail_policy"),
]


@pytest.mark.parametrize(
    "overrides,exc_type,match",
    _INVALID_CONFIG_CASES,
    ids=[
        "offload_chunksize_not_multiple_of_page_size",
        "unknown_host_backend",
        "invalid_offload_mode",
        "invalid_host_kvstorage_fail_policy",
    ],
)
def test_from_config_rejects_invalid_config(overrides, exc_type, match):
    # get_kvcache_config() itself performs no validation (host_kvstorage_backend
    # is just a free string there); the checks below all live in
    # KVCacheManager.from_config()/__init__(), so this locks down that contract.
    kwargs = dict(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=1024 * 2 * 32 * 4 * 128 * 2,
        max_batch_size=8,
        max_seq_len=4096,
        dtype=torch.bfloat16,
        device=torch.npu.current_device(),
        host_kvstorage_backend="native_host",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
    )
    kwargs.update(overrides)
    kvcache_config = get_kvcache_config(**kwargs)
    with pytest.raises(exc_type, match=match):
        KVCacheManager.from_config(kvcache_config)


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))

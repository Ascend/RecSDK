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
"""Unit tests for NPUKVCacheManager.

Mirrors GPU test patterns from recsys_kvcache_manager/test/test_native.py,
covering: construct, lookup, allocate, evict, put+get roundtrip,
check_for_offload, acquire/release_offload_pages.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from recsys_kvcache_manager_npu.npu_kvcache_manager import NPUKVCacheManager
from recsys_kvcache_manager_npu.kvcache_utils import KVLookupResult

NPU_AVAILABLE = False
try:
    torch.npu.current_device()
    NPU_AVAILABLE = True
except Exception:
    NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")

# Default configuration matching existing host test
NUM_LAYERS = 3
NUM_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 32
CHUNK_SIZE = 128
NUM_PRIMARY_PAGES = 128
NUM_BUFFER_PAGES = 4
MAX_BATCH_SIZE = 8
MAX_SEQ_LEN = 2048
DEVICE_IDX = 0
DTYPE = torch.bfloat16


def make_manager(**overrides):
    cfg = dict(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_tokens_per_page=PAGE_SIZE,
        num_tokens_per_chunk=CHUNK_SIZE,
        num_primary_cache_pages=NUM_PRIMARY_PAGES,
        num_buffer_pages=NUM_BUFFER_PAGES,
        max_batch_size=MAX_BATCH_SIZE,
        max_sequence_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        device_idx=DEVICE_IDX,
    )
    cfg.update(overrides)
    return NPUKVCacheManager(**cfg)


# ---------------------------------------------------------------------------
# 1. Construct / Destruct
# ---------------------------------------------------------------------------
def test_construct_destruct():
    mgr = make_manager()
    assert mgr.impl_ is not None
    assert mgr.npu_kvcache_tensor.shape == (
        NUM_LAYERS,
        NUM_PRIMARY_PAGES,
        2,
        PAGE_SIZE,
        NUM_HEADS,
        HEAD_DIM,
    )
    assert mgr.npu_kvcache_tensor.dtype == DTYPE
    assert len(mgr.npu_kvcache_tables) == NUM_LAYERS


# ---------------------------------------------------------------------------
# 2. Lookup empty
# ---------------------------------------------------------------------------
def test_lookup_empty():
    mgr = make_manager()
    uids = torch.tensor([10, 20, 30], dtype=torch.int64)
    result = mgr.lookup(uids)
    assert isinstance(result, KVLookupResult)
    assert torch.equal(result.user_ids, uids)
    assert result.npu_cached_lengths.tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# 3. Lookup after allocate
# ---------------------------------------------------------------------------
def test_lookup_after_allocate():
    mgr = make_manager()
    uids = torch.tensor([0, 1], dtype=torch.int64)
    seq_lens = torch.tensor([64, 96], dtype=torch.int32)
    lookup_res = mgr.lookup(uids)
    _metadata = mgr.allocate(uids, seq_lens, lookup_res)
    # Second lookup should report cached lengths
    lookup2 = mgr.lookup(uids)
    assert lookup2.npu_cached_lengths[0].item() == 64
    assert lookup2.npu_cached_lengths[1].item() == 96


# ---------------------------------------------------------------------------
# 4. Allocate + metadata verification
# ---------------------------------------------------------------------------
def test_allocate_metadata():
    mgr = make_manager()
    uids = torch.tensor([0, 1], dtype=torch.int64)
    seq_lens = torch.tensor([64, 33], dtype=torch.int32)  # 2 pages, 2 pages (33 > 32)
    lookup_res = mgr.lookup(uids)
    metadata = mgr.allocate(uids, seq_lens, lookup_res)

    # kv_indptr: [0, num_pages_user0, num_pages_user0+num_pages_user1]
    # user0: 64/32=2 pages, user1: ceil(33/32)=2 pages
    expected_indptr = [0, 2, 4]
    assert metadata.kv_indptr.tolist() == expected_indptr

    # kv_last_page_len: seq_len % page_size, with 0 -> page_size
    # user0: 64%32=0 -> 32, user1: 33%32=1
    expected_last_page_len = [32, 1]
    assert metadata.kv_last_page_len.tolist() == expected_last_page_len

    # max_seqlen should be max(seq_lens)
    assert metadata.max_seqlen == 64

    # kv_indices should be populated (non-negative page IDs)
    kv_indices = metadata.kv_indices[: expected_indptr[-1]]
    assert (kv_indices >= 0).all()


# ---------------------------------------------------------------------------
# 5. Allocate idempotent for already-cached
# ---------------------------------------------------------------------------
def test_allocate_already_cached():
    mgr = make_manager()
    uids = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([64], dtype=torch.int32)
    lookup1 = mgr.lookup(uids)
    _meta1 = mgr.allocate(uids, seq_lens, lookup1)

    # Second allocate with same seq_lens should be a no-op (0 new tokens)
    lookup2 = mgr.lookup(uids)
    meta2 = mgr.allocate(uids, seq_lens, lookup2)
    # new_history_nnz should be 0 (no new tokens to append)
    assert meta2.new_history_nnz == 0


# ---------------------------------------------------------------------------
# 6. Evict single uid
# ---------------------------------------------------------------------------
def test_evict_single():
    mgr = make_manager()
    uids = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([64], dtype=torch.int32)
    lookup_res = mgr.lookup(uids)
    mgr.allocate(uids, seq_lens, lookup_res)

    # Evict uid 0
    mgr.evict(torch.tensor([0], dtype=torch.int64))

    # Lookup should report 0 cached
    lookup2 = mgr.lookup(uids)
    assert lookup2.npu_cached_lengths[0].item() == 0


# ---------------------------------------------------------------------------
# 7. Evict all
# ---------------------------------------------------------------------------
def test_evict_all():
    mgr = make_manager()
    uids = torch.tensor([0, 1], dtype=torch.int64)
    seq_lens = torch.tensor([64, 96], dtype=torch.int32)
    lookup_res = mgr.lookup(uids)
    mgr.allocate(uids, seq_lens, lookup_res)

    mgr.evict_all()

    lookup2 = mgr.lookup(uids)
    assert lookup2.npu_cached_lengths.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# 8. Put + Get roundtrip (single user, single layer)
# ---------------------------------------------------------------------------
def test_put_get_roundtrip_single_user():
    mgr = make_manager()
    uid = torch.tensor([0], dtype=torch.int64)
    seq_len = 64  # exactly 2 pages
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    lookup_res = mgr.lookup(uid)
    metadata = mgr.allocate(uid, seq_lens, lookup_res)

    # Generate random k/v data: [num_new_tokens, num_heads, head_dim]
    num_new_tokens = seq_len
    k = torch.randn(num_new_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    v = torch.randn(num_new_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")

    for layer_idx in range(NUM_LAYERS):
        mgr.put(k, v, layer_idx, metadata)

    # Read back and verify
    page_ids = metadata.kv_indices[: metadata.kv_indptr[1]]
    last_page_len = metadata.kv_last_page_len[0]
    for layer_idx in range(NUM_LAYERS):
        k_out, v_out = mgr.get(page_ids, last_page_len, layer_idx)
        assert torch.allclose(k_out.cpu(), k.cpu(), atol=1e-2), f"Key mismatch at layer {layer_idx}"
        assert torch.allclose(v_out.cpu(), v.cpu(), atol=1e-2), f"Value mismatch at layer {layer_idx}"


# ---------------------------------------------------------------------------
# 9. Put + Get roundtrip (multiple users)
# ---------------------------------------------------------------------------
def test_put_get_roundtrip_multi_user():
    mgr = make_manager()
    uids = torch.tensor([0, 1], dtype=torch.int64)
    seq_lens = torch.tensor([64, 33], dtype=torch.int32)

    lookup_res = mgr.lookup(uids)
    metadata = mgr.allocate(uids, seq_lens, lookup_res)

    # Concatenated k/v for all new tokens across users
    num_new_tokens = metadata.new_history_nnz
    k = torch.randn(num_new_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    v = torch.randn(num_new_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")

    for layer_idx in range(NUM_LAYERS):
        mgr.put(k, v, layer_idx, metadata)

    # Verify per-user
    for i in range(2):
        page_start = metadata.kv_indptr[i]
        page_end = metadata.kv_indptr[i + 1]
        page_ids = metadata.kv_indices[page_start:page_end]
        last_page_len = metadata.kv_last_page_len[i]
        for layer_idx in range(NUM_LAYERS):
            k_out, v_out = mgr.get(page_ids, last_page_len, layer_idx)
            # Extract the portion of k/v belonging to user i
            # batch_indices maps new-token index -> user index
            user_mask = metadata.batch_indices[:num_new_tokens] == i
            user_positions = metadata.position[:num_new_tokens][user_mask]
            # Sort by position for correct ordering
            sorted_pos, sort_idx = user_positions.sort()
            num_user_tokens = seq_lens[i].item()
            expected_k = k[:num_new_tokens][user_mask][sort_idx][:num_user_tokens]
            expected_v = v[:num_new_tokens][user_mask][sort_idx][:num_user_tokens]
            assert torch.allclose(k_out.cpu(), expected_k.cpu(), atol=1e-2), f"Key mismatch user {i} layer {layer_idx}"
            assert torch.allclose(v_out.cpu(), expected_v.cpu(), atol=1e-2), (
                f"Value mismatch user {i} layer {layer_idx}"
            )


# ---------------------------------------------------------------------------
# 10. Put + Get roundtrip with partial update (extend sequence)
# ---------------------------------------------------------------------------
def test_put_get_roundtrip_partial_update():
    mgr = make_manager()
    uid = torch.tensor([0], dtype=torch.int64)
    # Phase 1: allocate 32 tokens (1 page)
    seq_len_1 = 32
    seq_lens_1 = torch.tensor([seq_len_1], dtype=torch.int32)
    lookup1 = mgr.lookup(uid)
    meta1 = mgr.allocate(uid, seq_lens_1, lookup1)

    k1 = torch.randn(seq_len_1, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    v1 = torch.randn(seq_len_1, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    for layer_idx in range(NUM_LAYERS):
        mgr.put(k1, v1, layer_idx, meta1)

    # Phase 2: extend to 64 tokens
    seq_len_2 = 64
    seq_lens_2 = torch.tensor([seq_len_2], dtype=torch.int32)
    lookup2 = mgr.lookup(uid)
    meta2 = mgr.allocate(uid, seq_lens_2, lookup2)

    # Only new tokens (32..63) need to be written
    new_tokens = seq_len_2 - seq_len_1
    k2 = torch.randn(new_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    v2 = torch.randn(new_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    for layer_idx in range(NUM_LAYERS):
        mgr.put(k2, v2, layer_idx, meta2)

    # Read back full 64 tokens
    page_ids = meta2.kv_indices[: meta2.kv_indptr[1]]
    last_page_len = meta2.kv_last_page_len[0]
    for layer_idx in range(NUM_LAYERS):
        k_out, v_out = mgr.get(page_ids, last_page_len, layer_idx)
        assert k_out.shape[0] == seq_len_2
        # First 32 tokens should match k1
        assert torch.allclose(k_out[:seq_len_1].cpu(), k1.cpu(), atol=1e-2), f"Old key mismatch at layer {layer_idx}"
        # Last 32 tokens should match k2
        assert torch.allclose(k_out[seq_len_1:].cpu(), k2.cpu(), atol=1e-2), f"New key mismatch at layer {layer_idx}"


# ---------------------------------------------------------------------------
# 11. Get cache tables
# ---------------------------------------------------------------------------
def test_get_cache_tables():
    mgr = make_manager()
    # All layers
    tables = mgr.get_cache_tables()
    assert len(tables) == NUM_LAYERS
    assert tables[0].shape == (NUM_PRIMARY_PAGES, 2, PAGE_SIZE, NUM_HEADS, HEAD_DIM)
    # Single layer
    table0 = mgr.get_cache_tables(layer_idx=0)
    assert table0.shape == (NUM_PRIMARY_PAGES, 2, PAGE_SIZE, NUM_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# 12. check_for_offload
# ---------------------------------------------------------------------------
def test_check_for_offload():
    mgr = make_manager()
    uids = torch.tensor([0, 1], dtype=torch.int64)
    # uid 0 has exactly one full chunk (CHUNK_SIZE tokens) cached and nothing
    # offloaded yet -> eligible (cached_end - offloaded_length >= chunk_size).
    # uid 1 has fewer tokens than one chunk -> not eligible. See
    # NPUKVCacheManagerImpl::check_for_offload.
    seq_lens = torch.tensor([CHUNK_SIZE, CHUNK_SIZE // 2], dtype=torch.int32)
    lookup_res = mgr.lookup(uids)
    mgr.allocate(uids, seq_lens, lookup_res)

    offload_uids = mgr.check_for_offload(uids)
    assert offload_uids.tolist() == [0], f"expected only uid 0 eligible for offload, got {offload_uids.tolist()}"

    # A uid below the chunk-size threshold must never be reported eligible.
    offload_uids_below = mgr.check_for_offload(torch.tensor([1], dtype=torch.int64))
    assert offload_uids_below.numel() == 0, "uid 1 has fewer tokens than one chunk and must not be offload-eligible"


# ---------------------------------------------------------------------------
# 13. acquire / release offload pages
# ---------------------------------------------------------------------------
def test_acquire_release_offload_pages():
    mgr = make_manager()
    uids = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([128], dtype=torch.int32)  # >= chunk_size to trigger offload
    lookup_res = mgr.lookup(uids)
    mgr.allocate(uids, seq_lens, lookup_res)

    offloaded_lengths = torch.tensor([0], dtype=torch.int64)
    start_indices, lengths, page_indices_list = mgr.acquire_offload_pages(uids, offloaded_lengths)
    assert start_indices.shape[0] == 1
    assert lengths.shape[0] == 1
    assert len(page_indices_list) == 1

    # Release (not actually offloaded to host)
    mgr.release_offload_pages(uids, start_indices, lengths, offloaded=False)


# ---------------------------------------------------------------------------
# 14. Evict then re-allocate
# ---------------------------------------------------------------------------
def test_evict_then_reallocate():
    mgr = make_manager()
    uid = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([64], dtype=torch.int32)

    # Allocate
    lookup1 = mgr.lookup(uid)
    meta1 = mgr.allocate(uid, seq_lens, lookup1)
    k1 = torch.randn(64, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    v1 = torch.randn(64, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    for layer_idx in range(NUM_LAYERS):
        mgr.put(k1, v1, layer_idx, meta1)

    # Evict
    mgr.evict(uid)

    # Re-allocate
    lookup2 = mgr.lookup(uid)
    meta2 = mgr.allocate(uid, seq_lens, lookup2)
    k2 = torch.randn(64, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    v2 = torch.randn(64, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=f"npu:{DEVICE_IDX}")
    for layer_idx in range(NUM_LAYERS):
        mgr.put(k2, v2, layer_idx, meta2)

    # Verify new data
    page_ids = meta2.kv_indices[: meta2.kv_indptr[1]]
    last_page_len = meta2.kv_last_page_len[0]
    for layer_idx in range(NUM_LAYERS):
        k_out, v_out = mgr.get(page_ids, last_page_len, layer_idx)
        assert torch.allclose(k_out.cpu(), k2.cpu(), atol=1e-2)
        assert torch.allclose(v_out.cpu(), v2.cpu(), atol=1e-2)


# ---------------------------------------------------------------------------
# 15. Multiple users exceed capacity trigger eviction
# ---------------------------------------------------------------------------
def test_multi_user_contention():
    mgr = make_manager(num_primary_cache_pages=16)
    # 16 pages * 32 tokens/page = 512 tokens total capacity.
    # uid 0 takes 8 pages (256 tokens), then uid 1 takes the remaining 8
    # pages (256 tokens): capacity is now exactly full and 0 pages are free.
    uids1 = torch.tensor([0], dtype=torch.int64)
    seq_lens1 = torch.tensor([256], dtype=torch.int32)
    lookup1 = mgr.lookup(uids1)
    meta1 = mgr.allocate(uids1, seq_lens1, lookup1)
    assert meta1.kv_indptr[1].item() == 8  # 256/32=8 pages

    uids2 = torch.tensor([1], dtype=torch.int64)
    seq_lens2 = torch.tensor([256], dtype=torch.int32)
    lookup2 = mgr.lookup(uids2)
    meta2 = mgr.allocate(uids2, seq_lens2, lookup2)
    assert meta2.kv_indptr[1].item() == 8  # 256/32=8 pages

    # uid 2 now requests 1 page with 0 pages free: alloc_single_sequence must
    # evict a cached uid via LRU (NPUKVCacheManagerImpl::getUIdToEvict walks
    # _lru_list from the front, i.e. least-recently-used first). uid 0 was
    # retained before uid 1, so it is LRU and must be the one evicted, while
    # uid 1 (more recently retained) must survive untouched.
    uids3 = torch.tensor([2], dtype=torch.int64)
    seq_lens3 = torch.tensor([32], dtype=torch.int32)
    lookup3 = mgr.lookup(uids3)
    meta3 = mgr.allocate(uids3, seq_lens3, lookup3)
    assert meta3.kv_indptr[1].item() == 1  # 32/32=1 page

    lookup_after = mgr.lookup(torch.tensor([0, 1], dtype=torch.int64))
    assert lookup_after.npu_cached_lengths.tolist() == [
        0,
        256,
    ], f"expected uid 0 evicted (LRU) and uid 1 untouched, got {lookup_after.npu_cached_lengths.tolist()}"


# ---------------------------------------------------------------------------
# 16. Last page len edge cases
# ---------------------------------------------------------------------------
def test_last_page_len_edge_cases():
    mgr = make_manager()
    uids = torch.tensor([0, 1, 2], dtype=torch.int64)
    # Exactly 1 page, partial last page, exactly 3 pages
    seq_lens = torch.tensor([32, 33, 96], dtype=torch.int32)
    lookup_res = mgr.lookup(uids)
    metadata = mgr.allocate(uids, seq_lens, lookup_res)

    # 32%32=0 -> last_page_len=32
    # 33%32=1 -> last_page_len=1
    # 96%32=0 -> last_page_len=32
    expected = [32, 1, 32]
    assert metadata.kv_last_page_len.tolist() == expected


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))

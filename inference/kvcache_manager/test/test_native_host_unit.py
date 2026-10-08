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

import time

import pytest
import torch
import torch_npu  # noqa: F401  required to register npu backend

from recsys_kvcache_manager_npu.native_host_kvcache_manager import NativeHostKVCacheManager
from recsys_kvcache_manager_npu.host_kvstorage_manager import HostKVTaskHandle, HostKVTaskStatus

NPU_AVAILABLE = False
try:
    torch.npu.current_device()
    NPU_AVAILABLE = True
except Exception:
    NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")


def make_config(
    num_layers=3,
    num_pages=128,
    page_size=32,
    num_heads=4,
    head_dim=128,
    capacity_per_layer=1024 * 2 * 32 * 4 * 128 * 2,
    max_batch_size=8,
    max_seq_len=2048,
    device_idx=0,
    num_tokens_per_chunk=128,
):
    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_tokens_per_page": page_size,
        "num_tokens_per_chunk": num_tokens_per_chunk,
        "bytes_capacity_per_layer": capacity_per_layer,
        "max_batch_size": max_batch_size,
        "max_sequence_length": max_seq_len,
        "device_idx": device_idx,
    }


def wait_offload_ready(mgr, task_handle, retries=100, interval_s=0.001):
    """Poll mgr.offload_kvcache_wait() until ready.

    Returns the final HostKVWaitResult once ready, or None if the retry
    budget is exhausted (callers must assert on the return value so a
    timeout fails loudly instead of silently falling through).
    """
    wait_res = None
    for _ in range(retries):
        wait_res = mgr.offload_kvcache_wait(task_handle)
        if wait_res.ready:
            return wait_res
        time.sleep(interval_s)
    return None


def wait_handle_ready(handle, retries=100, interval_s=0.001):
    """Poll a raw KVOffloadHandle.try_wait_layer(-1) until ready.

    Returns True once ready, False if the retry budget is exhausted.
    """
    for _ in range(retries):
        if handle.try_wait_layer(-1):
            return True
        time.sleep(interval_s)
    return False


def test_construct_destruct():
    print("Testing Case 1: Construct & Destruct ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    assert mgr.impl_ is not None
    del mgr
    print("Passed.")


def test_lookup_empty():
    print("Testing Case 2: Lookup Empty ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    uids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    cached_lengths = mgr.impl_.lookup(uids)
    assert cached_lengths.dtype == torch.int32
    assert cached_lengths.tolist() == [0, 0, 0, 0]
    print("Passed.")


def test_lru_eviction():
    print("Testing Case 3: LRU Eviction ... ", end="", flush=True)
    # Capacity for exactly 1 chunk per layer (num_tokens_per_chunk=128,
    # page_size=32 -> num_pages_per_chunk=4; unit_chunk_bytes =
    # num_tokens_per_chunk * 2*num_heads*head_dim * sizeof(bf16)
    # = 128 * 2*4*128 * 2 = 262144). Offloading a second uid then has no
    # free chunk left, forcing get_empty_pinned_chunks() to walk _lru_list
    # and evict the least-recently-used uid (the first one) before it can
    # proceed — this is the actual LRU eviction path, not just an idempotent
    # evict() call on a uid that was never present.
    cfg = make_config(capacity_per_layer=262144, num_tokens_per_chunk=128, page_size=32)
    mgr = NativeHostKVCacheManager(**cfg)
    fake = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))

    starts = torch.tensor([0], dtype=torch.int64)
    page_indices_a = [torch.arange(0, 4, dtype=torch.int32)]  # exactly 1 chunk (4 pages)
    page_indices_b = [torch.arange(4, 8, dtype=torch.int32)]  # exactly 1 chunk (4 pages)

    uid_a = torch.tensor([0], dtype=torch.int64)
    task_a = mgr.offload_kvcache_launch(uid_a, starts, page_indices_a)
    assert task_a is not None, "offload_kvcache_launch returned None for uid 0"
    assert wait_offload_ready(mgr, task_a) is not None, "offload for uid 0 timed out"
    mgr.finish_task(task_a)
    assert mgr.impl_.lookup(uid_a)[0].item() == 128, "uid 0 not registered after first offload"

    uid_b = torch.tensor([1], dtype=torch.int64)
    task_b = mgr.offload_kvcache_launch(uid_b, starts, page_indices_b)
    assert task_b is not None, "offload_kvcache_launch returned None for uid 1"
    assert wait_offload_ready(mgr, task_b) is not None, "offload for uid 1 timed out"
    mgr.finish_task(task_b)

    # uid 0 (least-recently-used, and the only candidate) must have been
    # evicted to free the chunk uid 1 needed; uid 1 must now be registered.
    lengths = mgr.impl_.lookup(torch.tensor([0, 1], dtype=torch.int64))
    assert lengths[0].item() == 0, "uid 0 (LRU) was not evicted to make room for uid 1"
    assert lengths[1].item() == 128, "uid 1 not registered after eviction freed its chunk"
    print("Passed.")


def test_double_buffering():
    print("Testing Case 4: Double Buffering Cross-Layer Dependency ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    # 构造 fake npu_kvcache_tensor
    fake = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    fake.uniform_(-1, 1)
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))

    # 模拟 offload 调用（page_ids_per_uid 已知）
    uids = torch.tensor([0], dtype=torch.int64)
    starts = torch.tensor([0], dtype=torch.int64)
    page_indices_list = [torch.arange(0, 22, dtype=torch.int32)]  # 1 个 uid, 22 pages

    task_handle = mgr.offload_kvcache_launch(uids, starts, page_indices_list)
    assert task_handle is not None, "offload_kvcache_launch returned None"

    wait_res = wait_offload_ready(mgr, task_handle)
    assert wait_res is not None, "offload_kvcache_wait timed out (100 retries)"
    assert wait_res.ready, f"offload not ready after polling 100 times, status={wait_res.status}"
    assert wait_res.status == HostKVTaskStatus.READY, f"expected status=READY, got {wait_res.status}"
    print("Passed.")


def test_timeout_cancel():
    print("Testing Case 5: Timeout + Cancel Offload ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    # [num_layers, num_primary_cache_pages, 2(k/v), page_size, num_heads, head_dim]
    fake = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))
    uids = torch.tensor([0], dtype=torch.int64)
    starts = torch.tensor([0], dtype=torch.int64)
    page_indices_list = [torch.arange(0, 22, dtype=torch.int32)]
    ret = mgr.offload_kvcache_launch(uids, starts, page_indices_list)
    result = mgr.cancel_task(ret)
    assert isinstance(result, list)
    assert all(r == 0 for r in result), "cancel should return all-zero lengths"
    print("Passed.")


def test_offload_failure_cancel():
    print("Testing Case 6: Offload Failure + Cancel ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    # [num_layers, num_primary_cache_pages, 2(k/v), page_size, num_heads, head_dim]
    fake = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))
    uids = torch.tensor([0], dtype=torch.int64)
    starts = torch.tensor([99999], dtype=torch.int64)  # 越界，触发失败
    page_indices_list = [torch.arange(0, 22, dtype=torch.int32)]

    from kvcache_npu_cpp import KVOffloadHandle

    handle = KVOffloadHandle(3)
    # offload 启动可能成功，finish 时校验失败
    mgr.impl_.offload_kvcache(uids, starts, page_indices_list, handle)
    assert wait_handle_ready(handle), "offload handle did not become ready within 100 retries"
    # finish_offload 检测到 offload_start_index 越界 -> 抛
    with pytest.raises(RuntimeError):
        mgr.impl_.finish_offload(handle)
    print("Passed.")


def test_onboard_skipped():
    print("Testing Case 7: Onboard SKIPPED ... ", end="", flush=True)
    # SKIPPED is decided in the Python wrapper (onboard_kvcache_launch returns
    # early with status=SKIPPED, handle=None when nothing needs onboarding),
    # not by KVOnloadHandle.no_onload: the C++ side unconditionally calls
    # init() as soon as onload_kvcache() is entered, so no_onload never stays
    # True there. This test exercises the real SKIPPED contract at the
    # manager level, matching the !3210 integration test's approach.
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    fake = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))

    from recsys_kvcache_manager_npu.kvcache_metadata import get_kvcache_metadata_buffer
    from recsys_kvcache_manager_npu.kvcache_utils import KVLookupResult
    from recsys_kvcache_manager_npu.npu_kvcache_manager import NPUKVCacheManager

    npu_mgr = NPUKVCacheManager(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        num_tokens_per_page=32,
        num_tokens_per_chunk=128,
        num_primary_cache_pages=128,
        num_buffer_pages=4,
        max_batch_size=8,
        max_sequence_length=2048,
        dtype=torch.bfloat16,
        device_idx=0,
    )
    uids = torch.tensor([0], dtype=torch.int64)
    npu_lookup = npu_mgr.lookup(uids)  # nothing allocated on NPU -> all-zero
    index_meta = mgr.build_index_meta(uids, torch.zeros(1, dtype=torch.int32))
    host_lookup = mgr.lookup_kvcache(index_meta)  # nothing offloaded to host -> all-zero
    merged = KVLookupResult.merge(npu_lookup, host_lookup)

    kvcache_metadata = get_kvcache_metadata_buffer(1, 0, 0, device=0)
    kvcache_metadata.metadata_npu_buffer.zero_()  # kv_indptr is a view into this buffer

    task_handle = mgr.onboard_kvcache_launch(index_meta, merged, kvcache_metadata)
    assert task_handle.status == HostKVTaskStatus.SKIPPED, f"expected SKIPPED, got {task_handle.status}"
    assert task_handle.handle is None, "SKIPPED task must carry no underlying C++ handle"
    print("Passed.")


def test_onboard_full_per_layer():
    print("Testing Case 8: Onboard Full + Per-Layer Wait ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    fake = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    fake.uniform_(-1, 1)
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))
    # 先 offload 填 host
    uids = torch.tensor([0], dtype=torch.int64)
    starts = torch.tensor([0], dtype=torch.int64)
    page_indices_list = [torch.arange(0, 22, dtype=torch.int32)]
    from kvcache_npu_cpp import KVOffloadHandle, KVOnloadHandle

    off_handle = KVOffloadHandle(3)
    mgr.impl_.offload_kvcache(uids, starts, page_indices_list, off_handle)
    assert wait_handle_ready(off_handle), "offload handle did not become ready within 100 retries"
    mgr.impl_.finish_offload(off_handle)
    # 再 onboard 搬回 NPU
    on_handle = KVOnloadHandle(3)
    page_indices_list = [t.to("npu") for t in page_indices_list]
    mgr.impl_.onload_kvcache(uids, page_indices_list, on_handle)
    for layer in range(3):
        on_handle.wait_layer(layer)  # 不阻塞
    print("Passed.")


def test_offload_full():
    print("Testing Case 9: Offload Full ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    num_layers, num_pages, page_size, H, D = 3, 128, 32, 4, 128
    fake = torch.randn(num_layers, num_pages, 2, page_size, H, D, dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))
    uids = torch.tensor([0], dtype=torch.int64)
    starts = torch.tensor([0], dtype=torch.int64)
    page_ids = torch.arange(0, 22, dtype=torch.int32)
    page_indices_list = [page_ids]
    from kvcache_npu_cpp import KVOffloadHandle

    handle = KVOffloadHandle(num_layers)

    ret = mgr.impl_.offload_kvcache(uids, starts, page_indices_list, handle)
    assert ret
    assert wait_handle_ready(handle), "offload handle did not become ready within 100 retries"

    mgr.impl_.finish_offload(handle)
    kvdata_t = mgr.impl_.get_kvdata_tensor([0], False)
    kvdata = kvdata_t[0]
    cached_k, cached_v = kvdata.unbind(dim=2)
    cached_k = cached_k.reshape(cached_k.size(0), -1, page_size, H, D)
    cached_v = cached_v.reshape(cached_v.size(0), -1, page_size, H, D)
    expected_k = fake[:, :22, 0, :, :, :].cpu()
    expected_v = fake[:, :22, 1, :, :, :].cpu()
    ok_k = torch.allclose(cached_k, expected_k)
    assert ok_k, "Key mismatch"
    ok_v = torch.allclose(cached_v, expected_v)
    assert ok_v, "Value mismatch"
    print("Passed.", flush=True)


def test_offload_partial_success():
    print("Testing Case 10: Offload Partial Success ... ", end="", flush=True)
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    num_layers, num_pages, page_size, H, D = 3, 128, 32, 4, 128
    fake = torch.randn(num_layers, num_pages, 2, page_size, H, D, dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))
    # 2 个 uid，pages 不同
    uids = torch.tensor([0, 1], dtype=torch.int64)
    starts = torch.tensor([0, 0], dtype=torch.int64)
    page_indices_list = [
        torch.arange(0, 22, dtype=torch.int32),
        torch.arange(22, 30, dtype=torch.int32),
    ]
    from kvcache_npu_cpp import KVOffloadHandle

    handle = KVOffloadHandle(num_layers)
    mgr.impl_.offload_kvcache(uids, starts, page_indices_list, handle)
    assert wait_handle_ready(handle), "offload handle did not become ready within 100 retries"
    result = mgr.impl_.finish_offload(handle)
    assert len(result) == 2
    assert result[0] > 0
    assert result[1] > 0
    print("Passed.")


def test_chunk_allocation_non_aligned():
    print("Testing Case 11: Chunk Allocation (Non-Aligned Reuse) ... ", end="", flush=True)
    cfg = make_config(num_tokens_per_chunk=128, page_size=32)  # 1 chunk = 4 pages
    mgr = NativeHostKVCacheManager(**cfg)
    num_layers, num_pages, page_size, H, D = 3, 128, 32, 4, 128
    fake = torch.randn(num_layers, num_pages, 2, page_size, H, D, dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))
    uids = torch.tensor([0], dtype=torch.int64)
    from kvcache_npu_cpp import KVOffloadHandle

    # Round 1: offload 1 page (< 1 chunk). start_pos=0 is chunk-aligned, so this
    # goes through the normal padded-allocation branch, not chunk reuse.
    starts_1 = torch.tensor([0], dtype=torch.int64)
    page_indices_1 = [torch.arange(0, 1, dtype=torch.int32)]
    handle_1 = KVOffloadHandle(num_layers)
    mgr.impl_.offload_kvcache(uids, starts_1, page_indices_1, handle_1)
    assert wait_handle_ready(handle_1), "round 1 offload handle did not become ready within 100 retries"
    mgr.impl_.finish_offload(handle_1)
    length_after_round1 = mgr.impl_.lookup(uids)[0].item()
    assert length_after_round1 == 32, f"expected 32 tokens (1 page) after round 1, got {length_after_round1}"

    # Round 2: offload starting at token 32, i.e. start_pos % num_tokens_per_chunk
    # == 32 != 0 -> this is the actual non-aligned-reuse branch in
    # get_empty_pinned_chunks(), which reuses the tail of round 1's chunk
    # (this uid's own _uid_to_chunks entry) instead of allocating a fresh one.
    starts_2 = torch.tensor([32], dtype=torch.int64)
    page_indices_2 = [torch.arange(1, 3, dtype=torch.int32)]  # 2 more pages
    handle_2 = KVOffloadHandle(num_layers)
    ok = mgr.impl_.offload_kvcache(uids, starts_2, page_indices_2, handle_2)
    assert ok, "round 2 offload_kvcache returned False: non-aligned reuse did not find a chunk to extend"
    assert wait_handle_ready(handle_2), "round 2 offload handle did not become ready within 100 retries"
    mgr.impl_.finish_offload(handle_2)

    length_after_round2 = mgr.impl_.lookup(uids)[0].item()
    assert length_after_round2 == 96, f"expected 96 tokens (3 pages) after non-aligned reuse, got {length_after_round2}"

    # The data written in round 1 must still be intact after round 2 reused
    # the same chunk's tail (i.e. round 2 did not clobber round 1's bytes).
    kvdata = mgr.impl_.get_kvdata_tensor([0], False)[0]
    cached_k, _cached_v = kvdata.unbind(dim=2)
    cached_k = cached_k.reshape(cached_k.size(0), -1, page_size, H, D)
    expected_k_round1 = fake[:, :1, 0, :, :, :].cpu()
    assert torch.allclose(cached_k[:, :1], expected_k_round1), "round 1 data corrupted by non-aligned chunk reuse"
    print("Passed.")


def test_acl_runtime_failure():
    print("Testing Case 12: Host-Side Failure Injection ... ", end="", flush=True)
    # Ascend's gather kernel (gather_paged_kvcache_kernel.h) does not bounds-check
    # page ids and offload_kvcache() only waits on a stream event rather than
    # synchronizing, so an out-of-range NPU page id is not guaranteed to raise a
    # Python exception (unlike the GPU reference's at::index_select). Instead of
    # asserting on that device-side assumption, inject a host-side failure that
    # is actually checked in this code path: an out-of-bounds offload_start_index,
    # which finish_offload() explicitly detects and raises on.
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    num_layers = 3
    fake = torch.randn(num_layers, 128, 2, 32, 4, 128, dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))
    uids = torch.tensor([0], dtype=torch.int64)
    starts = torch.tensor([99999], dtype=torch.int64)  # out of bounds
    page_indices_list = [torch.arange(0, 1, dtype=torch.int32)]
    from kvcache_npu_cpp import KVOffloadHandle

    handle = KVOffloadHandle(num_layers)
    mgr.impl_.offload_kvcache(uids, starts, page_indices_list, handle)
    assert wait_handle_ready(handle), "offload handle did not become ready within 100 retries"
    with pytest.raises(RuntimeError, match="chunks"):
        mgr.impl_.finish_offload(handle)
    print("Passed.")


def test_register_npu_cache_table_validation():
    print("Testing Case 13: register_npu_cache_table Validation ... ", end="", flush=True)
    cfg = make_config(num_layers=3)
    mgr = NativeHostKVCacheManager(**cfg)
    # 数量不匹配
    fake_wrong = torch.empty([2, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    with pytest.raises(RuntimeError, match="num_layers"):
        mgr.register_npu_cache_tables(list(fake_wrong.unbind(dim=0)))
    # 数量正确
    fake_ok = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake_ok.unbind(dim=0)))
    # 重复调用
    with pytest.raises(RuntimeError, match="already registered"):
        mgr.register_npu_cache_tables(list(fake_ok.unbind(dim=0)))
    print("Passed.")


def test_wrapper_finish_cancel_task_dispatch():
    print("Testing Case 14: Wrapper finish_task/cancel_task Dispatch ... ", end="", flush=True)
    # The preceding cases exercise offload/onboard exclusively through
    # mgr.impl_, bypassing NativeHostKVCacheManager's own finish_task/
    # cancel_task type dispatch entirely. This case drives that wrapper logic
    # directly: finish_task/cancel_task must reject an onload handle with
    # NotImplementedError, reject an unrecognized handle type with
    # ValueError, and finish_task on a real offload handle must return the
    # per-uid success list via the public API (not mgr.impl_.finish_offload).
    cfg = make_config()
    mgr = NativeHostKVCacheManager(**cfg)
    fake = torch.empty([3, 128, 2, 32, 4, 128], dtype=torch.bfloat16, device="npu")
    mgr.register_npu_cache_tables(list(fake.unbind(dim=0)))

    from kvcache_npu_cpp import KVOnloadHandle

    fake_onload_task = HostKVTaskHandle(backend="native", handle=KVOnloadHandle(3))
    with pytest.raises(NotImplementedError):
        mgr.finish_task(fake_onload_task)
    with pytest.raises(NotImplementedError):
        mgr.cancel_task(fake_onload_task)

    fake_bad_task = HostKVTaskHandle(backend="native", handle=object())
    with pytest.raises(ValueError):
        mgr.finish_task(fake_bad_task)

    uids = torch.tensor([0], dtype=torch.int64)
    starts = torch.tensor([0], dtype=torch.int64)
    page_indices_list = [torch.arange(0, 4, dtype=torch.int32)]
    task_handle = mgr.offload_kvcache_launch(uids, starts, page_indices_list)
    assert wait_offload_ready(mgr, task_handle) is not None, "offload timed out"

    result = mgr.finish_task(task_handle)
    assert isinstance(result, list)
    assert result[0] > 0, "finish_task via the public API did not report success"
    print("Passed.")


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))

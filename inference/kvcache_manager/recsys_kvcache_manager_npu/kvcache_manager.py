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

from typing import List, Optional, Tuple

import torch

from .host_kvstorage_manager import (
    HostKVStorageManagerBase,
    HostKVTaskHandle,
    HostKVTaskStatus,
    HostKVWaitResult,
)
from .kvcache_config import HostKVStorageBackend, HostKVStorageFailPolicy
from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVCacheOffloadMode, KVIndexMeta, KVLookupResult
from .npu_kvcache_manager import NPUKVCacheManager


class KVCacheManager:
    """顶层编排器：协调 NPU 侧页分配器与 Host 侧存储后端。

    推理系统通过此类与 KV cache 子系统交互，核心生命周期：
    1. lookup_kvcache: 同时查询 NPU 和 Host，合并结果
    2. allocate_kvcache: 在 NPU 上分配页
    3. onboard_launch/wait: 将 Host 数据异步搬入 NPU
    4. 推理使用 NPU 上的 KV cache
    5. offload_launch/try_wait: 将 NPU 数据异步搬回 Host
    """

    def __init__(
        self,
        npu_kvcache_manager: NPUKVCacheManager,
        host_kvstorage_manager: HostKVStorageManagerBase,
        offload_mode: str = KVCacheOffloadMode.LAZY.value,
        host_kvstorage_fail_policy: str = HostKVStorageFailPolicy.FAIL_OPEN.value,
    ):
        self.npu_kvcache_mgr = npu_kvcache_manager
        self.dummy_empty_tensor = torch.tensor([], dtype=torch.int32)

        self.host_kvstorage_manager = host_kvstorage_manager
        self.host_kvstorage_manager.register_npu_cache_tables(self.npu_kvcache_mgr.get_cache_tables())

        # offload 模式：lazy（按需 offload）或 eager（推理后立即 offload）
        # 失败策略：fail_open（失败继续）或 fail_close（失败抛异常）
        # 两者都做白名单校验并显式报错，不再对非法取值静默回退/放行。
        try:
            self.offload_mode = KVCacheOffloadMode(offload_mode)
        except ValueError as exc:
            raise ValueError(
                f"Unknown offload_mode {offload_mode!r}, expected one of {[m.value for m in KVCacheOffloadMode]}"
            ) from exc
        try:
            self.host_kvstorage_fail_policy = HostKVStorageFailPolicy(host_kvstorage_fail_policy)
        except ValueError as exc:
            raise ValueError(
                f"Unknown host_kvstorage_fail_policy {host_kvstorage_fail_policy!r}, expected one of "
                f"{[p.value for p in HostKVStorageFailPolicy]}"
            ) from exc
        self.ongoing_onboard_tasks: List[HostKVTaskHandle] = []
        self.ongoing_offload_tasks: List[HostKVTaskHandle] = []

    def lookup_kvcache(
        self,
        user_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Tuple[KVIndexMeta, KVLookupResult]:
        # 同时查询 NPU 和 Host，合并两侧的缓存状态
        npu_lookup_results = self.npu_kvcache_mgr.lookup(user_ids)
        index_meta = self.host_kvstorage_manager.build_index_meta(user_ids, sequence_lengths)
        host_lookup_results = self.host_kvstorage_manager.lookup_kvcache(index_meta)

        lookup_results = KVLookupResult.merge(npu_lookup_results, host_lookup_results)
        return index_meta, lookup_results

    def allocate_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> KVCacheMetadata:
        return self.npu_kvcache_mgr.allocate(
            index_meta.user_ids,
            index_meta.seq_lengths,
            lookup_results,
            output_kvcache_metadata=output_kvcache_metadata,
        )

    def onboard_launch(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> HostKVTaskHandle:
        # 启动异步 onboard（Host → NPU），推理线程稍后通过 onboard_wait 等待
        task_handle = self.host_kvstorage_manager.onboard_kvcache_launch(
            index_meta,
            lookup_result,
            kvcache_metadata,
        )
        kvcache_metadata.kv_onload_handle = task_handle
        return task_handle

    def _check_native_backend(self) -> None:
        if self.host_kvstorage_manager.backend_name != "native":
            raise NotImplementedError(f"Unknown host kvcache backend {self.host_kvstorage_manager.backend_name}")

    @staticmethod
    def _is_empty_handle(task_handle: Optional[HostKVTaskHandle]) -> bool:
        return task_handle is None or task_handle.handle is None or task_handle.status == HostKVTaskStatus.SKIPPED

    def onboard_try_wait(
        self,
        kv_index_meta: KVIndexMeta,  # unused: kept to mirror the upstream GPU manager's signature
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        self._check_native_backend()
        if self._is_empty_handle(task_handle):
            return HostKVWaitResult(
                status=HostKVTaskStatus.UNINITIALIZED,
                ready=False,
            )
        return self.host_kvstorage_manager.onboard_kvcache_wait(task_handle)

    def onboard_wait(
        self,
        kv_index_meta: KVIndexMeta,  # unused: kept to mirror the upstream GPU manager's signature
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        # 等待 onboard 完成；失败时根据策略回退页或抛异常
        self._check_native_backend()
        if self._is_empty_handle(task_handle):
            return HostKVWaitResult(
                status=HostKVTaskStatus.UNINITIALIZED,
                ready=False,
            )
        wait_result = self.host_kvstorage_manager.onboard_kvcache_wait(task_handle)

        if wait_result.status in (
            HostKVTaskStatus.FAILED,
            HostKVTaskStatus.TIMEOUT,
            HostKVTaskStatus.CANCELLED,
        ):
            self.npu_kvcache_mgr.revoke_onboard_pages(
                task_handle.user_ids,
                task_handle.metadata["onboard_start_indices"],
                task_handle.metadata["onboard_lengths"],
            )
            if self.host_kvstorage_fail_policy == HostKVStorageFailPolicy.FAIL_CLOSE:
                raise RuntimeError(
                    f"Onboarding failed for {wait_result.failed_user_ids}: "
                    f"status={wait_result.status.value}, msg={wait_result.message}"
                )
            else:
                print(
                    f"[WARNING] Onboarding failed for {wait_result.failed_user_ids}, "
                    "but continue with `fail_open` policy."
                )
        return wait_result

    def offload_launch(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ):
        """启动 offload（NPU → Host），两阶段锁定 NPU 页后异步搬运"""
        # 1. Get the set of users that have un-offloaded data on NPU
        uids_to_offload = self.npu_kvcache_mgr.check_for_offload(index_meta.user_ids)
        # 2. Lookup host again in case of multi-NPU instances
        _index_meta = self.host_kvstorage_manager.build_index_meta(
            uids_to_offload,
            torch.empty(0, dtype=torch.int32),  # dummy seq lengths since they are not used for lookup
        )
        offloaded_lengths = self.host_kvstorage_manager.lookup_kvcache(_index_meta).host_cached_lengths

        # 3. Acquire and lock NPU cache pages
        (
            offload_user_ids,
            offload_start_indices,
            offload_page_indices_list,
        ) = self.npu_kvcache_mgr.acquire_offload_pages(
            uids_to_offload,
            offloaded_lengths,
        )
        # returned with cache pages locked (per user).
        if offload_user_ids.size(0) == 0:
            return None

        # 4. Launch the offloading thru Host
        task_handle = self.host_kvstorage_manager.offload_kvcache_launch(
            offload_user_ids,
            offload_start_indices,
            offload_page_indices_list,
            index_meta=index_meta,
            kvcache_metadata=kvcache_metadata,
        )
        if task_handle is None or task_handle.handle is None or task_handle.status == HostKVTaskStatus.SKIPPED:
            # offload is rejected on the host side (e.g., due to overload), release the locks immediately.
            self.npu_kvcache_mgr.release_offload_pages(
                offload_user_ids,
                offload_start_indices,
                self.dummy_empty_tensor,
                offloaded=[0] * offload_user_ids.size(0),
            )
            return None

        self.ongoing_offload_tasks.append(task_handle)
        return task_handle

    def offload_try_wait(self) -> None:
        remain_tasks = []
        for task_handle in self.ongoing_offload_tasks:
            wait_result = self.host_kvstorage_manager.offload_kvcache_wait(task_handle)
            if wait_result.status == HostKVTaskStatus.LAUNCHED:
                remain_tasks.append(task_handle)
                continue
            if wait_result.status == HostKVTaskStatus.READY:
                offload_success = self.host_kvstorage_manager.finish_task(task_handle)
            elif wait_result.status == HostKVTaskStatus.SKIPPED:
                # The host declined the transfer, so nothing was written. Drop the page
                # lock taken in acquire_offload_pages() the same way offload_launch()
                # does for a launch-time rejection; the task is not retried, so skipping
                # the release would strand the lock and block eviction for these uids.
                print(f"Offload skipped for {task_handle.user_ids.tolist()}")
                self.npu_kvcache_mgr.release_offload_pages(
                    *(self.host_kvstorage_manager.get_offload_handle_metadata(task_handle)),
                    offloaded=[0] * task_handle.user_ids.size(0),
                )
                continue
            elif wait_result.status in (
                HostKVTaskStatus.FAILED,
                HostKVTaskStatus.TIMEOUT,
                HostKVTaskStatus.CANCELLED,
            ):
                if self.host_kvstorage_fail_policy == HostKVStorageFailPolicy.FAIL_CLOSE:
                    # Release the NPU page lock before raising: the lock was taken in
                    # acquire_offload_pages() and must be dropped on every exit path,
                    # otherwise this uid can never be evicted or offloaded again.
                    offload_success = [0] * task_handle.user_ids.size(0)
                    self.host_kvstorage_manager.cancel_task(task_handle)
                    self.npu_kvcache_mgr.release_offload_pages(
                        *(self.host_kvstorage_manager.get_offload_handle_metadata(task_handle)),
                        offloaded=offload_success,
                    )
                    raise RuntimeError(
                        f"Offloading failed for {wait_result.failed_user_ids}, "
                        f"fail_policy={self.host_kvstorage_fail_policy.value}"
                    )

                offload_success = [0] * task_handle.user_ids.size(0)
                self.host_kvstorage_manager.cancel_task(task_handle)
            else:
                raise RuntimeError(
                    f"Unexpected offload wait result status: {wait_result.status.value}, msg={wait_result.message}"
                )
            self.npu_kvcache_mgr.release_offload_pages(
                *(self.host_kvstorage_manager.get_offload_handle_metadata(task_handle)),
                offloaded=offload_success,
            )
        self.ongoing_offload_tasks = remain_tasks

    def evict(self, user_ids: torch.Tensor, for_npu: bool = False, for_host: bool = False):
        if not (for_npu or for_host):
            raise ValueError("evict() requires at least one of for_npu / for_host to be True")
        if for_npu:
            self.npu_kvcache_mgr.evict(user_ids)
        if for_host:
            self.host_kvstorage_manager.evict(user_ids)

    def evict_all(self, for_npu: bool = False, for_host: bool = False):
        if not (for_npu or for_host):
            raise ValueError("evict_all() requires at least one of for_npu / for_host to be True")
        if for_npu:
            self.npu_kvcache_mgr.evict_all()
        if for_host:
            self.host_kvstorage_manager.evict_all()

    @classmethod
    def from_config(cls, kvcache_config):
        if kvcache_config.offload_chunksize % kvcache_config.page_size != 0:
            raise ValueError(
                f"Require offload_chunksize ({kvcache_config.offload_chunksize}) to be a multiple of "
                f"page_size ({kvcache_config.page_size})"
            )

        # max_seq_len rounded up to a whole number of pages; NPU and host backends
        # must agree on this padded length, so derive it once here. Integer
        # arithmetic (no float division) avoids precision loss for large lengths.
        padded_max_seq_len = -(-kvcache_config.max_seq_len // kvcache_config.page_size) * kvcache_config.page_size

        npu_kvcache_mgr = NPUKVCacheManager(
            kvcache_config.num_layers,
            kvcache_config.num_heads,
            kvcache_config.head_dim,
            kvcache_config.page_size,
            kvcache_config.offload_chunksize,
            kvcache_config.num_primary_cache_pages,
            kvcache_config.num_buffer_pages,
            kvcache_config.max_batch_size,
            padded_max_seq_len,
            kvcache_config.dtype,
            kvcache_config.device,
        )

        if kvcache_config.host_kvstorage_backend == HostKVStorageBackend.NATIVE_HOST.value:
            from .native_host_kvcache_manager import NativeHostKVCacheManager

            host_kvcache_mgr = NativeHostKVCacheManager(
                kvcache_config.num_layers,
                kvcache_config.num_heads,
                kvcache_config.head_dim,
                kvcache_config.page_size,
                kvcache_config.offload_chunksize,
                kvcache_config.host_capacity_per_layer,
                kvcache_config.max_batch_size,
                padded_max_seq_len,
                kvcache_config.onload_timeout_ms,
                kvcache_config.offload_timeout_ms,
                kvcache_config.dtype,
                kvcache_config.device,
            )
        else:
            raise NotImplementedError(f"Unknown host kvcache backend {kvcache_config.host_kvstorage_backend}")

        return cls(
            npu_kvcache_mgr,
            host_kvcache_mgr,
            kvcache_config.offload_mode,
            kvcache_config.host_kvstorage_fail_policy,
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Dict, Optional, Tuple

import torch

from dynamic_emb.distributed.batched_dynamicemb_table import BatchedDynamicEmbeddingTablesV2
from dynamic_emb.distributed.dynamicemb_config import DynamicEmbPoolingMode, DynamicEmbTableOptions
from dynamic_emb.distributed.key_value_table import KeyValueTableFunction, KeyValueTableCachingFunction
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import EmbOptimType
from dynamic_emb.distributed.types import Storage
from key_value_table_test_helpers import run_caching_lookup


class MockOptimizer:
    def __init__(self):
        self.last_fused_update_shapes = None

    def get_state_dim(self, dim: int) -> int:
        return 0

    def get_initial_optim_states(self) -> float:
        return 0.0

    def fused_update(self, grads: torch.Tensor, values: torch.Tensor) -> None:
        self.last_fused_update_shapes = (tuple(grads.shape), tuple(values.shape))
        values[:, : grads.size(1)] += grads


class PyDictStorage(Storage):
    """In-memory external storage backend; data lives only in dict, not in HKV."""

    def __init__(self, options: DynamicEmbTableOptions, optimizer: MockOptimizer):
        self.options = options
        self.optimizer = optimizer
        self._emb_dim = options.dim
        self._value_dim = options.dim
        self._emb_dtype = options.embedding_dtype
        self._initial_optim_state = 0.0
        self.dict: Dict[int, torch.Tensor] = {}

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        missing_keys = []
        missing_indices = []
        found_flags = []
        for i in range(unique_keys.numel()):
            key = int(unique_keys[i].item())
            if key in self.dict:
                unique_vals[i, :] = self.dict[key]
                found_flags.append(True)
            else:
                found_flags.append(False)
                missing_keys.append(key)
                missing_indices.append(i)
        founds_tensor = torch.tensor(found_flags, dtype=torch.bool, device=unique_keys.device)
        if founds is not None:
            founds[:] = founds_tensor
        return (
            len(missing_keys),
            torch.tensor(missing_keys, dtype=unique_keys.dtype, device=unique_keys.device),
            torch.tensor(missing_indices, dtype=torch.long, device=unique_keys.device),
        )

    def find(self, unique_keys, unique_vals, founds=None):
        return self.find_impl(unique_keys, unique_vals, founds)

    def find_embeddings(self, unique_keys, unique_embs, founds=None):
        return self.find_impl(unique_keys, unique_embs, founds)

    def insert(self, keys, values, scores=None):
        for i in range(keys.numel()):
            self.dict[int(keys[i].item())] = values[i, :].clone()

    def update(self, keys, grads, return_missing=True):
        raise RuntimeError("PyDictStorage does not support storage.update")

    def enable_update(self) -> bool:
        return False

    def dump(self, meta_file_path, emb_key_path, embedding_file_path, optional_files=None):
        return

    def load(self, meta_file_path, emb_file_path, embedding_file_path, include_optim, optional_files=None):
        return

    def embedding_dtype(self):
        return self._emb_dtype

    def embedding_dim(self):
        return self._emb_dim

    def value_dim(self):
        return self._value_dim

    def init_optimizer_state(self):
        return self._initial_optim_state


class CacheStub:
    def __init__(self, emb_dim: int, value_dim: int, device: torch.device):
        self.emb_dim = emb_dim
        self.value_dim = value_dim
        self.device = device
        self.cache: Dict[int, torch.Tensor] = {}
        self.last_insert_and_evict_keys = None
        self.last_update_keys = None

    def find_embeddings(self, unique_keys, unique_embs, founds=None):
        missing_keys = []
        missing_indices = []
        for i in range(unique_keys.numel()):
            k = int(unique_keys[i].item())
            if k in self.cache:
                unique_embs[i, :] = self.cache[k][: self.emb_dim]
            else:
                missing_keys.append(k)
                missing_indices.append(i)
        return (
            len(missing_keys),
            torch.tensor(missing_keys, dtype=unique_keys.dtype, device=unique_keys.device),
            torch.tensor(missing_indices, dtype=torch.long, device=unique_keys.device),
        )

    def insert_and_evict(self, keys, values):
        self.last_insert_and_evict_keys = keys.clone()
        for i in range(keys.numel()):
            self.cache[int(keys[i].item())] = values[i, :].clone()
        return (
            0,
            torch.empty(0, dtype=keys.dtype, device=keys.device),
            torch.empty(0, self.value_dim, dtype=values.dtype, device=values.device),
            torch.empty(0, dtype=torch.int64, device=keys.device),
        )

    def update(self, unique_keys, unique_grads):
        self.last_update_keys = unique_keys.clone()
        missing_keys = []
        missing_indices = []
        for i in range(unique_keys.numel()):
            k = int(unique_keys[i].item())
            if k in self.cache:
                self.cache[k][: self.emb_dim] = self.cache[k][: self.emb_dim] + unique_grads[i, :]
            else:
                missing_keys.append(k)
                missing_indices.append(i)
        return (
            len(missing_keys),
            torch.tensor(missing_keys, dtype=unique_keys.dtype, device=unique_keys.device),
            torch.tensor(missing_indices, dtype=torch.long, device=unique_keys.device),
        )


class StorageEnableUpdateStub(PyDictStorage):
    def __init__(self, options, optimizer):
        super().__init__(options, optimizer)
        self.last_update_keys = None
        self.last_update_grads = None

    def enable_update(self) -> bool:
        return True

    def update(self, keys, grads, return_missing=True):
        self.last_update_keys = keys.clone()
        self.last_update_grads = grads.clone()
        for i in range(keys.numel()):
            k = int(keys[i].item())
            if k in self.dict:
                self.dict[k] = self.dict[k] + grads[i, :]
        return (
            0,
            torch.empty(0, dtype=keys.dtype, device=keys.device),
            torch.empty(0, dtype=torch.long, device=keys.device),
        )


def _external_storage_options(**kwargs) -> DynamicEmbTableOptions:
    defaults = dict(
        dim=2,
        embedding_dtype=torch.float32,
        index_type=torch.int64,
        training=True,
        caching=True,
        max_capacity=1024,
    )
    defaults.update(kwargs)
    return DynamicEmbTableOptions(**defaults)


def test_key_value_table_function_update_supports_external_storage_without_enable_update():
    optimizer = MockOptimizer()
    option = _external_storage_options()
    storage = PyDictStorage(option, optimizer)
    storage.insert(
        torch.tensor([11], dtype=torch.int64),
        torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    KeyValueTableFunction.update(
        storage,
        torch.tensor([11, 22], dtype=torch.int64),
        torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=torch.float32),
        optimizer,
    )

    assert 11 in storage.dict
    assert torch.allclose(storage.dict[11], torch.tensor([1.5, 2.5], dtype=torch.float32))
    assert 22 not in storage.dict
    assert optimizer.last_fused_update_shapes == ((1, 2), (1, 2))
    assert not hasattr(storage, "table")


def test_create_cache_storage_uses_external_storage(monkeypatch):
    from dynamic_emb.distributed import batched_dynamicemb_table as bdet_module

    class FakeKeyValueTable:
        def __init__(self, options, optimizer):
            self.options = options
            self.optimizer = optimizer

    monkeypatch.setattr(bdet_module, "KeyValueTable", FakeKeyValueTable)

    option = DynamicEmbTableOptions(
        dim=8,
        embedding_dtype=torch.float32,
        index_type=torch.int64,
        training=True,
        caching=True,
        max_capacity=4096,
        local_hbm_for_values=4 * 8 * 4096,
        external_storage=PyDictStorage,
    )

    bdet = BatchedDynamicEmbeddingTablesV2(
        table_options=[option],
        table_names=["t0"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.ADAM,
        device=torch.device("cpu"),
    )

    assert isinstance(bdet._caches[0], FakeKeyValueTable)
    assert isinstance(bdet._storages[0], PyDictStorage)
    assert not hasattr(bdet._storages[0], "table")


def test_key_value_table_caching_lookup_updates_cache_from_storage():
    option = _external_storage_options()
    optimizer = MockOptimizer()
    cache = CacheStub(emb_dim=2, value_dim=2, device=torch.device("cpu"))
    storage = PyDictStorage(option, optimizer)
    storage.insert(
        torch.tensor([7], dtype=torch.int64),
        torch.tensor([[3.0, 4.0]], dtype=torch.float32),
    )

    unique_keys = torch.tensor([7], dtype=torch.int64)
    unique_embs = torch.zeros((1, 2), dtype=torch.float32)

    def no_op_initializer(vals, missing_indices, keys):
        return

    run_caching_lookup(
        cache=cache,
        storage=storage,
        unique_keys=unique_keys,
        unique_embs=unique_embs,
        initializer=no_op_initializer,
        training=False,
    )

    assert torch.allclose(unique_embs, torch.tensor([[3.0, 4.0]], dtype=torch.float32))
    assert cache.last_insert_and_evict_keys is not None
    assert int(cache.last_insert_and_evict_keys[0].item()) == 7
    assert 7 in cache.cache


def test_key_value_table_caching_update_calls_storage_update_when_enabled():
    option = _external_storage_options()
    optimizer = MockOptimizer()
    cache = CacheStub(emb_dim=2, value_dim=2, device=torch.device("cpu"))
    storage = StorageEnableUpdateStub(option, optimizer)
    storage.insert(
        torch.tensor([5], dtype=torch.int64),
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    )

    KeyValueTableCachingFunction.update(
        cache=cache,
        storage=storage,
        unique_keys=torch.tensor([5], dtype=torch.int64),
        unique_grads=torch.tensor([[0.2, 0.3]], dtype=torch.float32),
        optimizer=optimizer,
    )

    assert storage.last_update_keys is not None
    assert int(storage.last_update_keys[0].item()) == 5


def test_key_value_table_caching_update_fallback_when_storage_update_disabled():
    option = _external_storage_options()
    optimizer = MockOptimizer()
    cache = CacheStub(emb_dim=2, value_dim=2, device=torch.device("cpu"))
    storage = PyDictStorage(option, optimizer)
    storage.insert(
        torch.tensor([31], dtype=torch.int64),
        torch.tensor([[2.0, 3.0]], dtype=torch.float32),
    )

    KeyValueTableCachingFunction.update(
        cache=cache,
        storage=storage,
        unique_keys=torch.tensor([31, 32], dtype=torch.int64),
        unique_grads=torch.tensor([[0.1, 0.2], [1.0, 1.0]], dtype=torch.float32),
        optimizer=optimizer,
    )

    assert 31 in storage.dict
    assert torch.allclose(storage.dict[31], torch.tensor([2.1, 3.2], dtype=torch.float32))
    assert 32 not in storage.dict
    assert optimizer.last_fused_update_shapes == ((1, 2), (1, 2))

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

import torch
import pytest
from typing import Optional, Tuple, Dict

from dynamic_emb.distributed.key_value_table import (
    KeyValueTableCachingFunction
)

npu_device = torch.device("npu:0")

class Cache:
    pass

class Storage:
    pass

class BaseDynamicEmbeddingOptimizerV2:
    """优化器基类"""
    pass

class BaseDynamicEmbInitializer:
    """初始化器基类"""
    pass

class MockCache(Cache):
    def __init__(self):
        self.cache_data: Dict[int, torch.Tensor] = {}
        self.score = 0

    def find_embeddings(self, unique_keys: torch.Tensor, unique_embs: torch.Tensor) \
        -> Tuple[int, torch.Tensor, torch.Tensor]:
        total_keys = unique_keys.numel()
        num_missing = total_keys // 2
        missing_keys = unique_keys[-num_missing:] \
            if num_missing > 0 else torch.tensor([], device=npu_device, dtype=torch.long)
        missing_indices = \
            torch.arange(total_keys - num_missing, total_keys, device=npu_device, dtype=torch.long) \
              if num_missing > 0 else torch.tensor([], device=npu_device, dtype=torch.long)

        unique_embs[:total_keys-num_missing] = torch.randn_like(unique_embs[:total_keys-num_missing])
        return num_missing, missing_keys, missing_indices

    def update(self, unique_keys: torch.Tensor, unique_grads: torch.Tensor) \
        -> Tuple[int, torch.Tensor, torch.Tensor]:
        total_keys = unique_keys.numel()
        num_missing = total_keys // 2
        missing_keys = unique_keys[-num_missing:] \
            if num_missing > 0 else torch.tensor([], device=npu_device, dtype=torch.long)
        missing_indices = \
            torch.arange(total_keys - num_missing, total_keys, device=npu_device, dtype=torch.long) \
              if num_missing > 0 else torch.tensor([], device=npu_device, dtype=torch.long)
        return num_missing, missing_keys, missing_indices

    def find_missed_keys(self, unique_keys: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        total_keys = unique_keys.numel()
        num_missing = total_keys // 2
        missing_keys = unique_keys[-num_missing:] \
            if num_missing > 0 else torch.tensor([], device=npu_device, dtype=torch.long)
        missing_indices = torch.arange(total_keys - num_missing, total_keys, device=npu_device, dtype=torch.long) \
            if num_missing > 0 else torch.tensor([], device=npu_device, dtype=torch.long)
        return num_missing, missing_keys, missing_indices

    def insert_and_evict(self, keys: torch.Tensor, values: torch.Tensor) \
        -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        return 0, torch.tensor([], device=npu_device), \
            torch.tensor([], device=npu_device), torch.tensor([], device=npu_device)

class MockStorage(Storage):
    def __init__(self, emb_dim: int = 16):
        self.emb_dim = emb_dim
        self.device = npu_device
        self.emb_dtype = torch.float32
        self.storage_data: Dict[int, torch.Tensor] = {}

    def embedding_dim(self) -> int:
        return self.emb_dim

    def embedding_dtype(self) -> torch.dtype:
        return self.emb_dtype

    def value_dim(self) -> int:
        return self.emb_dim * 2

    def find(self, keys: torch.Tensor, values: torch.Tensor, founds: Optional[torch.Tensor] = None) \
        -> Tuple[int, torch.Tensor, torch.Tensor]:
        if founds is not None:
            founds[:] = True

        values[:, :self.emb_dim] = torch.randn_like(values[:, :self.emb_dim])
        return 0, torch.tensor([], device=npu_device, dtype=torch.long), \
            torch.tensor([], device=npu_device, dtype=torch.long)

    def init_optimizer_state(self) -> torch.Tensor:
        return torch.zeros(self.emb_dim, dtype=self.emb_dtype, device=npu_device)

    def enable_update(self) -> bool:
        return False

    def update(self, keys: torch.Tensor, grads: torch.Tensor, return_missing: bool = False):
        pass

    def insert(self, keys: torch.Tensor, values: torch.Tensor, scores=None):
        pass

class MockOptimizer(BaseDynamicEmbeddingOptimizerV2):
    def fused_update(self, grads: torch.Tensor, values: torch.Tensor):
        pass

class MockInitializer(BaseDynamicEmbInitializer):
    def __call__(self, embs: torch.Tensor, indices: torch.Tensor, keys: torch.Tensor):
        embs[indices] = torch.randn_like(embs[indices])

def update_cache(cache: Cache, storage: Storage, missing_keys: torch.Tensor, missing_values: torch.Tensor):
    num_evicted, evicted_keys, evicted_values, evicted_scores = \
        cache.insert_and_evict(missing_keys, missing_values)
    if num_evicted != 0:
        storage.insert(evicted_keys, evicted_values, evicted_scores)


@pytest.mark.parametrize("training", [True, False])
def test_lookup_function(training: bool):
    cache = MockCache()
    storage = MockStorage(emb_dim=16)
    initializer = MockInitializer()

    unique_keys = torch.tensor([10, 20, 30], dtype=torch.long, device=npu_device)
    unique_embs = torch.zeros(3, 16, dtype=torch.float32, device=npu_device)

    KeyValueTableCachingFunction.lookup(
        cache=cache, storage=storage, unique_keys=unique_keys, unique_embs=unique_embs,
        initializer=initializer, enable_prefetch=False, training=training
    )

    assert not torch.all(unique_embs == 0)
    assert unique_embs.shape == (3, 16)
    print("√ lookup 测试通过")

def test_update_function():
    cache = MockCache()
    storage = MockStorage(emb_dim=16)
    optimizer = MockOptimizer()

    unique_keys = torch.tensor([5, 6, 7], dtype=torch.long, device=npu_device)
    unique_grads = torch.randn(3, 16, dtype=torch.float32, device=npu_device)

    KeyValueTableCachingFunction.update(
        cache=cache, storage=storage, unique_keys=unique_keys, unique_grads=unique_grads, optimizer=optimizer
    )
    print("√ update 测试通过")

@pytest.mark.parametrize("training", [True, False])
def test_prefetch_function(training: bool):
    cache = MockCache()
    storage = MockStorage(emb_dim=16)
    initializer = MockInitializer()

    unique_keys = torch.tensor([100, 200, 300], dtype=torch.long, device=npu_device)
    KeyValueTableCachingFunction.prefetch(
        cache=cache, storage=storage, unique_keys=unique_keys, initializer=initializer, training=training
    )
    print("√ prefetch 测试通过")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
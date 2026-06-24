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

import pytest
import torch

from dynamic_emb.distributed.key_value_table import KeyValueTableCachingFunction
from key_value_table_test_helpers import (
    DEFAULT_EMB_DIM,
    EVAL_INITIALIZER,
    TRAIN_INITIALIZER,
    create_training_cache_and_storage,
    insert_embeddings,
    npu_device,
    read_embeddings_from_storage,
    run_caching_lookup,
    sample_stored_keys_and_embs,
    sample_update_grads,
    sample_update_keys_and_embs,
)


@pytest.mark.parametrize(
    ("training", "initializer"),
    [
        (True, TRAIN_INITIALIZER),
        (False, EVAL_INITIALIZER),
    ],
)
def test_lookup_function(training: bool, initializer):
    device = npu_device()
    cache, storage, _ = create_training_cache_and_storage()

    stored_keys, stored_embs = sample_stored_keys_and_embs(device)
    insert_embeddings(storage, stored_keys, stored_embs)

    unique_keys = torch.tensor([10, 20, 30], dtype=torch.int64, device=device)
    unique_embs = torch.zeros(3, DEFAULT_EMB_DIM, dtype=torch.float32, device=device)

    run_caching_lookup(cache, storage, unique_keys, unique_embs, initializer, training)

    assert unique_embs.shape == (3, DEFAULT_EMB_DIM)
    torch.testing.assert_close(unique_embs[:2], stored_embs)

    cached_embs = torch.zeros(3, DEFAULT_EMB_DIM, dtype=torch.float32, device=device)
    num_missing, _, _ = cache.find_embeddings(unique_keys, cached_embs)
    if training:
        assert not torch.all(unique_embs[2] == 0)
        assert num_missing == 0
        torch.testing.assert_close(cached_embs, unique_embs)
    else:
        torch.testing.assert_close(unique_embs[2], torch.zeros(DEFAULT_EMB_DIM, device=device))
        assert num_missing == 1
        torch.testing.assert_close(cached_embs[:2], stored_embs)


def test_update_function():
    device = npu_device()
    cache, storage, optimizer = create_training_cache_and_storage()

    keys, init_embs = sample_update_keys_and_embs(device)
    insert_embeddings(storage, keys, init_embs)

    unique_grads = sample_update_grads(keys.numel(), DEFAULT_EMB_DIM, device)
    KeyValueTableCachingFunction.update(
        cache=cache,
        storage=storage,
        unique_keys=keys,
        unique_grads=unique_grads,
        optimizer=optimizer,
    )

    updated_from_storage = read_embeddings_from_storage(storage, keys)
    assert not torch.allclose(updated_from_storage, init_embs)


@pytest.mark.parametrize(
    ("training", "initializer"),
    [
        (True, TRAIN_INITIALIZER),
        (False, EVAL_INITIALIZER),
    ],
)
def test_prefetch_function(training: bool, initializer):
    device = npu_device()
    cache, storage, _ = create_training_cache_and_storage()

    unique_keys = torch.tensor([100, 200, 300], dtype=torch.int64, device=device)
    stored_embs = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5] * 4, [1.5, 1.5, 1.5, 1.5] * 4, [2.5, 2.5, 2.5, 2.5] * 4],
        dtype=torch.float32,
        device=device,
    )
    insert_embeddings(storage, unique_keys, stored_embs)

    KeyValueTableCachingFunction.prefetch(
        cache=cache,
        storage=storage,
        unique_keys=unique_keys,
        initializer=initializer,
        training=training,
    )

    cached_embs = torch.zeros(3, DEFAULT_EMB_DIM, dtype=torch.float32, device=device)
    num_missing, _, _ = cache.find_embeddings(unique_keys, cached_embs)
    assert num_missing == 0
    torch.testing.assert_close(cached_embs, stored_embs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

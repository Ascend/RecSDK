#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbEvictStrategy,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamic_emb.distributed.key_value_table import KeyValueTable
from dynamic_emb_extensions import OptimizerType, dyn_emb_is_pure_hbm_mode
from key_value_table_test_helpers import (
    DEVICE_ID,
    create_adam_optimizer,
    make_values,
    npu_device,
    read_embeddings_with_founds,
)

EMB_DIM = 10


def _create_kv_table(
    *,
    dim: int = EMB_DIM,
    local_hbm_for_values: int = 1024**3,
    max_capacity: int = 10000,
    init_capacity: int = 1024,
    evict_strategy: DynamicEmbEvictStrategy = DynamicEmbEvictStrategy.LRU,
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.TIMESTAMP,
) -> KeyValueTable:
    optimizer = create_adam_optimizer()
    options = DynamicEmbTableOptions(
        dim=dim,
        embedding_dtype=torch.float32,
        index_type=torch.int64,
        training=True,
        optimizer_type=OptimizerType.Adam,
        max_capacity=max_capacity,
        init_capacity=init_capacity,
        local_hbm_for_values=local_hbm_for_values,
        bucket_capacity=128,
        device_id=DEVICE_ID,
        evict_strategy=evict_strategy,
        score_strategy=score_strategy,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.NORMAL,
            value=0.0,
        ),
    )
    return KeyValueTable(options, optimizer)


def test_find_impl_partial_hit():
    device = npu_device()
    table = _create_kv_table()

    stored_keys = torch.tensor([101, 103], dtype=torch.int64, device=device)
    stored_embs = torch.tensor([[1.0] * EMB_DIM, [2.0] * EMB_DIM], dtype=torch.float32, device=device)
    table.insert(stored_keys, make_values(table, stored_embs))

    query_keys = torch.tensor([101, 102, 103, 104], dtype=torch.int64, device=device)
    query_embs = torch.zeros(4, EMB_DIM, dtype=torch.float32, device=device)

    num_missing, missing_keys, missing_indices = table.find_impl(query_keys, query_embs)

    assert num_missing == 2
    torch.testing.assert_close(missing_keys, torch.tensor([102, 104], dtype=torch.int64, device=device))
    torch.testing.assert_close(missing_indices, torch.tensor([1, 3], dtype=torch.int64, device=device))
    torch.testing.assert_close(query_embs[0], stored_embs[0])
    torch.testing.assert_close(query_embs[2], stored_embs[1])


@pytest.mark.parametrize(
    "local_hbm_for_values, expect_pure_hbm",
    [
        (1024**3, True),
        (0, False),
    ],
)
def test_find_impl_loads_inserted_embeddings(local_hbm_for_values, expect_pure_hbm):
    device = npu_device()
    table = _create_kv_table(local_hbm_for_values=local_hbm_for_values)

    assert dyn_emb_is_pure_hbm_mode(table.table) == expect_pure_hbm

    key = torch.tensor([201], dtype=torch.int64, device=device)
    emb = torch.full((1, EMB_DIM), 3.5, dtype=torch.float32, device=device)
    table.insert(key, make_values(table, emb))

    out_embs = torch.zeros(1, EMB_DIM, dtype=torch.float32, device=device)
    num_missing, _, _ = table.find_impl(key, out_embs)

    assert num_missing == 0
    torch.testing.assert_close(out_embs, emb)


def test_find_impl_no_hit():
    device = npu_device()
    table = _create_kv_table()

    query_keys = torch.tensor([301, 302, 303, 304], dtype=torch.int64, device=device)
    query_embs = torch.zeros(4, EMB_DIM, dtype=torch.float32, device=device)

    num_missing, missing_keys, missing_indices = table.find_impl(query_keys, query_embs)

    assert num_missing == 4
    torch.testing.assert_close(missing_keys, query_keys)
    torch.testing.assert_close(missing_indices, torch.arange(4, dtype=torch.int64, device=device))
    assert torch.all(query_embs == 0)


def test_insert_with_score():
    device = npu_device()
    table = _create_kv_table(
        evict_strategy=DynamicEmbEvictStrategy.CUSTOMIZED,
        score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
    )
    expected_score = 42
    table.set_score(expected_score)

    keys = torch.tensor([101, 102, 103], dtype=torch.int64, device=device)
    embs = torch.randn(3, EMB_DIM, dtype=torch.float32, device=device)
    table.insert(keys, make_values(table, embs))

    assert table.size() == 3
    read_embs, founds = read_embeddings_with_founds(table, keys)
    assert founds.all()
    torch.testing.assert_close(read_embs, embs)

    num_matched = torch.zeros(1, dtype=torch.int64, device=device)
    table.count_matched(expected_score - 1, num_matched)
    assert num_matched.item() == 3


def test_update_with_return_missing():
    device = npu_device()
    table = _create_kv_table()

    stored_keys = torch.tensor([101, 103], dtype=torch.int64, device=device)
    stored_embs = torch.tensor([[1.0] * EMB_DIM, [2.0] * EMB_DIM], dtype=torch.float32, device=device)
    table.insert(stored_keys, make_values(table, stored_embs))

    before_embs, _ = read_embeddings_with_founds(table, stored_keys)
    keys = torch.tensor([101, 102, 103, 104], dtype=torch.int64, device=device)
    grads = torch.full((4, EMB_DIM), 0.1, dtype=torch.float32, device=device)

    num_missing, missing_keys, missing_indices = table.update(keys, grads, return_missing=True)

    assert num_missing == 2
    torch.testing.assert_close(missing_keys, torch.tensor([102, 104], dtype=torch.int64, device=device))
    torch.testing.assert_close(missing_indices, torch.tensor([1, 3], dtype=torch.int64, device=device))

    after_embs, founds = read_embeddings_with_founds(table, stored_keys)
    assert founds.all()
    assert not torch.allclose(before_embs, after_embs)


def test_count_matched():
    device = npu_device()
    table = _create_kv_table(
        dim=16,
        evict_strategy=DynamicEmbEvictStrategy.CUSTOMIZED,
        score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
    )
    score = 100
    table.set_score(score)

    keys = torch.tensor([11, 22, 33], dtype=torch.int64, device=device)
    embs = torch.randn(3, 16, dtype=torch.float32, device=device)
    table.insert(keys, make_values(table, embs))

    num_matched = torch.zeros(1, dtype=torch.int64, device=device)
    table.count_matched(score - 1, num_matched)
    assert num_matched.item() == 3


def test_export_batch_matched():
    device = npu_device()
    table = _create_kv_table(
        dim=16,
        evict_strategy=DynamicEmbEvictStrategy.CUSTOMIZED,
        score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
    )
    score = 100
    table.set_score(score)

    keys = torch.tensor([11, 22, 33], dtype=torch.int64, device=device)
    embs = torch.randn(3, 16, dtype=torch.float32, device=device)
    table.insert(keys, make_values(table, embs))

    num_matched = torch.zeros(1, dtype=torch.int64, device=device)
    table.count_matched(score - 1, num_matched)
    expected = num_matched.item()

    batch_size = table.capacity()
    d_count = torch.zeros(1, dtype=torch.int64, device=device)
    d_keys = torch.empty(expected, dtype=torch.int64, device=device)
    d_vals = torch.empty(expected, 16, dtype=torch.float32, device=device)
    table.export_batch_matched(score - 1, batch_size, 0, d_count, d_keys, d_vals)

    assert d_count.item() == expected
    assert set(d_keys.cpu().tolist()) == {11, 22, 33}


def test_insert_and_evict():
    device = npu_device()
    table = _create_kv_table(max_capacity=256, init_capacity=256)

    num_keys = 300
    keys = torch.arange(1, num_keys + 1, dtype=torch.int64, device=device)
    values = torch.randn(num_keys, table.value_dim(), dtype=torch.float32, device=device)

    num_evicted, evicted_keys, evicted_values, evicted_scores = table.insert_and_evict(keys, values)

    assert num_evicted > 0
    assert table.size() <= 256
    assert evicted_keys.numel() == num_evicted
    assert evicted_values.size(0) == num_evicted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

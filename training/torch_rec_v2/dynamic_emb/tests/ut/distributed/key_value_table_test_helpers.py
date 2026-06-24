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

"""Shared helpers for KeyValueTable-related unit tests."""

from copy import deepcopy
from typing import Callable

import torch

from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
from dynamic_emb.distributed.initializers.dynamicemb_initializers import (
    ConstantInitializer,
    NormalInitializer,
)
from dynamic_emb.distributed.key_value_table import KeyValueTable, KeyValueTableCachingFunction
from dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer import AdamDynamicEmbeddingOptimizerV2
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import OptimizerArgs
from dynamic_emb_extensions import OptimizerType

DEVICE_ID = 0
DEFAULT_EMB_DIM = 16
CACHE_CAPACITY = 8192

TRAIN_INITIALIZER = NormalInitializer(DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.NORMAL))
EVAL_INITIALIZER = ConstantInitializer(DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.CONSTANT, value=0.0))


def npu_device() -> torch.device:
    return torch.device(f"npu:{DEVICE_ID}")


def create_adam_optimizer() -> AdamDynamicEmbeddingOptimizerV2:
    return AdamDynamicEmbeddingOptimizerV2(
        OptimizerArgs(
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )
    )


def create_training_cache_and_storage(
    emb_dim: int = DEFAULT_EMB_DIM,
) -> tuple[KeyValueTable, KeyValueTable, AdamDynamicEmbeddingOptimizerV2]:
    """Same setup as BatchedDynamicEmbeddingTablesV2._create_cache_storage."""
    optimizer = create_adam_optimizer()
    base_option = DynamicEmbTableOptions(
        dim=emb_dim,
        embedding_dtype=torch.float32,
        index_type=torch.int64,
        caching=True,
        training=True,
        optimizer_type=OptimizerType.Adam,
        max_capacity=8192,
        init_capacity=1024,
        local_hbm_for_values=1024**3,
        bucket_capacity=1024,
        device_id=DEVICE_ID,
        initializer_args=DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.NORMAL),
        eval_initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT,
            value=0.0,
        ),
    )

    cache_option = deepcopy(base_option)
    cache_option.bucket_capacity = 1024
    cache_option.max_capacity = CACHE_CAPACITY
    cache_option.init_capacity = CACHE_CAPACITY
    cache = KeyValueTable(cache_option, optimizer)

    storage_option = deepcopy(base_option)
    storage_option.local_hbm_for_values = 0
    storage_option.max_capacity = 10000
    storage_option.init_capacity = 1000
    storage_option.bucket_capacity = 128
    storage = KeyValueTable(storage_option, optimizer)

    return cache, storage, optimizer


def create_training_storage(
    emb_dim: int = DEFAULT_EMB_DIM,
) -> tuple[KeyValueTable, AdamDynamicEmbeddingOptimizerV2]:
    optimizer = create_adam_optimizer()
    option = DynamicEmbTableOptions(
        dim=emb_dim,
        embedding_dtype=torch.float32,
        index_type=torch.int64,
        caching=False,
        training=True,
        optimizer_type=OptimizerType.Adam,
        max_capacity=10000,
        init_capacity=1024,
        local_hbm_for_values=1024**3,
        bucket_capacity=128,
        device_id=DEVICE_ID,
        initializer_args=DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.NORMAL),
        eval_initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT,
            value=0.0,
        ),
    )
    return KeyValueTable(option, optimizer), optimizer


def make_values(table: KeyValueTable, embeddings: torch.Tensor) -> torch.Tensor:
    values = torch.zeros(
        embeddings.size(0),
        table.value_dim(),
        dtype=table.embedding_dtype(),
        device=embeddings.device,
    )
    values[:, : table.embedding_dim()] = embeddings
    return values


def insert_embeddings(
    table: KeyValueTable,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
) -> None:
    table.insert(keys, make_values(table, embeddings))


def read_embeddings_from_storage(
    storage: KeyValueTable,
    keys: torch.Tensor,
) -> torch.Tensor:
    val_dim = storage.value_dim()
    emb_dim = storage.embedding_dim()
    values = torch.zeros(
        keys.numel(),
        val_dim,
        dtype=storage.embedding_dtype(),
        device=keys.device,
    )
    founds = torch.empty(keys.numel(), dtype=torch.bool, device=keys.device)
    storage.find(keys, values, founds=founds)
    assert founds.all(), "expected all keys to exist in storage"
    return values[:, :emb_dim].clone()


def read_embeddings_with_founds(
    storage: KeyValueTable,
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embs = torch.zeros(keys.numel(), storage.embedding_dim(), dtype=storage.embedding_dtype(), device=keys.device)
    founds = torch.empty(keys.numel(), dtype=torch.bool, device=keys.device)
    storage.find_embeddings(keys, embs, founds=founds)
    return embs, founds


def sample_stored_keys_and_embs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stored_keys = torch.tensor([10, 20], dtype=torch.int64, device=device)
    stored_embs = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0] * 4, [5.0, 6.0, 7.0, 8.0] * 4],
        dtype=torch.float32,
        device=device,
    )
    return stored_keys, stored_embs


def sample_update_keys_and_embs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.tensor([5, 6, 7], dtype=torch.int64, device=device)
    init_embs = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0] * 4, [2.0, 0.0, 0.0, 0.0] * 4, [3.0, 0.0, 0.0, 0.0] * 4],
        dtype=torch.float32,
        device=device,
    )
    return keys, init_embs


def sample_update_grads(num_keys: int, emb_dim: int, device: torch.device) -> torch.Tensor:
    return torch.full((num_keys, emb_dim), 0.1, dtype=torch.float32, device=device)


def run_caching_lookup(
    cache,
    storage,
    unique_keys: torch.Tensor,
    unique_embs: torch.Tensor,
    initializer: Callable,
    training: bool,
) -> None:
    KeyValueTableCachingFunction.lookup(
        cache=cache,
        storage=storage,
        unique_keys=unique_keys,
        unique_embs=unique_embs,
        initializer=initializer,
        enable_prefetch=False,
        training=training,
    )

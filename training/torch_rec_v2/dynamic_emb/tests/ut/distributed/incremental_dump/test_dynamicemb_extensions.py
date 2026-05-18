#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import random
from dataclasses import astuple, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pytest
import torch
import torch_npu

from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    dyn_emb_to_torch,
)
from dynamic_emb_extensions import (
    DynamicEmbDataType,
    DynamicEmbTable,
    EvictStrategy,
    InitializerArgs,
    OptimizerType,
    device_timestamp,
    dyn_emb_rows,
    count_matched,
    export_batch_matched,
)

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


@dataclass
class ExtensionsTableOption:
    key_type: DynamicEmbDataType = DynamicEmbDataType.Int64
    value_type: DynamicEmbDataType = DynamicEmbDataType.Float32
    evict_strategy: EvictStrategy = EvictStrategy.kLru
    dim: int = 16
    init_capacity: int = 512 * 1024
    max_capacity: int = 512 * 1024
    local_hbm_for_values: int = 1024**3
    bucket_capacity: int = 128
    max_load_factor: float = 0.6
    block_size: int = 128
    io_block_size: int = 1024
    device_id: int = -1
    io_by_cpu: bool = False
    use_constant_memory: bool = False
    reserved_key_start_bit: int = 0
    num_of_buckets_par_alloc: int = 1
    initializer_args: InitializerArgs = field(
        default_factory=lambda: DynamicEmbInitializerArgs().as_ctype()
    )
    safe_check_mode: int = DynamicEmbCheckMode.IGNORE.value
    optimizer_type: OptimizerType = OptimizerType.Null


class ScoreAdaptor:
    def __init__(
        self, evict_strategy: EvictStrategy, dtype: torch.dtype, device: torch.device
    ):
        self.evict_strategy_ = evict_strategy
        self.dtype_ = dtype
        self.device_ = device

        min_step = 0
        if evict_strategy == EvictStrategy.kLru:
            self.min_score_: int = device_timestamp()
        elif evict_strategy == EvictStrategy.kLfu:
            # LFU mode: use monotonically increasing score
            self.min_score_: int = min_step
        else:
            self.min_score_: int = min_step
        self.step_: int = min_step + 1

    def min_score(self):
        return self.min_score_

    #   add LFU here same to customized
    def next_score(self) -> int:
        if self.evict_strategy_ == EvictStrategy.kLru:
            return device_timestamp()
        else:
            # LFU and other modes: return monotonically increasing score
            next_score = self.step_
            self.step_ += 1
            return next_score

    # add LFU here
    def score(self):
        if self.evict_strategy_ == EvictStrategy.kLru:
            return None
        elif self.evict_strategy_ == EvictStrategy.kLfu:
            return 1
        else:
            score = self.step_
            self.step_ += 1
            return score


class LFUSimulator:
    """Host-side LFU policy simulator for comparison with HKV table results"""

    def __init__(self, initial_capacity: int, dim: int):
        self.capacity = initial_capacity
        self.dim = dim

        # key -> (value, frequency, insertion_order)
        self.table: Dict[int, Tuple[torch.Tensor, int, int]] = {}
        self.insertion_counter = 0

    def size(self) -> int:
        return len(self.table)

    def find_or_insert(
        self, keys: torch.Tensor, values: torch.Tensor, score: int = 1
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Simulate find_or_insert operation
        Returns: (found_mask, evicted_keys)
        """
        batch_size = keys.size(0)
        found_mask = torch.zeros(batch_size, dtype=torch.bool)
        evicted_keys = []

        # Calculate number of new keys to insert
        new_keys_count = 0
        for key in keys:
            if key.item() not in self.table:
                new_keys_count += 1

        for i, key in enumerate(keys):
            key_item = key.item()
            self.insertion_counter += 1

            if key_item in self.table:
                # Key exists, update frequency and access time
                old_value, old_freq, _ = self.table[key_item]
                new_freq = old_freq + score
                self.table[key_item] = (
                    old_value.clone(),
                    new_freq,
                    self.insertion_counter,
                )
                values[i] = old_value
                found_mask[i] = True
            else:
                # Key doesn't exist, need to insert
                if len(self.table) < self.capacity:
                    # Has space, insert directly
                    self.table[key_item] = (
                        values[i].clone(),
                        score,
                        self.insertion_counter,
                    )
                else:
                    # Need to evict, find the lowest frequency key
                    min_freq = float("inf")
                    min_key = None
                    min_time = float("inf")

                    for k, (v, freq, time) in self.table.items():
                        # LFU policy: evict lowest frequency first, if same frequency then evict earliest accessed
                        if freq < min_freq or (freq == min_freq and time < min_time):
                            min_freq = freq
                            min_key = k
                            min_time = time

                    # Evict the least frequently used key
                    if min_key is not None:
                        del self.table[min_key]
                        evicted_keys.append(min_key)

                    # Insert new key
                    self.table[key_item] = (
                        values[i].clone(),
                        score,
                        self.insertion_counter,
                    )

                found_mask[i] = False

        return found_mask, evicted_keys

    def find(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Simulate find operation, don't update frequency
        """
        batch_size = keys.size(0)
        found_mask = torch.zeros(batch_size, dtype=torch.bool)

        for i, key in enumerate(keys):
            key_item = key.item()
            if key_item in self.table:
                value, freq, time = self.table[key_item]
                values[i] = value
                found_mask[i] = True
            else:
                found_mask[i] = False

        return found_mask

    def insert_and_evict(
        self, keys: torch.Tensor, values: torch.Tensor, score: int = 1
    ) -> Tuple[List[int], int]:
        """
        Simulate insert_and_evict operation
        Returns: (evicted_keys, num_evicted)
        """
        evicted_keys = []

        for i, key in enumerate(keys):
            key_item = key.item()
            self.insertion_counter += 1

            if key_item in self.table:
                # Key exists, update frequency
                old_value, old_freq, _ = self.table[key_item]
                new_freq = old_freq + score
                self.table[key_item] = (
                    values[i].clone(),
                    new_freq,
                    self.insertion_counter,
                )
            else:
                # Key doesn't exist, need to insert
                if len(self.table) >= self.capacity:
                    # Need to evict
                    min_freq = float("inf")
                    min_key = None
                    min_time = float("inf")

                    for k, (v, freq, time) in self.table.items():
                        if freq < min_freq or (freq == min_freq and time < min_time):
                            min_freq = freq
                            min_key = k
                            min_time = time

                    if min_key is not None:
                        del self.table[min_key]
                        evicted_keys.append(min_key)

                # Insert new key
                self.table[key_item] = (
                    values[i].clone(),
                    score,
                    self.insertion_counter,
                )

        return evicted_keys, len(evicted_keys)

    def get_keys_by_score_threshold(self, min_score: int) -> Set[int]:
        """Get all keys with frequency >= min_score"""
        return {k for k, (v, freq, time) in self.table.items() if freq >= min_score}

    def get_all_keys(self) -> Set[int]:
        """Get all keys"""
        return set(self.table.keys())

    def get_key_frequency(self, key: int) -> Optional[int]:
        """Get frequency of specified key"""
        if key in self.table:
            return self.table[key][1]
        return None


def random_indices(batch, min_index, max_index):
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


@pytest.fixture
def current_device():
    assert torch.npu.is_available()
    return torch.npu.current_device()


@pytest.fixture(name="ext_option")
def extentions_table_option(current_device):
    ext_table_option = ExtensionsTableOption()
    ext_table_option.device_id = current_device
    return ext_table_option


@pytest.fixture
def score_type():
    """
    It's safe to convert from uint64 to int64 for score:
      1. Under DynamicEmbScoreStrategy.TIMESTAMP mode, score is a device timestamp in nanosecond,
        and it takes nearly 300 years since GPU startup to make the highest bit to 1.
      2. Under DynamicEmbScoreStrategy.STEP mode, it will take longer,
        because score increase for 1 insertion and not for 1 nanosecond.
    """
    return torch.int64


@pytest.fixture
def counter_dtype():
    return torch.int64


@pytest.mark.parametrize(
    "evict_strategy",
    [EvictStrategy.kLru, EvictStrategy.kCustomized],
)
@pytest.mark.parametrize(
    "bucket_capacity, batch, capacity, num_iteration, dump_interval",
    [
        pytest.param(
            128, 128, 512 * 1024, 8192, 1024, id="Never evict keys from current batch"
        ),
        pytest.param(128, 65536, 512 * 1024, 32, 8),
        pytest.param(
            128, 1024 + 13, 1024, 12, 3, id="Always evict keys from current batch"
        ),
        pytest.param(
            128, 1024, 4 * 1024, 32, 4, id="Always evict keys from last dump_interval"
        ),
        pytest.param(512, 512, 512 * 1024, 2048, 256, id="Different bucket capacity"),
    ],
)
def test_dynamicemb_extensions(
    request,
    ext_option,
    score_type,
    counter_dtype,
    evict_strategy,
    bucket_capacity,
    batch,
    capacity,
    num_iteration,
    dump_interval,
):
    logger.info("Test: %s", request.node.name)
    # init
    ext_option.init_capacity = capacity
    ext_option.max_capacity = capacity
    ext_option.bucket_capacity = bucket_capacity
    ext_option.evict_strategy = evict_strategy
    ext_option.num_of_buckets_par_alloc = capacity // bucket_capacity

    assert ext_option.dim * ext_option.max_capacity <= ext_option.local_hbm_for_values

    table = DynamicEmbTable(*astuple(ext_option))
    device = torch.device(f"npu:{ext_option.device_id}")
    score_adaptor = ScoreAdaptor(evict_strategy, score_type, device)
    init_score: int = score_adaptor.min_score()

    # insert once: the first find_or_insert will insert all keys, and no eviction(batch = bucket_capacity)
    keys_set = random_indices(bucket_capacity, 0, (1 << 63) - 1)
    keys = torch.tensor(
        list(keys_set), dtype=dyn_emb_to_torch(ext_option.key_type), device=device
    )
    values = torch.empty(
        bucket_capacity,
        ext_option.dim,
        dtype=dyn_emb_to_torch(ext_option.value_type),
        device=device,
    )
    score = score_adaptor.score()
    scores = None
    if score is not None:
        scores = torch.full((bucket_capacity,), score, dtype=score_type, device=device)
    table.load(bucket_capacity, keys, values, scores, True, False)

    # check 1: count_matched works well
    d_num_matched = torch.zeros(1, dtype=counter_dtype, device=device)
    count_matched(table, init_score, d_num_matched)
    num_matched = d_num_matched.cpu().item()
    assert num_matched == bucket_capacity

    # check 2: export_batch_matched is consistent with count_matched
    dump_keys = torch.empty(num_matched, dtype=keys.dtype, device=device)
    dump_vals = torch.empty(
        num_matched, ext_option.dim, dtype=values.dtype, device=device
    )
    d_num_matched.fill_(0)
    export_batch_matched(
        table, init_score, table.get_max_capacity(), 0, d_num_matched, dump_keys, dump_vals
    )
    assert num_matched == d_num_matched.cpu().item()
    assert keys_set == set(dump_keys.cpu().tolist())


def export_all_keys_values_scores(table, device):
    """
    Export all keys, values and scores from HKV table
    Returns: (keys, values, scores) all on CPU
    """
    capacity = table.get_max_capacity()
    current_size = dyn_emb_rows(table)
    dim = table.get_emb_cols() # Get actual dimension from table

    if current_size == 0:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dim, dtype=torch.float32),
            torch.empty(0, dtype=torch.uint64),
        )

    # Prepare output tensors
    key_dtype = torch.int64  # key type
    value_dtype = torch.float32  # value type
    score_dtype = torch.uint64  # score type

    batch_size = min(65536, capacity)

    all_keys = []
    all_values = []
    all_scores = []

    offset = 0
    while offset < capacity:
        # Prepare batch tensors
        keys = torch.empty(batch_size, dtype=key_dtype, device=device)
        values = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
        scores = torch.empty(batch_size, dtype=score_dtype, device=device)
        d_counter = torch.zeros(1, dtype=torch.uint64, device=device)

        # Call export_batch
        table.export_batch(batch_size, offset, d_counter, keys, values, scores)

        # Get actual returned count
        actual_count = d_counter.cpu().item()

        if actual_count > 0:
            # Keep only valid data
            valid_keys = keys[:actual_count].cpu()
            valid_values = values[: actual_count * dim].view(actual_count, dim).cpu()
            valid_scores = scores[:actual_count].cpu()

            all_keys.append(valid_keys)
            all_values.append(valid_values)
            all_scores.append(valid_scores)

        offset += batch_size

        # If this batch returns no data, we've finished traversing
        if actual_count == 0:
            break

    if all_keys:
        return torch.cat(all_keys), torch.cat(all_values), torch.cat(all_scores)
    else:
        return (
            torch.empty(0, dtype=key_dtype),
            torch.empty(0, dim, dtype=value_dtype),
            torch.empty(0, dtype=score_dtype),
        )

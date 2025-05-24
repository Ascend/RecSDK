#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DataLoaderInput:
    dataset: torch.utils.data.Dataset
    batch_size: int
    world_size: int
    rank: int
    shuffle: bool
    prefetch_factor: int = 128
    num_workers: int = 8
    drop_last: bool = False


def create_data_loader(data_loader_input_params: DataLoaderInput) -> Tuple[
    Optional[torch.utils.data.distributed.DistributedSampler], torch.utils.data.DataLoader]:
    shuffle = data_loader_input_params.shuffle
    dataset = data_loader_input_params.dataset
    batch_size = data_loader_input_params.batch_size
    world_size = data_loader_input_params.world_size
    rank = data_loader_input_params.rank
    prefetch_factor = data_loader_input_params.prefetch_factor
    num_workers = data_loader_input_params.num_workers
    drop_last = data_loader_input_params.drop_last
    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=drop_last,
        )
    else:
        sampler = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
    )
    return sampler, data_loader

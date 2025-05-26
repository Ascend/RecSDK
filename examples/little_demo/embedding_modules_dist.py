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
import abc
import logging

logger = logging.getLogger(__name__)
import torch
import torch.distributed as dist

from initialization import truncated_normal
from .embedding_function_dist import DistributedEmbeddingFunction


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_items
        self.embedding_dim = item_embedding_dim
        self._item_embedding_dim = item_embedding_dim
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{dist.get_rank()}")
        elif torch.npu.is_available():
            self.device = torch.device(f"npu:{dist.get_rank()}")
        else:
            self.device = "cpu"

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Shard by embedding table
        self.shard_size = (self.num_embeddings + self.world_size - 1) // self.world_size
        self.shard_start = self.rank * self.shard_size
        self.shard_end = min((self.rank + 1) * self.shard_size, self.num_embeddings)
        self.local_dim = self.embedding_dim
        self.local_num_embeddings = self.shard_end - self.shard_start

        # Initialize local embedding
        self._item_emb = torch.nn.Embedding(
            self.local_num_embeddings, self.local_dim, padding_idx=0
        )
        self.reset_params()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return DistributedEmbeddingFunction.apply(
            input_ids, self._item_emb.weight, self.shard_start, self.shard_end
        )

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                logger.info(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                logger.info(f"Skipping initializing params {name} - not configured")

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

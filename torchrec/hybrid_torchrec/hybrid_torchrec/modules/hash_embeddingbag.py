#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from hybrid_torchrec.modules.embedding_config import HYBRID_SUPPORT_DEVICE
from hybrid_torchrec.modules.ids_process import IdsMapper
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    pooling_type_to_str,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


@torch.fx.wrap
def reorder_inverse_indices(
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]],
    feature_names: List[str],
) -> torch.Tensor:
    if inverse_indices is None:
        return torch.empty(0)
    index_per_name = {name: i for i, name in enumerate(inverse_indices[0])}
    index = torch.tensor(
        [index_per_name[name.split("@")[0]] for name in feature_names],
        device=inverse_indices[1].device,
    )
    return torch.index_select(inverse_indices[1], 0, index)


@torch.fx.wrap
def process_pooled_embeddings(
    pooled_embeddings: List[torch.Tensor],
    inverse_indices: torch.Tensor,
) -> torch.Tensor:
    if inverse_indices.numel() > 0:
        pooled_embeddings = torch.ops.fbgemm.group_index_select_dim0(
            pooled_embeddings, list(torch.unbind(inverse_indices))
        )
    return torch.cat(pooled_embeddings, dim=1)


class EvictStrategy(Enum):
    kLru = 0
    kLfu = 1
    kEpochLru = 2
    kEpochLfu = 3
    kCustomized = 4


@dataclass
class HashTableOption:
    init_capacity: int = 0
    max_capacity: int = 0
    max_npu_memory_for_vectors: int = 0
    max_bucket_size: int = 0
    dim: int = 64
    max_load_factor: float = 0.5
    block_size: int = 128
    io_block_size: int = 1024
    device_id: int = -1
    io_by_cpu: bool = False
    use_constant_memory = False


@dataclass
class HashEmbeddingBagConfig(EmbeddingBagConfig):
    pass


class HashEmbeddingBag(torch.nn.Module):
    def __init__(self, config: HashEmbeddingBagConfig, device: torch.device):
        pass

    def find_and_insert(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
        founds: torch.Tensor,
    ):
        return NotImplemented

    def forward(self, input_tensor: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        return NotImplemented


class HybridHashTable(torch.nn.Module):
    def __init__(self, config: HashEmbeddingBagConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.ids2slot_dict = IdsMapper(self.config.num_embeddings)
        self.vector_table = torch.nn.EmbeddingBag(
            self.config.num_embeddings,
            self.config.embedding_dim,
            mode=pooling_type_to_str(config.pooling),
            device=device,
            include_last_offset=True,
        )
        self.index = 0
        self.register_parameter("weight", self.vector_table.weight)

    def forward(
        self,
        input_tensor: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        per_sample_weights=None,
    ):
        raw_device = input_tensor.device
        ids_host = input_tensor.cpu()
        index_of_ids, _, _ = self.ids2slot_dict(ids_host)
        index_of_ids = index_of_ids.to(raw_device)
        values = self.vector_table(index_of_ids, offsets)
        return values


class HashEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    def __init__(
        self,
        tables: List[HashEmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._is_weighted = is_weighted
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        self._embedding_bag_configs = tables
        self._lengths_per_embedding: List[int] = []
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )

        table_names = set()
        for embedding_config in tables:
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            dtype = (
                torch.float32
                if embedding_config.data_type == DataType.FP32
                else torch.float16
            )
            is_hybrid_device = (isinstance(device, str) and device in HYBRID_SUPPORT_DEVICE
                               ) or (hasattr(device, 'type') and device.type in HYBRID_SUPPORT_DEVICE)
            if is_hybrid_device:
                self.embedding_bags[embedding_config.name] = HybridHashTable(
                    config=embedding_config,
                    device=self._device,
                )
            else:
                raise NotImplementedError(
                    f"HashEmbeddingBagCollection for {device} is not implemented, the device is {device}"
                )

            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            self._lengths_per_embedding.extend(
                len(embedding_config.feature_names) * [embedding_config.embedding_dim]
            )

        self._embedding_names: List[str] = [
            embedding
            for embeddings in get_embedding_names_by_table(tables)
            for embedding in embeddings
        ]
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self._device

    def is_weighted(self) -> bool:
        return self._is_weighted

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        flat_feature_names: List[str] = []
        for names in self._feature_names:
            flat_feature_names.extend(names)
        inverse_indices = reorder_inverse_indices(
            inverse_indices=features.inverse_indices_or_none(),
            feature_names=flat_feature_names,
        )
        pooled_embeddings: List[torch.Tensor] = []
        feature_dict = features.to_dict()
        for i, embedding_bag in enumerate(self.embedding_bags.values()):
            for feature_name in self._feature_names[i]:
                f = feature_dict[feature_name]
                res = embedding_bag(
                    input_tensor=f.values(),
                    offsets=f.offsets(),
                    per_sample_weights=f.weights() if self._is_weighted else None,
                ).float()
                pooled_embeddings.append(res)
        return KeyedTensor(
            keys=self._embedding_names,
            values=process_pooled_embeddings(
                pooled_embeddings=pooled_embeddings,
                inverse_indices=inverse_indices,
            ),
            length_per_key=self._lengths_per_embedding,
        )

    def reset_parameters(self) -> None:
        if isinstance(self.device, torch.device) and self.device.type == "meta":
            return
        if isinstance(self.device, str) and self.device == "meta":
            return
        # Initialize embedding bags weights with init_fn
        for table_config in self._embedding_bag_configs:
            if table_config.init_fn is None:
                raise AttributeError(
                    f"The init_fn is None, table name '{table_config.name}'"
                )
            param = self.embedding_bags[f"{table_config.name}"].weight
            # pyre-ignore
            table_config.init_fn(param)

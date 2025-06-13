#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional

import torch
from torch import nn

from hybrid_torchrec.modules.embedding_config import HYBRID_SUPPORT_DEVICE
from hybrid_torchrec.modules.hash_embeddingbag import HybridHashTable

from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingConfig,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor
from torchrec.modules.embedding_modules import (
    EmbeddingCollectionInterface,
    get_embedding_names_by_table,
)


class HashEmbeddingCollection(EmbeddingCollectionInterface):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        self._embedding_configs = tables
        self._embedding_dim: int = -1
        self._need_indices: bool = need_indices
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )

        table_names = set()
        for config in tables:
            if config.name in table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            table_names.add(config.name)
            self._embedding_dim = (
                config.embedding_dim if self._embedding_dim < 0 else self._embedding_dim
            )
            if self._embedding_dim != config.embedding_dim:
                raise ValueError(
                    "All tables in a EmbeddingCollection are required to have same embedding dimension."
                    + f" Violating case: {config.name}'s embedding_dim {config.embedding_dim} !="
                    + f" {self._embedding_dim}"
                )

            dtype = (
                torch.float32 if config.data_type == DataType.FP32 else torch.float16
            )

            is_hybrid_device = (isinstance(device, str) and device in HYBRID_SUPPORT_DEVICE
                               ) or (hasattr(device, 'type') and device.type in HYBRID_SUPPORT_DEVICE)

            if is_hybrid_device:
                self.embeddings[config.name] = HybridHashTable(
                    config=config,
                    device=self._device,
                )
            else:
                raise NotImplementedError(
                    f"HashEmbeddingBagCollection for {device} is not implemented, the device is {device}"
                )

            if config.init_fn is not None:
                config.init_fn(self.embeddings[config.name].weight)

            if not config.feature_names:
                config.feature_names = [config.name]

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(
            tables
        )
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        """
        Run the EmbeddingCollection forward pass. This method takes in a `KeyedJaggedTensor`
        and returns a `Dict[str, JaggedTensor]`, which is the result of the individual embeddings for each feature.

        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """

        feature_embeddings: Dict[str, JaggedTensor] = {}
        jt_dict: Dict[str, JaggedTensor] = features.to_dict()
        for i, emb_module in enumerate(self.embeddings.values()):
            feature_names = self._feature_names[i]
            embedding_names = self._embedding_names_by_table[i]
            for j, embedding_name in enumerate(embedding_names):
                feature_name = feature_names[j]
                f = jt_dict[feature_name]
                lookup = emb_module(
                    input=f.values(),
                ).float()
                feature_embeddings[embedding_name] = JaggedTensor(
                    values=lookup,
                    lengths=f.lengths(),
                    weights=f.values() if self._need_indices else None,
                )
        return feature_embeddings

    def need_indices(self) -> bool:
        return self._need_indices

    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return self._embedding_configs

    def embedding_names_by_table(self) -> List[List[str]]:
        return self._embedding_names_by_table

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the EmbeddingCollection. Parameter values
        are initialized based on the `init_fn` of each EmbeddingConfig if it exists.
        """
        is_meta_device = (isinstance(self.device, torch.device) and self.device.type == "meta") or (
            isinstance(self.device, str) and self.device == "meta"
        )
        if is_meta_device:
            return
        # Initialize embedding bags weights with init_fn
        for table_config in self._embedding_configs:
            if table_config.init_fn is not None:
                param = self.embeddings[f"{table_config.name}"].weight
                table_config.init_fn(param)

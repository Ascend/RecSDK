#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.modules.crossnet import LowRankCrossNet
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor, JaggedTensor
from torchrec.models.dlrm import (
    InteractionDCNArch,
    OverArch,
    DenseArch,
    InteractionArch,
)


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    return 0


class SparseArchEC(nn.Module):
    def __init__(self, embedding_collection: EmbeddingCollection, multi_hot_size) -> None:
        super().__init__()
        self.embedding_collection: EmbeddingCollection = embedding_collection
        if not self.embedding_collection.embedding_configs:
            raise ValueError("Embedding bag collection cannot be empty!")
        self.d: int = self.embedding_collection.embedding_configs()[0].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_collection.embedding_configs()
            for name in conf.feature_names
        ]

        self.f: int = len(self._sparse_feature_names)
        self.name2multi_hosts = {
            name: hotsize
            for name, hotsize in zip(self.sparse_feature_names, multi_hot_size)
        }

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """

        sparse_features: Dict[str, JaggedTensor] = self.embedding_collection(features)

        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            mutil_hot = self.name2multi_hosts[name]
            pooling_result = (
                sparse_features[name].values().reshape(-1, mutil_hot, self.d).sum(1)
            )
            sparse_values.append(pooling_result)

        return torch.cat(sparse_values, dim=1).reshape(-1, self.f, self.d)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DlrmDcnEc(nn.Module):
    def __init__(
        self,
        embedding_collection: EmbeddingCollection,
        multi_hot_sizes: List[int],
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__()

        for i in range(1, len(embedding_collection.embedding_configs())):
            conf_prev = embedding_collection.embedding_configs()[i - 1]
            conf = embedding_collection.embedding_configs()[i]
            if conf_prev.embedding_dim != conf.embedding_dim:
                raise ValueError("All EmbeddingConfigs must have the same dimension")
        embedding_dim: int = embedding_collection.embedding_configs()[0].embedding_dim
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"embedding_collection dimension ({embedding_dim}) and final dense "
                "arch layer size ({dense_arch_layer_sizes[-1]}) must match."
            )

        self.sparse_arch: SparseArchEC = SparseArchEC(embedding_collection, multi_hot_sizes)
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )

        self.inter_arch = InteractionArch(
            num_sparse_features=num_sparse_features,
        )

        over_in_features: int = (
                embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
        )

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )

        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        # Fix interaction and over arch for DLRM_DCN

        crossnet = LowRankCrossNet(
            in_features=(num_sparse_features + 1) * embedding_dim,
            num_layers=dcn_num_layers,
            low_rank=dcn_low_rank_dim,
        )

        self.inter_arch = InteractionDCNArch(
            num_sparse_features=num_sparse_features,
            crossnet=crossnet,
        )

        over_in_features: int = (num_sparse_features + 1) * embedding_dim

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )


    def forward(
            self,
            dense_features: torch.Tensor,
            sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits

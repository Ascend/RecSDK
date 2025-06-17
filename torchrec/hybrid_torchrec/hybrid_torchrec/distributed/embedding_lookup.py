#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn

from hybrid_torchrec.distributed.batched_embedding_kernel import (
    HybridBatchedFusedEmbeddingBag,
    HybridBatchedFusedEmbedding
)
from torchrec.distributed.batched_embedding_kernel import KeyValueEmbeddingBag, KeyValueEmbedding
from torchrec.distributed.comm_ops import get_gradient_division
from torchrec.distributed.embedding_kernel import BaseEmbedding
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    BaseEmbeddingLookup,
)

from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    GroupedEmbeddingsLookup,
)
from torchrec.distributed.types import rank_device, ShardingType


class HybridGroupedPooledEmbeddingsLookup(GroupedPooledEmbeddingsLookup):
    """
    Lookup modules for Pooled embeddings (i.e EmbeddingBags)
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        pg: Optional[dist.ProcessGroup] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
        scale_weight_gradients: bool = True,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
            device: Optional[torch.device] = None,
        ) -> BaseEmbedding:
            if config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return NotImplemented
            elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
                return HybridBatchedFusedEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    sharding_type=sharding_type,
                )
            elif config.compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE,
            }:
                return KeyValueEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    sharding_type=sharding_type,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        BaseEmbeddingLookup.__init__(self)
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        self._need_prefetch = False
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config, device))

        self._feature_splits: List[int] = []
        for config in grouped_configs:
            self._feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs
        self._feature_processor = feature_processor

        self._scale_gradient_factor: int = (
            dist.get_world_size(pg)
            if scale_weight_gradients and get_gradient_division()
            else 1
        )


class HybridGroupedEmbeddingsLookup(GroupedEmbeddingsLookup):
    """
    Lookup modules for Sequence embeddings (i.e Embeddings)
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
        ) -> BaseEmbedding:
            for table in config.embedding_tables:
                if (
                    table.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING
                    or table.compute_kernel == EmbeddingComputeKernel.KEY_VALUE
                ):
                    self._need_prefetch = True

            if config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return NotImplemented
            elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
                return HybridBatchedFusedEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
            elif config.compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE,
            }:
                return KeyValueEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        BaseEmbeddingLookup.__init__(self)
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        self._need_prefetch: bool = False
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config))

        self._feature_splits: List[int] = []
        for config in grouped_configs:
            self._feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs

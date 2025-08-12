#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
)

import torch
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.embedding import EmbeddingCollectionSharder

from torchrec_embcache.distributed.embedding_bag import (
    EmbCacheShardedEmbeddingBagCollection,
    EmbCacheEmbeddingBagCollection,
)
from torchrec_embcache.distributed.embedding import (
    EmbCacheShardedEmbeddingCollection,
    EmbCacheEmbeddingCollection,
)


class EmbCacheEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    """
    This implementation uses non-fused `EmbCacheEmbeddingBagCollection`
    """

    def __init__(
        self,
        cpu_device: torch.device,
        cpu_env: ShardingEnv,
        npu_device: torch.device,
        npu_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            fused_params=fused_params,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
        self._cpu_device = cpu_device
        self._cpu_env = cpu_env
        self._npu_device = npu_device
        self._npu_env = npu_env

    def shard(
        self,
        module: EmbCacheEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> EmbCacheShardedEmbeddingBagCollection:
        return EmbCacheShardedEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            fused_params=self.fused_params,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            cpu_env=self._cpu_env,
            cpu_device=self._cpu_device,
            npu_device=self._npu_device,
            npu_env=self._npu_env,
            module_fqn=module_fqn,
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.ROW_WISE.value,
        ]

    @property
    def module_type(self) -> Type[EmbCacheEmbeddingBagCollection]:
        return EmbCacheEmbeddingBagCollection


class EmbCacheEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    """
    This implementation uses non-fused `EmbCacheEmbeddingCollection`
    """

    def __init__(
        self,
        cpu_device: torch.device,
        cpu_env: ShardingEnv,
        npu_device: torch.device,
        npu_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            fused_params=fused_params,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
        self._cpu_device = cpu_device
        self._cpu_env = cpu_env
        self._npu_device = npu_device
        self._npu_env = npu_env

    def shard(
        self,
        module: EmbCacheEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> EmbCacheShardedEmbeddingCollection:
        return EmbCacheShardedEmbeddingCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            fused_params=self.fused_params,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            cpu_env=self._cpu_env,
            cpu_device=self._cpu_device,
            npu_device=self._npu_device,
            npu_env=self._npu_env,
            module_fqn=module_fqn,
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.ROW_WISE.value,
        ]

    @property
    def module_type(self) -> Type[EmbCacheEmbeddingCollection]:
        return EmbCacheEmbeddingCollection

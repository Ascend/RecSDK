#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, cast, Dict, List, Optional, Type, TypeVar

import torch
from torch import nn
from hybrid_torchrec.distributed.sharding.hybrid_tw_sequence_sharding import (
    HybridHashTwSequenceEmbeddingSharding,
)
from hybrid_torchrec.distributed.embedding import HybridShardedEmbeddingCollection
from hybrid_torchrec.distributed.sharding.hybrid_rw_sequence_sharding import (
    HybridHashRwSequenceEmbeddingSharding,
)
from hybrid_torchrec.modules.hash_embedding import HashEmbeddingCollection
from hybrid_torchrec.modules.ids_process import HashMapBase
from hybrid_torchrec.distributed.embedding_types import (
    kjt_list_to_device,
)

from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import BaseEmbeddingSharder, KJTList
from torchrec.distributed.types import (
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.distributed.embedding import (
    EmbeddingCollectionAwaitable,
)


Out = TypeVar("Out")


class HybridShardedHashEmbeddingCollection(HybridShardedEmbeddingCollection):
    """
    Sharded implementation of EmbeddingCollection.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: HashEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        host_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
    ) -> None:
        self.table2hashmap: Dict[str, HashMapBase] = self.create_table2hashmap(module)
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            env,
            host_env,
            fused_params,
            device,
            qcomm_codecs_registry=qcomm_codecs_registry,
            use_index_dedup=use_index_dedup
        )

    def create_hybrid_embedding_sharding(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        host_env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        sharding_type = sharding_infos[0].param_sharding.sharding_type
        if sharding_type == ShardingType.TABLE_WISE.value:
            return HybridHashTwSequenceEmbeddingSharding(
                sharding_infos,
                self.table2hashmap,
                env,
                host_env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            return HybridHashRwSequenceEmbeddingSharding(
                sharding_infos,
                self.table2hashmap,
                env,
                self._host_env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            raise ValueError(
                f"Sharding type not supported {sharding_type} for hybrid mode"
            )

    def create_table2hashmap(self, module):
        table2hashmap = {}
        for name in module.embeddings.keys():
            hashmap = module.embeddings[name].ids2slot_dict
            table2hashmap[name] = hashmap
        return table2hashmap


class HybridHashEmbeddingCollectionSharder(
    BaseEmbeddingSharder[EmbeddingCollection]
):
    """
    This implementation uses non-fused `EmbeddingCollection`
    """

    def __init__(
        self,
        host_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            fused_params=fused_params, qcomm_codecs_registry=qcomm_codecs_registry
        )
        self._host_env = host_env

    @property
    def module_type(self) -> Type[HashEmbeddingCollection]:
        return HashEmbeddingCollection

    def shard(
        self,
        module: EmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> HybridShardedHashEmbeddingCollection:
        return HybridShardedHashEmbeddingCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            host_env=self._host_env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )

    def shardable_parameters(
        self, module: EmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embeddings.named_parameters()
        }

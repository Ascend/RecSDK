#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import os
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Mapping, Union, Type, Tuple
from collections import defaultdict
import logging
import numpy as np

import torch_npu
import torch
from torch import distributed as dist, nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hybrid_torchrec.distributed.embeddingbag import (
    _apply_mean_pooling,
    _create_mean_pooling_divisor,
    MeanPoolingConfig,
)
from hybrid_torchrec.constants import MAX_CACHINE_MEM_SIZE
from hybrid_torchrec.modules.ids_process import IdsMapper
from hybrid_torchrec.modules.ids_process import HashMapBase
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    EMPTY_POST_INPUT_DIST,
    PostInputKJTListAwaitable,
)
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
)
from hybrid_torchrec.utils import check
from torchrec_embcache.distributed.configs import (
    EmbCacheEmbeddingBagConfig,
    check_valid_value,
    check_create_table_params,
    check_multi_hot_sizes
)
from torchrec_embcache.distributed.sharding.rw_sharding import (
    EmbCacheRwPooledEmbeddingSharding,
)
from torchrec_embcache.distributed.utils import get_embedding_optim_num
from torchrec_embcache.embcache_pybind import (
    EmbcacheManager,
    EmbConfig,
    AsyncSwapInfo,
    AsyncSwapinTensor,
    InitializerType as CppInitType,
    SwapInfo,
)

from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.distributed.model_parallel import (
    DistributedDataParallel,
)
from torchrec.sparse.jagged_tensor import KeyedTensor, KeyedJaggedTensor
from torchrec.distributed.embedding_types import (
    ShardingType,
    KJTList,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    QuantizedCommCodecs,
    EmbeddingModuleShardingPlan,
    ShardingEnv,
    ShardedTensor,
    ParameterSharding,
)
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    EmbeddingShardingContext,
    KJTListSplitsAwaitable,
)
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    pooling_type_to_str,
    PoolingType,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.distributed.embeddingbag import (
    ShardedEmbeddingBagCollection,
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionAwaitable,
    create_sharding_infos_by_sharding,
)


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ShardingConfig:
    sharding_type: str
    table2hashmap: Dict[str, HashMapBase]
    sharding_infos: List[EmbeddingShardingInfo]
    cpu_env: ShardingEnv
    cpu_device: Optional[torch.device] = None
    permute_embeddings: bool = False
    qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None
    npu_device: Optional[torch.device] = None
    npu_env: Optional[ShardingEnv] = None


def create_embcache_embedding_bag_sharding(
    sharding_config: ShardingConfig,
) -> EmbeddingSharding[
    EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
]:
    if sharding_config.sharding_type == ShardingType.ROW_WISE.value:
        return EmbCacheRwPooledEmbeddingSharding(
            sharding_config.sharding_infos,
            sharding_config.table2hashmap,
            cpu_env=sharding_config.cpu_env,
            cpu_device=sharding_config.cpu_device,
            npu_device=sharding_config.npu_device,
            npu_env=sharding_config.npu_env,
            qcomm_codecs_registry=sharding_config.qcomm_codecs_registry,
        )
    else:
        raise ValueError(f"Sharding type not supported {sharding_config.sharding_type}")


class EmbCacheHashTable(torch.nn.Module):
    def __init__(self, config: EmbeddingBagConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.ids2slot_dict = IdsMapper(self.config.num_embeddings, only_device_memory=False)
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
        input_feat: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        per_sample_weights=None,
    ):
        raw_device = input_feat.device
        ids_host = input_feat.cpu()
        index_of_ids, _, _ = self.ids2slot_dict(ids_host)
        index_of_ids = index_of_ids.to(raw_device)
        values = self.vector_table(index_of_ids, offsets)
        return values


class EmbCacheEmbeddingBagCollection(EmbeddingBagCollection):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).

    It processes sparse data in the form of `KeyedJaggedTensor` with values of the form
    [F X B X L] where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (jagged)

    and outputs a `KeyedTensor` with values of the form [B * (F * D)] where:

    * F: features (keys)
    * D: each feature's (key's) embedding dimension
    * B: batch size

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        device (Optional[torch.device]): default compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[table_0, table_1])

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]   <-- key=f1, and jagged tensor
        # "f2"   [3]    [4]    [5,6,7]  <-- key=f2, and jagged tensor
        #  ^
        # feature

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[-0.8899, -0.1342, -1.9060, -0.0905, -0.2814, -0.9369, -0.7783],
            [ 0.0000,  0.0000,  0.0000,  0.1598,  0.0695,  1.3265, -0.1011],
            [-0.4256, -1.1846, -2.1648, -1.0893,  0.3590, -1.9784, -0.7681]],
            grad_fn=<CatBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        tensor([0, 3, 7])
    """

    def __init__(
        self,
        tables: List[EmbCacheEmbeddingBagConfig | EmbeddingBagConfig],
        world_size: int,
        batch_size: int,
        multi_hot_sizes: List[int], 
        is_weighted: bool = False,
        need_accumulate_offset: bool = True,
        device: Optional[torch.device] = None,
        embedding_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
    ) -> None:
        check_multi_hot_sizes(multi_hot_sizes, tables)
        check(all([isinstance(item, (EmbCacheEmbeddingBagConfig, EmbeddingBagConfig)) for item in tables]),
              "all element type in 'tables' must be EmbCacheEmbeddingBagConfig or EmbeddingBagConfig object")
        check_create_table_params(batch_size, embedding_optimizer_cls, multi_hot_sizes, tables, world_size)
        check_valid_value(isinstance(is_weighted, bool) and not is_weighted, "is_weighted must be False")

        super().__init__(tables, is_weighted, device)
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._is_weighted = is_weighted
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        self.need_accumulate_offset: bool = need_accumulate_offset
        self._convert_2_cache_embedding_bag_config(tables)
        self._embedding_bag_configs = tables
        self._lengths_per_embedding: List[int] = []
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )
        self._optim_num = get_embedding_optim_num(embedding_optimizer_cls)
        logger.debug("======  _optim_num: %s", self._optim_num)

        # 16GB -> 16*1024*1024*1024 -> 17179869184
        try:
            embcache_size_on_device_mem = int(os.getenv("EMBCACHE_SIZE_ON_DEVICE_MEM", "17179869184"))
        except ValueError as err:
            logger.error("environ EMBCACHE_SIZE_ON_DEVICE_MEM must be int: %s", err)
            raise err
        if embcache_size_on_device_mem > MAX_CACHINE_MEM_SIZE:
            raise ValueError(f"EMBCACHE_SIZE_ON_DEVICE_MEM is greater than MAX_CACHINE_MEM_SIZE {MAX_CACHINE_MEM_SIZE}")
        logger.debug("======  embcache_size_on_device_mem: %s", embcache_size_on_device_mem)

        cache_num_embeddings = self._calculate_caches(
            tables, embcache_size_on_device_mem, multi_hot_sizes, batch_size, world_size
        )
        logger.debug("table_num_embeddings: %s", cache_num_embeddings)
        table_names = set()
        for index, embedding_config in enumerate(tables):
            # Use the cache_num_embeddings to embedding config
            embedding_config.num_embeddings = int(cache_num_embeddings[index])
            embedding_config.cache = int(cache_num_embeddings[index])

            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            self.embedding_bags[embedding_config.name] = EmbCacheHashTable(
                config=embedding_config, device=self._device
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

    @staticmethod
    def _convert_2_cache_embedding_bag_config(tables: List[EmbCacheEmbeddingBagConfig | EmbeddingBagConfig]):
        for i, ori_config in enumerate(tables):
            if isinstance(ori_config, EmbCacheEmbeddingBagConfig):
                continue
            emb_cache_config = EmbCacheEmbeddingBagConfig(embedding_dim=ori_config.embedding_dim,
                                                          num_embeddings=ori_config.num_embeddings)
            emb_cache_config.__dict__.update(ori_config.__dict__)
            tables[i] = emb_cache_config

    def _calculate_caches(
        self,
        tables: List[EmbeddingBagConfig],
        max_device_mem_for_vectors: int,
        multi_hot_sizes: List[int],
        batch_size: int,
        world_size: int,
    ) -> List[int]:
        embedding_dims = []
        for embedding_config in tables:
            embedding_dims.append(embedding_config.embedding_dim)
        dtype_size = 4  # default fp32
        weight_and_optim_count = self._optim_num + 1
        # 由于同时训练和换出，最少要能放下2倍batch_size的emb+optim
        min_mem = np.sum(
            np.dot(
                np.multiply(embedding_dims, multi_hot_sizes),
                dtype_size * 2 * batch_size * weight_and_optim_count,
            )
        )
        if max_device_mem_for_vectors < min_mem:
            raise ValueError(
                f"max_device_mem_for_vectors {max_device_mem_for_vectors} < min_mem:{min_mem}"
            )

        table_num_embeddings = np.trunc(
            np.dot(
                multi_hot_sizes,
                (1.0 * max_device_mem_for_vectors / min_mem) * 2 * batch_size * world_size,
            )
        ).astype(int)
        return table_num_embeddings


class EmbCacheShardedEmbeddingBagCollection(ShardedEmbeddingBagCollection):
    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        npu_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        npu_device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        cpu_device: Optional[torch.device] = None,
        cpu_env: Optional[ShardingEnv] = None,
        module_fqn: Optional[str] = None,
    ) -> None:
        super(EmbCacheShardedEmbeddingBagCollection.__bases__[0], self).__init__(
            qcomm_codecs_registry=qcomm_codecs_registry
        )

        self._module_fqn = module_fqn

        self.table2hashmap: Dict[str, HashMapBase] = self.create_table2hashmap(module)
        self._embedding_bag_configs: List[EmbeddingBagConfig] = (
            module.embedding_bag_configs()
        )

        self._table_names: List[str] = []
        self._pooling_type_to_rs_features: Dict[str, List[str]] = defaultdict(list)
        self._table_name_to_config: Dict[str, EmbeddingBagConfig] = {}
        self._post_input_dists: List[nn.Module] = []

        for config in self._embedding_bag_configs:
            self._table_names.append(config.name)
            self._table_name_to_config[config.name] = config

            if table_name_to_parameter_sharding[config.name].sharding_type in [
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.ROW_WISE.value,
            ]:
                self._pooling_type_to_rs_features[config.pooling.value].extend(
                    config.feature_names
                )

        self.module_sharding_plan: EmbeddingModuleShardingPlan = cast(
            EmbeddingModuleShardingPlan,
            {
                table_name: parameter_sharding
                for table_name, parameter_sharding in table_name_to_parameter_sharding.items()
                if table_name in self._table_names
            },
        )
        self._env = npu_env
        # output parameters as DTensor in state dict
        self._output_dtensor: bool = npu_env.output_dtensor

        self.sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module,
            table_name_to_parameter_sharding,
            "embedding_bags.",
            fused_params,
        )
        self._sharding_types: List[str] = list(
            self.sharding_type_to_sharding_infos.keys()
        )
        self._embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = [
            create_embcache_embedding_bag_sharding(
                ShardingConfig(
                    sharding_type,
                    self.table2hashmap,
                    embedding_configs,
                    cpu_env=cpu_env,
                    cpu_device=cpu_device,
                    permute_embeddings=True,
                    qcomm_codecs_registry=self.qcomm_codecs_registry,
                    npu_env=npu_env,
                    npu_device=npu_device,
                )
            )
            for sharding_type, embedding_configs in self.sharding_type_to_sharding_infos.items()
        ]

        self._is_weighted: bool = module.is_weighted()
        self._device = npu_device
        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._create_lookups()
        self._output_dists: List[nn.Module] = []
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []
        self._uncombined_embedding_names: List[str] = []
        self._uncombined_embedding_dims: List[int] = []
        self._inverse_indices_permute_indices: Optional[torch.Tensor] = None
        self._has_uninitialized_post_input_dist: bool = True
        # to support mean pooling callback hook
        self._has_mean_pooling_callback: bool = PoolingType.MEAN.value in self._pooling_type_to_rs_features
        self._dim_per_key: Optional[torch.Tensor] = None
        self._kjt_key_indices: Dict[str, int] = {}
        self._kjt_inverse_order: Optional[torch.Tensor] = None
        self._kt_key_ordering: Optional[torch.Tensor] = None
        # to support the FP16 hook
        self._create_output_dist()
        self._dim_per_key_cpu = torch.tensor(self._embedding_dims, device="cpu")

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_features_permute: bool = True
        # Get all fused optimizers and combine them.
        optims = []
        for lookup in self._lookups:
            for _, tbe_module in lookup.named_modules():
                if isinstance(tbe_module, FusedOptimizerModule):
                    # modify param keys to match EmbeddingBagCollection
                    params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in tbe_module.fused_optimizer.params.items():
                        # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`
                        params["embedding_bags." + param_key] = weight
                    tbe_module.fused_optimizer.params = params
                    optims.append(("", tbe_module.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

        for i, (sharding, lookup) in enumerate(
            zip(self._embedding_shardings, self._lookups)
        ):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                self._lookups[i] = DistributedDataParallel(
                    module=lookup,
                    device_ids=(
                        [self._device]
                        if self._device is not None
                        and (self._device.type in {"cuda", "mtia"})
                        else None
                    ),
                    process_group=npu_env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    static_graph=True,
                )

        # BUG Fix 注册两次
        self._state_dict_pre_hooks.clear()
        self._state_dict_hooks.clear()
        self._load_state_dict_pre_hooks.clear()

        if npu_env.process_group and dist.get_backend(npu_env.process_group) != "fake":
            self._initialize_torch_state()

        if module.device not in ["meta", "cpu"] and module.device.type not in [
            "meta",
            "cpu",
        ]:
            self.load_state_dict(module.state_dict(), strict=False)

        self._memcpy_stream: Optional[torch_npu.npu.streams.Stream] = (
            torch_npu.npu.Stream(priority=-1)
        )
        self._embcache_mgr = self._create_embcache_mgr(module.need_accumulate_offset)

    @property
    def embcache_mgr(self):
        return self._embcache_mgr
    
    def _init_mean_pooling_callback(
        self,
        input_feature_names: List[str],
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]],
    ) -> None:
        # account for shared features
        feature_names: List[str] = [
            feature_name
            for sharding in self._embedding_shardings
            for feature_name in sharding.feature_names()
        ]

        for i, key in enumerate(feature_names):
            if key not in self._kjt_key_indices:  # index of first occurence
                self._kjt_key_indices[key] = i

        keyed_tensor_ordering = []
        for key in self._embedding_names:
            if "@" in key:
                key = key.split("@")[0]
            keyed_tensor_ordering.append(self._kjt_key_indices[key])
        self._kt_key_ordering = torch.tensor(keyed_tensor_ordering, device=self._device)
        self._kt_key_ordering_cpu = torch.tensor(keyed_tensor_ordering, device="cpu")
        if inverse_indices:
            key_to_inverse_index = {
                name: i
                for i, name in enumerate(inverse_indices[0])
            }
            self._kjt_inverse_order = torch.tensor(
                [key_to_inverse_index[key] for key in feature_names],
                device=self._device,
            )

    def create_table2hashmap(self, module):
        table2hashmap = {}
        for name in module.embedding_bags.keys():
            hashmap = module.embedding_bags[name].ids2slot_dict
            table2hashmap[name] = hashmap
        return table2hashmap

    def input_dist(
        self, ctx: EmbeddingBagCollectionAwaitable, features: KeyedJaggedTensor
    ) -> Awaitable[Awaitable[KJTList]]:
        """
        feature的顺序按照Dict[str, list[]]  shardType -> [t.feature_name for t in tables]
        """
        features = features.to("cpu")
        ctx.variable_batch_per_feature = features.variable_stride_per_key()
        ctx.inverse_indices = features.inverse_indices_or_none()
        if self._has_uninitialized_input_dist:
            self._create_input_dist(features.keys())
            self._has_uninitialized_input_dist = False
            if ctx.variable_batch_per_feature:
                self._create_inverse_indices_permute_indices(ctx.inverse_indices)
            if self._has_mean_pooling_callback:
                self._init_mean_pooling_callback(features.keys(), ctx.inverse_indices)
        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    self._features_order_tensor,
                )
            if self._has_mean_pooling_callback:
                ctx.divisor = _create_mean_pooling_divisor(MeanPoolingConfig(
                    lengths=features.lengths(),
                    stride=features.stride(),
                    keys=features.keys(),
                    offsets=features.offsets(),
                    pooling_type_to_rs_features=self._pooling_type_to_rs_features,
                    stride_per_key=features.stride_per_key(),
                    dim_per_key=self._dim_per_key_cpu,
                    embedding_names=self._embedding_names,
                    embedding_dims=self._embedding_dims,
                    variable_batch_per_feature=ctx.variable_batch_per_feature,
                    kjt_inverse_order=self._kjt_inverse_order,
                    kjt_key_indices=self._kjt_key_indices,
                    kt_key_ordering=self._kt_key_ordering_cpu,
                    inverse_indices=ctx.inverse_indices,
                    weights=features.weights_or_none(),
                ))

            features_by_shards = features.split(
                self._feature_splits,
            )
            awaitables = []
            for input_dist, features_by_shard in zip(
                self._input_dists, features_by_shards
            ):
                awaitables.append(input_dist(features_by_shard))
                ctx.sharding_contexts.append(
                    EmbeddingShardingContext(
                        batch_size_per_feature_pre_a2a=features_by_shard.stride_per_key(),
                        variable_batch_per_feature=features_by_shard.variable_stride_per_key(),
                    )
                )
            return KJTListSplitsAwaitable(awaitables, ctx)

    def compute_and_output_dist(
        self, ctx: EmbeddingBagCollectionContext, input_feat: KJTList
    ) -> LazyAwaitable[KeyedTensor]:
        awaitables = []
        for lookup, out_dist, sharding_ctx, features in zip(
            self._lookups,
            self._output_dists,
            ctx.sharding_contexts,
            input_feat,
        ):
            awaitables.append(out_dist(lookup(features), sharding_ctx))

        awaitable = EmbeddingBagCollectionAwaitable(
            awaitables=awaitables,
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

        # register callback if there are features that need mean pooling
        if self._has_mean_pooling_callback:
            awaitable.callbacks.append(
                partial(_apply_mean_pooling, divisor=ctx.divisor)
            )

        return awaitable

    def post_input_dist(
        self, ctx: EmbeddingBagCollectionAwaitable, features: KJTList
    ) -> PostInputKJTListAwaitable:
        """
        feature的顺序按照Dict[str, list[]]  shardType -> [t.feature_name for t in tables]
        """
        if self._has_uninitialized_post_input_dist:
            self._create_post_input_dist()
            self._has_uninitialized_post_input_dist = False
        with torch.no_grad():
            await_list = []
            for post_dist, features_by_shard in zip(self._post_input_dists, features):
                await_list.append(post_dist(features_by_shard))
            return PostInputKJTListAwaitable(await_list)

    def _create_post_input_dist(
        self,
    ) -> None:
        for sharding in self._embedding_shardings:
            if hasattr(sharding, "create_post_input_dist"):
                self._post_input_dists.append(sharding.create_post_input_dist())
            else:
                self._post_input_dists.append(EMPTY_POST_INPUT_DIST)

    def compute_swap_info_async(
        self, sparse_features_after_dist: KJTList
    ) -> AsyncSwapInfo:
        if isinstance(sparse_features_after_dist[0], KeyedJaggedTensorWithLookHelper):
            return self._embcache_mgr.compute_swap_info_async(
                sparse_features_after_dist[0]._unique_ids,
                sparse_features_after_dist[0]._unique_offset_list_single,
            )
        else:
            return self._embcache_mgr.compute_swap_info_async(
                sparse_features_after_dist[0].values(),
                sparse_features_after_dist[0].offset_per_key(),
            )

    def host_embedding_update_async(
        self,
        swap_info: SwapInfo,
        swapout_embs: torch.Tensor,
        swapout_optims: torch.Tensor,
    ) -> None:
        return self._embcache_mgr.embedding_update_async(
            swap_info, swapout_embs, swapout_optims
        )

    def record_host_emb_update_times(self):
        self._embcache_mgr.record_embedding_update_times()

    def host_embedding_lookup_async(self, swap_info: SwapInfo) -> AsyncSwapinTensor:
        return self._embcache_mgr.embedding_lookup_async(swap_info)

    def get_batched_embedding_kernels(
        self,
    ) -> List[List[SplitTableBatchedEmbeddingBagsCodegen]]:
        batched_embedding_kernels = []
        for lookup in self._lookups:
            modules = []
            for emb_module in lookup._emb_modules:
                modules.append(emb_module._emb_module)
            batched_embedding_kernels.append(modules)
        return batched_embedding_kernels

    def _create_embcache_mgr(self, need_accumulate_offset: bool) -> EmbcacheManager:
        emb_configs = []
        for _, sharding_infos in self.sharding_type_to_sharding_infos.items():
            for sharding_info in sharding_infos:
                embedding_config = sharding_info.embedding_config
                emb_original_config = self._table_name_to_config[embedding_config.name]
                cpp_initializer_type = getattr(CppInitType, emb_original_config.initializer_type.name)
                optim_num = 0
                if (
                    sharding_info.fused_params["optimizer"]
                    == EmbOptimType.EXACT_ADAGRAD
                ):
                    optim_num = 1
                elif sharding_info.fused_params["optimizer"] == EmbOptimType.ADAM:
                    optim_num = 2
                else:
                    raise NotImplementedError(
                        f"Getting optimizer states is not supported "
                        f"for {sharding_info.fused_params['optimizer']}"
                    )

                local_shard_size = 0
                rank_str = os.environ.get("LOCAL_RANK", "0")
                if not rank_str.isdigit():
                    raise ValueError(
                        f"Param error, LOCAL_RANK must be a number but got {rank_str}."
                    )
                rank = int(rank_str)
                for shard_metadata in sharding_info.param_sharding.sharding_spec.shards:
                    # 解析 placement 字符串以获取 rank
                    placement_str = str(shard_metadata.placement)
                    # 尝试提取 rank
                    rank_part = placement_str.split("/")[0]  # 获取 "rank:N" 部分
                    shard_rank = int(rank_part.split(":")[1])  # 获取 N
                    if shard_rank == rank:
                        # 找到了当前 rank 对应的 shard
                        local_shard_size = shard_metadata.shard_sizes[
                            0
                        ]  # 获取第一个维度的大小
                        break

                emb_configs.append(
                    EmbConfig(
                        table_name=embedding_config.name, initializer_type=cpp_initializer_type,
                        emb_dim=embedding_config.embedding_dim,
                        optim_num=optim_num,
                        cache_size=local_shard_size,
                        weight_init_min=embedding_config.get_weight_init_min(),
                        weight_init_max=embedding_config.get_weight_init_max(),
                        weight_init_mean=emb_original_config.weight_init_mean,
                        weight_init_stddev=emb_original_config.weight_init_stddev,
                        num_features=emb_original_config.num_features(),
                    )
                )
        return EmbcacheManager(emb_configs, need_accumulate_offset)

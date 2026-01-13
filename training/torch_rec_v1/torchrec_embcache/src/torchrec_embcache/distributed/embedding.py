#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Dict,
    List,
    MutableMapping,
    Optional,
    Union as TypeUnion,
    Type,
)
import logging

import numpy as np
import torch_npu
import torch
from torch import distributed as dist, nn, Tensor

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)

from hybrid_torchrec.constants import MAX_CACHINE_MEM_SIZE
from hybrid_torchrec.modules.ids_process import IdsMapper
from hybrid_torchrec.modules.ids_process import HashMapBase
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    EMPTY_POST_INPUT_DIST,
    PostInputKJTListAwaitable,
)
from hybrid_torchrec.distributed.sharding.sequence_sharding import (
    HybridSequenceShardingContext,
)
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
)
from hybrid_torchrec.utils import check
from torchrec_embcache.distributed.configs import (
    EmbCacheEmbeddingConfig,
    check_multi_hot_sizes,
    check_create_table_params
)
from torchrec_embcache.distributed.sharding.rw_sequence_sharding import (
    EmbCacheRwSequenceEmbeddingSharding,
)
from torchrec_embcache.sparse.jagged_tensor_with_timestamp import (
    KeyedJaggedTensorWithTimestamp,
)
from torchrec_embcache.distributed.utils import get_embedding_optim_num
from torchrec_embcache.distributed.sharding.rw_sharding import EmbCacheRwSparseFeaturesDistAwaitable
from torchrec_embcache.embcache_pybind import (
    EmbcacheManager,
    EmbConfig,
    AdmitAndEvictConfig,
    AsyncSwapInfo,
    AsyncSwapinTensor,
    InitializerType as CppInitType,
    SwapInfo,
    restore_async,
)

from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    Out,
    QuantizedCommCodecs,
    EmbeddingModuleShardingPlan,
    ShardingEnv,
    ShardedTensor,
    ParameterSharding,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    EmbeddingShardingContext,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.sharding.dp_sequence_sharding import (
    DpSequenceEmbeddingSharding,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.distributed.model_parallel import (
    DistributedDataParallel,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor
from torchrec.distributed.embedding_types import (
    ShardingType,
    KJTList
)
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingConfig,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer
from torchrec.modules.embedding_modules import get_embedding_names_by_table
from torchrec.distributed.embedding import (
    ShardedEmbeddingCollection,
    EmbeddingCollectionContext,
    EmbeddingCollectionAwaitable,
    create_sharding_infos_by_sharding,
    pad_vbe_kjt_lengths,
    get_ec_index_dedup,
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
    enable_admit: bool = False


def create_embcache_embedding_sharding(
    sharding_config: ShardingConfig,
) -> EmbeddingSharding[
    EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
]:
    if sharding_config.sharding_type == ShardingType.ROW_WISE.value:
        return EmbCacheRwSequenceEmbeddingSharding(
            sharding_config.sharding_infos,
            sharding_config.table2hashmap,
            cpu_env=sharding_config.cpu_env,
            cpu_device=sharding_config.cpu_device,
            npu_device=sharding_config.npu_device,
            npu_env=sharding_config.npu_env,
            qcomm_codecs_registry=sharding_config.qcomm_codecs_registry,
            enable_admit=sharding_config.enable_admit,
        )
    else:
        raise ValueError(f"Sharding type not supported {sharding_config.sharding_type}")


class EmbCacheHashTable(torch.nn.Module):
    def __init__(self, config: EmbeddingConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.ids2slot_dict = IdsMapper(self.config.num_embeddings, only_device_memory=False)
        self.vector_table = nn.Embedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            device=device,
            dtype=torch.float32 if config.data_type == DataType.FP32 else torch.float16,
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
        index_of_ids, _, _ = self.ids2slot_dict(ids_host, high_precison=True)
        index_of_ids = index_of_ids.to(raw_device)
        values = self.vector_table(index_of_ids, offsets)
        return values


class EmbCacheEmbeddingCollection(EmbeddingCollection):
    def __init__(
        self,
        tables: List[EmbCacheEmbeddingConfig | EmbeddingConfig],
        world_size: int,
        batch_size: int,
        multi_hot_sizes: List[int],
        need_indices: bool = False,
        need_accumulate_offset: bool = True,
        device: Optional[torch.device] = None,
        embedding_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
    ) -> None:
        check_multi_hot_sizes(multi_hot_sizes, tables)
        check(all([isinstance(item, (EmbCacheEmbeddingConfig, EmbeddingConfig)) for item in tables]),
              "all element type in 'tables' must be EmbCacheEmbeddingConfig or EmbeddingConfig object")
        check_create_table_params(batch_size, embedding_optimizer_cls, multi_hot_sizes, tables, world_size)
        
        super().__init__(tables, device, need_indices)
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        self.need_accumulate_offset: bool = need_accumulate_offset
        self._convert_2_cache_embedding_config(tables)
        self._embedding_configs = tables
        self._embedding_dim: int = -1
        self._need_indices: bool = need_indices
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )
        self._optim_num = get_embedding_optim_num(embedding_optimizer_cls)
        logger.debug("======  _optim_num: %d", self._optim_num)

        evict_step_intervals = set(
            config.admit_and_evict_config.evict_step_interval for config in tables
        )
        if len(evict_step_intervals) > 1:
            raise ValueError("all table must have the same evict_step_interval param.")

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
        # 开启准入时会预留offset 0位置，手动给计算后的表大小加1
        for i, _ in enumerate(cache_num_embeddings):
            if self._embedding_configs[i].admit_and_evict_config.is_feature_admit_enabled():
                cache_num_embeddings[i] += 1

        logger.debug("table_num_embeddings: %s ", cache_num_embeddings)
        table_names = set()
        for index, config in enumerate(tables):
            # Use the cache_num_embeddings to embedding config
            config.num_embeddings = int(cache_num_embeddings[index])
            config.cache = int(cache_num_embeddings[index])

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
            self.embeddings[config.name] = EmbCacheHashTable(
                config=config, device=self.device
            )
            if config.init_fn is not None:
                config.init_fn(self.embeddings[config.name].weight)

            if not config.feature_names:
                config.feature_names = [config.name]

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(tables)
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]

    @staticmethod
    def _convert_2_cache_embedding_config(tables: List[EmbCacheEmbeddingConfig | EmbeddingConfig]):
        for i, ori_config in enumerate(tables):
            if isinstance(ori_config, EmbCacheEmbeddingConfig):
                continue

            emb_cache_config = EmbCacheEmbeddingConfig(embedding_dim=ori_config.embedding_dim,
                                                       num_embeddings=ori_config.num_embeddings)
            emb_cache_config.__dict__.update(ori_config.__dict__)
            tables[i] = emb_cache_config

    def _calculate_caches(
        self,
        tables: List[EmbeddingConfig],
        max_device_mem_for_vectors: int,
        multi_hot_sizes: List[int],
        batch_size: int,
        world_size: int,
    ) -> List[int]:
        embedding_dims = [embedding_config.embedding_dim for embedding_config in tables]
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


class EmbCacheShardedEmbeddingCollection(ShardedEmbeddingCollection):
    def __init__(
        self,
        module: EmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        npu_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        npu_device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        cpu_device: Optional[torch.device] = None,
        cpu_env: Optional[ShardingEnv] = None,
        use_index_dedup: bool = False,
        module_fqn: Optional[str] = None,
    ) -> None:
        super(EmbCacheShardedEmbeddingCollection.__bases__[0], self).__init__(
            qcomm_codecs_registry=qcomm_codecs_registry
        )
        # re-init the followings (because we will create self._embedding_shardings,
        # and following variables might depend on it)
        self._module_fqn = module_fqn
        self._embedding_configs: List[EmbCacheEmbeddingConfig] = (
            module.embedding_configs()
        )
        self._table_names: List[str] = [
            config.name 
            for config in self._embedding_configs
        ]
        self._table_name_to_config: Dict[str, EmbCacheEmbeddingConfig] = {
            config.name: config 
            for config in self._embedding_configs
        }
        self.module_sharding_plan: EmbeddingModuleShardingPlan = cast(
            EmbeddingModuleShardingPlan,
            {
                table_name: parameter_sharding
                for table_name, parameter_sharding in table_name_to_parameter_sharding.items()
                if table_name in self._table_names
            },
        )
        self._output_dtensor: bool = (
            fused_params.get("output_dtensor", False) if fused_params else False
        )
        self._env = npu_env
        self._use_index_dedup: bool = use_index_dedup or get_ec_index_dedup()

        self.sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module,
            table_name_to_parameter_sharding,
            fused_params,
        )
        self.table2hashmap: Dict[str, HashMapBase] = self.create_table2hashmap(module)
        self._enable_admit = any(
            hasattr(emb_config, "admit_and_evict_config")
            and emb_config.admit_and_evict_config.is_feature_admit_enabled()
            for emb_config in self._embedding_configs
        )

        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
            ],
        ] = {
            sharding_type: create_embcache_embedding_sharding(
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
                    enable_admit=self._enable_admit,
                )
            )
            for sharding_type, embedding_configs in self.sharding_type_to_sharding_infos.items()
        }

        self._device = npu_device
        self._cpu_device = cpu_device
        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._create_lookups()
        self._output_dists: List[nn.Module] = []
        self._create_output_dist()

        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        self._has_uninitialized_input_dist: bool = True

        optims = []
        for lookup in self._lookups:
            for _, m in lookup.named_modules():
                if isinstance(m, FusedOptimizerModule):
                    # modify param keys to match EmbeddingCollection
                    params: MutableMapping[
                        str, TypeUnion[torch.Tensor, ShardedTensor]
                    ] = {}
                    for param_key, weight in m.fused_optimizer.params.items():
                        params["embeddings." + param_key] = weight
                    m.fused_optimizer.params = params
                    optims.append(("", m.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

        self._embedding_dim: int = module.embedding_dim()
        self._embedding_names_per_sharding: List[List[str]] = []

        for sharding in self._sharding_type_to_sharding.values():
            self._embedding_names_per_sharding.append(sharding.embedding_names())
        self._local_embedding_dim: int = self._embedding_dim
        self._features_to_permute_indices: Dict[str, List[int]] = {}

        self._need_indices: bool = module.need_indices()
        self._inverse_indices_permute_per_sharding: Optional[List[torch.Tensor]] = None

        for index, (sharding, lookup) in enumerate(
            zip(
                self._sharding_type_to_sharding.values(),
                self._lookups,
            )
        ):
            if isinstance(sharding, DpSequenceEmbeddingSharding):
                self._lookups[index] = DistributedDataParallel(
                    module=lookup,
                    device_ids=(
                        [self._device]
                        if self._device is not None and self._device.type in {"cuda", "mtia"}
                        else None
                    ),
                    process_group=npu_env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    static_graph=True,
                )

        if npu_env.process_group and dist.get_backend(npu_env.process_group) != "fake":
            self._initialize_torch_state()

        if module.device not in ["meta", "cpu"] and module.device.type not in ["meta", "cpu"]:
            self.load_state_dict(module.state_dict(), strict=False)

        self._memcpy_stream: Optional[torch_npu.npu.streams.Stream] = (
            torch_npu.npu.Stream(priority=-1)
        )
        self._embcache_mgr = self._create_embcache_mgr(module.need_accumulate_offset)
        self._set_cache_mgr_for_ids_mapper()
        self._has_uninitialized_post_input_dist: bool = True
        self._post_input_dists: List[nn.Module] = []

    @property
    def embcache_mgr(self):
        return self._embcache_mgr
    
    @property
    def config_list(self):
        return self._embedding_configs

    def _set_cache_mgr_for_ids_mapper(self):
        for ids_mapper in self.table2hashmap.values():
            ids_mapper.set_cache_mgr(self._embcache_mgr)

    def compute_and_output_dist(
        self, ctx: EmbeddingCollectionContext, input_feature: KJTList
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for lookup, out_dist, sharding_ctx, features, sharding_type in zip(
            self._lookups,
            self._output_dists,
            ctx.sharding_contexts,
            input_feature,
            self._sharding_type_to_sharding,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )

            lookup_ret = lookup(features)
            if self._has_enable_feature_admit():
                lookup_ret = self._reset_embedding_for_not_admitted_ids(
                    features, lookup, lookup_ret
                )
            embedding_dim = self._embedding_dim_for_sharding_type(sharding_type)

            awaitables.append(
                out_dist(lookup_ret.view(-1, embedding_dim), sharding_ctx)
            )

            features_before_all2all_per_sharding.append(
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
            ctx=ctx,
        )

    def _reset_embedding_for_not_admitted_ids(
        self, features: KeyedJaggedTensor, lookup, lookup_ret: Tensor
    ) -> Tensor:
        emb_dims: List[int] = [
            emb_table.embedding_dim
            for emb_table in lookup.grouped_configs[0].embedding_tables
        ]
        emb_names: List[str] = [
            emb_table.name 
            for emb_table in lookup.grouped_configs[0].embedding_tables
        ]
        emb_not_admitted_default_value: List[float] = []
        for emb_name in emb_names:
            emb_not_admitted_default_value.append(
                self._table_name_to_config[
                    emb_name
                ].admit_and_evict_config.not_admitted_default_value
            )
        features_offset_per_key: List[int] = features.offset_per_key()
        feature_key_num = len(features_offset_per_key) - 1
        table_num = len(emb_dims)
        if feature_key_num != table_num:
            raise RuntimeError(
                f"Admit current only support same number of feature key and table,"
                f" but got feature_key_num:{feature_key_num}, table_num:{table_num}"
            )

        lookup_ret_by_feature: List[Tensor] = []
        lookup_ret_offset = 0
        for i in range(feature_key_num):
            lookup_ret_size = emb_dims[i] * (
                features_offset_per_key[i + 1] - features_offset_per_key[i]
            )
            lookup_ret_by_feature.append(
                lookup_ret[lookup_ret_offset: lookup_ret_offset + lookup_ret_size]
            )
            lookup_ret_offset += lookup_ret_size
        for i in range(feature_key_num):
            ids_offset_tensor = features.values()[
                features_offset_per_key[i]: features_offset_per_key[i + 1]
            ]
            feature_key_offset_musk = ids_offset_tensor == 0
            true_value_num = torch.sum(feature_key_offset_musk).item()
            lookup_ret_with_default = lookup_ret_by_feature[i].view(-1, emb_dims[i])
            if true_value_num > 0:
                default_emb = torch.full(
                    (emb_dims[i],),
                    emb_not_admitted_default_value[i],
                    dtype=lookup_ret.dtype,
                    device=lookup_ret.device,
                )
                lookup_ret_with_default[feature_key_offset_musk] = default_emb
            lookup_ret_by_feature[i] = lookup_ret_with_default.view(-1)
        lookup_ret = torch.cat(lookup_ret_by_feature, dim=-1)
        return lookup_ret

    def _has_enable_feature_admit(self):
        return any(
            emb_config.admit_and_evict_config.admit_threshold != -1
            for emb_config in self._embedding_configs
        )

    def _embedding_dim_for_sharding_type(self, sharding_type: str) -> int:
        return (
            self._local_embedding_dim
            if sharding_type == ShardingType.COLUMN_WISE.value
            else self._embedding_dim
        )

    def compute_swap_info_async(
        self, sparse_features_after_dist: KJTList
    ) -> AsyncSwapInfo:
        if isinstance(sparse_features_after_dist[0], KeyedJaggedTensorWithLookHelper):
            return self._embcache_mgr.compute_swap_info_async(
                sparse_features_after_dist[0].unique_ids,
                sparse_features_after_dist[0].unique_offset_host,
            )
        else:
            return self._embcache_mgr.compute_swap_info_async(
                sparse_features_after_dist[0].values(),
                sparse_features_after_dist[0].offset_per_key(),
            )
 
    def record_batch_keys_async(
        self, sparse_features_after_dist: KJTList
    ) -> None:
        if isinstance(sparse_features_after_dist[0], KeyedJaggedTensorWithLookHelper):
            return self._embcache_mgr.record_batch_keys_async(
                sparse_features_after_dist[0]._unique_ids,
                sparse_features_after_dist[0]._unique_offset_list_single,
            )
        else:
            return self._embcache_mgr.record_batch_keys_async(
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

    def host_embedding_evict(self) -> None:
        self._embcache_mgr.evict_features()

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

    @staticmethod
    def _build_admit_and_evict_config(cache_ec_config: EmbCacheEmbeddingConfig):
        aaec_py = cache_ec_config.admit_and_evict_config
        logging.info("admit_and_evict_config info:%s", aaec_py)
        aaec = AdmitAndEvictConfig(
            admit_threshold=aaec_py.admit_threshold,
            not_admitted_default_value=aaec_py.not_admitted_default_value,
            evict_threshold=aaec_py.evict_threshold,
            evict_step_interval=aaec_py.evict_step_interval,
        )
        return aaec

    def _create_embcache_mgr(self, need_accumulate_offset: bool) -> EmbcacheManager:
        emb_configs = []
        for _, sharding_infos in self.sharding_type_to_sharding_infos.items():
            for sharding_info in sharding_infos:
                embedding_config = sharding_info.embedding_config
                emb_original_config = self._table_name_to_config[embedding_config.name]
                cpp_initializer_type = getattr(CppInitType, emb_original_config.initializer_type.name)
                if sharding_info.fused_params["optimizer"] == EmbOptimType.EXACT_ADAGRAD:
                    optim_num = 1
                elif sharding_info.fused_params["optimizer"] == EmbOptimType.ADAM:
                    optim_num = 2
                else:
                    raise NotImplementedError(
                        f"Getting optimizer states is not supported for {sharding_info.fused_params['optimizer']}"
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
                        local_shard_size = shard_metadata.shard_sizes[0]  # 获取第一个维度的大小
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
                        admit_and_evict_config=self._build_admit_and_evict_config(emb_original_config),
                        num_features=emb_original_config.num_features(),
                        is_incremental=emb_original_config.is_incremental
                    )
                )
        return EmbcacheManager(emb_configs, need_accumulate_offset)

    def create_table2hashmap(self, module):
        table2hashmap = {}
        for name in module.embeddings.keys():
            hashmap = module.embeddings[name].ids2slot_dict
            table2hashmap[name] = hashmap
        return table2hashmap

    def post_input_dist(
        self, ctx: EmbeddingCollectionContext, features: KJTList
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
        for sharding_key in self._sharding_type_to_sharding:
            sharding = self._sharding_type_to_sharding[sharding_key]
            if hasattr(sharding, "create_post_input_dist"):
                self._post_input_dists.append(sharding.create_post_input_dist())
            else:
                self._post_input_dists.append(EMPTY_POST_INPUT_DIST)

    def input_dist(
        self,
        ctx: EmbeddingCollectionAwaitable,
        features: KeyedJaggedTensorWithTimestamp,
    ) -> Awaitable[Awaitable[KJTList]]:
        """
        feature的顺序按照Dict[str, list[]]  shardType -> [t.feature_name for t in tables]
        """
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys())
            self._has_uninitialized_input_dist = False

        with torch.no_grad():
            unpadded_features = None
            if features.variable_stride_per_key():
                unpadded_features = features
                features = pad_vbe_kjt_lengths(unpadded_features)

            if self._features_order:
                if self._features_order_tensor is not None and self._features_order_tensor.device != features.device():
                    features_order_tensor = self._features_order_tensor.to(features.device()) 
                else:
                    features_order_tensor = self._features_order_tensor

                features = features.permute(
                    self._features_order,
                    features_order_tensor,
                )

            self._record_timestamp_data(features)

            features_by_shards = features.split(self._feature_splits)
            if self._use_index_dedup:
                features_by_shards = self._dedup_indices(ctx, features_by_shards)

            awaitables = []
            for input_dist, features in zip(self._input_dists, features_by_shards):
                shard_context = HybridSequenceShardingContext(
                    features_before_input_dist=features
                )
                awaitables.append(input_dist(features, shard_context))

                ctx.sharding_contexts.append(shard_context)
            if unpadded_features is not None:
                self._compute_sequence_vbe_context(ctx, unpadded_features)
        return KJTListSplitsAwaitable(awaitables, ctx)

    def _record_timestamp_data(self, features: KeyedJaggedTensorWithTimestamp):
        # 记录淘汰要用的timestamp数据
        is_evict_enabled = any(
            emb_config.admit_and_evict_config.is_feature_evict_enabled()
            for emb_config in self._embedding_configs
        )
        if is_evict_enabled and hasattr(features, "_timestamps"):
            self._embcache_mgr.record_timestamp(
                features.values(), features.offset_per_key(), features.timestamps
            )

    def forward(self, *args, **kwargs) -> LazyAwaitable[Out]:
        """
        Executes the input dist, compute, and output dist steps.

        Args:
            *args: input.
            **kwargs: keyword arguments.

        Returns:
            LazyAwaitable[Out]: awaitable of output from output dist.
        """
        ctx = self.create_context()
        kjt_list_split_awaitable = self.input_dist(ctx, *args, **kwargs)
        for ind, awaitable in enumerate(kjt_list_split_awaitable.awaitables):
            if isinstance(awaitable, EmbCacheRwSparseFeaturesDistAwaitable):
                kjt_list_split_awaitable.awaitables[ind] = awaitable.wait()

        dist_input = kjt_list_split_awaitable.wait().wait()
        post_dist_input = self.post_input_dist(ctx, dist_input).wait()
        swap_info = self.compute_swap_info_async(post_dist_input).get()

        _stb_eb_codegen = self.get_batched_embedding_kernels()[0][0]
        # swapout and host_update
        swapout_offs = swap_info.swapout_offs.to(self._device, non_blocking=True)
        swapout_embs = _stb_eb_codegen.gather_embs(swapout_offs).to(self._cpu_device)
        swapout_optims = []
        for momentum in _stb_eb_codegen.gather_momentum(swapout_offs):
            swapout_optims.append(momentum.to(self._cpu_device))
        self.host_embedding_update_async(swap_info, swapout_embs, swapout_optims).get()

        # host_lookup and swapin
        swapin_tensor = self.host_embedding_lookup_async(swap_info).get()
        swap_embs = swapin_tensor.swapin_embs.to(self._device, non_blocking=True)
        swapin_optims = []
        for optim in swapin_tensor.swapin_optims:
            swapin_optims.append(optim.to(self._device, non_blocking=True))
        swapin_offs = swap_info.swapin_offs.to(self._device, non_blocking=True)
        _stb_eb_codegen.scatter_update_embs(swapin_offs, swap_embs)
        _stb_eb_codegen.scatter_update_momentum(swapin_offs, swapin_optims)

        # restore and key2offset
        restore_async(
            swap_info.batch_offs,
            post_dist_input[0].unique_inverse,
            post_dist_input[0].unique_offset,
            post_dist_input[0].offset_per_key(),
            post_dist_input[0].hash_indices,
        ).get()
        post_dist_input[0].unique_indices = swap_info.batch_offs
        post_dist_input[0] = post_dist_input[0].to(self._device, non_blocking=True)

        for sharding_ctx in ctx.sharding_contexts:
            sharding_ctx.sparse_features_recat = None
            # 因查表前卸载到cpu,查表时要to device.
            if sharding_ctx.unbucketize_permute_tensor is not None:
                sharding_ctx.unbucketize_permute_tensor = (
                    sharding_ctx.unbucketize_permute_tensor.to(self._device, non_blocking=True)
                )
            sharding_ctx.features_before_input_dist = (
                sharding_ctx.features_before_input_dist.to(self._device, non_blocking=True)
            )
        
        return self.compute_and_output_dist(ctx, post_dist_input)

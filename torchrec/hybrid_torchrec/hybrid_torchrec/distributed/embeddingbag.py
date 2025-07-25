#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, cast, Dict, List, Mapping, Optional, Type, Union, TypeVar, Tuple

import torch
from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
from torch import distributed as dist, nn
from torch.autograd.profiler import record_function
from torch.distributed._tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from hybrid_torchrec.distributed.embedding_types import kjt_list_to_device
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import (
    HybridRwPooledEmbeddingSharding,
)
from hybrid_torchrec.distributed.sharding.hybrid_tw_sharding import (
    HybridTwPooledEmbeddingSharding,
)
from hybrid_torchrec.distributed.sharding.post_input_dist import EMPTY_POST_INPUT_DIST, PostInputKJTListAwaitable
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    replace_placement_with_meta_device,
    create_sharding_infos_by_sharding,
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionAwaitable,
    VariableBatchEmbeddingBagCollectionAwaitable,
)
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.types import (
    Awaitable,
    EmbeddingModuleShardingPlan,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
)
from torchrec.optim.fused import EmptyFusedOptimizer, FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.sparse.jagged_tensor import _to_offsets, KeyedJaggedTensor, KeyedTensor

Out = TypeVar("Out")


def device_is_in(device, check_device: list[str]):
    if isinstance(device, torch.device):
        return device.type in check_device
    else:
        return device in check_device


def _pin_and_move(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (
        tensor
        if device.type == "cpu"
        else tensor.pin_memory().to(device=device, non_blocking=True)
    )


class HybridShardedEmbeddingBagCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        KeyedTensor,
        EmbeddingBagCollectionContext,
    ],
    FusedOptimizerModule,
):
    """
    Sharded implementation of EmbeddingBagCollection.
    This is part of the public API to allow for manual data dist pipelining.
    """
    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        host_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        module_fqn: Optional[str] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._module_fqn = module_fqn
        self._embedding_bag_configs: List[EmbeddingBagConfig] = (
            module.embedding_bag_configs()
        )
        self._table_names: List[str] = []
        self._pooling_type_to_rs_features: Dict[str, List[str]] = defaultdict(list)
        self._table_name_to_config: Dict[str, EmbeddingBagConfig] = {}

        self._init_sharding_plan(table_name_to_parameter_sharding)
        self._env = env
        self._host_env = host_env
        # output parameters as DTensor in state dict
        self._output_dtensor: bool = (fused_params.get("output_dtensor", False) if fused_params else False)
        self._init_embedding_shardings(device, fused_params, module, table_name_to_parameter_sharding)

        self._is_weighted: bool = module.is_weighted()
        self._device = device
        self._input_dists: List[nn.Module] = []
        self._post_input_dists: List[nn.Module] = []
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
        # to support mean pooling callback hook
        self._has_mean_pooling_callback: bool = (
            PoolingType.MEAN.value in self._pooling_type_to_rs_features
        )
        self._dim_per_key: Optional[torch.Tensor] = None
        self._kjt_key_indices: Dict[str, int] = {}
        self._kjt_inverse_order: Optional[torch.Tensor] = None
        self._kt_key_ordering: Optional[torch.Tensor] = None
        # to support the FP16 hook
        self._create_output_dist()
        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_post_input_dist: bool = True
        self._has_features_permute: bool = True
        # Get all fused optimizers and combine them.
        self._init_optim()
        self._init_lookups(device, env)
        if env.process_group and dist.get_backend(env.process_group) != "fake":
            self._initialize_torch_state()
        if not device_is_in(module.device, ["meta", "cpu"]):
            self.load_state_dict(module.state_dict(), strict=False)

    def _init_lookups(self, device, env):
        for i, (sharding, lookup) in enumerate(
            zip(self._embedding_shardings, self._lookups)
        ):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                self._lookups[i] = DistributedDataParallel(
                    module=lookup,
                    device_ids=(
                        [device]
                        if self._device and (self._device.type in {"cuda", "mtia"})
                        else None
                    ),
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    static_graph=True,
                )

    def _init_optim(self):
        optims = []
        for lookup in self._lookups:
            for _, tbe_module in lookup.named_modules():
                if isinstance(tbe_module, FusedOptimizerModule):
                    # Modify param keys to match EmbeddingBagCollection
                    params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in tbe_module.fused_optimizer.params.items():
                        params["embedding_bags." + param_key] = weight
                    tbe_module.fused_optimizer.params = params
                    optims.append(("", tbe_module.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

    def _init_embedding_shardings(self, device, fused_params, module, table_name_to_parameter_sharding):
        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(module, table_name_to_parameter_sharding,
                                                                            "embedding_bags.", fused_params, )
        self._embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = [
            self.create_hybrid_embedding_bag_sharding(
                embedding_configs,
                self._env,
                self._host_env,
                device,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for embedding_configs in sharding_type_to_sharding_infos.values()
        ]

    def _init_sharding_plan(self, table_name_to_parameter_sharding):
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

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    @staticmethod
    def _pre_state_dict_hook(
        self: "HybridShardedEmbeddingBagCollection",
        prefix: str = "",
        keep_vars: bool = False,
    ) -> None:
        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            lookup.flush()

    @staticmethod
    def _pre_load_state_dict_hook(
        self: "HybridShardedEmbeddingBagCollection",
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        Modify the destination state_dict for model parallel
        to transform from ShardedTensors into tensors
        """
        for table_name in self._model_parallel_name_to_local_shards.keys():
            key = f"{prefix}embedding_bags.{table_name}.weight"
            # gather model shards from both DTensor and ShardedTensor maps
            model_shards_sharded_tensor = self._model_parallel_name_to_local_shards[
                table_name
            ]
            model_shards_dtensor = self._model_parallel_name_to_shards_wrapper[
                table_name
            ]
            # If state_dict[key] is already a ShardedTensor, use its local shards
            if isinstance(state_dict[key], ShardedTensor):
                HybridShardedEmbeddingBagCollection._pre_load_state_dict_with_shared_tensor(key, state_dict)
            elif isinstance(state_dict[key], DTensor):
                HybridShardedEmbeddingBagCollection._pre_load_state_dict_with_dtensor(key, state_dict)
            elif isinstance(state_dict[key], torch.Tensor):
                HybridShardedEmbeddingBagCollection._pre_load_state_dict_with_torch_tensor(key, model_shards_dtensor,
                                                                                           model_shards_sharded_tensor,
                                                                                           state_dict)
            else:
                raise RuntimeError(
                    f"Unexpected state_dict key type {type(state_dict[key])} found for {key}"
                )
        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            lookup.purge()

    @staticmethod
    def _pre_load_state_dict_with_torch_tensor(key, model_shards_dtensor, model_shards_sharded_tensor, state_dict):
        local_shards = []
        if model_shards_sharded_tensor:
            # splice according to sharded tensor metadata
            for shard in model_shards_sharded_tensor:
                # Extract shard size and offsets for splicing
                shard_size = shard.metadata.shard_sizes
                shard_offset = shard.metadata.shard_offsets
                # Prepare tensor by splicing and placing on appropriate device
                spliced_tensor = state_dict[key][
                                 shard_offset[0]: shard_offset[0] + shard_size[0],
                                 shard_offset[1]: shard_offset[1] + shard_size[1],
                                 ]
                # Append spliced tensor into local shards
                local_shards.append(spliced_tensor)
        elif model_shards_dtensor:
            # splice according to dtensor metadata
            for tensor, shard_offset in zip(
                    model_shards_dtensor["local_tensors"],
                    model_shards_dtensor["local_offsets"],
            ):
                shard_size = tensor.size()
                spliced_tensor = state_dict[key][
                                 shard_offset[0]: shard_offset[0] + shard_size[0],
                                 shard_offset[1]: shard_offset[1] + shard_size[1],
                                 ]
                local_shards.append(spliced_tensor)
        state_dict[key] = (
            torch.empty(0)
            if not local_shards
            else torch.cat(local_shards, dim=0)
        )

    @staticmethod
    def _pre_load_state_dict_with_dtensor(key, state_dict):
        shards_wrapper = state_dict[key].to_local()
        local_shards = shards_wrapper.local_shards()
        if len(local_shards) == 0:
            state_dict[key] = torch.empty(0)
        else:
            dim = shards_wrapper.local_sizes()[0][1]
            # CW multiple shards are merged
            if len(local_shards) > 1:
                state_dict[key] = torch.cat(
                    [s.view(-1) for s in local_shards], dim=0
                ).view(-1, dim)
            else:
                state_dict[key] = local_shards[0].view(-1, dim)

    @staticmethod
    def _pre_load_state_dict_with_shared_tensor(key, state_dict):
        local_shards = state_dict[key].local_shards()
        if len(local_shards) == 0:
            state_dict[key] = torch.empty(0)
        else:
            dim = state_dict[key].metadata().shards_metadata[0].shard_sizes[1]
            # CW multiple shards are merged
            if len(local_shards) > 1:
                state_dict[key] = torch.cat(
                    [s.tensor.view(-1) for s in local_shards], dim=0
                ).view(-1, dim)
            else:
                state_dict[key] = local_shards[0].tensor.view(-1, dim)

    def create_context(self) -> EmbeddingBagCollectionContext:
        return EmbeddingBagCollectionContext()

    def create_hybrid_embedding_bag_sharding(
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
        if device is not None and device.type == "meta":
            replace_placement_with_meta_device(sharding_infos)
        if sharding_type == ShardingType.TABLE_WISE.value:
            return HybridTwPooledEmbeddingSharding(
                sharding_infos,
                env,
                host_env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            return HybridRwPooledEmbeddingSharding(
                sharding_infos,
                env,
                self._host_env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            raise ValueError(
                f"Unsupported sharding type: {sharding_type}. "
                f"Valid options: {', '.join(ShardingType.get_values())}"
            )

    def forward(self, *input_tensor, **kwargs) -> LazyAwaitable[Out]:
        if len(input_tensor) < 1:
            raise ValueError(f"input must be kjt in 0, but got {input_tensor}")
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input_tensor, **kwargs).wait().wait()
        dist_post_input = self.post_input_dist(ctx, dist_input).wait()
        dist_post_input = kjt_list_to_device(dist_post_input, self._device)
        return self.compute_and_output_dist(ctx, dist_post_input)

    def reset_parameters(self) -> None:
        if self._device and self._device.type == "meta":
            return

        # Initialize embedding bags weights with init_fn
        for table_config in self._embedding_bag_configs:
            if self.module_sharding_plan[table_config.name].compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE.value,
            }:
                continue
            if table_config.init_fn is None:
                raise ValueError(
                    f"table_config init_fn is None, table name {table_config.name}"
                )
            param = self.embedding_bags[f"{table_config.name}"].weight
            table_config.init_fn(param)

            sharding_type = self.module_sharding_plan[table_config.name].sharding_type
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                pg = self._env.process_group
                with torch.no_grad():
                    dist.broadcast(param.data, src=0, group=pg)

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
                    dim_per_key=self._dim_per_key_cpu,  # pyre-ignore[6]
                    embedding_names=self._embedding_names,
                    embedding_dims=self._embedding_dims,
                    variable_batch_per_feature=ctx.variable_batch_per_feature,
                    kjt_inverse_order=self._kjt_inverse_order,  # pyre-ignore[6]
                    kjt_key_indices=self._kjt_key_indices,
                    kt_key_ordering=self._kt_key_ordering_cpu,  # pyre-ignore[6]
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

    def compute(
        self,
        ctx: EmbeddingBagCollectionAwaitable,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        return [lookup(features) for lookup, features in zip(self._lookups, dist_input)]

    def output_dist(
        self,
        ctx: EmbeddingBagCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:
        batch_size_per_feature_pre_a2a = []
        awaitables = []
        for out_dist, sharding_ctx, embeddings_list in zip(
            self._output_dists,
            ctx.sharding_contexts,
            output,
        ):
            awaitables.append(out_dist(embeddings_list, sharding_ctx))
            if sharding_ctx:
                batch_size_per_feature_pre_a2a.extend(
                    sharding_ctx.batch_size_per_feature_pre_a2a
                )

        if ctx.variable_batch_per_feature:
            if ctx.inverse_indices is None:
                raise (
                    "inverse indices must be provided from KJT if using variable batch size per feature."
                )
            awaitable = VariableBatchEmbeddingBagCollectionAwaitable(
                awaitables=awaitables,
                inverse_indices=ctx.inverse_indices,
                inverse_indices_permute_indices=self._inverse_indices_permute_indices,
                batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
                uncombined_embedding_dims=self._uncombined_embedding_dims,
                embedding_names=self._embedding_names,
                embedding_dims=self._embedding_dims,
                permute_op=self._permute_op,
            )
        else:
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

    def compute_and_output_dist(
        self, ctx: EmbeddingBagCollectionContext, input_tensor: KJTList
    ) -> LazyAwaitable[KeyedTensor]:
        batch_size_per_feature_pre_a2a = []
        awaitables = []

        # No usage of zip for dynamo
        for lookup, output_dist, sharding_context, features in zip(self._lookups, self._output_dists,
                                                                   ctx.sharding_contexts, input_tensor):
            awaitables.append(output_dist(lookup(features), sharding_context))
            if sharding_context:
                batch_size_per_feature_pre_a2a.extend(
                    sharding_context.batch_size_per_feature_pre_a2a
                )

        if ctx.variable_batch_per_feature:
            if ctx.inverse_indices is None:
                raise TypeError(
                    "inverse indices must be provided from KJT if using variable batch size per feature."
                )
            awaitable = VariableBatchEmbeddingBagCollectionAwaitable(
                awaitables=awaitables,
                inverse_indices=ctx.inverse_indices,
                inverse_indices_permute_indices=self._inverse_indices_permute_indices,
                batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
                uncombined_embedding_dims=self._uncombined_embedding_dims,
                embedding_names=self._embedding_names,
                embedding_dims=self._embedding_dims,
                permute_op=self._permute_op,
            )
        else:
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

    def _initialize_torch_state(self) -> None:  # noqa
        """
        This provides consistency between this class and the EmbeddingBagCollection's
        nn.Module API calls (state_dict, named_modules, etc)
        """
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        for table_name in self._table_names:
            self.embedding_bags[table_name] = nn.Module()

        self._model_parallel_name_to_local_shards = OrderedDict()
        self._model_parallel_name_to_shards_wrapper = OrderedDict()
        self._model_parallel_name_to_sharded_tensor = OrderedDict()
        self._model_parallel_name_to_dtensor = OrderedDict()

        model_parallel_name_to_compute_kernel: Dict[str, str] = {}
        for (table_name, parameter_sharding,) in self.module_sharding_plan.items():
            if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            self._model_parallel_name_to_local_shards[table_name] = []
            self._model_parallel_name_to_shards_wrapper[table_name] = OrderedDict(
                [("local_tensors", []), ("local_offsets", [])]
            )
            model_parallel_name_to_compute_kernel[table_name] = (parameter_sharding.compute_kernel)

        self._name_to_table_size = {}
        for table in self._embedding_bag_configs:
            self._name_to_table_size[table.name] = (table.num_embeddings, table.embedding_dim,)

        for lookup, sharding in zip(self._lookups, self._embedding_shardings):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                # unwrap DDP
                lookup = lookup.module
            else:
                # save local_shards for transforming MP params to DTensor
                for key, v in lookup.state_dict().items():
                    table_name = key[: -len(".weight")]
                    if isinstance(v, DTensor):
                        shards_wrapper = self._model_parallel_name_to_shards_wrapper[table_name]
                        local_shards_wrapper = v._local_tensor
                        shards_wrapper["local_tensors"].extend(local_shards_wrapper.local_shards())
                        shards_wrapper["local_offsets"].extend(local_shards_wrapper.local_offsets())
                        shards_wrapper["global_size"] = v.size()
                        shards_wrapper["global_stride"] = v.stride()
                        shards_wrapper["placements"] = v.placements
                    elif isinstance(v, ShardedTensor):
                        self._model_parallel_name_to_local_shards[table_name].extend(v.local_shards())
            for (table_name, tbe_slice,) in lookup.named_parameters_by_table():
                self.embedding_bags[table_name].register_parameter("weight", tbe_slice)

        self._initialize_shards_state(model_parallel_name_to_compute_kernel)

        def _post_state_dict_hook(
            module: HybridShardedEmbeddingBagCollection,
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            # Adjust dense MP
            for (
                table_name,
                sharded_t,
            ) in module._model_parallel_name_to_sharded_tensor.items():
                destination_key = f"{prefix}embedding_bags.{table_name}.weight"
                destination[destination_key] = sharded_t
            for (
                table_name,
                d_tensor,
            ) in module._model_parallel_name_to_dtensor.items():
                destination_key = f"{prefix}embedding_bags.{table_name}.weight"
                destination[destination_key] = d_tensor

        self.register_state_dict_pre_hook(self._pre_state_dict_hook)
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )
        self.reset_parameters()

    def _initialize_shards_state(self, model_parallel_name_to_compute_kernel):
        for table_name in self._model_parallel_name_to_local_shards.keys():
            local_shards = self._model_parallel_name_to_local_shards[table_name]
            shards_wrapper_map = self._model_parallel_name_to_shards_wrapper[table_name]
            # for shards that don't exist on this rank, register with empty tensor
            if not hasattr(self.embedding_bags[table_name], "weight"):
                self.embedding_bags[table_name].register_parameter("weight", nn.Parameter(torch.empty(0)))
                if (model_parallel_name_to_compute_kernel[table_name] != EmbeddingComputeKernel.DENSE.value):
                    self.embedding_bags[table_name].weight._in_backward_optimizers = [EmptyFusedOptimizer()]
            if model_parallel_name_to_compute_kernel[table_name] in {EmbeddingComputeKernel.KEY_VALUE.value}:
                continue

            if self._output_dtensor:
                if shards_wrapper_map["local_tensors"]:
                    self._model_parallel_name_to_dtensor[table_name] = (DTensor.from_local(
                        local_tensor=LocalShardsWrapper(local_shards=shards_wrapper_map["local_tensors"],
                                                        local_offsets=shards_wrapper_map["local_offsets"], ),
                        device_mesh=self._env.device_mesh, placements=shards_wrapper_map["placements"],
                        shape=shards_wrapper_map["global_size"], stride=shards_wrapper_map["global_stride"],
                        run_check=False, ))
                else:
                    # empty shard case
                    self._model_parallel_name_to_dtensor[table_name] = (
                        DTensor.from_local(local_tensor=LocalShardsWrapper(local_shards=[], local_offsets=[], ),
                                           device_mesh=self._env.device_mesh, run_check=False, ))
            else:
                # created ShardedTensors once in init, use in post_state_dict_hook
                self._model_parallel_name_to_sharded_tensor[table_name] = (
                    ShardedTensor._init_from_local_shards(local_shards, self._name_to_table_size[table_name],
                                                          process_group=self._env.process_group, ))

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:
        feature_names: List[str] = []
        for sharding in self._embedding_shardings:
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))

        if feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(
                    self._features_order, device=torch.device("cpu"), dtype=torch.int32
                ),
                persistent=False,
            )

    def _create_post_input_dist(
        self,
    ) -> None:
        for sharding in self._embedding_shardings:
            if hasattr(sharding, "create_post_input_dist"):
                self._post_input_dists.append(sharding.create_post_input_dist())
            else:
                self._post_input_dists.append(EMPTY_POST_INPUT_DIST)

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

    def _create_lookups(
        self,
    ) -> None:
        for sharding in self._embedding_shardings:
            self._lookups.append(sharding.create_lookup())

    def _create_inverse_indices_permute_indices(
        self, inverse_indices: Optional[Tuple[List[str], torch.Tensor]]
    ) -> None:
        if inverse_indices is None:
            raise TypeError(
                "inverse indices must be provided from KJT if using variable batch size per feature."
            )
        index_per_name = {name: i for i, name in enumerate(inverse_indices[0])}
        permute_indices = [
            index_per_name[name.split("@")[0]]
            for name in self._uncombined_embedding_names
        ]
        if len(permute_indices) != len(index_per_name) or permute_indices != sorted(
            permute_indices
        ):
            self._inverse_indices_permute_indices = _pin_and_move(
                torch.tensor(permute_indices),
                inverse_indices[1].device,
            )

    def _create_output_dist(self) -> None:
        embedding_shard_metadata: List[Optional[ShardMetadata]] = []
        for sharding in self._embedding_shardings:
            self._output_dists.append(sharding.create_output_dist(device=self._device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())
            self._uncombined_embedding_names.extend(
                sharding.uncombined_embedding_names()
            )
            self._uncombined_embedding_dims.extend(sharding.uncombined_embedding_dims())
            embedding_shard_metadata.extend(sharding.embedding_shard_metadata())
        self._dim_per_key = torch.tensor(self._embedding_dims, device=self._device)
        self._dim_per_key_cpu = torch.tensor(self._embedding_dims, device="cpu")
        embedding_shard_offsets: List[int] = [
            meta.shard_offsets[1] if meta is not None else 0
            for meta in embedding_shard_metadata
        ]
        embedding_name_order: Dict[str, int] = {}
        for i, name in enumerate(self._uncombined_embedding_names):
            embedding_name_order.setdefault(name, i)

        def sort_key(input_tensor: Tuple[int, str]) -> Tuple[int, int]:
            index, name = input_tensor
            return (embedding_name_order[name], embedding_shard_offsets[index])

        permute_indices = []
        for i, _ in sorted(enumerate(self._uncombined_embedding_names), key=sort_key):
            permute_indices.append(i)

        self._permute_op: PermutePooledEmbeddings = PermutePooledEmbeddings(
            self._uncombined_embedding_dims, permute_indices, self._device
        )


class HybridEmbeddingBagCollectionSharder(BaseEmbeddingSharder[EmbeddingBagCollection]):
    """
    This implementation uses non-fused `EmbeddingBagCollection`
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
    def module_type(self) -> Type[EmbeddingBagCollection]:
        return EmbeddingBagCollection

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> HybridShardedEmbeddingBagCollection:
        return HybridShardedEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            host_env=self._host_env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            module_fqn=module_fqn,
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]
        if compute_device_type in {"cuda", "npu", "cpu"}:
            types += [
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
            ]
        return types

    def shardable_parameters(
        self, module: EmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }


@dataclass
class MeanPoolingConfig:
    lengths: torch.Tensor
    keys: List[str]
    offsets: torch.Tensor
    stride: int
    stride_per_key: List[int]
    dim_per_key: torch.Tensor
    pooling_type_to_rs_features: Dict[str, List[str]]
    embedding_names: List[str]
    embedding_dims: List[int]
    variable_batch_per_feature: bool
    kjt_inverse_order: torch.Tensor
    kjt_key_indices: Dict[str, int]
    kt_key_ordering: torch.Tensor
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None
    weights: Optional[torch.Tensor] = None


def _create_mean_pooling_divisor(config: MeanPoolingConfig) -> torch.Tensor:
    with record_function("## ebc create mean pooling callback ##"):
        batch_size = (
            none_throws(config.inverse_indices)[1].size(dim=1)
            if config.variable_batch_per_feature
            else config.stride
        )

        if config.weights is not None:
            # if we have weights, lengths is the sum of weights by offsets for feature
            config.lengths = torch.ops.fbgemm.segment_sum_csr(1, config.offsets.int(), config.weights)

        if config.variable_batch_per_feature:
            inverse_indices = none_throws(config.inverse_indices)
            device = inverse_indices[1].device
            inverse_indices_t = inverse_indices[1]
            if len(config.keys) != len(inverse_indices[0]):
                inverse_indices_t = torch.index_select(
                    inverse_indices[1], 0, config.kjt_inverse_order
                )
            offsets = _to_offsets(torch.tensor(config.stride_per_key, device=device))[
                :-1
            ].unsqueeze(-1)
            indices = (inverse_indices_t + offsets).flatten()
            config.lengths = torch.index_select(input=config.lengths, dim=0, index=indices)

        # only convert the sum pooling features to be 1 length
        for feature in config.pooling_type_to_rs_features[PoolingType.SUM.value]:
            feature_index = config.kjt_key_indices[feature]
            feature_index = feature_index * batch_size
            config.lengths[feature_index: feature_index + batch_size] = 1

        if len(config.embedding_names) != len(config.keys):
            config.lengths = torch.index_select(
                config.lengths.reshape(-1, batch_size),
                0,
                config.kt_key_ordering,
            ).reshape(-1)

        # transpose to align features with keyed tensor dim_per_key
        config.lengths = config.lengths.reshape(-1, batch_size).T  # [batch_size, num_features]
        output_size = sum(config.embedding_dims)

        divisor = torch.repeat_interleave(
            input=config.lengths,
            repeats=config.dim_per_key,
            dim=1,
            output_size=output_size,
        )
        eps = 1e-6  # used to safe guard against 0 division
        divisor = divisor + eps
        return divisor.detach()


def _apply_mean_pooling(
    keyed_tensor: KeyedTensor, divisor: torch.Tensor
) -> KeyedTensor:
    """
    Apply mean pooling to pooled embeddings in RW/TWRW sharding schemes.
    This function is applied as a callback to the awaitable
    """
    with record_function("## ebc apply mean pooling ##"):
        _keyed_tensor_value = keyed_tensor.values().clone()
        mean_pooled_values = (
            _keyed_tensor_value.to("cpu") / divisor
        )  # [batch size, num_features * embedding dim]
        return KeyedTensor(
            keys=keyed_tensor.keys(),
            values=mean_pooled_values,
            length_per_key=keyed_tensor.length_per_key(),
            key_dim=1,
        )
#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict, deque, OrderedDict
from itertools import accumulate
import logging
from typing import (
    Any,
    cast,
    Dict,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union as TypeUnion,
)

import torch
from torch import distributed as dist, nn, Tensor
from torch.autograd.profiler import record_function
from torch.distributed._tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from hybrid_torchrec.distributed.embedding_types import (
    kjt_list_to_device,
)
from hybrid_torchrec.distributed.sharding.hybrid_tw_sequence_sharding import (
    HybridTwSequenceEmbeddingSharding,
)
from hybrid_torchrec.distributed.sharding.hybrid_rw_sequence_sharding import (
    HybridRwSequenceEmbeddingSharding,
)
from hybrid_torchrec.distributed.sharding.post_input_dist import EMPTY_POST_INPUT_DIST, PostInputKJTListAwaitable
from hybrid_torchrec.distributed.sharding.sequence_sharding import HybridSequenceShardingContext

from torchrec.distributed.embedding import (
    create_sharding_infos_by_sharding,
    EmbeddingCollectionContext,
    EmbeddingCollectionAwaitable,
)
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
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.sharding.dp_sequence_sharding import DpSequenceEmbeddingSharding
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
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.modules.utils import SequenceVBEContext
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingCollection,
)
from torchrec.optim.fused import EmptyFusedOptimizer, FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.sparse.jagged_tensor import _to_offsets, KeyedJaggedTensor, KeyedTensor, JaggedTensor


Out = TypeVar("Out")

logger: logging.Logger = logging.getLogger(__name__)

EC_INDEX_DEDUP: bool = False


def get_ec_index_dedup() -> bool:
    global EC_INDEX_DEDUP
    return EC_INDEX_DEDUP


def pad_vbe_kjt_lengths(features: KeyedJaggedTensor) -> KeyedJaggedTensor:
    max_stride = max(features.stride_per_key())
    new_lengths = torch.zeros(
        max_stride * len(features.keys()),
        device=features.device(),
        dtype=features.lengths().dtype,
    )
    cum_stride = 0
    for i, stride in enumerate(features.stride_per_key()):
        new_lengths[i * max_stride:i * max_stride + stride] = features.lengths()[
            cum_stride:cum_stride + stride
        ]
        cum_stride += stride

    return KeyedJaggedTensor(
        keys=features.keys(),
        values=features.values(),
        lengths=new_lengths,
        stride=max_stride,
        length_per_key=features.length_per_key(),
        offset_per_key=features.offset_per_key(),
    )


def device_is_in(device, check_deivce: list[str]) -> bool:
    if isinstance(device, torch.device):
        return device.type in check_deivce
    else:
        return device in check_deivce


def _pin_and_move(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (
        tensor
        if device.type == "cpu"
        else tensor.pin_memory().to(device=device, non_blocking=True)
    )


class HybridShardedEmbeddingCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        KeyedTensor,
        EmbeddingCollectionContext,
    ],
    FusedOptimizerModule,
):
    """
    Sharded implementation of EmbeddingCollection.
    This is part of the public API to allow for manual data dist pipelining.
    """
    def __init__(
        self,
        module: EmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        host_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
        module_fqn: Optional[str] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._module_fqn = module_fqn
        self._embedding_configs: List[EmbeddingConfig] = (
            module.embedding_configs()
        )
        self._table_names: List[str] = [config.name for config in self._embedding_configs]

        self._table_name_to_config: Dict[str, EmbeddingConfig] = {
            config.name: config for config in self._embedding_configs
        }

        self.module_sharding_plan: EmbeddingModuleShardingPlan = cast(
            EmbeddingModuleShardingPlan,
            {
                table_name: parameter_sharding
                for table_name, parameter_sharding in table_name_to_parameter_sharding.items()
                if table_name in self._table_names
            },
        )
        self._env = env
        self._host_env = host_env
        self._use_index_dedup: bool = use_index_dedup or get_ec_index_dedup()

        # output parameters as DTensor in state dict
        self._output_dtensor: bool = (
            fused_params.get("output_dtensor", False) if fused_params else False
        )

        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module,
            table_name_to_parameter_sharding,
            fused_params,
        )

        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                SequenceShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor
            ],
        ] = {
            sharding_type: self.create_hybrid_embedding_sharding(
                sharding_infos=embedding_confings,
                env=env,
                host_env=host_env,
                device=device,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
        }

        self._post_input_dists: List[nn.Module] = []

        self._device = device
        self._input_dists: List[nn.Module] = []

        self._lookups: List[nn.Module] = []
        self._create_lookups()
        self._output_dists: List[nn.Module] = []
        self._create_output_dist()


        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_post_input_dist: bool = True
        logger.info(f"EC index dedup enabled: {self._use_index_dedup}.")

        # Get all fused optimizers and combine them.
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
        if ShardingType.COLUMN_WISE.value in self._sharding_type_to_sharding:
            sharding = self._sharding_type_to_sharding[ShardingType.COLUMN_WISE.value]
            # CW partition must be same for all CW sharded parameters
            self._local_embedding_dim = cast(
                ShardMetadata, sharding.embedding_shard_metadata()[0]
            ).shard_sizes[1]
            self._generate_permute_indices_per_feature(
                module.embedding_configs(), table_name_to_parameter_sharding
            )
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
                        [device]
                        if self._device and (self._device.type == "cuda" or self._device.type == "npu")
                        else None
                    ),
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    static_graph=True,
                )
        self._initialize_torch_state()

        if not device_is_in(module.device, ["meta", "cpu"]):
            self.load_state_dict(module.state_dict(), strict=False)

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    @staticmethod
    def _pre_state_dict_hook(
        self: "HybridShardedEmbeddingCollection",
        prefix: str = "",
        keep_vars: bool = False,
    ) -> None:
        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            lookup.flush()

    @staticmethod
    def _pre_load_state_dict_hook(
        self: "HybridShardedEmbeddingCollection",
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        Modify the destination state_dict for model parallel
        to transform from ShardedTensors into tensors
        """
        for table_name in self._model_parallel_name_to_local_shards.keys():
            key = f"{prefix}embeddings.{table_name}.weight"
            # gather model shards from both DTensor and ShardedTensor maps
            model_shards_sharded_tensor = self._model_parallel_name_to_local_shards[
                table_name
            ]
            model_shards_dtensor = self._model_parallel_name_to_shards_wrapper[
                table_name
            ]
            # If state_dict[key] is already a ShardedTensor, use its local shards
            if isinstance(state_dict[key], ShardedTensor):
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
            elif isinstance(state_dict[key], DTensor):
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
            elif isinstance(state_dict[key], torch.Tensor):
                local_shards = []
                if model_shards_sharded_tensor:
                    # splice according to sharded tensor metadata
                    for shard in model_shards_sharded_tensor:
                        # Extract shard size and offsets for splicing
                        shard_size = shard.metadata.shard_sizes
                        shard_offset = shard.metadata.shard_offsets
                        # Prepare tensor by splicing and placing on appropriate device
                        spliced_tensor = state_dict[key][
                            shard_offset[0]:shard_offset[0] + shard_size[0],
                            shard_offset[1]:shard_offset[1] + shard_size[1],
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
                            shard_offset[0]:shard_offset[0] + shard_size[0],
                            shard_offset[1]:shard_offset[1] + shard_size[1],
                        ]
                        local_shards.append(spliced_tensor)
                state_dict[key] = (
                    torch.empty(0)
                    if not local_shards
                    else torch.cat(local_shards, dim=0)
                )
            else:
                raise RuntimeError(
                    f"Unexpected state_dict key type {type(state_dict[key])} found for {key}"
                )
        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            lookup.purge()

    def create_context(self) -> EmbeddingCollectionContext:
        return EmbeddingCollectionContext()

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
            return HybridTwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                host_env=host_env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            return HybridRwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                host_env=self._host_env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            raise ValueError(
                f"Sharding type not supported {sharding_type} for hybrid mode"
            )

    def forward(self, *input_tensor, **kwargs) -> LazyAwaitable[Out]:
        if len(input_tensor) < 1:
            raise ValueError(f"input must be kjt in 0, but got {input_tensor}")
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input_tensor, **kwargs).wait().wait()
        dist_post_input = self.post_input_dist(ctx, dist_input).wait()
        dist_post_input = kjt_list_to_device(dist_post_input, self._device)
        for ind, _ in enumerate(ctx.sharding_contexts):
            ctx.sharding_contexts[ind] = ctx.sharding_contexts[ind].to(self._device)
        return self.compute_and_output_dist(ctx, dist_post_input)

    def reset_parameters(self) -> None:
        if self._device and self._device.type == "meta":
            return

        # Initialize embedding weights with init_fn
        for table_config in self._embedding_configs:
            if self.module_sharding_plan[table_config.name].compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE.value,
            }:
                continue
            if table_config.init_fn is None:
                raise ValueError(
                    f"table_config init_fn is None, table name {table_config.name}"
                )
            param = self.embeddings[f"{table_config.name}"].weight
            table_config.init_fn(param)

            sharding_type = self.module_sharding_plan[table_config.name].sharding_type
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                pg = self._env.process_group
                with torch.no_grad():
                    dist.broadcast(param.data, src=0, group=pg)

    def input_dist(
        self,
        ctx: EmbeddingCollectionAwaitable,
        features: KeyedJaggedTensor
    ) -> Awaitable[Awaitable[KJTList]]:
        """
        feature的顺序按照Dict[str, list[]]  shardType -> [t.feature_name for t in tables]
        """
        features = features.to("cpu")
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys())
            self._has_uninitialized_input_dist = False

        with torch.no_grad():
            unpadded_features = None
            if features.variable_stride_per_key():
                unpadded_features = features
                features = pad_vbe_kjt_lengths(unpadded_features)

            if self._features_order:
                features = features.permute(
                    self._features_order,
                    self._features_order_tensor,
                )
            features_by_shards = features.split(self._feature_splits)
            if self._use_index_dedup:
                features_by_shards = self._dedup_indices(ctx, features_by_shards)

            awaitables = []
            for input_dist, features in zip(self._input_dists, features_by_shards):
                shard_context = HybridSequenceShardingContext(
                    features_before_input_dist=features
                )
                awaitables.append(input_dist(features, shard_context))
                ctx.sharding_contexts.append(
                    shard_context
                )
            if unpadded_features is not None:
                self._compute_sequence_vbe_context(ctx, unpadded_features)
        return KJTListSplitsAwaitable(awaitables, ctx)

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

    def compute(
        self, ctx: EmbeddingCollectionContext, dist_input: KJTList
    ) -> List[torch.Tensor]:
        ret: List[torch.Tensor] = []
        for lookup, features, sharding_ctx, sharding_type in zip(
            self._lookups,
            dist_input,
            ctx.sharding_contexts,
            self._sharding_type_to_sharding,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
            embedding_dim = self._embedding_dim_for_sharding_type(sharding_type)
            ret.append(lookup(features).view(-1, embedding_dim))
        return ret

    def output_dist(
        self, ctx: EmbeddingCollectionContext, output: List[torch.Tensor]
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for odist, embeddings, sharding_ctx in zip(
            self._output_dists,
            output,
            ctx.sharding_contexts,
        ):
            awaitables_per_sharding.append(odist(embeddings, sharding_ctx))
            features_before_all2all_per_sharding.append(
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
            ctx=ctx,
        )

    def compute_and_output_dist(
        self, ctx: EmbeddingCollectionContext, sparse_features: KJTList
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for lookup, odist, features, sharding_ctx, sharding_type in zip(
            self._lookups,
            self._output_dists,
            sparse_features,
            ctx.sharding_contexts,
            self._sharding_type_to_sharding,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
            embedding_dim = self._embedding_dim_for_sharding_type(sharding_type)
            awaitables_per_sharding.append(
                odist(lookup(features).view(-1, embedding_dim), sharding_ctx)
            )
            features_before_all2all_per_sharding.append(
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
            ctx=ctx,
        )

    def _embedding_dim_for_sharding_type(self, sharding_type: str) -> int:
        return (
            self._local_embedding_dim
            if sharding_type == ShardingType.COLUMN_WISE.value
            else self._embedding_dim
        )

    def _initialize_torch_state(self) -> None:
        """
        This provides consistency between this class and the EmbeddingCollection's
        nn.Module API calls (state_dict, named_modules, etc)
        """
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        for table_name in self._table_names:
            self.embeddings[table_name] = nn.Module()
        self._model_parallel_name_to_local_shards = OrderedDict()
        self._model_parallel_name_to_shards_wrapper = OrderedDict()
        self._model_parallel_name_to_sharded_tensor = OrderedDict()
        self._model_parallel_name_to_dtensor = OrderedDict()
        model_parallel_name_to_compute_kernel: Dict[str, str] = {}
        for (
            table_name,
            parameter_sharding,
        ) in self.module_sharding_plan.items():
            if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            self._model_parallel_name_to_local_shards[table_name] = []
            self._model_parallel_name_to_shards_wrapper[table_name] = OrderedDict(
                [("local_tensors", []), ("local_offsets", [])]
            )
            model_parallel_name_to_compute_kernel[table_name] = (
                parameter_sharding.compute_kernel
            )

        self._name_to_table_size = {}
        for table in self._embedding_configs:
            self._name_to_table_size[table.name] = (
                table.num_embeddings,
                table.embedding_dim,
            )

        for sharding_type, lookup in zip(
            self._sharding_type_to_sharding.keys(), self._lookups
        ):
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                # unwrap DDP
                lookup = lookup.module
            else:
                # save local_shards for transforming MP params to shardedTensor
                for key, v in lookup.state_dict().items():
                    table_name = key[: -len(".weight")]
                    if isinstance(v, DTensor):
                        shards_wrapper = self._model_parallel_name_to_shards_wrapper[
                            table_name
                        ]
                        local_shards_wrapper = v._local_tensor
                        shards_wrapper["local_tensors"].extend(
                            local_shards_wrapper.local_shards()
                        )
                        shards_wrapper["local_offsets"].extend(
                            local_shards_wrapper.local_offsets()
                        )
                        shards_wrapper["global_size"] = v.size()
                        shards_wrapper["global_stride"] = v.stride()
                        shards_wrapper["placements"] = v.placements
                    elif isinstance(v, ShardedTensor):
                        self._model_parallel_name_to_local_shards[table_name].extend(
                            v.local_shards()
                        )
            for (
                table_name,
                tbe_slice,
            ) in lookup.named_parameters_by_table():
                self.embeddings[table_name].register_parameter("weight", tbe_slice)
        for table_name in self._model_parallel_name_to_local_shards.keys():
            local_shards = self._model_parallel_name_to_local_shards[table_name]
            shards_wrapper_map = self._model_parallel_name_to_shards_wrapper[table_name]

            # for shards that don't exist on this rank, register with empty tensor
            if not hasattr(self.embeddings[table_name], "weight"):
                self.embeddings[table_name].register_parameter(
                    "weight", nn.Parameter(torch.empty(0))
                )
                if (
                    model_parallel_name_to_compute_kernel[table_name]
                    != EmbeddingComputeKernel.DENSE.value
                ):
                    self.embeddings[table_name].weight._in_backward_optimizers = [
                        EmptyFusedOptimizer()
                    ]

            if model_parallel_name_to_compute_kernel[table_name] in {
                EmbeddingComputeKernel.KEY_VALUE.value
            }:
                continue
            if self._output_dtensor:
                if shards_wrapper_map["local_tensors"]:
                    self._model_parallel_name_to_dtensor[table_name] = (
                        DTensor.from_local(
                            local_tensor=LocalShardsWrapper(
                                local_shards=shards_wrapper_map["local_tensors"],
                                local_offsets=shards_wrapper_map["local_offsets"],
                            ),
                            device_mesh=self._env.device_mesh,
                            placements=shards_wrapper_map["placements"],
                            shape=shards_wrapper_map["global_size"],
                            stride=shards_wrapper_map["global_stride"],
                            run_check=False,
                        )
                    )
                else:
                    # empty shard case
                    self._model_parallel_name_to_dtensor[table_name] = (
                        DTensor.from_local(
                            local_tensor=LocalShardsWrapper(
                                local_shards=[],
                                local_offsets=[],
                            ),
                            device_mesh=self._env.device_mesh,
                            run_check=False,
                        )
                    )
            else:
                # created ShardedTensors once in init, use in post_state_dict_hook
                self._model_parallel_name_to_sharded_tensor[table_name] = (
                    ShardedTensor._init_from_local_shards(
                        local_shards,
                        self._name_to_table_size[table_name],
                        process_group=self._env.process_group,
                    )
                )

        def post_state_dict_hook(
                module: HybridShardedEmbeddingCollection,
                destination: Dict[str, torch.Tensor],
                prefix: str,
                _local_metadata: Dict[str, Any],
        ) -> None:
            # Adjust dense MP
            for (
                    table_name,
                    sharded_t,
            ) in module._model_parallel_name_to_sharded_tensor.items():
                destination_key = f"{prefix}embeddings.{table_name}.weight"
                destination[destination_key] = sharded_t
            for (
                    table_name,
                    d_tensor,
            ) in module._model_parallel_name_to_dtensor.items():
                destination_key = f"{prefix}embeddings.{table_name}.weight"
                destination[destination_key] = d_tensor

        self.register_state_dict_pre_hook(self._pre_state_dict_hook)
        self._register_state_dict_hook(post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )
        self.reset_parameters()

    def _generate_permute_indices_per_feature(
            self,
            embedding_configs: List[EmbeddingConfig],
            table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    ) -> None:
        """
        Generates permute indices per feature for column-wise sharding.

        Since outputs are stored in order of rank, column-wise shards of a table on the
        same rank will be seen as adjacent, which may not be correct.

        The permute indices store the correct ordering of outputs relative to the
        provided ordering.

        Example::
            rank_0 = [f_0(shard_0), f_0(shard_2)]
            rank_1 = [f_0(shard_1)]
            output = [f_0(shard_0), f_0(shard_2), f_0(shard_1)]

            shard_ranks = [0, 1, 0]
            output_ranks = [0, 0, 1]

            # To get the correct order from output_ranks -> shard_ranks
            permute_indices = [0, 2, 1]
        """
        shared_feature: Dict[str, bool] = {}
        for table in embedding_configs:
            for feature_name in table.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True

        for table in embedding_configs:
            sharding = table_name_to_parameter_sharding[table.name]
            if sharding.sharding_type != ShardingType.COLUMN_WISE.value:
                continue
            ranks = cast(List[int], sharding.ranks)
            rank_to_indices = defaultdict(deque)
            for i, rank in enumerate(sorted(ranks)):
                rank_to_indices[rank].append(i)
            permute_indices = [rank_to_indices[rank].popleft() for rank in ranks]
            for feature_name in table.feature_names:
                if shared_feature[feature_name]:
                    self._features_to_permute_indices[
                        feature_name + "@" + table.name
                        ] = permute_indices
                else:
                    self._features_to_permute_indices[feature_name] = permute_indices

    def _create_hash_size_info(
        self,
        feature_names: List[str],
    ) -> None:
        feature_index = 0
        for i, sharding in enumerate(self._sharding_type_to_sharding.values()):
            feature_hash_size: List[int] = []
            feature_hash_size_lengths: List[int] = []
            for table in sharding.embedding_tables():
                table_hash_size = [0] * table.num_features()
                table_hash_size[-1] = table.num_embeddings
                feature_hash_size.extend(table_hash_size)

                table_hash_size = [0] * table.num_features()
                table_hash_size[0] = table.num_features()
                feature_hash_size_lengths.extend(table_hash_size)

                # Sanity check for feature orders
                for f in range(table.num_features()):
                    if feature_names[feature_index + f] != table.feature_names[f]:
                        raise ValueError(
                            f"Feature name mismatch at index {feature_index + f}: expected {table.feature_names[f]},"
                            f" got {feature_names[feature_index + f]}"
                        )
                feature_index += table.num_features()

            feature_hash_size_cumsum: List[int] = [0] + list(
                accumulate(feature_hash_size)
            )
            feature_hash_size_offset: List[int] = [0] + list(
                accumulate(feature_hash_size_lengths)
            )

            # Register buffers for this shard
            self.register_buffer(
                f"_hash_size_cumsum_tensor_{i}",
                torch.tensor(
                    feature_hash_size_cumsum, device=self._device, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                f"_hash_size_offset_tensor_{i}",
                torch.tensor(
                    feature_hash_size_offset, device=self._device, dtype=torch.int64
                ),
                persistent=False,
            )

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:
        feature_names: List[str] = []
        self._feature_splits: List[int] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))
        self._features_order: List[int] = []
        for f in feature_names:
            self._features_order.append(input_feature_names.index(f))
        self._features_order = (
            []
            if self._features_order == list(range(len(self._features_order)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(self._features_order, device=torch.device("cpu"), dtype=torch.int32),
            persistent=False,
        )

        if self._use_index_dedup:
            self._create_hash_size_info(feature_names)

    def _create_post_input_dist(
        self,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            if hasattr(sharding, "create_post_input_dist"):
                self._post_input_dists.append(sharding.create_post_input_dist())
            else:
                self._post_input_dists.append(EMPTY_POST_INPUT_DIST)

    def _create_lookups(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup())

    def _create_output_dist(
        self,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist())

    def _dedup_indices(
        self,
        ctx: EmbeddingCollectionContext,
        input_feature_splits: List[KeyedJaggedTensor],
    ) -> List[KeyedJaggedTensor]:
        with record_function("## dedup_ec_indices ##"):
            features_by_shards = []
            for i, input_feature in enumerate(input_feature_splits):
                hash_size_cumsum = self.get_buffer(f"_hash_size_cumsum_tensor_{i}")
                hash_size_offset = self.get_buffer(f"_hash_size_offset_tensor_{i}")
                (
                    lengths,
                    offsets,
                    unique_indices,
                    reverse_indices,
                ) = torch.ops.fbgemm.jagged_unique_indices(
                    hash_size_cumsum,
                    hash_size_offset,
                    input_feature.offsets().to(torch.int64),
                    input_feature.values().to(torch.int64),
                )
                dedup_features = KeyedJaggedTensor(
                    keys=input_feature.keys(),
                    lengths=lengths,
                    offsets=offsets,
                    values=unique_indices,
                )

                ctx.input_features.append(input_feature)
                ctx.reverse_indices.append(reverse_indices)
                features_by_shards.append(dedup_features)

        return features_by_shards

    def _create_inverse_indices_permute_per_sharding(
        self, inverse_indices: Tuple[List[str], torch.Tensor]
    ) -> None:
        if (
            len(self._embedding_names_per_sharding) == 1
            and self._embedding_names_per_sharding[0] == inverse_indices[0]
        ):
            return
        index_per_name = {name: i for i, name in enumerate(inverse_indices[0])}
        permute_per_sharding = []
        for emb_names in self._embedding_names_per_sharding:
            permute = _pin_and_move(
                torch.tensor(
                    [index_per_name[name.split("@")[0]] for name in emb_names]
                ),
                inverse_indices[1].device,
            )
            permute_per_sharding.append(permute)
        self._inverse_indices_permute_per_sharding = permute_per_sharding

    def _compute_sequence_vbe_context(
        self,
        ctx: EmbeddingCollectionContext,
        unpadded_features: KeyedJaggedTensor,
    ) -> None:
        if unpadded_features.inverse_indices_or_none() is None:
            raise ValueError("inverse indices must be provided from KJT if using variable batch size per feature.")
        inverse_indices = unpadded_features.inverse_indices()
        stride = inverse_indices[1].numel() // len(inverse_indices[0])
        if self._inverse_indices_permute_per_sharding is None:
            self._create_inverse_indices_permute_per_sharding(inverse_indices)

        if self._features_order:
            unpadded_features = unpadded_features.permute(
                self._features_order,
                self._features_order_tensor,
            )

        features_by_sharding = unpadded_features.split(self._feature_splits)
        for i, feature in enumerate(features_by_sharding):
            if self._inverse_indices_permute_per_sharding is not None:
                permute = self._inverse_indices_permute_per_sharding[i]
                permuted_indices = torch.index_select(inverse_indices[1], 0, permute)
            else:
                permuted_indices = inverse_indices[1]
            stride_per_key = _pin_and_move(
                torch.tensor(feature.stride_per_key()), feature.device()
            )
            offsets = _to_offsets(stride_per_key)[:-1].unsqueeze(-1)
            recat = (permuted_indices + offsets).flatten().int()

            if self._need_indices:
                reindexed_lengths, reindexed_values, _ = (
                    torch.ops.fbgemm.permute_1D_sparse_data(
                        recat,
                        feature.lengths(),
                        feature.values(),
                    )
                )
            else:
                reindexed_lengths = torch.index_select(feature.lengths(), 0, recat)
                reindexed_values = None

            reindexed_lengths = reindexed_lengths.view(-1, stride)
            reindexed_length_per_key = torch.sum(reindexed_lengths, dim=1).tolist()

            ctx.seq_vbe_ctx.append(
                SequenceVBEContext(
                    recat=recat,
                    unpadded_lengths=feature.lengths(),
                    reindexed_lengths=reindexed_lengths,
                    reindexed_length_per_key=reindexed_length_per_key,
                    reindexed_values=reindexed_values,
                )
            )


class HybridEmbeddingCollectionSharder(BaseEmbeddingSharder[EmbeddingCollection]):
    def __init__(
        self,
        host_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
    ) -> None:
        super().__init__(
            fused_params=fused_params, qcomm_codecs_registry=qcomm_codecs_registry
        )
        self._host_env = host_env
        self._use_index_dedup = use_index_dedup

    @property
    def module_type(self) -> Type[EmbeddingCollection]:
        return EmbeddingCollection

    def shard(
        self,
        module: EmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> HybridShardedEmbeddingCollection:
        return HybridShardedEmbeddingCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            host_env=self._host_env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
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
        self, module: EmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embeddings.named_parameters()
        }

#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import torch
import torch.distributed as dist
from torch import nn, Tensor

from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    PoolingMode,
    ComputeDevice,
    EmbeddingLocation,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import CacheAlgorithm
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    is_torchdynamo_compiling,
)

import hybrid_torchrec.hybrid_lookup_invoke as invokers
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
)
import hybrid_torchrec.hybrid_lookup_invoke as invokers
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
)

from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbeddingBag,
    EmbeddingFusedOptimizer,
    _gen_named_parameters_by_table_fused,
)
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_types import (
    compute_kernel_to_embedding_location,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.types import (
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
)
from torchrec.optim.fused import (
    EmptyFusedOptimizer,
    FusedOptimizer,
    FusedOptimizerModule,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
    BaseBatchedEmbeddingBag,
    EmbeddingFusedOptimizer,
    _gen_named_parameters_by_table_fused,
)


class HybridSplitTableBatchedEmbeddingBagsCodegen(
    SplitTableBatchedEmbeddingBagsCodegen
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        is_mixed_dim = False
        first_dim = self.dims[0]
        for d in self.dims:
            if d != first_dim:
                is_mixed_dim = True
                break

        self.is_mixed_dim = is_mixed_dim
        optimizer_type = kwargs["optimizer"]
        if optimizer_type in (OptimType.ADAM,):
            self._optim_num = 2
        elif optimizer_type in (OptimType.EXACT_ADAGRAD,):
            self._optim_num = 1
        else:
            raise ValueError(f"{optimizer_type} is not support")

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        hash_indices: torch.Tensor = None,
        unique_indices: torch.Tensor = None,
        unique_offset: torch.Tensor = None,
        unique_inverse: torch.Tensor = None,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
        # 2D tensor of batch size for each rank and feature.
        # Shape (number of features, number of ranks)
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        total_unique_indices: Optional[int] = None,
    ) -> Tensor:
        (
            indices,
            offsets,
            per_sample_weights,
            vbe_metadata,
        ) = self.prepare_inputs(
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
            force_cast_input_types=True,
        )
        # Print input stats if enable (for debugging purpose only)
        self._debug_print_input_stats(indices, offsets, per_sample_weights)

        if not is_torchdynamo_compiling():
            # Mutations of nn.Module attr forces dynamo restart of Analysis which increases compilation time

            # Storing tensors for linear_cache_indices recomputation
            self._indices = indices
            self._offsets = offsets
            self._vbe_b_offsets = vbe_metadata.B_offsets
            self._vbe_max_b = vbe_metadata.max_B

            self.step += 1
            self._report_io_size_count("fwd_input", indices)
            self._report_tbe_mem_usage()

        if len(self.timesteps_prefetched) == 0:
            # In forward, we don't enable multi-pass prefetch as we want the process
            # to be as fast as possible and memory usage doesn't matter (will be recycled
            # by dense fwd/bwd)
            self._prefetch(
                indices, offsets, vbe_metadata, multipass_prefetch_config=None
            )

        if len(self.timesteps_prefetched) > 0:
            self.timesteps_prefetched.pop(0)

        self.lxu_cache_locations = (
            self.lxu_cache_locations_empty
            if len(self.lxu_cache_locations_list) == 0
            else self.lxu_cache_locations_list.pop(0)
        )
        common_args = invokers.lookup_args.HybridCommonArgs(
            placeholder_autograd_tensor=self.placeholder_autograd_tensor,
            dev_weights=self.weights_dev,
            host_weights=self.weights_host,
            uvm_weights=self.weights_uvm,
            lxu_cache_weights=self.lxu_cache_weights,
            weights_placements=self.weights_placements,
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_D=self.max_D,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=indices,
            offsets=offsets,
            hash_indices=hash_indices,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_inverse=unique_inverse,
            hash_indices2address=None,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
            feature_requires_grad=feature_requires_grad,
            lxu_cache_locations=self.lxu_cache_locations,
            uvm_cache_stats=(
                self.local_uvm_cache_stats
                if (
                    self.gather_uvm_cache_stats
                    # Unique conflict misses are only collected when using CacheAlgorithm.LRU
                    and self.cache_algorithm == CacheAlgorithm.LRU
                )
                else None
            ),
            output_dtype=self.output_dtype,
            vbe_metadata=vbe_metadata,
            is_experimental=self.is_experimental,
            use_uniq_cache_locations_bwd=self.use_uniq_cache_locations_bwd,
            use_homogeneous_placements=self.use_homogeneous_placements,
        )

        if not isinstance(self.optimizer, OptimType):
            raise ValueError(f"Invalid OptimType: {self.optimizer}")

        momentum1 = invokers.lookup_args.Momentum(
            dev=self.momentum1_dev,
            host=self.momentum1_host,
            uvm=self.momentum1_uvm,
            offsets=self.momentum1_offsets,
            placements=self.momentum1_placements,
        )

        if not self.iter.is_cpu:
            self.iter = self.iter.cpu()
        self.iter[0] += 1

        momentum2 = invokers.lookup_args.Momentum(
            dev=self.momentum2_dev,
            host=self.momentum2_host,
            uvm=self.momentum2_uvm,
            offsets=self.momentum2_offsets,
            placements=self.momentum2_placements,
        )

        if self.optimizer == OptimType.EXACT_ADAGRAD:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_adagrad.invoke(
                    common_args, self.optimizer_args, momentum1
                ),
            )
        elif self.optimizer == OptimType.ADAM:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_adam.invoke(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                    momentum2,
                    # pyre-fixme[6]: Expected `int` for 5th param but got `Union[float,
                    #  int]`.
                    self.iter.item(),
                ),
            )
        else:
            return NotImplemented

    def prepare_inputs(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        force_cast_input_types: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], invokers.lookup_args.VBEMetadata]:
        """
        Prepare TBE inputs as follows:

        (1) Create VBE metadata
        (2) Convert input types if `force_cast_input_types=True`
        (3) Run `bounds_check_indices` if `bounds_check_mode` is not
            BoundsCheckMode.NONE

        Args:
            indices (Tensor): Input indices
            offsets (Tensor): Input offsets
            per_sample_weights (Optional[Tensor]): Input per sample
                weights
            batch_size_per_feature_per_rank
                (Optional[List[List[int]]]): A 2D tensor of batch size
                for each rank and feature. Shape = (number of
                features, number of ranks)
            force_cast_input_types (bool): A flag to force convert
                input types if set to True

        Returns:
            A tuple of indices, offsets, per_sample_weights, and VBE
            metadata
        """

        # Generate VBE metadata
        vbe_metadata = self._generate_vbe_metadata(
            offsets, batch_size_per_feature_per_rank
        )

        # type
        force_cast_input_types = (
            indices.dtype != offsets.dtype or force_cast_input_types
        )

        if force_cast_input_types:
            # Force casting indices and offsets to long
            (indices, offsets) = indices.long(), offsets.long()

            # Force casting per_sample_weights to float
            if per_sample_weights is not None:
                per_sample_weights = per_sample_weights.float()

        return indices, offsets, per_sample_weights, vbe_metadata

    def scatter_update_embs(self, indices, updates):
        if not self.is_mixed_dim:
            self.weights_dev.reshape(-1, self.dims[0]).index_put_(
                [indices], updates.reshape(-1, self.dims[0])
            )
        else:
            raise ValueError(f"Mixed dimensions are not supported.")
        return

    def gather_embs(self, indices) -> Tensor:
        if not self.is_mixed_dim:
            return torch.index_select(
                self.weights_dev.reshape(-1, self.dims[0]), 0, indices
            ).reshape(-1)
        else:
            raise ValueError(f"Mixed dimensions are not supported.")

    def gather_momentum(self, indices: torch.Tensor) -> Tensor:
        if not self.is_mixed_dim:
            result = []
            if self._optim_num > 0:
                moment1 = torch.index_select(
                    self.momentum1_dev.reshape(-1, self.dims[0]), 0, indices
                ).reshape(-1)
                result.append(moment1)
            if self._optim_num > 1:
                moment2 = torch.index_select(
                    self.momentum2_dev.reshape(-1, self.dims[0]), 0, indices
                ).reshape(-1)
                result.append(moment2)
            return result
        else:
            raise ValueError(f"Mixed dimensions are not supported.")

    def scatter_update_momentum(
        self, indices: torch.Tensor, updates: List[torch.Tensor]
    ):
        if not self.is_mixed_dim:
            if self._optim_num > 0:
                self.momentum1_dev.reshape(-1, self.dims[0]).index_put_(
                    [indices], updates[0].reshape(-1, self.dims[0])
                )
            if self._optim_num > 1:
                self.momentum2_dev.reshape(-1, self.dims[0]).index_put_(
                    [indices], updates[1].reshape(-1, self.dims[0])
                )
        else:
            raise ValueError(f"Mixed dimensions are not supported.")


class HybridBatchedFusedEmbeddingBag(
    BaseBatchedEmbeddingBag[torch.Tensor], FusedOptimizerModule
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

        managed: List[EmbeddingLocation] = []
        compute_devices: List[ComputeDevice] = []
        for table in config.embedding_tables:
            if table.local_cols % 4 != 0:
                raise ValueError(
                    f"table {table.name} has local_cols={table.local_cols} "
                    "not divisible by 4. "
                )
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            elif device is not None and device.type == "mtia":
                compute_devices.append(ComputeDevice.MTIA)
                # Set EmbeddingLocation.HOST to make embedding op in FBGEMM choose CPU path.
                # But the tensor will still be created on MTIA with device type "mtia".
                managed.append(EmbeddingLocation.HOST)
            elif device is not None and device.type == "npu":
                compute_devices.append(ComputeDevice.NPU)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        if "cache_precision" not in fused_params:
            fused_params["cache_precision"] = weights_precision

        self._emb_module: HybridSplitTableBatchedEmbeddingBagsCodegen = (
            HybridSplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=list(
                    zip(self._local_rows, self._local_cols, managed, compute_devices)
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                weights_precision=weights_precision,
                device=device,
                **fused_params,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_fused(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> HybridSplitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        hash_indices = None
        unique_indices = None
        unique_offset = None
        unique_inverse = None
        if isinstance(features, KeyedJaggedTensorWithLookHelper):
            features: KeyedJaggedTensorWithLookHelper
            hash_indices = features.hash_indices
            unique_indices = features.unique_indices
            unique_offset = features.unique_offset
            unique_inverse = features.unique_inverse

        weights = features.weights_or_none()
        if weights is not None and not torch.is_floating_point(weights):
            weights = None
        if features.variable_stride_per_key() and isinstance(
            self.emb_module, SplitTableBatchedEmbeddingBagsCodegen
        ):
            return self.emb_module(
                indices=features.values().long(),
                offsets=features.offsets().long(),
                hash_indices=hash_indices,
                unique_indices=None,
                unique_offset=None,
                unique_inverse=None,
                per_sample_weights=weights,
                batch_size_per_feature_per_rank=features.stride_per_key_per_rank(),
            )
        else:
            return self.emb_module(
                indices=features.values().long(),
                offsets=features.offsets().long(),
                hash_indices=hash_indices,
                unique_indices=unique_indices,
                unique_offset=unique_offset,
                unique_inverse=unique_inverse,
                per_sample_weights=weights,
            )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            param = nn.Parameter(tensor)
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()

    def purge(self) -> None:
        self._emb_module.reset_cache_states()


class HybridBatchedFusedEmbedding(
    BaseBatchedEmbedding[torch.Tensor], FusedOptimizerModule
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        compute_devices: List[ComputeDevice] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            elif device is not None and device.type == "mtia":
                compute_devices.append(ComputeDevice.MTIA)
                # Set EmbeddingLocation.HOST to make embedding op in FBGEMM choose CPU path.
                # But the tensor will still be created on MTIA with device type "mtia".
                managed.append(EmbeddingLocation.HOST)
            elif device is not None and device.type == "npu":
                compute_devices.append(ComputeDevice.NPU)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)

        weights_precision = data_type_to_sparse_type(config.data_type)

        fused_params = config.fused_params or {}
        if "cache_precision" not in fused_params:
            fused_params["cache_precision"] = weights_precision

        self._emb_module: HybridSplitTableBatchedEmbeddingBagsCodegen = (
            HybridSplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=list(
                    zip(self._local_rows, self._local_cols, managed, compute_devices)
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=PoolingMode.NONE,
                weights_precision=weights_precision,
                device=device,
                table_names=[t.name for t in config.embedding_tables],
                **fused_params,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_fused(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> HybridSplitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        hash_indices = None
        unique_indices = None
        unique_offset = None
        unique_inverse = None
        if isinstance(features, KeyedJaggedTensorWithLookHelper):
            features: KeyedJaggedTensorWithLookHelper
            hash_indices = features.hash_indices
            unique_indices = features.unique_indices
            unique_offset = features.unique_offset
            unique_inverse = features.unique_inverse

        weights = features.weights_or_none()
        if weights is not None and not torch.is_floating_point(weights):
            weights = None
        if features.variable_stride_per_key() and isinstance(
            self.emb_module, SplitTableBatchedEmbeddingBagsCodegen
        ):
            return self.emb_module(
                indices=features.values().long(),
                offsets=features.offsets().long(),
                hash_indices=hash_indices,
                unique_indices=None,
                unique_offset=None,
                unique_inverse=None,
                per_sample_weights=weights,
                batch_size_per_feature_per_rank=features.stride_per_key_per_rank(),
            )
        else:
            return self.emb_module(
                indices=features.values().long(),
                offsets=features.offsets().long(),
                hash_indices=hash_indices,
                unique_indices=unique_indices,
                unique_offset=unique_offset,
                unique_inverse=unique_inverse,
                per_sample_weights=weights,
            )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after SEA deprecation
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()

    def purge(self) -> None:
        self._emb_module.reset_cache_states()

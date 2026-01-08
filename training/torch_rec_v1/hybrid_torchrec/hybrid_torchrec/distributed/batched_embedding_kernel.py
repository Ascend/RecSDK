#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Tuple, Any,
)

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import CacheAlgorithm
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    PoolingMode,
    ComputeDevice,
    EmbeddingLocation,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import is_torchdynamo_compiling
from torch import nn, Tensor

import hybrid_torchrec.hybrid_lookup_invoke as invokers
from hybrid_torchrec import IS_TORCH_REC_120
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
)
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
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

INIT_BUFFER = 100  # 初始buffer为table大小的100分之1
RESET_BUFFER = 10  # 扩大开始的10倍

if hasattr(torch, 'npu') and torch.npu.is_available():
    DEVICE = 'npu'
else:
    DEVICE = 'cpu'


@dataclass
class CommonArgsInput:
    indices: Tensor
    offsets: Tensor
    vbe_metadata: Any
    feature_requires_grad: Optional[Tensor] = None
    hash_indices: torch.Tensor = None
    per_sample_weights: Optional[Tensor] = None
    unique_indices: torch.Tensor = None
    unique_inverse: torch.Tensor = None
    unique_offset: torch.Tensor = None
    table_grad_accumulate_offsets: torch.Tensor = None
    grad_accumulate: List[torch.Tensor] = None
    grad_accumulate_offsets: torch.Tensor = None
    use_optimize: bool = True


@dataclass
class CommonArgsAggregationInput:
    indices: Tensor
    offsets: Tensor
    vbe_metadata: Any
    feature_requires_grad: Optional[Tensor] = None
    hash_indices: torch.Tensor = None
    per_sample_weights: Optional[Tensor] = None
    unique_indices: torch.Tensor = None
    unique_inverse: torch.Tensor = None
    unique_offset: torch.Tensor = None
    table_grad_accumulate_offsets: torch.Tensor = None
    grad_accumulate: List[torch.Tensor] = None
    grad_accumulate_offsets: torch.Tensor = None
    use_optimize: bool = True
    table_offsets_multi: torch.Tensor = None
    indices_multi_step: torch.Tensor = None
    offsets_multi_step: torch.Tensor = None
    unique_multi_step: torch.Tensor = None
    unique_offset_multi_step: torch.Tensor = None
    unique_inverse_multi_step: torch.Tensor = None


class GradientAccumulator(nn.Module):
    def __init__(self, table_shapes, device):
        super().__init__()
        self.buffers = nn.ModuleDict()
        self.buffer_names = []
        self.device = device
        self.current_accumulate_step = 0

        # 为每个参数创建显存缓冲区
        for i, shape in enumerate(table_shapes):
            buffer_name = f"grad_acc_{i}"
            self.buffer_names.append(buffer_name)

            self.register_buffer(
                buffer_name,
                torch.zeros(shape, device=device, dtype=torch.float32),
                persistent=False  # 不保存到状态字典
            )
        self.total_index_size = [0 for _ in self.buffer_names]
        self.total_index_size_pre = [0 for _ in self.buffer_names]
        self.indice_multi_step = [None for _ in self.buffer_names]
        self.unique_indice_multi_step = 0
        self.unique_inverse_multi_step = 0

    def get_buffer(self):
        """返回所有缓冲区的指针和元数据"""
        handles = []
        for name, buf in self.named_buffers():
            handles.append(buf)
        return handles

    def updata_total_index_size(self, size_list, table_dims):
        for i in range(len(size_list) - 1):
            self.total_index_size[i] += int(size_list[i + 1] - size_list[i]) * table_dims[i]

    def zero_grad(self):
        """重置所有梯度缓冲区"""
        with torch.no_grad():
            for buffer_name, buffer in self.named_buffers():
                self.register_buffer(
                    buffer_name,
                    torch.zeros(buffer.numel(), device=self.device, dtype=torch.float32),
                    persistent=False
                ) # 需同时重置累积步数计数器

    def zero_parameters(self):
        self.total_index_size = [0 for _ in self.buffer_names]
        self.total_index_size_pre = [0 for _ in self.buffer_names]
        self.indice_multi_step = [None for _ in self.buffer_names]
        self.unique_indice_multi_step = 0
        self.unique_inverse_multi_step = 0
        self.current_accumulate_step = 0

    def get_buffer_size(self, buffer_name):
        """获取单个缓冲区的显存大小（字节）"""
        if not hasattr(self, buffer_name):
            raise ValueError(f"Buffer {buffer_name} not found")

        buffer = getattr(self, buffer_name)
        # 计算显存占用：元素数量 × 每个元素字节大小
        return buffer.numel() * buffer.element_size()

    def resize_buffer(self, new_shape):
        for name, buf in self.named_buffers():
            current_buffer_size = self.get_buffer_size(name)
            new_buffer_size = new_shape[name] * buf.element_size()
            if new_buffer_size > current_buffer_size:
                new_buffer = torch.zeros(
                    new_shape[name] * RESET_BUFFER,
                    device=self.device,
                    dtype=torch.float32
                )
                new_buffer[:buf.numel()].copy_(buf)
                self.register_buffer(name, new_buffer, persistent=False)

    def store_buffer_shape(self, table_shapes):
        table_shape_dict = {}
        for i, shape in enumerate(table_shapes):
            table_shape_dict[self.buffer_names[i]] = shape
        return table_shape_dict

    def get_split_lookup_input(self, values: torch.Tensor, offset_per_key: torch.Tensor, ):
        len_per_key = (offset_per_key[1:] - offset_per_key[:-1]).tolist()
        split_values = torch.split(values.long(), len_per_key)
        return split_values

    def concat_multi_step(self, values):
        for i, value in enumerate(values):
            if self.indice_multi_step[i] is None:
                cur = value
            else:
                cur = torch.cat((self.indice_multi_step[i], value))
            self.indice_multi_step[i] = cur

    def do_multi_step_unique(self, ):
        unique_values_list = []
        inverse_indices_list = []
        counts_list = [0]
        for i, table_indices in enumerate(self.indice_multi_step):
            unique_values, inverse_indices = torch.unique(
                table_indices,
                return_inverse=True
            )
            unique_values_list.append(unique_values)
            inverse_indices_list.append(inverse_indices)
            counts_list.append(unique_values.shape[0] + counts_list[-1])
        unique = torch.cat(unique_values_list)
        unique_inverse = torch.cat(inverse_indices_list)
        unique_offset = torch.tensor(counts_list)
        return unique, unique_inverse, unique_offset

    def do_table_offsets(self, last_step, offsets):
        table_offsets = [0]
        if last_step:
            tmp = 0
            for i, table_indices in enumerate(self.indice_multi_step):
                tmp += len(table_indices)
                table_offsets.append(tmp)
        else:
            b = int((len(offsets) - 1) / len(self.buffer_names))
            for i in range(len(self.buffer_names)):
                table_offsets.append(offsets[(i + 1) * b])
        return torch.Tensor(table_offsets)




class HybridSplitTableBatchedEmbeddingBagsCodegen(
    SplitTableBatchedEmbeddingBagsCodegen
):
    def __init__(
        self,
        embedding_specs: List[
            Tuple[int, int, EmbeddingLocation, ComputeDevice]
        ],
        use_accumulate=False,
        accumulate_step=1,
        **kwargs
    ) -> None:
        super().__init__(embedding_specs, **kwargs)

        is_mixed_dim = False
        first_dim = self.dims[0]
        for d in self.dims:
            if d != first_dim:
                is_mixed_dim = True
                break

        self.is_mixed_dim = is_mixed_dim
        optimizer_type = kwargs.get("optimizer", self.optimizer)
        if optimizer_type in (OptimType.ADAM,):
            self._optim_num = 2
        elif optimizer_type in (OptimType.EXACT_ADAGRAD,):
            self._optim_num = 1
        else:
            self._optim_num = 0

        table_shapes = []
        self.feature_map_dims = []
        for table_index in self.feature_table_map:
            table_shape = embedding_specs[table_index]
            table_shapes.append(int(table_shape[0] * table_shape[1] / INIT_BUFFER))
            self.feature_map_dims.append(table_shape[1])

        self.grad_accum = GradientAccumulator(table_shapes, self.current_device)
        self.use_accumulate = use_accumulate
        self.accumulate_step = accumulate_step

    def clear_after_accumulate(self, grad):
        # 这里可以调用您的自定义函数
        self.grad_accum.zero_grad()
        self.grad_accum.zero_parameters()
        # 注意：钩子函数需要返回梯度（可以是修改后的或原始的）
        return grad

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
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> Tensor:
        (indices, offsets, per_sample_weights, vbe_metadata,) = self.prepare_inputs(
            indices, offsets, per_sample_weights, batch_size_per_feature_per_rank, force_cast_input_types=True, )
        # Print input stats if enable (for debugging purpose only)
        self._debug_print_input_stats(indices, offsets, per_sample_weights)

        self.check_preprocess(indices, offsets, vbe_metadata)

        table_grad_accumulate_offsets = self.grad_accum.do_table_offsets(False, offsets)
        table_grad_accumulate_offsets = table_grad_accumulate_offsets.to(DEVICE, dtype=torch.int64)

        if self.use_accumulate and self.training:
            self.grad_accum.current_accumulate_step += 1
            split_values = self.grad_accum.get_split_lookup_input(unique_indices, unique_offset)
            # 将当前step拼接到每个表的多step索引中并存储
            self.grad_accum.concat_multi_step(split_values)

            self.grad_accum.total_index_size_pre = torch.tensor(self.grad_accum.total_index_size)
            self.grad_accum.updata_total_index_size(unique_offset, self.feature_map_dims)
            table_shapes = self.grad_accum.total_index_size
            table_dict = self.grad_accum.store_buffer_shape(table_shapes)
            self.grad_accum.resize_buffer(table_dict)
            grad_accumulate = self.grad_accum.get_buffer()
            grad_accumulate_offsets = self.grad_accum.total_index_size_pre
            use_optimize = False
            if self.grad_accum.current_accumulate_step == self.accumulate_step:
                self.iter[0] += 1
        else:
            grad_accumulate = [torch.tensor([0]).to(DEVICE)]
            grad_accumulate_offsets = torch.tensor([0])
            use_optimize = True
            if self.training:
                self.iter[0] += 1

        common_args = self.create_common_args(
            CommonArgsInput(indices, offsets, vbe_metadata, feature_requires_grad, hash_indices, per_sample_weights,
                            unique_indices, unique_inverse, unique_offset, table_grad_accumulate_offsets,
                            grad_accumulate, grad_accumulate_offsets, use_optimize))

        if not isinstance(self.optimizer, OptimType):
            raise ValueError(f"Invalid OptimType: {self.optimizer}")

        momentum1, momentum2 = self.create_momentum()

        # use_accumulate and last step
        if self.use_accumulate and self.grad_accum.current_accumulate_step == self.accumulate_step:
            indices_multi_step = torch.cat(self.grad_accum.indice_multi_step)
            offsets_multi_step = torch.arange(indices_multi_step.shape[0] + 1).to(DEVICE)

            unique_multi_step, unique_inverse_multi_step, unique_offset_multi_step = (
                self.grad_accum.do_multi_step_unique()
            )
            unique_offset_multi_step = unique_offset_multi_step.to(DEVICE)
            table_offsets_multi = self.grad_accum.do_table_offsets(True, offsets)
            table_offsets_multi = table_offsets_multi.to(DEVICE).to(torch.int64)

            common_args = self.create_common_args_aggregation(
                CommonArgsAggregationInput(indices, offsets, vbe_metadata, feature_requires_grad, hash_indices,
                                           per_sample_weights, unique_indices, unique_inverse, unique_offset,
                                           table_grad_accumulate_offsets, grad_accumulate, grad_accumulate_offsets,
                                           use_optimize, table_offsets_multi, indices_multi_step,
                                           offsets_multi_step, unique_multi_step, unique_offset_multi_step,
                                           unique_inverse_multi_step)
            )
            if self.optimizer == OptimType.EXACT_ADAGRAD:
                result = invokers.lookup_adagrad.invoke_grad_aggregation(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                    iteration=self.iter[0],
                )
            elif self.optimizer == OptimType.ADAM:
                result = invokers.lookup_adam.invoke_grad_aggregation(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                    momentum2,
                    iteration=self.iter[0],
                )
            elif self.optimizer == OptimType.EXACT_SGD:
                result = invokers.lookup_sgd.invoke_grad_aggregation(
                    common_args,
                    self.optimizer_args,
                    iteration=self.iter[0],
                )
            else:
                return NotImplemented
            self.grad_accum.current_accumulate_step = 0
            if result.requires_grad:
                result.register_hook(self.clear_after_accumulate)
            return self._report_io_size_count("fwd_output", result)

        if self.optimizer == OptimType.EXACT_ADAGRAD:
            result = invokers.lookup_adagrad.invoke(
                common_args, self.optimizer_args, momentum1
            )
        elif self.optimizer == OptimType.ADAM:
            result = invokers.lookup_adam.invoke(
                common_args, self.optimizer_args, momentum1, momentum2, iteration=self.iter[0],
            )
        elif self.optimizer == OptimType.EXACT_SGD:
            result = invokers.lookup_sgd.invoke(
                common_args, self.optimizer_args, iteration=self.iter[0],
            )
        else:
            return NotImplemented
        return self._report_io_size_count("fwd_output", result)


    def create_momentum(self):
        momentum1 = invokers.lookup_args.Momentum(
            dev=self.momentum1_dev,
            host=self.momentum1_host,
            uvm=self.momentum1_uvm,
            offsets=self.momentum1_offsets,
            placements=self.momentum1_placements,
        )
        if not self.iter.is_cpu:
            self.iter = self.iter.cpu()
        momentum2 = invokers.lookup_args.Momentum(
            dev=self.momentum2_dev,
            host=self.momentum2_host,
            uvm=self.momentum2_uvm,
            offsets=self.momentum2_offsets,
            placements=self.momentum2_placements,
        )
        return momentum1, momentum2

    def create_common_args(self, args_input: CommonArgsInput):
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
            rows_per_table=self.rows_per_table,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=args_input.indices,
            offsets=args_input.offsets,
            hash_indices=args_input.hash_indices,
            unique_indices=args_input.unique_indices,
            unique_offset=args_input.unique_offset,
            unique_inverse=args_input.unique_inverse,
            hash_indices2address=None,
            pooling_mode=self.pooling_mode,
            indice_weights=args_input.per_sample_weights,
            feature_requires_grad=args_input.feature_requires_grad,
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
            vbe_metadata=args_input.vbe_metadata,
            is_experimental=self.is_experimental,
            use_uniq_cache_locations_bwd=self.use_uniq_cache_locations_bwd,
            use_homogeneous_placements=self.use_homogeneous_placements,
            learning_rate=self.get_learning_rate() if IS_TORCH_REC_120 else 0.0,
            table_grad_accumulate_offsets=args_input.table_grad_accumulate_offsets,
            grad_accumulate=args_input.grad_accumulate,
            grad_accumulate_offsets=args_input.grad_accumulate_offsets,
            use_optimize=args_input.use_optimize
        )
        return common_args


    def create_common_args_aggregation(self, args_input: CommonArgsAggregationInput):
        common_args = invokers.lookup_args.HybridCommonArgsAggregation(
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
            rows_per_table=self.rows_per_table,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=args_input.indices,
            offsets=args_input.offsets,
            hash_indices=args_input.hash_indices,
            unique_indices=args_input.unique_indices,
            unique_offset=args_input.unique_offset,
            unique_inverse=args_input.unique_inverse,
            hash_indices2address=None,
            pooling_mode=self.pooling_mode,
            indice_weights=args_input.per_sample_weights,
            feature_requires_grad=args_input.feature_requires_grad,
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
            vbe_metadata=args_input.vbe_metadata,
            is_experimental=self.is_experimental,
            use_uniq_cache_locations_bwd=self.use_uniq_cache_locations_bwd,
            use_homogeneous_placements=self.use_homogeneous_placements,
            learning_rate=self.get_learning_rate() if IS_TORCH_REC_120 else 0.0,
            table_grad_accumulate_offsets=args_input.table_grad_accumulate_offsets,
            grad_accumulate=args_input.grad_accumulate,
            grad_accumulate_offsets=args_input.grad_accumulate_offsets,
            use_optimize=args_input.use_optimize,
            table_offsets_multi=args_input.table_offsets_multi,
            indices_multi_step=args_input.indices_multi_step,
            offsets_multi_step=args_input.offsets_multi_step,
            unique_multi_step=args_input.unique_multi_step,
            unique_offset_multi_step=args_input.unique_offset_multi_step,
            unique_inverse_multi_step=args_input.unique_inverse_multi_step
        )
        return common_args


    def check_preprocess(self, indices, offsets, vbe_metadata):
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
            self._prefetch(indices, offsets, vbe_metadata, multipass_prefetch_config=None)
        if len(self.timesteps_prefetched) > 0:
            self.timesteps_prefetched.pop(0)
        self.lxu_cache_locations = (self.lxu_cache_locations_empty if len(self.lxu_cache_locations_list) == 0
                                    else self.lxu_cache_locations_list.pop(0))

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
        if self.is_mixed_dim:
            raise ValueError(f"Mixed dimensions are not supported.")

        self.weights_dev.reshape(-1, self.dims[0]).index_put_(
            [indices], updates.reshape(-1, self.dims[0])
        )

    def gather_embs(self, indices) -> Tensor:
        if self.is_mixed_dim:
            raise ValueError(f"Mixed dimensions are not supported.")

        return torch.index_select(
            self.weights_dev.reshape(-1, self.dims[0]), 0, indices
        ).reshape(-1)

    def gather_momentum(self, indices: torch.Tensor) -> Tensor:
        if self.is_mixed_dim:
            raise ValueError(f"Mixed dimensions are not supported.")

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

    def scatter_update_momentum(
        self, indices: torch.Tensor, updates: List[torch.Tensor]
    ):
        if self.is_mixed_dim:
            raise ValueError(f"Mixed dimensions are not supported.")

        if self._optim_num > 0:
            self.momentum1_dev.reshape(-1, self.dims[0]).index_put_(
                [indices], updates[0].reshape(-1, self.dims[0])
            )
        if self._optim_num > 1:
            self.momentum2_dev.reshape(-1, self.dims[0]).index_put_(
                [indices], updates[1].reshape(-1, self.dims[0])
            )

    def get_momentum(self) -> List[torch.Tensor]:
        result = []
        if self._optim_num > 0:
            result.append(self.momentum1_dev)
        if self._optim_num > 1:
            result.append(self.momentum2_dev)
        return result


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
                raise ValueError(f"table {table.name} has local_cols={table.local_cols} " "not divisible by 4. ")
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(compute_kernel_to_embedding_location(table.compute_kernel))
            elif device is not None and device.type == "mtia":
                compute_devices.append(ComputeDevice.MTIA)
                # Set EmbeddingLocation.HOST to make embedding op in FBGEMM choose CPU path.
                # But the tensor will still be created on MTIA with device type "mtia".
                managed.append(EmbeddingLocation.HOST)
            elif device is not None and device.type == "npu":
                compute_devices.append(ComputeDevice.NPU)
                managed.append(compute_kernel_to_embedding_location(table.compute_kernel))
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        if "cache_precision" not in fused_params:
            fused_params["cache_precision"] = weights_precision

        self._emb_module: HybridSplitTableBatchedEmbeddingBagsCodegen = (HybridSplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=list(zip(self._local_rows, self._local_cols, managed, compute_devices)),
            feature_table_map=self._feature_table_map, pooling_mode=self._pooling, weights_precision=weights_precision,
            device=device, **fused_params, ))
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(config, self._emb_module, pg,)
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_fused(emb_module=self._emb_module,
                                                 table_name_to_count=self.table_name_to_count.copy(),
                                                 config=self._config, pg=pg, ))
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=too-many-lines

import os
import enum
import logging
import warnings
from copy import deepcopy
import glob
from typing import List, Tuple, cast, Optional, Union, Dict
from dataclasses import dataclass, field
from itertools import accumulate
from functools import partial

import torch
import torch.distributed as dist
import torch_npu
from torch import Tensor, nn

from dynamic_emb.distributed.utils import tabulate
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    BaseDynamicEmbeddingOptimizerV2,
    OptimizerArgs,
    EmbOptimType,
    convert_optimizer_type,
)
from dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer import (
    AdamDynamicEmbeddingOptimizer,
    AdamDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.adamw_dynamicemb_optimizer import (
    AdamWDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.sgd_dynamicemb_optimizer import (
    SGDDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.adagrad_dynamicemb_optimizer import (
    AdagradDynamicEmbeddingOptimizer,
    AdagradDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.row_wise_adagrad_dynamicemb_optimizer import (
    RowWiseAdagradDynamicEmbeddingOptimizer,
    RowWiseAdagradDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbPoolingMode,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbEvictStrategy,
    get_constraint_capacity,
    BATCH_SIZE_PER_DUMP,
)
from dynamic_emb.distributed.batched_dynamicemb_function import (
    DynamicEmbeddingFunctionV2,
    DynamicEmbeddingBagFunction,
    DynamicEmbeddingFunctionV2Config,
    dynamicemb_prefetch,
)
from dynamic_emb.distributed.types import (
    Cache,
    Storage,
    StorageOptionalFilePaths,
)
from dynamic_emb.distributed.key_value_table import KeyValueTable, batched_export_keys_values
from dynamic_emb.distributed.initializers.dynamicemb_initializers import (
    NormalInitializer,
    ConstantInitializer,
    UniformInitializer,
    TruncatedNormalInitializer,
    DebugInitializer,
)
from dynamic_emb_extensions import device_timestamp, OptimizerType, DynamicEmbTable

logger = logging.getLogger(__name__)


def warning_for_cstm_score() -> bool:
    """Return whether to warn when a new CUSTOMIZED score is less than the old one."""
    return os.environ.get("DYNAMICEMB_CSTM_SCORE_CHECK", "1") != "0"


class WeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2
    COUNTER = 3
    COWCLIP = 4


class CounterWeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2


class LearningRateMode(enum.IntEnum):
    EQUAL = -1
    TAIL_ID_LR_INCREASE = 0
    TAIL_ID_LR_DECREASE = 1
    COUNTER_SGD = 2


class GradSumDecay(enum.IntEnum):
    NO_DECAY = -1
    CTR_DECAY = 0


@dataclass
class TailIdThreshold:
    val: float = 0.0
    is_ratio: bool = False


@dataclass
class CounterBasedRegularizationDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    adjustment_iter: int = -1
    adjustment_ub: float = 1.0
    learning_rate_mode: LearningRateMode = LearningRateMode.EQUAL
    grad_sum_decay: GradSumDecay = GradSumDecay.NO_DECAY
    tail_id_threshold: TailIdThreshold = field(default_factory=TailIdThreshold)
    max_counter_update_freq: int = 1000


@dataclass
class CowClipDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    weight_norm_coefficient: float = 0.0
    lower_bound: float = 0.0


@dataclass
class EmbGradWeightConfig:
    per_sample_weights: Optional[Tensor] = None
    feature_requires_grad: Optional[Tensor] = None


@dataclass
class CreateOptimizerConfig:
    optimizer_type: EmbOptimType
    stochastic_rounding: bool
    gradient_clipping: bool
    max_gradient: float
    max_norm: float
    learning_rate: float
    eps: float
    initial_accumulator_value: float
    beta1: float
    beta2: float
    weight_decay: float
    eta: float
    momentum: float
    weight_decay_mode: WeightDecayMode
    counter_based_regularization: Optional[CounterBasedRegularizationDefinition] = None
    cowclip_regularization: Optional[CowClipDefinition] = None


def encode_meta_json_file_path(root_path: str, table_name: str) -> str:
    return os.path.join(root_path, f"{table_name}_opt_args.json")


def encode_checkpoint_file_path(root_path: str, table_name: str, rank: int, world_size: int, item: str) -> str:
    valid_items = ["keys", "values", "scores", "opt_values"]
    if item not in valid_items:
        raise ValueError(f"Invalid item '{item}'. Must be one of: {', '.join(valid_items)}")
    return os.path.join(root_path, f"{table_name}_emb_{item}.rank_{rank}.world_size_{world_size}")


def find_files(root_path: str, table_name: str, suffix: str) -> Tuple[List[str], int]:
    suffix_to_encode_file_path_func = {
        "emb_keys": partial(encode_checkpoint_file_path, item="keys"),
        "emb_values": partial(encode_checkpoint_file_path, item="values"),
        "emb_scores": partial(encode_checkpoint_file_path, item="scores"),
        "opt_values": partial(encode_checkpoint_file_path, item="opt_values"),
    }
    if suffix not in suffix_to_encode_file_path_func:
        raise RuntimeError(f"Invalid suffix: {suffix}")
    encode_file_path_func = suffix_to_encode_file_path_func[suffix]

    # v2 version
    files = glob.glob(encode_file_path_func(root_path, table_name, "*", "*"))
    if len(files) == 0:
        return [], 0
    files = sorted(files)

    # /model/checkpoints/user_embedding_emb_values.rank_0.world_size_1 --> world_size_[1] --> 1
    world_size = int(files[0].split(".")[-1].split("_")[-1])
    if len(files) != world_size:
        raise RuntimeError(
            f"Checkpoints is corrupted. Found {len(files)} under path {root_path} for table {table_name}, \
            but the number of checkpointed world size is {world_size}."
        )

    for i in range(world_size):
        expected_file_path = encode_file_path_func(root_path, table_name, i, world_size)
        if expected_file_path not in set(files):
            raise RuntimeError(
                f"Checkpoints is corrupted. Expected file path {expected_file_path} for table {table_name}, \
                but it is not found."
            )

    return files, len(files)


def get_loading_files(
    root_path: str,
    name: str,
    rank: int,
    world_size: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if not os.path.exists(root_path):
        raise RuntimeError(f"can't find path to load, path: {root_path}")

    key_files, num_key_files = find_files(root_path, name, "emb_keys")
    value_files, num_value_files = find_files(root_path, name, "emb_values")
    score_files, num_score_files = find_files(root_path, name, "emb_scores")
    opt_files, num_opt_files = find_files(root_path, name, "opt_values")

    if num_key_files != num_value_files:
        if num_key_files == 0:
            raise RuntimeError(f"No key files found under path {root_path} for table {name}")
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of value files."
        )

    if world_size == num_key_files:
        return (
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "keys")],
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "values")],
            (
                [encode_checkpoint_file_path(root_path, name, rank, world_size, "scores")]
                if num_score_files == num_key_files
                else []
            ),
            (
                [encode_checkpoint_file_path(root_path, name, rank, world_size, "opt_values")]
                if num_opt_files == num_key_files
                else []
            ),
        )
    return (
        key_files,
        value_files,
        score_files,
        opt_files,
    )


OPTIMIZER_STATE_ALIGNMENT_BYTES = 16
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024


def _print_memory_consume(
    table_names: Optional[List[str]],
    dynamicemb_options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizerV2,
    device_id: int,
) -> None:
    subtitle = [
        "",
        "total",
        "embedding",
        "optim_state",
        "total",
        "embedding",
        "optim_state",
        "total",
        "embedding",
        "optim_state",
    ]
    table_consume = []
    table_consume.append(subtitle)

    def _get_optimizer_state_dim(optimizer_type, dim, element_size):
        if optimizer_type == OptimizerType.RowWiseAdaGrad:
            return OPTIMIZER_STATE_ALIGNMENT_BYTES // element_size
        elif optimizer_type == OptimizerType.Adam:
            return dim * 2
        elif optimizer_type == OptimizerType.AdamW:
            return dim * 2
        elif optimizer_type == OptimizerType.AdaGrad:
            return dim
        else:
            return 0

    DTYPE_NUM_BYTES: Dict[torch.dtype, int] = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }

    def MB_(x) -> int:
        return x // BYTES_PER_MB

    def KB_(x) -> int:
        return x // BYTES_PER_KB

    F = None

    for table_name, table_option in zip(table_names, dynamicemb_options):
        element_size = DTYPE_NUM_BYTES[table_option.embedding_dtype]
        emb_dim = table_option.dim
        if optimizer is not None:
            optim_state_dim = optimizer.get_state_dim(emb_dim)
        else:
            optim_state_dim = _get_optimizer_state_dim(table_option.optimizer_type, emb_dim, element_size)
        total_dim = emb_dim + optim_state_dim
        total_memory = table_option.max_capacity * element_size * total_dim
        if F is None:
            if total_memory // BYTES_PER_MB != 0:
                F = MB_
            else:
                F = KB_
        local_hbm_for_values = min(table_option.local_hbm_for_values, total_memory)
        local_dram_for_values = total_memory - local_hbm_for_values
        table_consume.append(
            [
                table_name,
                F(total_memory),
                F(table_option.max_capacity * element_size * emb_dim),
                F(table_option.max_capacity * element_size * optim_state_dim),
                F(local_hbm_for_values),
                F(int(local_hbm_for_values * emb_dim // total_dim)),
                F(int(local_hbm_for_values * optim_state_dim // total_dim)),
                F(local_dram_for_values),
                F(int(local_dram_for_values * emb_dim // total_dim)),
                F(int(local_dram_for_values * optim_state_dim // total_dim)),
            ]
        )
    unit = "MB" if F is MB_ else "KB"
    title = [
        "table name",
        "",
        f"memory({unit})",
        "",
        "",
        f"hbm({unit})/cuda:{device_id}",
        "",
        "",
        f"dram({unit})",
        "",
    ]
    output = "\n\n" + tabulate(table_consume, title, sub_headers=True)
    logging.info(output)


def _export_matched_and_gather(
    dynamic_table: KeyValueTable,
    threshold: int,
    pg: Optional[dist.ProcessGroup] = None,
    batch_size: int = BATCH_SIZE_PER_DUMP,
) -> Tuple[Tensor, Tensor]:
    if not torch_npu.npu.is_available():
        raise ValueError("No available npu device.")
    # Get the rank of the current process
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    device = torch.device(f"npu:{torch_npu.npu.current_device()}")

    d_num_matched = torch.zeros(1, dtype=torch.int64, device=device).to(torch.uint64)
    dynamic_table.count_matched(threshold, d_num_matched)

    gathered_num_matched = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_num_matched, d_num_matched.to(dtype=torch.int64), group=pg)

    total_matched = sum(t.item() for t in gathered_num_matched)  # t is on device.
    key_dtype = dynamic_table.key_type()
    value_dtype = dynamic_table.value_type()
    dim: int = dynamic_table.embedding_dim()

    ret_keys = torch.empty(total_matched, dtype=key_dtype, device="cpu")
    ret_vals = torch.empty(total_matched * dim, dtype=value_dtype, device="cpu")
    ret_offset = 0

    search_offset = 0
    search_capacity = dynamic_table.capacity()
    batch_size = batch_size if batch_size < search_capacity else search_capacity
    logger.debug(
        "[_export_matched_and_gather] rank=%s world_size=%s device=%s threshold=%d count_matched=%d capacity=%d batch_size=%d",
        rank,
        world_size,
        device,
        threshold,
        total_matched,
        search_capacity,
        batch_size,
    )

    d_keys = torch.empty(batch_size, dtype=key_dtype, device=device)
    d_vals = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
    d_count = torch.zeros(1, dtype=torch.int64, device=device).to(torch.uint64)

    # Gather keys and values for all ranks
    gathered_keys = [torch.empty_like(d_keys) for _ in range(world_size)]
    gathered_vals = [torch.empty_like(d_vals) for _ in range(world_size)]
    gathered_counts = [torch.empty_like(d_count, dtype=torch.int64) for _ in range(world_size)]

    while search_offset < search_capacity:
        dynamic_table.export_batch_matched(threshold, batch_size, search_offset, d_count, d_keys, d_vals)

        dist.all_gather(gathered_keys, d_keys, group=pg)
        dist.all_gather(gathered_vals, d_vals, group=pg)
        dist.all_gather(gathered_counts, d_count.to(dtype=torch.int64), group=pg)

        for d_keys_, d_vals_, d_count_ in zip(gathered_keys, gathered_vals, gathered_counts):
            h_count = d_count_.cpu().item()
            ret_keys[ret_offset : ret_offset + h_count] = d_keys_[0:h_count].cpu()
            ret_vals[ret_offset * dim : (ret_offset + h_count) * dim] = d_vals_[0 : h_count * dim].cpu()
            ret_offset += h_count

        search_offset += batch_size
        d_count = torch.zeros(1, dtype=torch.int64, device=device).to(torch.uint64)

    logger.debug(
        "[_export_matched_and_gather] export finished num_keys=%d num_vals=%d", ret_keys.numel(), ret_vals.numel()
    )
    return ret_keys, ret_vals


def _export_matched(
    dynamic_table: KeyValueTable,
    threshold: int,
    batch_size: int = BATCH_SIZE_PER_DUMP,
) -> Tuple[Tensor, Tensor]:
    if not torch_npu.npu.is_available():
        raise ValueError("No available npu device.")
    device = torch.device(f"npu:{torch_npu.npu.current_device()}")
    d_num_matched = torch.zeros(1, dtype=torch.int64, device=device).to(torch.uint64)
    dynamic_table.count_matched(threshold, d_num_matched)

    total_matched = d_num_matched.cpu().item()
    key_dtype = dynamic_table.key_type()
    value_dtype = dynamic_table.value_type()
    dim: int = dynamic_table.embedding_dim()

    ret_keys = torch.empty(total_matched, dtype=key_dtype, device="cpu")
    ret_vals = torch.empty(total_matched * dim, dtype=value_dtype, device="cpu")
    ret_offset = 0

    search_offset = 0
    search_capacity = dynamic_table.capacity()
    batch_size = batch_size if batch_size < search_capacity else search_capacity
    logger.debug(
        "[_export_matched] device=%s threshold=%d count_matched=%d capacity=%d batch_size=%d",
        device,
        threshold,
        total_matched,
        search_capacity,
        batch_size,
    )

    d_keys = torch.empty(batch_size, dtype=key_dtype, device=device)
    d_vals = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
    d_count = torch.zeros(1, dtype=torch.int64, device=device).to(torch.uint64)

    while search_offset < search_capacity:
        dynamic_table.export_batch_matched(threshold, batch_size, search_offset, d_count, d_keys, d_vals)

        h_count = d_count.cpu().item()
        ret_keys[ret_offset : ret_offset + h_count] = d_keys[0:h_count].cpu()
        ret_vals[ret_offset * dim : (ret_offset + h_count) * dim] = d_vals[0 : h_count * dim].cpu()
        ret_offset += h_count

        search_offset += batch_size
        d_count = torch.zeros(1, dtype=torch.int64, device=device).to(torch.uint64)

    logger.debug("[_export_matched] export finished num_keys=%s num_vals=%s", ret_keys.numel(), ret_vals.numel())
    return ret_keys, ret_vals


class BatchedDynamicEmbeddingTablesV2(nn.Module):
    """
    Dynamic Embedding is based on [HKV](https://github.com/NVIDIA-Merlin/HierarchicalKV/tree/master).
    Looks up one or more dynamic embedding tables. The module is application for training.

    Its optional to fuse the optimizer with the backward operator by parameter *update_grads_explicitly*.
    """

    optimizer_args: OptimizerArgs

    def __init__(  # pylint: disable=keyword-arg-before-vararg
        self,
        table_options: List[DynamicEmbTableOptions],
        table_names: Optional[List[str]] = None,
        feature_table_map: Optional[List[int]] = None,  # [T]
        use_index_dedup: bool = False,
        prefetch_pipeline: bool = False,  # we set the arg name same as FBGEMM TBE to align with it
        pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.SUM,
        output_dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        optimizer: EmbOptimType = EmbOptimType.ADAM,
        # General Optimizer args
        stochastic_rounding: bool = True,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        max_norm: float = 0.0,
        learning_rate: float = 0.01,
        eps: float = 1.0e-8,
        initial_accumulator_value: float = 0.0,
        momentum: float = 0.9,  # used by LARS-SGD
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
        eta: float = 0.001,  # used by LARS-SGD,
        beta1: float = 0.9,  # used by LAMB and ADAM
        beta2: float = 0.999,  # used by LAMB and ADAM
        counter_based_regularization: Optional[CounterBasedRegularizationDefinition] = None,  # used by Rowwise Adagrad
        cowclip_regularization: Optional[CowClipDefinition] = None,  # used by Rowwise Adagrad
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if len(table_options) == 0:
            raise ValueError("table_options size must be greater than 0")
        table_option = table_options[0]
        for other_option in table_options:
            if table_option != other_option:
                raise ValueError("All tables must match in grouped keys.")
        self._dynamicemb_options = table_options
        self.initializer_args = table_option.initializer_args
        self.index_type = table_option.index_type
        self.embedding_dtype = table_option.embedding_dtype
        self.output_dtype = output_dtype
        self.pooling_mode = pooling_mode
        self.use_index_dedup = use_index_dedup
        self._enable_prefetch = prefetch_pipeline
        self.prefetch_stream = None
        self.num_prefetch_ahead = 0
        self._table_names = table_names
        self._create_score()
        self.device_id = 0
        if device.type == "npu":
            if device is not None:
                self.device_id = int(str(device)[-1])
            else:
                if not torch_npu.npu.is_available():
                    raise ValueError("No available npu device.")
                self.device_id = torch_npu.npu.current_device()
            if table_option.device_id is None:
                for option in self._dynamicemb_options:
                    option.device_id = self.device_id
        # get npu device config

        self.dims: List[int] = [option.dim for option in self._dynamicemb_options]
        # mixed D is not supported by sequence embedding.
        mixed_D = False
        D = self.dims[0]
        for d in self.dims:
            if d != D:
                mixed_D = True
                break
        if mixed_D:
            if self.pooling_mode == DynamicEmbPoolingMode.NONE:
                raise ValueError("Mixed dimension tables only supported for pooling tables.")

        # physical table number.
        T_ = len(self._dynamicemb_options)
        if T_ == 0:
            raise ValueError(f"T_ ({T_}) must be greater than 0")
        self.feature_table_map: List[int] = feature_table_map if feature_table_map is not None else list(range(T_))
        # logical table number.
        T = len(self.feature_table_map)
        if T_ > T:
            raise ValueError(f"T_ ({T_}) must not be greater than T ({T})! Please ensure T_ ≤ T.")
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        if not all(table_has_feature):
            raise ValueError("Each physical table must appear at least once in feature_table_map!")

        feature_dims = [self.dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(feature_dims))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(self.dims)

        self.feature_num = len(self.feature_table_map)
        self.table_offsets_in_feature: List[int] = []
        old_table_id = -1
        for idx, table_id in enumerate(self.feature_table_map):
            if table_id != old_table_id:
                self.table_offsets_in_feature.append(idx)
                old_table_id = table_id
        self.table_offsets_in_feature.append(self.feature_num)
        self.feature_offsets = torch.tensor(
            self.table_offsets_in_feature,
            device=torch.device(self.device_id),
            dtype=torch.int64,
        )

        for option in self._dynamicemb_options:
            if option.init_capacity is None:
                option.init_capacity = option.max_capacity

        self._optimizer: Union[BaseDynamicEmbeddingOptimizerV2] = None
        optimizer_config = CreateOptimizerConfig(
            optimizer_type=optimizer,
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            max_norm=max_norm,
            learning_rate=learning_rate,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eta=eta,
            momentum=momentum,
            weight_decay_mode=weight_decay_mode,
            counter_based_regularization=counter_based_regularization,
            cowclip_regularization=cowclip_regularization,
        )
        self._create_optimizer(optimizer_config)
        self._storage_externel = table_option.external_storage is not None
        self._create_cache_storage()
        if self.pooling_mode != DynamicEmbPoolingMode.NONE:
            self._tables = []
            for storage in self._storages:
                if not isinstance(storage, KeyValueTable):
                    raise TypeError("The storage should be KeyValueTable when pooling mode is not None.")
                kvtable = cast(KeyValueTable, storage)
                self._tables.append(kvtable.table)
            self._create_bag_optimizer(self._optimizer_type, self._optimizer_args, self._tables)
        self._initializers = []
        self._eval_initializers = []
        self._create_initializers()

        self._empty_tensor = nn.Parameter(
            torch.empty(
                10,
                requires_grad=True,
                device=torch.device(self.device_id),
                dtype=self.embedding_dtype,
            )
        )
        self._unique_op = None

    def _create_cache_storage(self) -> None:
        self._storages: List[Storage] = []
        self._caches: List[Cache] = []
        self._caching = self._dynamicemb_options[0].caching

        for option in self._dynamicemb_options:
            if option.training and option.optimizer_type == OptimizerType.Null:
                option.optimizer_type = convert_optimizer_type(self._optimizer_type)
            elif not option.training and option.optimizer_type != OptimizerType.Null:
                option.optimizer_type = OptimizerType.Null
                warnings.warn("Set OptimizerType to Null as not on training mode.", UserWarning)

            if option.caching and option.training:
                cache_option = deepcopy(option)
                cache_option.bucket_capacity = 1024
                capacity = get_constraint_capacity(
                    option.local_hbm_for_values,
                    option.embedding_dtype,
                    option.dim,
                    option.optimizer_type,
                    cache_option.bucket_capacity,
                )
                if capacity == 0:
                    raise ValueError("Can't use caching mode as the reserved HBM size is too small.")

                cache_option.max_capacity = capacity
                cache_option.init_capacity = capacity
                self._caches.append(KeyValueTable(cache_option, self._optimizer))

                storage_option = deepcopy(option)
                storage_option.local_hbm_for_values = 0
                PS = storage_option.external_storage
                self._storages.append(
                    PS(storage_option, self._optimizer) if PS else KeyValueTable(storage_option, self._optimizer)
                )
            else:
                self._caches.append(None)
                self._storages.append(KeyValueTable(option, self._optimizer))
        _print_memory_consume(self._table_names, self._dynamicemb_options, self._optimizer, self.device_id)

    def _create_initializers(self) -> None:
        def _get_initializer(initializer_args):
            mode = initializer_args.mode
            if mode == DynamicEmbInitializerMode.NORMAL:
                initializer = NormalInitializer(initializer_args)
            elif mode == DynamicEmbInitializerMode.CONSTANT:
                initializer = ConstantInitializer(initializer_args)
            elif mode == DynamicEmbInitializerMode.UNIFORM:
                initializer = UniformInitializer(initializer_args)
            elif mode == DynamicEmbInitializerMode.TRUNCATED_NORMAL:
                initializer = TruncatedNormalInitializer(initializer_args)
            elif mode == DynamicEmbInitializerMode.DEBUG:
                initializer = DebugInitializer(initializer_args)
            else:
                raise ValueError(f"Not supported initializer type({mode}) {type(mode)} {mode.value}.")
            return initializer

        for option in self._dynamicemb_options:
            initializer = _get_initializer(option.initializer_args)
            self._initializers.append(initializer)
            eval_initializer = _get_initializer(option.eval_initializer_args)
            self._eval_initializers.append(eval_initializer)

    def _create_optimizer(
        self,
        config: CreateOptimizerConfig,
    ) -> None:
        self._optimizer_type = config.optimizer_type
        self.stochastic_rounding = config.stochastic_rounding

        self.weight_decay_mode = config.weight_decay_mode
        if (config.weight_decay_mode == WeightDecayMode.COUNTER) != (config.counter_based_regularization is not None):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COUNTER together with valid counter_based_regularization"
            )
        if (config.weight_decay_mode == WeightDecayMode.COWCLIP) != (config.cowclip_regularization is not None):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COWCLIP together with valid cowclip_regularization"
            )

        optimizer_args = OptimizerArgs(
            learning_rate=config.learning_rate,
            eps=config.eps,
            initial_accumulator_value=config.initial_accumulator_value,
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
        self._optimizer_args = optimizer_args
        self._optimizer = self._create_optimizer_instance(config.optimizer_type, optimizer_args)

    def _create_bag_optimizer(
        self,
        optimizer_type: EmbOptimType,
        optimizer_args: OptimizerArgs,
        tables: List[DynamicEmbTable],
    ) -> None:
        if optimizer_type == EmbOptimType.SGD:
            self._bag_optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                tables,
            )
        elif optimizer_type == EmbOptimType.EXACT_SGD:
            self._bag_optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                tables,
            )
        elif optimizer_type == EmbOptimType.ADAM:
            self._bag_optimizer = AdamDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                tables,
            )
        elif optimizer_type == EmbOptimType.EXACT_ADAGRAD:
            self._bag_optimizer = AdagradDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                tables,
            )
        elif optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            self._bag_optimizer = RowWiseAdagradDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                tables,
            )
        else:
            raise ValueError(
                f"Not supported optimizer type ,optimizer type = {optimizer_type} {type(optimizer_type)} {optimizer_type.value}."
            )

    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, _ in enumerate(self._dynamicemb_options):
            splits.append(torch.empty((1, 1), device=torch.device("npu"), dtype=self.embedding_dtype))
        return splits

    def flush(self) -> None:
        self.num_prefetch_ahead = 0
        if self.pooling_mode == DynamicEmbPoolingMode.NONE and self._caching:
            for cache, storage in zip(self._caches, self._storages):
                cache.flush(storage)

    def reset_cache_states(self) -> None:
        if self.pooling_mode == DynamicEmbPoolingMode.NONE and self._caching:
            for cache in self._caches:
                cache.reset()

    @property
    def table_names(self) -> List[str]:
        return self._table_names

    @property
    def optimizer(
        self,
    ) -> Union[BaseDynamicEmbeddingOptimizerV2]:
        return self._optimizer

    @property
    def tables(self) -> List[KeyValueTable]:
        return self._storages

    @property
    def caches(self) -> List[Cache]:
        return self._caches

    def set_record_cache_metrics(self, record: bool) -> None:
        for cache in self._caches:
            cache.set_record_cache_metrics(record)

    def set_learning_rate(self, lr: float) -> None:
        if self.pooling_mode != DynamicEmbPoolingMode.NONE:
            self._bag_optimizer.set_learning_rate(lr)
        else:
            self._optimizer.set_learning_rate(lr)

    @property
    def enable_prefetch(
        self,
    ) -> bool:
        return self._enable_prefetch

    @enable_prefetch.setter
    def enable_prefetch(self, value: bool):
        self._enable_prefetch = value
        self.num_prefetch_ahead = 0

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        grad_weight_config: Optional[EmbGradWeightConfig] = None,
        # 2D tensor of batch size for each rank and feature.
        # Shape (number of features, number of ranks)
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        total_unique_indices: Optional[int] = None,
    ) -> List[Tensor]:
        if self._enable_prefetch:
            self.num_prefetch_ahead -= 1

        if indices.dtype != self.index_type:
            indices = indices.to(self.index_type)

        if any(not option.training for option in self._dynamicemb_options) and self.training:
            raise RuntimeError(
                "BatchedDynamicEmbeddingTables does not support training when some tables are in eval mode."
            )

        scores = []
        for table_name in self._table_names:
            if table_name not in self._scores:
                raise RuntimeError(f"Must set score for table '{table_name}' whose score_strategy is customized.")
            scores.append(self._scores[table_name])

        if self.pooling_mode == DynamicEmbPoolingMode.NONE:
            for i, cache in enumerate(self._caches):
                if isinstance(cache, KeyValueTable):
                    table = cast(KeyValueTable, cache)
                    table.score_update = True
                    table.set_score(self._scores[self.table_names[i]])
            for i, storage in enumerate(self._storages):
                if isinstance(storage, KeyValueTable):
                    table = cast(KeyValueTable, storage)
                    # if not training and not caching, we don't need to update score.
                    table.score_update = self.training or self._caching
                    table.set_score(self._scores[self.table_names[i]])

            emb_config = DynamicEmbeddingFunctionV2Config(
                indices=indices,
                offsets=offsets,
                caches=self._caches,
                storages=self._storages,
                feature_offsets=self.feature_offsets,
                output_dtype=self.output_dtype,
                initializers=self._initializers if self.training else self._eval_initializers,
                optimizer=self._optimizer,
                enable_prefetch=self._enable_prefetch,
                input_dist_dedup=self.use_index_dedup,
                training=self.training,
            )
            res = DynamicEmbeddingFunctionV2.apply(
                emb_config,
                self._empty_tensor,
            )
            for cache in self._caches:
                if isinstance(cache, KeyValueTable):
                    table = cast(KeyValueTable, cache)
                    table.score_update = False
            for storage in self._storages:
                if isinstance(storage, KeyValueTable):
                    table = cast(KeyValueTable, storage)
                    table.score_update = False
        else:
            res = DynamicEmbeddingBagFunction.apply(
                indices,
                offsets,
                self.use_index_dedup,
                self.table_offsets_in_feature,
                self._tables,
                scores,
                self.total_D,
                self.dims,
                self.feature_table_map,
                self.embedding_dtype,
                self.output_dtype,
                self.pooling_mode,
                self._unique_op,
                torch.device(self.device_id),
                self._bag_optimizer,
                self.training,
                [option.eval_initializer_args for option in self._dynamicemb_options],
                self._empty_tensor,
            )
        # We have to update cache's core in eval mode.
        if self.training or self._caching:
            self._update_score()

        return res

    def prefetch(
        self,
        indices: Tensor,
        offsets: Tensor,
        forward_stream: Optional[torch.npu.Stream] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> None:
        if self.pooling_mode != DynamicEmbPoolingMode.NONE:
            raise RuntimeError("only support prefetch for sequence embedding.")
        if not self._enable_prefetch:
            raise RuntimeError("Prefetch is not enabled.")
        if not self._caching:
            logging.warning("Caching is not enabled, prefetch will do nothing.")
        if self.prefetch_stream is None and forward_stream is not None:
            # Set the prefetch stream to the current stream
            self.prefetch_stream = torch.npu.current_stream()
            if self.prefetch_stream == forward_stream:
                raise RuntimeError("prefetch_stream and forward_stream should not be the same stream.")

            current_stream = torch.npu.current_stream()
            # Record tensors on the current stream
            indices.record_stream(current_stream)
            offsets.record_stream(current_stream)

        if self._enable_prefetch:
            self.num_prefetch_ahead += 1
            if self.num_prefetch_ahead < 1:
                raise RuntimeError("Prefetch context mismatches.")

        prefetch_scores = self._get_prefetch_score()

        for i, cache in enumerate(self._caches):
            if isinstance(cache, KeyValueTable):
                table = cast(KeyValueTable, cache)
                table.score_update = True
                table.set_score(prefetch_scores[i])
        for i, storage in enumerate(self._storages):
            if isinstance(storage, KeyValueTable):
                table = cast(KeyValueTable, storage)
                table.score_update = True
                table.set_score(prefetch_scores[i])

        dynamicemb_prefetch(
            indices,
            offsets,
            self._caches,
            self._storages,
            self.feature_offsets,
            self._initializers if self.training else self._eval_initializers,
            self._unique_op,
            self.training,
            forward_stream,
        )

        for cache in self._caches:
            if isinstance(cache, KeyValueTable):
                table = cast(KeyValueTable, cache)
                table.score_update = False
        for storage in self._storages:
            if isinstance(storage, KeyValueTable):
                table = cast(KeyValueTable, storage)
                table.score_update = False

    def set_score(
        self,
        named_score: Dict[str, int],
    ) -> None:
        table_names: List[str] = named_score.keys()
        table_scores: List[int] = named_score.values()
        for table_name, table_score in zip(table_names, table_scores):
            if not isinstance(table_score, int):
                raise ValueError(f"Table's score is expect to int but got {type(table_score)}")
            if table_score == 0:
                raise ValueError("Can't set table's score to 0.")
            index = self._table_names.index(table_name)
            if self._dynamicemb_options[index].score_strategy != DynamicEmbScoreStrategy.CUSTOMIZED:
                raise ValueError(
                    "Can only set score for table whose score_strategy is DynamicEmbScoreStrategy.CUSTOMIZED."
                )

            if table_name in self._scores and self._scores[table_name] > table_score:
                if warning_for_cstm_score():
                    warnings.warn(
                        f"New set score is less than the old one for table '{table_name}': "
                        f"{table_score} < {self._scores[table_name]}",
                        UserWarning,
                    )
            self._scores[table_name] = table_score

    def get_score(self) -> Dict[str, int]:
        return self._scores.copy()

    def _create_score(self):
        self._scores: Dict[str, int] = {}
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                option.evict_strategy = DynamicEmbEvictStrategy.LRU
                self._scores[table_name] = device_timestamp()
            elif option.score_strategy == DynamicEmbScoreStrategy.STEP:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
                self._scores[table_name] = 1
            elif option.score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
            elif option.score_strategy == DynamicEmbScoreStrategy.LFU:
                option.evict_strategy = DynamicEmbEvictStrategy.LFU
                self._scores[table_name] = 1

    def _update_score(self):
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            old_score = self._scores[table_name]
            if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                new_score = device_timestamp()
                if new_score < old_score:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is less than old one({old_score}).",
                        UserWarning,
                    )
                self._scores[table_name] = new_score
            elif option.score_strategy == DynamicEmbScoreStrategy.STEP:
                max_step_score = torch.iinfo(torch.int64).max
                new_score = old_score + 1
                if new_score > max_step_score:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is out of range, reset to 0.",
                        UserWarning,
                    )
                    self._scores[table_name] = 0
                else:
                    self._scores[table_name] = new_score
            elif option.score_strategy == DynamicEmbScoreStrategy.LFU:
                self._scores[table_name] = 1

    def _get_prefetch_score(
        self,
    ):
        ret_scores = []
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            cur_score = self._scores[table_name]
            if self.enable_prefetch and option.score_strategy == DynamicEmbScoreStrategy.STEP:
                max_uint64 = (2**64) - 1
                new_score = cur_score + self.num_prefetch_ahead - 1
                if new_score > max_uint64:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is out of range, reset to 0.",
                        UserWarning,
                    )
                    new_score = 0
            else:
                new_score = cur_score

            ret_scores.append(new_score)
        return ret_scores

    def dump(
        self,
        save_dir: str,
        optim: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        if table_names is None:
            table_names = self._table_names

        if pg is None:
            if not dist.is_initialized():
                raise RuntimeError("Distributed is not initialized.")
            pg = dist.group.WORLD
        rank = dist.get_rank(group=pg)
        world_size = dist.get_world_size(group=pg)

        self.flush()
        for table_name, storage in zip(self._table_names, self._storages):
            if table_name not in set(table_names):
                continue

            meta_file_path = encode_meta_json_file_path(save_dir, table_name)
            emb_key_path = encode_checkpoint_file_path(save_dir, table_name, rank, world_size, "keys")
            emb_value_path = encode_checkpoint_file_path(save_dir, table_name, rank, world_size, "values")
            emb_score_path = encode_checkpoint_file_path(save_dir, table_name, rank, world_size, "scores")
            opt_value_path = encode_checkpoint_file_path(save_dir, table_name, rank, world_size, "opt_values")

            if isinstance(storage, KeyValueTable) and not storage._use_score:
                dist.barrier()  # sync global timestamp
                cast(KeyValueTable, storage).update_timestamp()
            optional_files = StorageOptionalFilePaths(score_file_path=emb_score_path, opt_file_path=opt_value_path)
            storage.dump(
                meta_file_path,
                emb_key_path,
                emb_value_path,
                optional_files,
                include_optim=optim,
                include_meta=(rank == 0),
            )

    def load(
        self,
        save_dir: str,
        optim: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ):
        if table_names is None:
            table_names = self._table_names

        if pg is None and not dist.is_initialized():  # for inference load
            rank = 0
            world_size = 1
        else:
            rank = dist.get_rank(group=pg)
            world_size = dist.get_world_size(group=pg)

        for table_name, storage in zip(self._table_names, self._storages):
            if table_name not in set(table_names):
                continue
            (
                emb_key_files,
                emb_value_files,
                emb_score_files,
                opt_value_files,
            ) = get_loading_files(
                save_dir,
                table_name,
                rank=rank,
                world_size=world_size,
            )
            meta_json_file = encode_meta_json_file_path(save_dir, table_name)

            if isinstance(storage, KeyValueTable) and not storage._use_score:
                cast(KeyValueTable, storage).update_timestamp()
            num_key_files = len(emb_key_files)
            for i in range(num_key_files):
                optional_files = StorageOptionalFilePaths(
                    score_file_path=emb_score_files[i] if len(emb_score_files) > 0 else None,
                    opt_file_path=opt_value_files[i] if len(opt_value_files) > 0 else None,
                )
                storage.load(
                    meta_json_file,
                    emb_key_files[i],
                    emb_value_files[i],
                    include_optim=optim,
                    optional_files=optional_files,
                )

    def export_keys_values(
        self, table_name: str, device: torch.device, batch_size: int = 65536
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys_list = []
        values_list = []
        self.flush()
        for dynamic_table_name, dynamic_table in zip(self.table_names, self.tables):
            if table_name != dynamic_table_name:
                continue

            for keys, embeddings, _, _ in batched_export_keys_values(dynamic_table.table, device, batch_size):
                keys_list.append(keys)
                values_list.append(embeddings)

        return torch.cat(keys_list), torch.cat(values_list, dim=0)

    def _create_optimizer_instance(
        self, optimizer_type: EmbOptimType, optimizer_args: OptimizerArgs
    ) -> BaseDynamicEmbeddingOptimizerV2:
        if optimizer_type == EmbOptimType.ADAM:
            return AdamDynamicEmbeddingOptimizerV2(optimizer_args)
        elif optimizer_type == EmbOptimType.ADAMW:
            return AdamWDynamicEmbeddingOptimizerV2(optimizer_args)
        elif optimizer_type == EmbOptimType.SGD:
            return SGDDynamicEmbeddingOptimizerV2(optimizer_args)
        elif optimizer_type == EmbOptimType.EXACT_SGD:
            return SGDDynamicEmbeddingOptimizerV2(optimizer_args)
        elif optimizer_type == EmbOptimType.EXACT_ADAGRAD:
            return AdagradDynamicEmbeddingOptimizerV2(optimizer_args)
        elif optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            return RowWiseAdagradDynamicEmbeddingOptimizerV2(optimizer_args, emb_dtype=self.embedding_dtype)
        else:
            raise ValueError(
                f"Not supported optimizer type, optimizer type = {optimizer_type} {type(optimizer_type)} "
                f"{optimizer_type.value}."
            )

    def incremental_dump(
        self,
        named_thresholds: Dict[str, int] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[Dict[str, Tuple[Tensor, Tensor]], Dict[str, int]]:
        if named_thresholds is None:
            named_thresholds = {}
        table_names: List[str] = list(named_thresholds.keys())
        table_thresholds: List[int] = list(named_thresholds.values())
        ret_tensors: Dict[str, Tuple[Tensor, Tensor]] = {}
        ret_scores: Dict[str, int] = {}

        def _export_matched_per_table(pg, table, threshold):
            if not dist.is_initialized() or dist.get_world_size(group=pg) == 1:
                key, value = _export_matched(table, threshold)
            else:
                key, value = _export_matched_and_gather(table, threshold, pg)
            return key, value

        for table_name, threshold in zip(table_names, table_thresholds):
            index = self._table_names.index(table_name)

            storage = self._storages[index]
            if not isinstance(storage, KeyValueTable):
                raise RuntimeError("Only KeyValueTable is supported for incremental dump")
            key, value = _export_matched_per_table(pg, storage, threshold)
            logger.debug("[incremental_dump] table=%s storage export num_keys=%d", table_name, key.numel())

            if self._caches[index] is not None:
                cache = self._caches[index]
                key_c, value_c = _export_matched_per_table(pg, cache, threshold)
                logger.debug("[incremental_dump] table=%s cache export num_keys=%d", table_name, key_c.numel())

                mask = ~torch.isin(key, key_c)
                if key.numel() != 0:
                    if mask.sum() != 0:
                        value = value.view(key.numel(), -1)[mask, :].contiguous().view(-1)
                        key = key[mask].contiguous()
                        key = torch.cat((key_c, key), dim=0).contiguous()
                        value = torch.cat((value_c, value), dim=0).contiguous()
                    else:
                        key = key_c
                        value = value_c
                else:
                    key = key_c
                    value = value_c

            ret_tensors[table_name] = (key, value)
            ret_scores[table_name] = self._scores[table_name]
        return ret_tensors, ret_scores

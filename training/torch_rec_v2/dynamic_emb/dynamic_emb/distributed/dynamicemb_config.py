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

import enum
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from torchrec.tensor_types import check

from rec_sdk_common.constants.constants import NumCheckValueMethod
from rec_sdk_common.validator.safe_checker import class_safe_check, float_safe_check, int_safe_check

from dynamic_emb_extensions import (
    InitializerArgs,
    OptimizerType,
    EvictStrategy,
    DynamicEmbTable,
    DynamicEmbDataType,
)
from dynamic_emb.distributed.types import Storage


DEFAULT_INDEX_TYPE = torch.int64
DynamicEmbKernel = "DynamicEmb"
DEFAULT_BATCH_SIZE = 512
MIN_BATCH_SIZE = 0
MAX_BATCH_SIZE = 1000000
DTYPE_NUM_BYTES: Dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}
_MAX_INIT_ARGS_VALUE = 1.0
_MIN_INIT_ARGS_VALUE = 0.0


class DynamicEmbPoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2


class DynamicEmbInitializerMode(enum.Enum):
    """
    Enumeration for different modes of initializing dynamic embedding vector values.

    Attributes
    ----------
    NORMAL : str
        Normal Distribution.
    TRUNCATED_NORMAL : str
        TRUNCATED NORMAL distribution of random values.
    UNIFORM : str
        Uniform distribution of random values.
    CONSTANT : str
        All dynamic embedding vector values are a given constant.
    DEBUG : str
        Debug value generation mode for testing.
    """

    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    DEBUG = "debug"


def dyn_emb_to_torch(data_type: DynamicEmbDataType) -> torch.dtype:
    if data_type == DynamicEmbDataType.Float32:
        return torch.float32
    elif data_type == DynamicEmbDataType.BFloat16:
        return torch.bfloat16
    elif data_type == DynamicEmbDataType.Float16:
        return torch.float16
    elif data_type == DynamicEmbDataType.Int64:
        return torch.int64
    elif data_type == DynamicEmbDataType.Int32:
        return torch.int32
    elif data_type == DynamicEmbDataType.Size_t:
        return torch.int64  # Size_t to int64
    else:
        raise ValueError(f"Unsupported DynamicEmbDataType: {data_type}")


def torch_to_dyn_emb(torch_dtype: torch.dtype) -> DynamicEmbDataType:
    if torch_dtype == torch.float32:
        return DynamicEmbDataType.Float32
    elif torch_dtype == torch.bfloat16:
        return DynamicEmbDataType.BFloat16
    elif torch_dtype == torch.float16:
        return DynamicEmbDataType.Float16
    elif torch_dtype == torch.int64:
        return DynamicEmbDataType.Int64
    elif torch_dtype == torch.int32:
        return DynamicEmbDataType.Int32
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


@enum.unique
class DynamicEmbEvictStrategy(enum.Enum):
    LRU = EvictStrategy.kLru
    LFU = EvictStrategy.kLfu
    EPOCH_LRU = EvictStrategy.kEpochLru
    EPOCH_LFU = EvictStrategy.kEpochLfu
    CUSTOMIZED = EvictStrategy.kCustomized


@dataclass
class DynamicEmbInitializerArgs:
    """
    Arguments for initializing dynamic embedding vector values.

    Attributes
    ----------
    mode : DynamicEmbInitializerMode
        The mode of initialization, one of the DynamicEmbInitializerMode values.
    mean : float, optional
        The mean value for (truncated) normal distributions. Defaults to 0.0.
    std_dev : float, optional
        The standard deviation for (truncated) normal distributions. Defaults to 1.0.
    lower : float, optional
        The lower bound for uniform/truncated_normal distribution. Defaults to 0.0.
    upper : float, optional
        The upper bound for uniform/truncated_normal distribution. Defaults to 1.0.
    value : float, optional
        The constant value for constant initialization. Defaults to 0.0.
    """

    mode: DynamicEmbInitializerMode = DynamicEmbInitializerMode.NORMAL
    mean: float = 0.0
    std_dev: float = 1.0
    lower: float = 0.0
    upper: float = 1.0
    value: float = 0.0

    def __post_init__(self):
        class_safe_check("mode", self.mode, (DynamicEmbInitializerMode,))
        float_safe_check("mean", self.mean, min_value=_MIN_INIT_ARGS_VALUE, max_value=_MAX_INIT_ARGS_VALUE)
        float_safe_check("std_dev", self.std_dev, min_value=_MIN_INIT_ARGS_VALUE, max_value=_MAX_INIT_ARGS_VALUE)
        float_safe_check("lower", self.lower, min_value=_MIN_INIT_ARGS_VALUE, max_value=_MAX_INIT_ARGS_VALUE)
        float_safe_check("upper", self.upper, min_value=_MIN_INIT_ARGS_VALUE, max_value=_MAX_INIT_ARGS_VALUE)
        float_safe_check("value", self.value, min_value=_MIN_INIT_ARGS_VALUE, max_value=_MAX_INIT_ARGS_VALUE)

    def __eq__(self, other):
        if not isinstance(other, DynamicEmbInitializerArgs):
            return NotImplementedError
        if self.mode == DynamicEmbInitializerMode.NORMAL:
            return self.mean == other.mean and self.std_dev == other.std_dev
        return True

    def __ne__(self, other):
        if not isinstance(other, DynamicEmbInitializerArgs):
            return NotImplementedError
        return not (self == other)

    def as_ctype(self) -> InitializerArgs:
        return InitializerArgs(
            self.mode.value,
            self.mean,
            self.std_dev,
            self.lower,
            self.upper,
            self.value,
        )


class DynamicEmbScoreStrategy(enum.IntEnum):
    """
    Enumeration for different modes to set index-embedding's score.
    The index-embedding pair with smaller scores will be more likely to be evicted from the embedding table 
    when the table is full.

    dynamicemb allows configuring scores by table.
    For a table, the scores in the subsequent forward passes are larger than those 
    in the previous ones for modes TIMESTAMP and STEP.
    Users can also provide customized score(mode CUSTOMIZED) for each table's forward pass.
    Attributes
    ----------
    TIMESTAMP:
        In a forward pass, embedding table's scores will be set to global nanosecond timer of device,
        and due to the timing of NPU scheduling, different scores may have slight differences.
        Users must not set scores under TIMESTAMP mode.
    STEP:
        Each embedding table has a member `step` which will increment for every forward pass.
        All scores in each forward pass are the same which is step's value.
        Users must not set scores under STEP mode.
    CUSTOMIZED:
        Each embedding table's score are managed by users.
        Users have to set the score before every forward pass using `set_score` interface.
    LFU:
        Least frequently used.
    """

    TIMESTAMP = 0
    STEP = 1
    CUSTOMIZED = 2
    LFU = 3


@enum.unique
class DynamicEmbCheckMode(enum.IntEnum):
    """
    Enumeration for different modes of checking dynamic embedding's insertion behaviors.
    DynamicEmb uses a hashtable as the backend. 
    If the embedding table capacity is small and the number of indices in a single lookup is large,
    it is easy for too many indices to be allocated to the same hash table bucket in one lookup, 
    resulting in the inability to insert indices into the hashtable.
    DynamicEmb resolves this issue by setting the lookup results of indices that cannot be inserted to 0.
    Fortunately, in a hashtable with a large capacity, such insertion failures are very rare and almost never occur.
    This issue is more frequent in hashtables with small capacities, which can affect training accuracy.
    Therefore, we do not recommend using dynamic embedding tables for very small embedding tables.

    To prevent this behavior from affecting training without user awareness, DynamicEmb provides a safe check mode.
    Users can set whether to enable safe check when configuring DynamicEmbTableOptions.
    Enabling safe check will add some overhead, 
    but it can provide insights into whether the hash table frequently fails to insert indices.
    If the number of insertion failures is high and the proportion of affected indices is large,
    it is recommended to either increase the dynamic embedding capacity 
    or avoid using dynamic embedding tables for small embedding tables.

    Attributes
    ----------
    ERROR : int
        When there are indices that can't be inserted successfully:
            This mode will throw a runtime error indicating how many indices failed to insert.
            The program will crash.
    WARNING : int
        When there are indices that can't be inserted successfully:
            This mode will give a warning about how many indices failed to insert.
            The program will continue. For uninserted indices, their embeddings' values will be set to 0.0.
    IGNORE : int
        Don't check whether insertion is successful or not, therefore it doesn't bring additional checking overhead.
        For uninserted indices, their embeddings' values will be set to 0.0 silently.
    """

    ERROR = 0
    WARNING = 1
    IGNORE = 2


@dataclass
class _ContextOptions:
    """
    Parameters in InternalConfigs is not configurable when using dynamicemb by DistributedModelParallel.
    Internal Configurations that including three parts:

    1. Fixed
        score_type : torch.dtype
            Score represents how important an embedding item is. This specifies the type of the score.

    2. Inferred from the context: from params already defined in torchrec, or from the runtime env, e.g. world size.
        embedding_dtype : Optional[torch.dtype], optional
            Data (weight) type of dynamic embedding table.
        dim : Optional[int], optional
            The dimensionality of the value vectors. Default is -1, indicating it should be set explicitly.
        max_capacity : Optional[int], optional
                The maximum capacity of the shard of the embedding table on a single GPU. 
                Automatically set in the shared planner.
                It is not configurable, but it's important for the total memory consumption.
                It will be automatically inferred from EmbeddingConfig.num_embeddings and the world size, 
                rounded up to a power of 2，
                    and minimized to the size of bucket capacity of the HKV.
                If init_capacity is set, max_capacity will not be smaller than init_capacity.
        evict_strategy : DynamicEmbEvictStrategy
            Strategy used for evicting entries when the table exceeds its capacity. 
            Default is DynamicEmbEvictStrategy.LRU.
        local_hbm_for_values : int
            High-bandwidth memory allocated for local values, in bytes. Default is 0.
        num_aligned_embedding_per_rank: int
                Number of aligned embedding per rank when the `num_embeddings` does not meet our alignment requirements, 
                default to None.
        device_id : Optional[int], optional
                CUDA device index.
        optimizer_type: OptimizerType, used internally to determine how much memory optimizer states will consume,
            and default to `OptimizerType.Null`.

    3. Will be removed in the feature.
        block_size : int
            The size of blocks used during operations. Default is 128.
        io_block_size : int
            The size of input/output blocks during data transfer operations. Default is 1024.
        io_by_cpu : bool
            Flag indicating whether to use CPU for handling IO operations. Default is False.
        use_constant_memory : bool
            Flag to indicate if constant memory should be utilized. Default is False.
        reserved_key_start_bit : int
            Bit offset for reserved keys in the key space. Default is 0.
        num_of_buckets_per_alloc : int
            Number of buckets allocated per memory allocation request. Default is 1.

    """

    # Fixed.
    score_type: torch.dtype = torch.int32

    # Inferred from the context.
    embedding_dtype: Optional[torch.dtype] = None
    dim: Optional[int] = None
    max_capacity: Optional[int] = None
    evict_strategy: DynamicEmbEvictStrategy = DynamicEmbEvictStrategy.LRU
    local_hbm_for_values: int = 0  # in bytes
    num_aligned_embedding_per_rank: int = None
    device_id: Optional[int] = None
    optimizer_type: OptimizerType = OptimizerType.Null

    # Will be removed in the future, please ignore them.
    block_size: int = 128
    io_block_size: int = 1024
    io_by_cpu: bool = False  # use cpu to deal with the value copy.
    use_constant_memory: bool = False
    reserved_key_start_bit: int = 0
    num_of_buckets_per_alloc: int = 1


@dataclass
class DynamicEmbTableOptions(_ContextOptions):
    training: bool = True
    initializer_args: DynamicEmbInitializerArgs = field(default_factory=DynamicEmbInitializerArgs)
    eval_initializer_args: DynamicEmbInitializerArgs = field(
        default_factory=lambda: DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.NORMAL,
            value=0.0,
        )
    )
    init_capacity: Optional[int] = None  # if not set then set to max_capacity after sharded
    max_load_factor: float = 0.5  # max load factor before rehash(double capacity)
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.TIMESTAMP
    bucket_capacity: int = 128
    safe_check_mode: DynamicEmbCheckMode = DynamicEmbCheckMode.IGNORE
    # 若为默认值0，则在planner中会被设置为：值类型的字节数 * 表的行数(对齐到2的幂次) * 表的列数
    global_hbm_for_values: int = 0  # in bytes

    caching: bool = False
    external_storage: Storage = None
    index_type: torch.dtype = torch.int64

    def __post_init__(self):
        class_safe_check("training", self.training, (bool,))
        class_safe_check("initializer_args", self.initializer_args, (DynamicEmbInitializerArgs,))
        class_safe_check("eval_initializer_args", self.eval_initializer_args, (DynamicEmbInitializerArgs,))
        if self.eval_initializer_args.mode != DynamicEmbInitializerMode.NORMAL:
            raise ValueError("eval_initializer_args must be constant initialization")
        class_safe_check("init_capacity", self.init_capacity, (int, type(None)))
        if self.init_capacity == 0:
            self.init_capacity = None
        if self.init_capacity is not None:
            int_safe_check("init_capacity", self.init_capacity, min_value=0)
            target_init_capacity = next_power_of_2(self.init_capacity)
            if self.init_capacity != target_init_capacity:
                self.init_capacity = target_init_capacity
        float_safe_check(
            "max_load_factor",
            self.max_load_factor,
            min_value=0.0,
            max_value=1.0,
            method=NumCheckValueMethod.OPEN_INTERVAL.value,
        )
        class_safe_check("score_strategy", self.score_strategy, (DynamicEmbScoreStrategy,))
        int_safe_check("bucket_capacity", self.bucket_capacity, min_value=2, max_value=1024)
        target_bucket_capacity = next_power_of_2(self.bucket_capacity)
        if self.bucket_capacity != target_bucket_capacity:
            self.bucket_capacity = target_bucket_capacity
        class_safe_check("safe_check_mode", self.safe_check_mode, (DynamicEmbCheckMode,))
        int_safe_check("global_hbm_for_values", self.global_hbm_for_values, min_value=0)

        class_safe_check("caching", self.caching, (bool,))
        check(not self.caching, "caching should be False")
        check(self.external_storage is None, "external_storage should be None")
        check(self.index_type == torch.int64, "index_type should be torch.int64")

    def __eq__(self, other):
        if not isinstance(other, DynamicEmbTableOptions):
            return NotImplementedError
        self_group_keys = self.get_grouped_key()
        other_group_keys = other.get_grouped_key()
        return self_group_keys == other_group_keys

    def __ne__(self, other):
        if not isinstance(other, DynamicEmbTableOptions):
            return NotImplementedError
        return not (self == other)

    def get_grouped_key(self):
        grouped_key = {
            "training": self.training,
            "caching": self.caching,
            "external_storage": self.external_storage,
            "index_type": self.index_type,
        }
        return grouped_key

    def __hash__(self):
        group_keys = self.get_grouped_key()
        return hash(tuple(group_keys.items()))


class DistType(enum.Enum):
    CONTINUOUS = "continuous"
    ROUNDROBIN = "roundrobin"


def next_power_of_2(n: int) -> int:
    # Handle the case where n is 0
    if n == 0:
        return 1

    # If n is already a power of 2, return n
    if (n & (n - 1)) == 0:
        return n

    # Find the next power of 2
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32  # This line is necessary for 64-bit integers
    return n + 1


def create_dynamicemb_table(table_options: DynamicEmbTableOptions) -> DynamicEmbTable:
    if not table_options.training:
        table_options.optimizer_type = OptimizerType.Null
    return DynamicEmbTable(
        torch_to_dyn_emb(table_options.index_type),
        torch_to_dyn_emb(table_options.embedding_dtype),
        table_options.evict_strategy.value,
        table_options.dim,
        table_options.init_capacity,
        table_options.max_capacity,
        # We are verifying the logic of this section and will remove the multiplication by 10 later
        table_options.local_hbm_for_values * 10,
        table_options.bucket_capacity,
        table_options.max_load_factor,
        table_options.block_size,
        table_options.io_block_size,
        table_options.device_id,
        table_options.io_by_cpu,
        table_options.use_constant_memory,
        table_options.reserved_key_start_bit,
        table_options.num_of_buckets_per_alloc,
        table_options.initializer_args.as_ctype(),
        table_options.safe_check_mode.value,
        table_options.optimizer_type,
    )


def get_optimizer_state_dim(optimizer_type: OptimizerType, dim: int, dtype: torch.dtype) -> int:
    if optimizer_type == OptimizerType.RowWiseAdaGrad:
        # 16 is row-wise-adagrad optimizer dim
        return 16 // DTYPE_NUM_BYTES[dtype]
    if optimizer_type == OptimizerType.Adam:
        return dim * 2  # m and v
    if optimizer_type == OptimizerType.AdaGrad:
        return dim
    return 0

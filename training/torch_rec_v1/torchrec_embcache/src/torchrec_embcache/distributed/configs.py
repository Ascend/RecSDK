#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Type
import torch

from torchrec import EmbeddingConfig, EmbeddingBagConfig
from hybrid_torchrec.constants import (
    MAX_MULTI_HOT_SIZE,
    MAX_NUM_TABLES,
    MAX_WORLD_SIZE,
    MAX_BATCH_SIZE
)
from hybrid_torchrec.modules.hash_embeddingbag import check_embedding_config_valid


_DEFAULT_ADMIT_THRESHOLD: int = -1
_DEFAULT_EVICT_THRESHOLD: int = 0


@dataclass
class AdmitAndEvictConfig:
    """
    AdmitAndEvictConfig is a dataclass that represents an admit and evict config of a single embedding table.

    Args:
        admit_threshold (Optional[int]): feature admit threshold. Feature (which after input dist) will be admitted
            when repeat time is greater than `admit_threshold`.
            Default value is -1, and indicates that feature admit function is not enabled.
        not_admitted_default_value (Optional[float]): the embedding value of not admitted feature ids.
            Default value is 0.0, and take effect only when `admit_threshold` is a non-default value.
        evict_threshold (Optional[int]): feature evict threshold, unit: seconds.
            Default value is 0, and indicates that feature evict function is not enabled.
        evict_step_interval(Optional[int]): the step interval of feature evict function.
            Default value is 0, and take effect only when `evict_threshold` is a non-default value.
    """

    admit_threshold: Optional[int] = _DEFAULT_ADMIT_THRESHOLD
    not_admitted_default_value: Optional[float] = 0.0

    evict_threshold: Optional[int] = _DEFAULT_EVICT_THRESHOLD  # unit: seconds
    evict_step_interval: Optional[int] = 0

    def __post_init__(self):
        """后置校验函数，确保参数类型和值的正确性"""
        # 校验admit_threshold类型和值
        if self.admit_threshold is not None:
            if not isinstance(self.admit_threshold, int):
                raise TypeError(
                    f"admit_threshold must be int or None, but got {type(self.admit_threshold)}"
                )
            if self.admit_threshold < -1:
                raise ValueError(
                    f"admit_threshold must be >= -1, but got {self.admit_threshold}"
                )

        # 校验not_admitted_default_value类型和值
        if self.not_admitted_default_value is not None:
            if not isinstance(self.not_admitted_default_value, float):
                raise TypeError(
                    f"not_admitted_default_value must be float or None, but got {type(self.not_admitted_default_value)}"
                )

        # 校验evict_threshold类型和值
        if self.evict_threshold is not None:
            if not isinstance(self.evict_threshold, int):
                raise TypeError(
                    f"evict_threshold must be int or None, but got {type(self.evict_threshold)}"
                )
            if self.evict_threshold < 0:
                raise ValueError(
                    f"evict_threshold must be >= 0, but got {self.evict_threshold}"
                )

        # 校验evict_step_interval类型和值
        if self.evict_step_interval is not None:
            if not isinstance(self.evict_step_interval, int):
                raise TypeError(
                    f"evict_step_interval must be int or None, but got {type(self.evict_step_interval)}"
                )
            if self.evict_step_interval < 0:
                raise ValueError(
                    f"evict_step_interval must be >= 0, but got {self.evict_step_interval}"
                )

    def is_feature_admit_enabled(self) -> bool:
        return self.admit_threshold != _DEFAULT_ADMIT_THRESHOLD

    def is_feature_evict_enabled(self) -> bool:
        return self.evict_threshold != _DEFAULT_EVICT_THRESHOLD

    def is_feature_filter_enabled(self) -> bool:
        return self.is_feature_admit_enabled() or self.is_feature_evict_enabled()


class InitializerType(str, Enum):
    LINEAR = "linear"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"


def check_valid_value(is_valid: bool, message: str):
    if not is_valid:
        raise ValueError(message)


def check_embedding_optimizer(optimizer: Type[torch.optim.Optimizer]):
    if optimizer not in [torch.optim.Adagrad, torch.optim.Adam, torch.optim.SGD]:
        raise ValueError(
            f"The optimizer should be one of [torch.optim.Adagrad, torch.optim.Adam, torch.optim.SGD]"
        )


def check_multi_hot_sizes(multi_hot_sizes: List[int], tables: List[EmbeddingBagConfig | EmbeddingConfig]):
    if not isinstance(multi_hot_sizes, list):
        raise ValueError(f"The 'multi_hot_sizes' should be a list")
    if not isinstance(tables, list):
        raise ValueError(f"The 'tables' should be a list")
    if len(tables) <= 0 or len(tables) > MAX_NUM_TABLES:
        raise ValueError(
            f"The length of tables should be in range: [1, {MAX_NUM_TABLES}]"
        )
    if len(multi_hot_sizes) != len(tables):
        raise ValueError(
            f"The multi_hot_sizes length should be equal to the length of tables"
        )
    for hot_size in multi_hot_sizes:
        if not type(hot_size) is int:
            raise ValueError(f"The multi_hot_sizes should be a list of int")
        if not (1 <= hot_size <= MAX_MULTI_HOT_SIZE):
            raise ValueError(
                f"The multi_hot_sizes element value should be in [1, {MAX_MULTI_HOT_SIZE}]"
            )


def check_create_table_params(batch_size, embedding_optimizer_cls, multi_hot_sizes, tables, world_size):
    check_valid_value(
        type(world_size) is int and 0 < world_size <= MAX_WORLD_SIZE,
        f"world_size must be greater than 0 and less than or equal to {MAX_WORLD_SIZE}",
    )
    
    for config in tables:
        check_embedding_config_valid(config)
    if not any(config.num_embeddings >= world_size for config in tables):
        raise ValueError(
            f"The max num_embeddings should be greater than world_size, "
            f"but all configs have num_embeddings < {world_size}"
        )
    check_embedding_optimizer(embedding_optimizer_cls)
    
    check_valid_value(
        type(batch_size) is int and 0 < batch_size <= MAX_BATCH_SIZE,
        f"batch_size must be greater than 0 and less than or equal to {MAX_BATCH_SIZE}",
    )
    check_valid_value(multi_hot_sizes is not None, "multi_hot_sizes must be not None")


@dataclass
class EmbCacheEmbeddingBagConfig(EmbeddingBagConfig):
    weight_init_mean: Optional[float] = 0.0  # used for InitializerType.UNIFORM
    weight_init_stddev: Optional[float] = 0.05  # used for InitializerType.UNIFORM
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )
    is_incremental: bool = False

    def __post_init__(self):
        check_embedding_config_valid(self)
        super().__post_init__()


@dataclass
class EmbCacheEmbeddingConfig(EmbeddingConfig):
    weight_init_mean: Optional[float] = 0.0  # used for InitializerType.UNIFORM
    weight_init_stddev: Optional[float] = 0.05  # used for InitializerType.UNIFORM
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )
    is_incremental: bool = False


    def __post_init__(self):
        check_embedding_config_valid(self)
        super().__post_init__()

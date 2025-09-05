#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from torchrec import EmbeddingConfig, EmbeddingBagConfig
from hybrid_torchrec.constants import EMBEDDINGS_DIM_ALIGNMENT, MAX_EMBEDDINGS_DIM, MAX_NUM_EMBEDDINGS


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


def check_embedding_config(config: EmbeddingConfig):
    if config.num_embeddings < 1 or config.num_embeddings > MAX_NUM_EMBEDDINGS:
        raise ValueError(
            f"The num_embeddings should be in [1, {MAX_NUM_EMBEDDINGS}], but is {config.num_embeddings}"
        )

    if config.embedding_dim < EMBEDDINGS_DIM_ALIGNMENT or config.embedding_dim > MAX_EMBEDDINGS_DIM:
        raise ValueError(
            f"The embedding dim should be in [{EMBEDDINGS_DIM_ALIGNMENT}, {MAX_EMBEDDINGS_DIM}], "
            f"but is {config.embedding_dim}"
        )
    
    if config.embedding_dim % EMBEDDINGS_DIM_ALIGNMENT != 0:
        raise ValueError(
            f"The embedding dim should be a multiple of {EMBEDDINGS_DIM_ALIGNMENT}, but is {config.embedding_dim}"
        )
    
    if config.weight_init_min is None:
        config.weight_init_min = 0.0
    
    if config.weight_init_max is None:
        config.weight_init_max = 1.0
    
    if config.weight_init_min >= config.weight_init_max:
        raise ValueError(
            f"The weight_init_min should be less than weight_init_max, "
            f"but is {config.weight_init_min} >= {config.weight_init_max}"
        )
    

@dataclass
class EmbCacheEmbeddingBagConfig(EmbeddingBagConfig):
    weight_init_mean: Optional[float] = 0.0  # used for InitializerType.UNIFORM
    weight_init_stddev: Optional[float] = 0.05  # used for InitializerType.UNIFORM
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )

    def __post_init__(self):
        check_embedding_config(self)
        super().__post_init__()


@dataclass
class EmbCacheEmbeddingConfig(EmbeddingConfig):
    weight_init_mean: Optional[float] = 0.0  # used for InitializerType.UNIFORM
    weight_init_stddev: Optional[float] = 0.05  # used for InitializerType.UNIFORM
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )

    def __post_init__(self):
        check_embedding_config(self)
        super().__post_init__()

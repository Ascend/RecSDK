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


_DEFAULT_ADMIT_THRESHOLD: int = -1
_DEFAULT_EVICT_THRESHOLD: int = 0


@dataclass
class AdmitAndEvictConfig:
    """
    AdmitAndEvictConfig is a dataclass that represents an admit and evict config of a single embedding table.

    Args:
        admit_threshold (Optional[int]): feature admit threshold. Feature (which after input dist) whill de admitted
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


class InitializerType(str, Enum):
    LINEAR = "linear"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"


@dataclass
class EmbCacheEmbeddingBagConfig(EmbeddingBagConfig):
    weight_init_mean: Optional[float] = 0.0  # used for InitializerType.UNIFORM
    weight_init_stddev: Optional[float] = 0.05  # used for InitializerType.UNIFORM
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )


@dataclass
class EmbCacheEmbeddingConfig(EmbeddingConfig):
    weight_init_mean: Optional[float] = 0.0  # used for InitializerType.UNIFORM
    weight_init_stddev: Optional[float] = 0.05  # used for InitializerType.UNIFORM
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )

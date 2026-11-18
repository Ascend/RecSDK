#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Optional
import dataclasses
from dataclasses import dataclass, field

from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingType
from dynamic_emb.distributed.dynamicemb_config import DynamicEmbTableOptions
from rec_sdk_common.validator.safe_checker import class_safe_check


@dataclass
class DynamicEmbParameterConstraints(ParameterConstraints):
    """
    DynamicEmb-specific parameter constraints that extend ParameterConstraints.

    Attributes
    ----------
    use_dynamicemb : Optional[bool]
        A flag indicating whether to use DynamicEmb storage. Defaults to True.
    dynamicemb_options : Optional[DynamicEmbTableOptions]
        Including HKV Configs and Initializer Args. The initialization method for the parameters.
        Common choices include "uniform", "normal", etc. Defaults to "uniform".
    """

    use_dynamicemb: Optional[bool] = True
    dynamicemb_options: Optional[DynamicEmbTableOptions] = field(default_factory=DynamicEmbTableOptions)

    def __post_init__(self):
        class_safe_check("use_dynamicemb", self.use_dynamicemb, (bool,))
        if not self.use_dynamicemb:
            raise ValueError("use_dynamicemb should be True")
        class_safe_check("dynamicemb_options", self.dynamicemb_options, (DynamicEmbTableOptions,))

        if self.sharding_types != [ShardingType.ROW_WISE.value]:
            raise ValueError(
                "DynamicEmbParameterConstraints only supports sharding_types=[ShardingType.ROW_WISE.value]"
            )
        if self.compute_kernels != ["fused"]:
            raise ValueError("DynamicEmbParameterConstraints only supports compute_kernels=['fused']")

        try:
            default_constraints = ParameterConstraints(
                sharding_types=self.sharding_types,
                compute_kernels=self.compute_kernels,
            )
        except (TypeError, ValueError):
            default_constraints = ParameterConstraints()
        for constraint_field in dataclasses.fields(ParameterConstraints):
            field_name = constraint_field.name
            if field_name in ("sharding_types", "compute_kernels"):
                continue
            if getattr(self, field_name) != getattr(default_constraints, field_name):
                raise ValueError(f"DynamicEmbParameterConstraints only supports the default value for {field_name}")

        super_post_init = getattr(super(), "__post_init__", None)
        if super_post_init is not None:
            super_post_init()

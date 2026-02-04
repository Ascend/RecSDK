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
from dataclasses import dataclass, field

from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.tensor_types import check

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
        check(self.use_dynamicemb, "use_dynamicemb should be True")
        class_safe_check("dynamicemb_options", self.dynamicemb_options, (DynamicEmbTableOptions,))
        super().__post_init__()

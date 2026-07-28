#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import os
import warnings

from .data import TestDataGenerator
from .record import Record, BenchmarkRecord
from .seq_stats import SeqStats

__all__ = [
    "Record",
    "BenchmarkRecord",
    "SeqStats",
]

DEFAULT_HSTU_CUSTOM_OPP_PATH = "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/hstu_attn_metadata_transformer"


def create_data_generator(seed, seq_all_equal, seq_max_ratio=1):
    return TestDataGenerator(seed, seq_all_equal, seq_max_ratio)


def ensure_hstu_custom_opp_path():
    """检查 HSTU metadata vendor 路径，并在未配置时设置默认值。"""
    custom_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").strip()
    if not custom_opp_path:
        warnings.warn(
            f"ASCEND_CUSTOM_OPP_PATH is not configured; using default path: {DEFAULT_HSTU_CUSTOM_OPP_PATH}",
            RuntimeWarning,
            stacklevel=2,
        )
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = DEFAULT_HSTU_CUSTOM_OPP_PATH
        return

    configured_paths = [path for path in custom_opp_path.split(os.pathsep) if path]
    if not any(os.path.isdir(os.path.join(path, "op_impl")) for path in configured_paths):
        warnings.warn(
            "ASCEND_CUSTOM_OPP_PATH is configured but no entry contains op_impl/: "
            f"{custom_opp_path}. CANN may skip the custom operator path.",
            RuntimeWarning,
            stacklevel=2,
        )

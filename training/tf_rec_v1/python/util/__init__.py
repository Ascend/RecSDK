#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

__all__ = [
    "init", "terminate_config_initializer", "get_dense_and_sparse_variable", "hccl_ops", "import_host_pipeline_ops",
    "ConfigInitializer", 
]

from rec_sdk_common.util.tf_adapter import hccl_ops
from mx_rec.util.initialize import init, terminate_config_initializer, ConfigInitializer
from mx_rec.util.variable import get_dense_and_sparse_variable
from mx_rec.util.ops import import_host_pipeline_ops

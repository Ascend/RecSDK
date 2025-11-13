#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from typing import List

from mxrec.python.utils.loader import load_dynamic_lib, find_binding_path
from mxrec.python.utils.decorator import make_singleton
from rec_sdk_common.communication.hccl.hccl_info import get_device_id

host = load_dynamic_lib("host", find_binding_path())


@make_singleton
class RuntimeManager:  # pragma: no cover
    def __init__(self):
        self._runtime_manager = host.RuntimeManager(get_device_id())

    def start_count_filter(self, table_name: str, min_used_times: int):
        self._runtime_manager.start_count_filter(table_name, min_used_times)

    def start_time_evictor(self, table_name: str, max_cold_secs: int):
        self._runtime_manager.start_time_evictor(table_name, max_cold_secs)

    def save_count_filter(self, table_name: str, save_path: str):
        self._runtime_manager.save_count_filter(table_name, save_path)

    def save_time_evictor(self, table_name: str, save_path: str):
        self._runtime_manager.save_time_evictor(table_name, save_path)

    def load_count_filter(self, table_name: str, load_path: str):
        self._runtime_manager.load_count_filter(table_name, load_path)

    def load_time_evictor(self, table_name: str, load_path: str):
        self._runtime_manager.load_time_evictor(table_name, load_path)

    def get_evicted_keys(self, table_name: str) -> List[int]:
        return self._runtime_manager.get_evicted_keys(table_name)

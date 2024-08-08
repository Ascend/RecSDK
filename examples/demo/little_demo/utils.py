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

import json
import os
import stat
from datetime import datetime, timezone

from mx_rec.util.communication.hccl_ops import get_rank_id, get_rank_size

LOCAL_RANK_ID = get_rank_id()
RANK_ZERO = 0
GLOBAL_RANK_SIZE = get_rank_size()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TIME = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
PRECISION_CHECK_PATH = os.path.join(SCRIPT_DIR, 'precision_check', CURRENT_TIME)
PRECISION_DUMP_STEP = [1, 2]


class PrecisionDumpInfo:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.dump_info_dict = {}
        self.dump_info_path = os.path.join(PRECISION_CHECK_PATH, "dump_info.json")

    @classmethod
    def add_item(cls, key, value):
        cls.get_instance().instance_add_item(key, value)  # 委托给实例的方法

    def instance_add_item(self, key, value):
        if self.dump_info_dict.get(key):
            raise ValueError(f"Key [{key}] already set in dump_info_dict and no need to be set again.")
        self.dump_info_dict[key] = value

    @classmethod
    def write_dump_info(cls):
        cls.get_instance().instance_write_dump_info()  # 委托给实例的方法

    def instance_write_dump_info(self):

        if os.path.exists(self.dump_info_path):
            return

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        modes = stat.S_IWUSR | stat.S_IRUSR

        with os.fdopen(os.open(self.dump_info_path, flags, modes), 'w') as fout:
            json.dump(self.dump_info_dict, fout)


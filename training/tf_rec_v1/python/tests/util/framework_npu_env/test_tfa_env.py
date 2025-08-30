#!/usr/bin/env python3
# coding: UTF-8
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import unittest
from unittest import mock

from mx_rec.util.framework_npu_env.tfa_env import set_ascend_env
from core.mock_class import MockGlobalEnv


class TestBindCpu(unittest.TestCase):
    @mock.patch.multiple(
        "mx_rec.util.framework_npu_env.tfa_env",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.util.framework_npu_env.tfa_env.global_env", MockGlobalEnv(rank_table_file=True))
    def test_set_ascend_env_ok(self):
        set_ascend_env()
        self.assertTrue(callable(set_ascend_env))

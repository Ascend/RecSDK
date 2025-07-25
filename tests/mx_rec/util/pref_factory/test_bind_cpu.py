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

from mx_rec.util.perf_factory.bind_cpu import bind_cpu


class TestBindCpu(unittest.TestCase):
    @mock.patch.multiple(
        "mx_rec.util.perf_factory.bind_cpu",
        get_local_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
    )
    @bind_cpu
    def test_bind_cpu_ok(self):
        self.assertTrue(callable(bind_cpu))

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

import pytest

import mxrec
from mxrec.python.config.parser import TomlParser
from mxrec.python.tests.ut.ut_utils import mock_get_device_id


class TestInit:
    """Test for 'mxrec.python.initializer.initializer.init'."""

    @staticmethod
    def teardown_method():
        TomlParser._instance = None

    @staticmethod
    def test_init_with_ranktable_ok(monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        try:
            mxrec.init("./ut_test.toml")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_init_without_ranktable_ok(monkeypatch):
        def _mock_get_use_ranktable():
            return False

        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_use_ranktable", _mock_get_use_ranktable)
        try:
            mxrec.init("./ut_test.toml")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_init_twice_err(monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        with pytest.raises(RuntimeError) as excinfo:
            mxrec.init("./ut_test.toml")
            mxrec.init("./ut_test.toml")
        assert "TomlParser has been initialized once, twice initialization was forbidden" in str(excinfo.value)

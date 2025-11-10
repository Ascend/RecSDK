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
import toml

import mxrec
from mxrec.python.config.parser import TomlParser
from mxrec.python.config.config import (
    get_log_level,
    get_comm_node_info,
    get_use_ranktable,
)
from mxrec.python.tests.ut.ut_utils import mock_get_device_id


class TestGetLogLevel:
    """Test for 'from mxrec.python.config.config.get_log_level'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        TomlParser._instance = None

    @staticmethod
    def test_ok():
        try:
            get_log_level()
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_config_not_none():
        try:
            get_log_level()
            # On the second call, it is not None.
            get_log_level()
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_parser_instance_none_err():
        TomlParser._instance = None

        with pytest.raises(RuntimeError) as excinfo:
            get_log_level()
        assert "the TomlParser instance is None" in str(excinfo.value)


class TestGetCommNodeInfo:
    """Test for 'from mxrec.python.config.config.get_comm_node_info'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        TomlParser._instance = None

    @staticmethod
    def test_ok():
        try:
            get_comm_node_info()
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_config_not_none():
        try:
            get_comm_node_info()
            # On the second call, it is not None.
            get_comm_node_info()
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_parser_instance_none_err():
        TomlParser._instance = None

        with pytest.raises(RuntimeError) as excinfo:
            get_comm_node_info()
        assert "the TomlParser instance is None" in str(excinfo.value)


class TestGetUseRanktable:
    """Test for 'from mxrec.python.config.config.get_use_ranktable'."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("mxrec.python.initializer.initializer.get_device_id", mock_get_device_id)
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        TomlParser._instance = None

    @staticmethod
    def test_use_ranktable_is_false(monkeypatch):
        mock_toml = """
                    [mxrec]
                    use_ranktable = false
                    """
        config = toml.loads(mock_toml).get("mxrec")
        TomlParser.get_instance().use_ranktable = None
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(ValueError) as excinfo:
            get_use_ranktable()
        assert "currently, the use_ranktable only supports True" in str(excinfo.value)

    @staticmethod
    def test_use_ranktable_is_true(monkeypatch):
        mock_toml = """
                    [mxrec]
                    use_ranktable = true
                    """
        config = toml.loads(mock_toml).get("mxrec")
        TomlParser.get_instance().use_ranktable = None
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        assert get_use_ranktable()

    @staticmethod
    def test_use_ranktable_missing_key(monkeypatch):
        mock_toml = """
                    [mxrec]
                    """
        config = toml.loads(mock_toml).get("mxrec")
        TomlParser.get_instance().use_ranktable = None
        monkeypatch.setattr("mxrec.python.config.parser.TomlParser.config", config)

        with pytest.raises(KeyError) as excinfo:
            get_use_ranktable()
        assert "the use_ranktable field is missing" in str(excinfo.value)

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

from mxrec.python.config.parser import parse_field, parse_env_field


class TestParseField:
    """Test for 'from mxrec.python.config.parser.parse_field'."""

    @staticmethod
    def test_ok():
        mock_toml = """
                    [mxrec]
                    log_level = "INFO"
                    """
        config = toml.loads(mock_toml).get("mxrec")
        field = parse_field(config, "log_level")
        assert field == "INFO"

    @staticmethod
    def test_field_not_exist_err():
        mock_toml = """
                    [mxrec]
                    log_level = "INFO"
                    """
        config = toml.loads(mock_toml).get("mxrec")

        with pytest.raises(KeyError) as excinfo:
            parse_field(config, "cm-node-info")
        assert "the cm-node-info field is missing" in str(excinfo.value)


class TestParseEnvField:
    """Test for 'from mxrec.python.config.parser.parse_env_field'."""

    @staticmethod
    def test_ok():
        mock_toml = """
                    [mxrec]
                    log_level = "INFO"
                    """
        config = toml.loads(mock_toml).get("mxrec")
        field = parse_env_field(config, "log_level")
        assert field == "INFO"

    @staticmethod
    def test_field_env_is_not_none(monkeypatch):
        def _mock_os_getenv(*args, **kwargs):
            return "INFO"

        monkeypatch.setattr("mxrec.python.config.parser.os.getenv", _mock_os_getenv)

        mock_toml = """
                    [mxrec]
                    log_level = "INFO"
                    """
        config = toml.loads(mock_toml).get("mxrec")
        field = parse_env_field(config, "log_level")
        assert field == "INFO"

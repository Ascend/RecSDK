# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for configuration management system.

Tests the simplified ConfigManager and related components.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import toml

from ngo.core.config import (
    ConfigError, ConfigManager, ConfigValidationError
)


class TestConfigManager:
    """Test cases for ConfigManager."""

    def test_create_manager(self):
        """Test creating ConfigManager."""
        manager = ConfigManager()
        assert manager.config is not None
        # ConfigManager starts with empty config, loads config from file
        assert isinstance(manager.config, dict)

    def test_get_set_values(self):
        """Test getting and setting configuration values."""
        manager = ConfigManager()

        # Test getting nonexistent values
        assert manager.get("nonexistent.key") is None
        assert manager.get("nonexistent.key", "default") == "default"

        # Test setting values
        manager.set("core.log_level", "DEBUG")
        assert manager.get("core.log_level") == "DEBUG"

        # Test setting nested values
        manager.set("new.section.value", 42)
        assert manager.get("new.section.value") == 42

    def test_load_save_file(self):
        """Test loading and saving configuration files."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            test_config = {
                "core": {"log_level": "DEBUG", "max_workers": 8},
                "test": {"value": 42},
            }
            toml.dump(test_config, f)
            temp_file = f.name

        try:
            # Test loading file
            manager.load_file(temp_file)
            assert manager.get("core.log_level") == "DEBUG"
            assert manager.get("core.max_workers") == 8
            assert manager.get("test.value") == 42

            # Test saving file
            manager.set("test.new_value", "hello")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False
            ) as f:
                save_file = f.name

            manager.save_file(save_file)

            # Verify saved content
            with open(save_file, "r", encoding='utf-8') as f:
                saved_config = toml.load(f)
            assert saved_config["test"]["new_value"] == "hello"

            os.unlink(save_file)

        finally:
            os.unlink(temp_file)

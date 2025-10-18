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
import unittest
from unittest.mock import patch

import toml

from ngo.core.config import (
    ConfigError, ConfigManager, ConfigSecurityError
)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""

    def test_create_manager(self):
        """Test creating ConfigManager."""
        manager = ConfigManager()
        self.assertIsNotNone(manager.config)
        self.assertIsInstance(manager.config, dict)

    def test_get_set_values(self):
        """Test getting and setting configuration values."""
        manager = ConfigManager()

        # Test getting nonexistent values
        self.assertIsNone(manager.get("nonexistent.key"))
        self.assertEqual(manager.get("nonexistent.key", "default"), "default")

        # Test setting values
        manager.set("core.log_level", "DEBUG")
        self.assertEqual(manager.get("core.log_level"), "DEBUG")

        # Test setting nested values
        manager.set("new.section.value", 42)
        self.assertEqual(manager.get("new.section.value"), 42)

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
            self.assertEqual(manager.get("core.log_level"), "DEBUG")
            self.assertEqual(manager.get("core.max_workers"), 8)
            self.assertEqual(manager.get("test.value"), 42)

            # Test saving file
            manager.set("test.new_value", "hello")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False
            ) as f:
                save_file = f.name

            manager.save_file(save_file)

            # Verify saved content
            with open(save_file, "r", encoding="utf-8") as f:
                saved_config = toml.load(f)
            self.assertEqual(saved_config["test"]["new_value"], "hello")

            os.unlink(save_file)

        finally:
            os.unlink(temp_file)

    def test_from_dict(self):
        """Test loading configuration from dictionary."""
        manager = ConfigManager()

        test_config = {
            "core": {"log_level": "DEBUG", "max_workers": 8},
            "test": {"value": 42},
        }

        manager.from_dict(test_config)

        self.assertEqual(manager.get("core.log_level"), "DEBUG")
        self.assertEqual(manager.get("core.max_workers"), 8)
        self.assertEqual(manager.get("test.value"), 42)

    def test_security_validation_file_not_found(self):
        """Test security validation for non-existent file."""
        manager = ConfigManager()

        with self.assertRaises(ConfigSecurityError):
            manager.load_file("/nonexistent/path/config.toml")

    def test_invalid_toml_format(self):
        """Test handling of invalid TOML format."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [unclosed bracket")
            temp_file = f.name

        try:
            with self.assertRaises(ConfigError):
                manager.load_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_save_file_no_path(self):
        """Test save file with no path specified."""
        manager = ConfigManager()

        with self.assertRaises(ConfigError):
            manager.save_file()

    def test_set_value_no_change(self):
        """Test setting value that hasn't changed."""
        manager = ConfigManager()

        manager.set("test.key", "value")
        # Set same value again - should not error and should skip processing
        manager.set("test.key", "value")

        self.assertEqual(manager.get("test.key"), "value")

    def test_get_nested_keys(self):
        """Test getting nested configuration keys."""
        manager = ConfigManager()

        # Set nested values
        manager.set("a.b.c.d", "deep_value")
        manager.set("a.b.x", "sibling_value")

        self.assertEqual(manager.get("a.b.c.d"), "deep_value")
        self.assertEqual(manager.get("a.b.x"), "sibling_value")
        self.assertIsNone(manager.get("a.b.nonexistent"))
        self.assertEqual(manager.get("a.b.nonexistent", "default"), "default")


if __name__ == "__main__":
    unittest.main()
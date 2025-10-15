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

"""NGO日志模块单元测试."""

import inspect
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ngo.utils.logger import (
    NGOFormatter,
    NGOLogger,
    get_logger,
    configure_logging,
    configure_from_config,
    _logger_manager,
)


class TestNGOFormatter(unittest.TestCase):
    """测试NGOFormatter类."""

    def setUp(self):
        """设置测试环境."""
        self.formatter = NGOFormatter("test_project")

    def test_init(self):
        """测试初始化."""
        self.assertEqual(self.formatter.project_name, "test_project")
        self.assertIn("%(project_name)s", self.formatter.format_string)

    def test_format(self):
        """测试日志格式化."""
        # 创建模拟的日志记录
        record = logging.LogRecord(
            name="test_module",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="测试消息",
            args=(),
            exc_info=None,
        )

        # 格式化日志记录
        formatted = self.formatter.format(record)

        # 验证格式化结果
        self.assertIn("test_project", formatted)
        self.assertIn("test_module", formatted)
        self.assertIn("测试消息", formatted)
        self.assertIn("INFO", formatted)
        self.assertIn("path.py:42", formatted)

    def test_format_with_different_project_name(self):
        """测试不同项目名称的格式化."""
        formatter = NGOFormatter("my_project")
        record = logging.LogRecord(
            name="test_module",
            level=logging.DEBUG,
            pathname="/test/path.py",
            lineno=10,
            msg="调试消息",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        self.assertIn("my_project", formatted)
        self.assertIn("DEBUG", formatted)


class TestNGOLogger(unittest.TestCase):
    """测试NGOLogger类."""

    def setUp(self):
        """设置测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}
        self.logger_manager = NGOLogger()
        # 确保根日志记录器存在
        if "ngo" not in self.logger_manager._loggers:
            self.logger_manager._setup_root_logger()

    def tearDown(self):
        """清理测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}

    def test_singleton_pattern(self):
        """测试单例模式."""
        logger1 = NGOLogger()
        logger2 = NGOLogger()
        self.assertIs(logger1, logger2)

    def test_get_logger(self):
        """测试获取日志记录器."""
        logger = self.logger_manager.get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "ngo.test_module")

    def test_get_logger_with_ngo_prefix(self):
        """测试获取已包含ngo前缀的日志记录器."""
        logger = self.logger_manager.get_logger("ngo.test_module")
        self.assertEqual(logger.name, "ngo.test_module")

    def test_get_logger_caching(self):
        """测试日志记录器缓存."""
        logger1 = self.logger_manager.get_logger("test_module")
        logger2 = self.logger_manager.get_logger("test_module")
        self.assertIs(logger1, logger2)

    def test_update_config(self):
        """测试更新配置."""
        config = {
            "level": "DEBUG",
            "file": "test.log",
            "enable_console": False,
            "enable_file": True,
            "max_file_size": 1024,
            "backup_count": 3,
        }

        self.logger_manager.update_config(config)
        self.assertEqual(self.logger_manager.log_level, "DEBUG")
        self.assertEqual(self.logger_manager.log_file, "test.log")
        self.assertFalse(self.logger_manager.enable_console)
        self.assertTrue(self.logger_manager.enable_file)

    @patch("ngo.utils.logger.inspect.currentframe")
    def test_get_current_logger(self, mock_currentframe):
        """测试获取当前调用者的日志记录器."""
        # 模拟调用栈
        mock_frame = Mock()
        mock_caller_frame = Mock()
        mock_module = Mock()
        mock_module.__name__ = "ngo.test_module"

        mock_currentframe.return_value = mock_frame
        mock_frame.f_back.f_back.f_back = mock_caller_frame
        mock_caller_frame.__module__ = mock_module
        with patch("inspect.getmodule", return_value=mock_module):
            logger = self.logger_manager.get_current_logger()
            self.assertEqual(logger.name, "ngo.test_module")

    @patch("ngo.utils.logger.inspect.currentframe")
    def test_get_current_logger_unknown_module(self, mock_currentframe):
        """测试获取未知模块的日志记录器."""
        # 模拟调用栈
        mock_frame = Mock()
        mock_caller_frame = Mock()
        mock_module = Mock()
        mock_module.__name__ = "unknown_module"

        mock_currentframe.return_value = mock_frame
        mock_frame.f_back.f_back.f_back = mock_caller_frame
        mock_caller_frame.__module__ = mock_module
        with patch("inspect.getmodule", return_value=mock_module):
            logger = self.logger_manager.get_current_logger()
            self.assertEqual(logger.name, "ngo.unknown")


class TestGetLoggerFunction(unittest.TestCase):
    """测试get_logger函数."""

    def setUp(self):
        """设置测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}
        # 确保全局日志管理器正确初始化
        from ngo.utils.logger import _logger_manager

        _logger_manager._setup_root_logger()

    def tearDown(self):
        """清理测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}

    def test_get_logger_with_name(self):
        """测试使用指定名称获取日志记录器."""
        logger = get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "ngo.test_module")

    @patch("ngo.utils.logger._logger_manager.get_current_logger")
    def test_get_logger_without_name(self, mock_get_current_logger):
        """测试不指定名称获取日志记录器."""
        mock_logger = Mock()
        mock_get_current_logger.return_value = mock_logger

        logger = get_logger()
        mock_get_current_logger.assert_called_once()
        self.assertEqual(logger, mock_logger)


class TestConfigureLoggingFunction(unittest.TestCase):
    """测试configure_logging函数."""

    def setUp(self):
        """设置测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}

    def tearDown(self):
        """清理测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}

    @patch("ngo.utils.logger._logger_manager.update_config")
    def test_configure_logging(self, mock_update_config):
        """测试配置日志系统."""
        configure_logging(
            level="DEBUG",
            log_file="test.log",
            enable_console=False,
            enable_file=True,
            max_file_size=1024,
            backup_count=3,
        )

        expected_config = {
            "level": "DEBUG",
            "file": "test.log",
            "enable_console": False,
            "enable_file": True,
            "max_file_size": 1024,
            "backup_count": 3,
        }
        mock_update_config.assert_called_once_with(expected_config)

    @patch("ngo.utils.logger._logger_manager.update_config")
    def test_configure_logging_default_values(self, mock_update_config):
        """测试配置日志系统默认值."""
        configure_logging()

        expected_config = {
            "level": "INFO",
            "file": "ngo.log",
            "enable_console": True,
            "enable_file": True,
            "max_file_size": 10 * 1024 * 1024,
            "backup_count": 5,
        }
        mock_update_config.assert_called_once_with(expected_config)


class TestConfigureFromConfigFunction(unittest.TestCase):
    """测试configure_from_config函数."""

    def setUp(self):
        """设置测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}

    def tearDown(self):
        """清理测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}

    @patch("ngo.utils.logger.configure_logging")
    def test_configure_from_config(self, mock_configure_logging):
        """测试从配置字典配置日志系统."""
        config = {
            "logging": {
                "level": "WARNING",
                "file": "app.log",
                "enable_console": True,
                "enable_file": False,
                "max_file_size": 2048,
                "backup_count": 10,
            }
        }

        configure_from_config(config)

        mock_configure_logging.assert_called_once_with(
            level="WARNING",
            log_file="app.log",
            enable_console=True,
            enable_file=False,
            max_file_size=2048,
            backup_count=10,
        )

    @patch("ngo.utils.logger.configure_logging")
    def test_configure_from_config_empty_logging(self, mock_configure_logging):
        """测试从空日志配置字典配置日志系统."""
        config = {"logging": {}}

        configure_from_config(config)

        mock_configure_logging.assert_called_once_with(
            level="INFO",
            log_file="ngo.log",
            enable_console=True,
            enable_file=True,
            max_file_size=10 * 1024 * 1024,
            backup_count=5,
        )

    @patch("ngo.utils.logger.configure_logging")
    def test_configure_from_config_no_logging_key(self, mock_configure_logging):
        """测试从没有logging键的配置字典配置日志系统."""
        config = {"other_key": "value"}

        configure_from_config(config)

        mock_configure_logging.assert_called_once_with(
            level="INFO",
            log_file="ngo.log",
            enable_console=True,
            enable_file=True,
            max_file_size=10 * 1024 * 1024,
            backup_count=5,
        )


class TestLoggerIntegration(unittest.TestCase):
    """测试日志系统集成功能."""

    def setUp(self):
        """设置测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}
        # 确保全局日志管理器正确初始化
        from ngo.utils.logger import _logger_manager

        _logger_manager._setup_root_logger()

    def tearDown(self):
        """清理测试环境."""
        # 重置单例实例
        NGOLogger._instance = None
        NGOLogger._loggers = {}

    def test_logger_creation_and_logging(self):
        """测试日志记录器创建和日志记录."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            # 配置日志系统
            configure_logging(
                level="DEBUG", log_file=log_file, enable_console=False, enable_file=True
            )

            # 获取日志记录器并记录日志
            logger = get_logger("test_module")
            logger.info("这是一条测试信息")
            logger.debug("这是一条调试信息")
            logger.warning("这是一条警告信息")

            # 验证日志文件是否创建
            self.assertTrue(os.path.exists(log_file))

            # 读取并验证日志内容
            with open(log_file, "r", encoding="utf-8") as f:
                log_content = f.read()
                self.assertIn("这是一条测试信息", log_content)
                self.assertIn("这是一条调试信息", log_content)
                self.assertIn("这是一条警告信息", log_content)
                self.assertIn("test_module", log_content)

    def test_multiple_loggers(self):
        """测试多个日志记录器."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module1")  # 应该返回相同的实例

        self.assertIsNot(logger1, logger2)
        self.assertIs(logger1, logger3)
        self.assertEqual(logger1.name, "ngo.module1")
        self.assertEqual(logger2.name, "ngo.module2")


if __name__ == "__main__":
    unittest.main()

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
NGO 的配置管理器。

该模块提供了一个简化的 ConfigManager 类来处理基本配置管理。
"""

import copy
import logging
import os
import stat
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import toml


class ConfigError(Exception):
    """配置错误的基础异常。"""

    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""

    pass


class ConfigSecurityError(ConfigError):
    """Exception raised when configuration security checks fail."""

    pass


class ConfigManager:
    """Simplified configuration manager for NGO."""

    def __init__(self):
        """Initialize configuration manager."""
        self.config: Dict[str, Any] = {}
        self.file_path: Optional[str] = None
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # 配置日志系统
        try:
            from ngo.utils.logger import configure_from_config

            configure_from_config(self.config)
        except ImportError:
            # 如果logger模块还没准备好，使用默认配置
            pass

    def _validate_file_security(self, file_path: str) -> None:
        """验证文件路径的安全性，包括软连接和权限检查。

        Args:
            file_path: 要验证的文件路径

        Raises:
            ConfigSecurityError: 如果文件安全性检查失败
        """
        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            raise ConfigSecurityError(f"Configuration file not found: {file_path}")

        # 检查是否为软连接
        if path.is_symlink():
            raise ConfigSecurityError(
                f"Configuration file cannot be a symbolic link: {file_path}"
            )

        # 检查文件权限 - 确保文件对当前用户可读
        self._ensure_file_permissions(file_path)

        # 检查文件大小 (防止过大的配置文件)
        try:
            file_size = path.stat().st_size
            max_size = 10 * 1024 * 1024  # 10MB 限制
            if file_size > max_size:
                raise ConfigSecurityError(
                    f"Configuration file too large ({file_size} bytes > {max_size} bytes): {file_path}"
                )
        except OSError as e:
            raise ConfigSecurityError(f"Error checking file size for {file_path}: {e}")

    def _ensure_directory_permissions(self, dir_path: str) -> None:
        """确保目录具有正确的权限 (750)。

        Args:
            dir_path: 目录路径

        Raises:
            ConfigSecurityError: 如果无法设置目录权限
        """
        try:
            path = Path(dir_path)
            if path.exists():
                # 检查当前权限
                current_mode = path.stat().st_mode
                expected_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP  # 750

                # 如果权限不正确，尝试修正
                if (current_mode & 0o777) != expected_mode:
                    try:
                        path.chmod(expected_mode)
                        self._logger.info(
                            f"Directory permissions set to 750 for {dir_path}"
                        )
                    except PermissionError:
                        self._logger.warning(
                            f"Cannot set directory permissions for {dir_path}, current permissions: {oct(current_mode & 0o777)}"
                        )
        except OSError as e:
            raise ConfigSecurityError(
                f"Error setting directory permissions for {dir_path}: {e}"
            )

    def _ensure_file_permissions(self, file_path: str) -> None:
        """确保文件具有正确的权限 (640)。

        Args:
            file_path: 文件路径

        Raises:
            ConfigSecurityError: 如果无法设置文件权限
        """
        try:
            path = Path(file_path)
            if path.exists():
                # 检查当前权限
                current_mode = path.stat().st_mode
                expected_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP  # 640

                # 如果权限不正确，尝试修正
                if (current_mode & 0o777) != expected_mode:
                    try:
                        path.chmod(expected_mode)
                        self._logger.info(
                            f"File permissions set to 640 for {file_path}"
                        )
                    except PermissionError:
                        self._logger.warning(
                            f"Cannot set file permissions for {file_path}, current permissions: {oct(current_mode & 0o777)}"
                        )
        except OSError as e:
            raise ConfigSecurityError(
                f"Error setting file permissions for {file_path}: {e}"
            )

    def _create_secure_directory(self, dir_path: str) -> None:
        """创建具有安全权限的目录。

        Args:
            dir_path: 要创建的目录路径

        Raises:
            ConfigSecurityError: 如果无法创建目录或设置权限
        """
        try:
            path = Path(dir_path)

            # 如果目录不存在，创建它
            if not path.exists():
                # 设置 umask 以确保新目录有正确的权限
                old_umask = os.umask(0o027)  # 750 = 777 - 027
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self._logger.info(
                        f"Directory created with secure permissions: {dir_path}"
                    )
                finally:
                    os.umask(old_umask)
            else:
                # 目录已存在，确保权限正确
                self._ensure_directory_permissions(dir_path)

        except OSError as e:
            raise ConfigSecurityError(
                f"Error creating secure directory {dir_path}: {e}"
            )

    def load_file(self, file_path: str) -> None:
        """Load configuration from a TOML file.

        Args:
            file_path: Path to the configuration file

        Raises:
            ConfigSecurityError: If file security checks fail
            ConfigError: If file cannot be loaded or parsed
        """
        try:
            with self._lock:
                # 首先进行安全性验证
                self._validate_file_security(file_path)

                path = Path(file_path)

                # Load TOML file
                with open(path, "r", encoding="utf-8") as f:
                    file_config = toml.load(f)

                # Update configuration
                self.config = file_config
                self.file_path = file_path

                # 配置日志系统
                try:
                    from ngo.utils.logger import configure_from_config

                    configure_from_config(self.config)
                except ImportError:
                    self._logger.warning("logger module not found, using default configuration")

                self._logger.info(f"Configuration loaded from {file_path}")

        except ConfigSecurityError:
            # 重新抛出安全错误
            raise
        except toml.TomlDecodeError as e:
            raise ConfigError(f"Invalid TOML format in {file_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration from {file_path}: {e}")

    def save_file(self, file_path: Optional[str] = None) -> None:
        """Save configuration to a TOML file with secure permissions.

        Args:
            file_path: Path to save the configuration file. If None, uses current file_path.

        Raises:
            ConfigSecurityError: If security checks fail
            ConfigError: If file cannot be saved
        """
        save_path = file_path or self.file_path
        if not save_path:
            raise ConfigError("No file path specified for saving configuration")

        try:
            with self._lock:
                path = Path(save_path)

                # 创建具有安全权限的目录
                if path.parent != path:  # 确保不是根目录
                    self._create_secure_directory(str(path.parent))

                # 保存 TOML 文件，使用安全的 umask
                old_umask = os.umask(0o137)  # 640 = 777 - 137
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        toml.dump(self.config, f)

                    # 确保文件权限正确
                    self._ensure_file_permissions(save_path)

                finally:
                    os.umask(old_umask)

                self.file_path = save_path
                self._logger.info(
                    f"Configuration saved to {save_path} with secure permissions"
                )

        except ConfigSecurityError:
            # 重新抛出安全错误
            raise
        except Exception as e:
            raise ConfigError(f"Error saving configuration to {save_path}: {e}")

    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path.

        Args:
            path: Configuration path (e.g., "core.log_level")
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        with self._lock:
            keys = path.split(".")
            value = self.config

            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return default

    def set(self, path: str, value: Any) -> None:
        """Set a configuration value by path.

        Args:
            path: Configuration path (e.g., "core.log_level")
            value: Value to set
        """
        with self._lock:
            # Get old value for change detection
            old_value = self.get(path)

            # Skip if value hasn't changed
            if old_value == value:
                return

            # Set the value
            keys = path.split(".")
            config = self.config

            # Navigate to parent
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]

            # Set the value
            config[keys[-1]] = value

    def from_dict(self, config: Dict[str, Any]) -> None:
        """Load configuration from dictionary.

        Args:
            config: Configuration dictionary
        """
        with self._lock:
            # Update configuration
            self.config = copy.deepcopy(config)

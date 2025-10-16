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
NGO统一日志模块 - 并发安全版本。
"""

import inspect
import logging
import logging.handlers
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


class NGOFormatter(logging.Formatter):
    """NGO自定义日志格式化器."""

    def __init__(self, project_name: str = "ngo"):
        """初始化格式化器.

        Args:
            project_name: 项目名称
        """
        self.project_name = project_name
        # 定义日志格式
        self.format_string = (
            "%(asctime)s | %(project_name)s | %(process)d | "
            "%(filename)s:%(lineno)d | %(levelname)s | "
            "%(module_name)s | %(message)s"
        )
        super().__init__(self.format_string)

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录.

        Args:
            record: 日志记录

        Returns:
            格式化后的日志字符串
        """
        # 添加项目名称
        record.project_name = self.project_name

        # 添加模块名（使用模块名而不是函数名）
        record.module_name = record.name

        return super().format(record)


class NGOLogger:
    """NGO统一日志管理器 - 并发安全版本."""

    _instance: Optional["NGOLogger"] = None
    _lock = threading.Lock()
    _initialized = False
    _config_lock = threading.RLock()
    _loggers: Dict[str, logging.Logger] = {}
    _loggers_lock = threading.RLock()
    _handlers: Dict[str, logging.Handler] = {}
    _handlers_lock = threading.RLock()

    def __new__(cls) -> "NGOLogger":
        """线程安全的单例模式."""
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定模式
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化日志管理器."""
        # 使用锁确保初始化只执行一次
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialized = True
                    self._config = {}
                    self._load_config()
                    self._setup_root_logger()

    def _load_config(self):
        """加载日志配置."""
        # 从配置字典加载配置
        self.log_level = self._config.get("level", "INFO")
        self.log_file = self._config.get("file", "ngo.log")
        self.max_file_size = self._config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        self.backup_count = self._config.get("backup_count", 5)
        self.enable_console = self._config.get("enable_console", True)
        self.enable_file = self._config.get("enable_file", True)

    def _setup_root_logger(self):
        """设置根日志记录器."""
        # 创建根日志记录器
        root_logger = logging.getLogger("ngo")
        root_logger.setLevel(getattr(logging, self.log_level.upper()))

        # 清除已有的处理器
        root_logger.handlers.clear()

        # 创建格式化器
        formatter = NGOFormatter()

        # 控制台处理器
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.log_level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            with self._handlers_lock:
                self._handlers["console"] = console_handler

        # 文件处理器
        if self.enable_file:
            # 确保日志目录存在
            log_file_path = Path(self.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # 使用轮转文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, self.log_level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            with self._handlers_lock:
                self._handlers["file"] = file_handler

        # 禁止向上传播
        root_logger.propagate = False

        # 存储根日志记录器
        with self._loggers_lock:
            self._loggers["ngo"] = root_logger

    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器.

        Args:
            name: 日志记录器名称

        Returns:
            日志记录器实例
        """
        # 完整名称以 ngo 为前缀
        full_name = f"ngo.{name}" if not name.startswith("ngo.") else name

        # 使用锁保护字典操作
        with self._loggers_lock:
            if full_name not in self._loggers:
                # 创建新的日志记录器
                logger = logging.getLogger(full_name)
                logger.setLevel(getattr(logging, self.log_level.upper()))

                # 为每个日志器创建独立的处理器副本
                logger.handlers = []

                with self._handlers_lock:
                    for handler_name, handler in self._handlers.items():
                        # 创建处理器的深拷贝
                        try:
                            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.handlers.RotatingFileHandler):
                                new_handler = logging.StreamHandler(sys.stdout)
                            elif isinstance(handler, logging.handlers.RotatingFileHandler):
                                new_handler = logging.handlers.RotatingFileHandler(
                                    handler.baseFilename,
                                    maxBytes=handler.maxBytes,
                                    backupCount=handler.backupCount,
                                    encoding=handler.encoding,
                                )
                            else:
                                logging.warning(f"Unsupported handler type: {type(handler)}")
                                continue
                            # 设置处理器属性
                            new_handler.setLevel(handler.level)
                            new_handler.setFormatter(handler.formatter)
                            logger.addHandler(new_handler)
                        except Exception as e:
                            logging.error(f"Failed to create handler: {e}")

                # 禁止向上传播
                logger.propagate = False

                self._loggers[full_name] = logger

            return self._loggers[full_name]

    def update_config(self, config: Dict[str, Any]):
        """更新日志配置.

        Args:
            config: 新的配置字典
        """
        # 使用锁保护配置更新操作
        with self._config_lock:
            # 更新配置
            self._config.update(config)

            # 重新加载配置
            self._load_config()

            # 清理现有处理器
            with self._handlers_lock:
                for handler in self._handlers.values():
                    handler.close()
                self._handlers.clear()

            # 重新设置根日志记录器
            self._setup_root_logger()

            # 更新所有已创建的日志记录器
            with self._loggers_lock:
                for name, logger in self._loggers.items():
                    if name != "ngo":  # 跳过根日志记录器
                        logger.setLevel(getattr(logging, self.log_level.upper()))
                        # 更新处理器
                        logger.handlers.clear()
                        with self._handlers_lock:
                            for handler_name, handler in self._handlers.items():
                                # 创建处理器的深拷贝
                                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.handlers.RotatingFileHandler):
                                    new_handler = logging.StreamHandler(sys.stdout)
                                elif isinstance(handler, logging.handlers.RotatingFileHandler):
                                    new_handler = logging.handlers.RotatingFileHandler(
                                        handler.baseFilename,
                                        maxBytes=handler.maxBytes,
                                        backupCount=handler.backupCount,
                                        encoding=handler.encoding,
                                    )
                                else:
                                    continue

                                new_handler.setLevel(handler.level)
                                new_handler.setFormatter(handler.formatter)
                                logger.addHandler(new_handler)

    def get_current_logger(self) -> logging.Logger:
        """获取当前调用者的日志记录器.

        Returns:
            当前模块的日志记录器
        """
        # 获取调用者的信息
        frame = inspect.currentframe()
        if frame is None:
            return self.get_logger("unknown")

        # 跳过几个调用栈
        caller_frame = frame.f_back
        if caller_frame is None:
            return self.get_logger("unknown")

        caller_frame = caller_frame.f_back
        if caller_frame is None:
            return self.get_logger("unknown")

        caller_frame = caller_frame.f_back
        if caller_frame is None:
            return self.get_logger("unknown")

        # 获取模块名称
        module = inspect.getmodule(caller_frame)
        if module is None:
            return self.get_logger("unknown")

        module_name = module.__name__

        # 转换为相对于ngo包的路径
        if module_name.startswith("ngo."):
            return self.get_logger(module_name)
        else:
            return self.get_logger("unknown")

# 全局日志管理器实例
_logger_manager = NGOLogger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器.

    Args:
        name: 日志记录器名称。如果为None，则自动获取调用者的模块名

    Returns:
        日志记录器实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条信息")

        >>> logger = get_logger()  # 自动获取调用者模块名
        >>> logger.debug("这是一条调试信息")
    """
    if name is None:
        return _logger_manager.get_current_logger()
    return _logger_manager.get_logger(name)


def configure_logging(
    level: str = "INFO",
    log_file: str = "ngo.log",
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
):
    """配置日志系统.

    Args:
        level: 日志级别
        log_file: 日志文件路径
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        max_file_size: 最大文件大小（字节）
        backup_count: 备份文件数量

    Examples:
        >>> configure_logging(
        ...     level='DEBUG',
        ...     log_file='debug.log',
        ...     enable_console=True,
        ...     enable_file=True
        ... )
    """
    config = {
        "level": level,
        "file": log_file,
        "enable_console": enable_console,
        "enable_file": enable_file,
        "max_file_size": max_file_size,
        "backup_count": backup_count,
    }
    _logger_manager.update_config(config)


def configure_from_config(config: Dict[str, Any]):
    """从配置字典配置日志系统.

    Args:
        config: 配置字典，包含logging相关的配置项
    """
    log_config = config.get("logging", {})
    configure_logging(
        level=log_config.get("level", "INFO"),
        log_file=log_config.get("file", "ngo.log"),
        enable_console=log_config.get("enable_console", True),
        enable_file=log_config.get("enable_file", True),
        max_file_size=log_config.get("max_file_size", 10 * 1024 * 1024),
        backup_count=log_config.get("backup_count", 5),
    )

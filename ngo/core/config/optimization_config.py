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
NGO 的优化配置管理。

该模块为优化 pass 和 pattern 提供专门的配置管理，包括自动生成和加载 optimization.toml 文件。
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ngo.passes.base import BasePass
from ngo.patterns.base import BasePattern
from ngo.utils.logger import configure_from_config, get_logger
from ngo.core.base import ComponentPriority
from ngo.core.unified_registry import UnifiedRegistry, RegistrationInfo
from ngo.core.config.manager import ConfigError, ConfigManager


@dataclass
class PassConfigEntry:
    """Pass 的配置条目。"""

    name: str
    enabled: bool = True
    priority: str = "NORMAL"
    description: str = ""


@dataclass
class PatternConfigEntry:
    """Pattern 的配置条目。"""

    name: str
    enabled: bool = True
    priority: str = "NORMAL"
    description: str = ""


class OptimizationConfigManager:
    """优化组件的专门配置管理器。"""

    def __init__(self, config_dir: str = "config", auto_sync: bool = True):
        """
        初始化优化配置管理器。

        Args:
            config_dir: 配置文件目录
            auto_sync: 是否在初始化时自动与注册表同步
        """
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "optimization.toml"
        self.config_manager = ConfigManager()
        self.logger = get_logger(__name__)
        self.auto_sync = auto_sync

        # 如果配置目录不存在则创建
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 加载现有配置或创建默认配置
        self._load_or_create_config()

        # 如果启用，则自动与注册表同步
        if self.auto_sync:
            self._auto_sync_with_registry()

    def _auto_sync_with_registry(self) -> None:
        """自动同步注册表信息到配置文件"""
        try:
            from ngo.core.unified_registry import get_registry

            registry = get_registry()

            # 只有当注册表中有组件时才进行同步
            if len(registry._registrations) > 0:
                self.logger.info(
                    f"auto save {len(registry._registrations)} components to config file"
                )
                self.generate_config_from_registry(registry)
        except Exception as e:
            self.logger.warning(f"auto save components to config file failed: {e}")

    def _load_or_create_config(self) -> None:
        """加载现有配置或创建默认配置。"""
        if self.config_file.exists():
            try:
                self.config_manager.load_file(str(self.config_file))
                configure_from_config(self.config_manager.config)
                self.logger.info(
                    f"Loaded optimization configuration from {self.config_file}"
                )
            except ConfigError as e:
                self.logger.warning(
                    f"Failed to load config file: {e}, creating default"
                )
                self._create_default_config()
        else:
            self.logger.info("No optimization configuration found, creating default")
            self._create_default_config()

    def _create_default_config(self) -> None:
        """创建默认优化配置。"""
        default_config = {
            "passes": {},
            "patterns": {},
            "logging": {
                "level": "INFO",
                "file": "ngo.log",
                "enable_console": True,
                "enable_file": True,
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5,
            },
        }
        self.config_manager.from_dict(default_config)
        self.save_config()

    def register_pass_config(self, pass_component: BasePass) -> None:
        """
        Register or update pass configuration.

        Args:
            pass_component: Pass component to register
        """
        if not isinstance(pass_component, BasePass):
            raise ValueError(f"Expected BasePass, got {type(pass_component)}")

        pass_name = pass_component.metadata.name
        enabled = pass_component.config.enabled
        priority = pass_component.config.priority.name

        # Check if configuration already exists
        existing_config = self.config_manager.get(f"passes.{pass_name}")
        if existing_config:
            # Use existing enabled setting but update other fields
            enabled = existing_config.get("enabled", enabled)

        config_entry = PassConfigEntry(
            name=pass_name,
            enabled=enabled,
            priority=priority,
            description=pass_component.metadata.description,
        )

        self.config_manager.set(f"passes.{pass_name}", asdict(config_entry))
        self.logger.debug(
            f"Registered pass config: {pass_name} (enabled={enabled}, priority={priority})"
        )

    def register_pattern_config(self, pattern_component: BasePattern) -> None:
        """
        Register or update pattern configuration.

        Args:
            pattern_component: Pattern component to register
        """
        if not isinstance(pattern_component, BasePattern):
            raise ValueError(f"Expected BasePattern, got {type(pattern_component)}")

        pattern_name = pattern_component.metadata.name

        # Check if configuration already exists
        existing_config = self.config_manager.get(f"patterns.{pattern_name}")
        if existing_config:
            enabled = existing_config.get("enabled", True)
        else:
            enabled = True  # Default to enabled for patterns

        # Use NORMAL as default priority for patterns
        priority = "NORMAL"

        config_entry = PatternConfigEntry(
            name=pattern_name,
            enabled=enabled,
            priority=priority,
            description=pattern_component.metadata.description,
        )

        self.config_manager.set(f"patterns.{pattern_name}", asdict(config_entry))
        self.logger.debug(
            f"Registered pattern config: {pattern_name} (enabled={enabled}, priority={priority})"
        )

    def get_pass_config(self, pass_name: str) -> Optional[PassConfigEntry]:
        """
        Get pass configuration.

        Args:
            pass_name: Name of the pass

        Returns:
            PassConfigEntry or None if not found
        """
        config_dict = self.config_manager.get(f"passes.{pass_name}")
        if config_dict:
            # Ensure name field is set
            if "name" not in config_dict:
                config_dict["name"] = pass_name
            return PassConfigEntry(**config_dict)
        return None

    def get_pattern_config(self, pattern_name: str) -> Optional[PatternConfigEntry]:
        """
        Get pattern configuration.

        Args:
            pattern_name: Name of the pattern

        Returns:
            PatternConfigEntry or None if not found
        """
        config_dict = self.config_manager.get(f"patterns.{pattern_name}")
        if config_dict:
            # Ensure name field is set
            if "name" not in config_dict:
                config_dict["name"] = pattern_name
            return PatternConfigEntry(**config_dict)
        return None

    def apply_pass_config(self, pass_component: BasePass) -> None:
        """
        Apply configuration to a pass component.

        Args:
            pass_component: Pass component to configure
        """
        if not isinstance(pass_component, BasePass):
            raise ValueError(f"Expected BasePass, got {type(pass_component)}")

        pass_name = pass_component.metadata.name
        config = self.get_pass_config(pass_name)

        if config:
            # Apply enabled setting
            pass_component.config.enabled = config.enabled

            # Apply priority setting
            try:
                priority = ComponentPriority[config.priority]
                pass_component.config.priority = priority
            except KeyError:
                self.logger.warning(
                    f"Invalid priority '{config.priority}' for pass {pass_name}, using NORMAL"
                )
                pass_component.config.priority = ComponentPriority.NORMAL

            self.logger.debug(
                f"Applied config to pass {pass_name}: enabled={config.enabled}, priority={config.priority}"
            )
        else:
            # Create default config if none exists
            self.register_pass_config(pass_component)

    def apply_pattern_config(self, pattern_component: BasePattern) -> None:
        """
        Apply configuration to a pattern component.

        Args:
            pattern_component: Pattern component to configure
        """
        if not isinstance(pattern_component, BasePattern):
            raise ValueError(f"Expected BasePattern, got {type(pattern_component)}")

        pattern_name = pattern_component.metadata.name
        config = self.get_pattern_config(pattern_name)

        if config:
            # Apply enabled setting to pattern component
            if hasattr(pattern_component, "config") and hasattr(
                pattern_component.config, "enabled"
            ):
                pattern_component.config.enabled = config.enabled
                self.logger.debug(
                    f"Applied config to pattern {pattern_name}: enabled={config.enabled}"
                )
            else:
                # If pattern doesn't have config.enabled, just log the configuration
                self.logger.debug(
                    f"Pattern {pattern_name} config: enabled={config.enabled} (no config.enabled attribute to apply)"
                )
        else:
            # Create default config if none exists
            self.register_pattern_config(pattern_component)

    def save_config(self) -> None:
        """保存配置到文件。"""
        try:
            self.config_manager.save_file(str(self.config_file))
            self.logger.info(f"Saved optimization configuration to {self.config_file}")
        except ConfigError as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def generate_config_from_registry(self, registry: UnifiedRegistry) -> None:
        """
        从所有注册的组件生成配置

        Args:
            registry: 组件注册表
        """
        self.logger.info("generate optimization config from registry")

        # 处理所有注册的组件
        try:
            # 获取所有注册信息
            registrations = registry._registrations

            for component_name, registration_info in registrations.items():
                if registration_info is None:
                    continue

                component = registration_info.component_class
                if component is None:
                    continue

                if registration_info.component_type == "pass":
                    self._register_pass_from_registration(
                        component_name, registration_info
                    )
                elif registration_info.component_type == "pattern":
                    self._register_pattern_from_registration(
                        component_name, registration_info
                    )

        except Exception as e:
            self.logger.warning(
                f"generate optimization config from registry failed: {e}"
            )

        # Save the configuration
        self.save_config()
        self.logger.info(
            f"Generated optimization configuration with {len(self.config_manager.get('passes', {}))} passes and {len(self.config_manager.get('patterns', {}))} patterns"
        )

    def _register_pass_from_registration(
        self, name: str, registration_info: RegistrationInfo
    ) -> None:
        """
        根据注册信息注册pass配置
        """
        # Check if configuration already exists
        existing_config = self.config_manager.get(f"passes.{name}")
        if existing_config:
            enabled = existing_config.get("enabled", registration_info.enabled)
        else:
            enabled = registration_info.enabled

        config_entry = PassConfigEntry(
            name=name,
            enabled=enabled,
            priority=self._priority_to_string(registration_info.priority),
            description=(
                registration_info.metadata.description
                if registration_info.metadata
                else f"Pass {name}"
            ),
        )

        self.config_manager.set(f"passes.{name}", asdict(config_entry))
        self.logger.debug(
            f"generate pass config from registry: {name} (enabled={enabled}, priority={config_entry.priority})"
        )

    def _register_pattern_from_registration(self, name: str, registration_info) -> None:
        """
        根据注册信息注册pattern配置
        """
        # Check if configuration already exists
        existing_config = self.config_manager.get(f"patterns.{name}")
        if existing_config:
            enabled = existing_config.get("enabled", registration_info.enabled)
        else:
            enabled = registration_info.enabled

        config_entry = PatternConfigEntry(
            name=name,
            enabled=enabled,
            priority="NORMAL",  # Patterns使用默认优先级
            description=(
                registration_info.metadata.description
                if registration_info.metadata
                else f"Pattern {name}"
            ),
        )

        self.config_manager.set(f"patterns.{name}", asdict(config_entry))
        self.logger.debug(
            f"generate pattern config from registry: {name} (enabled={enabled})"
        )

    def _priority_to_string(self, priority: int) -> str:
        """将优先级数字转换为字符串"""
        return ComponentPriority(priority).name

    def get_enabled_passes(self) -> List[str]:
        """获取启用的 pass 名称列表。"""
        passes_config = self.config_manager.get("passes", {})
        return [
            name
            for name, config in passes_config.items()
            if config.get("enabled", True)
        ]

    def get_enabled_patterns(self) -> List[str]:
        """获取启用的 pattern 名称列表。"""
        patterns_config = self.config_manager.get("patterns", {})
        return [
            name
            for name, config in patterns_config.items()
            if config.get("enabled", True)
        ]

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要。"""
        passes_config = self.config_manager.get("passes", {})
        patterns_config = self.config_manager.get("patterns", {})
        settings = self.config_manager.get("settings", {})

        enabled_passes = [
            name
            for name, config in passes_config.items()
            if config.get("enabled", True)
        ]
        enabled_patterns = [
            name
            for name, config in patterns_config.items()
            if config.get("enabled", True)
        ]

        return {
            "total_passes": len(passes_config),
            "enabled_passes": len(enabled_passes),
            "total_patterns": len(patterns_config),
            "enabled_patterns": len(enabled_patterns),
            "enabled_pass_names": enabled_passes,
            "enabled_pattern_names": enabled_patterns,
            "settings": settings,
        }


# Global optimization configuration manager instance
_global_opt_config_manager = None


def get_global_optimization_config_manager(
    config_dir: str = "config",
) -> OptimizationConfigManager:
    """
    Get the global optimization configuration manager instance.

    Args:
        config_dir: Directory for configuration files

    Returns:
        Global OptimizationConfigManager instance
    """
    global _global_opt_config_manager
    # If config_dir changes or manager doesn't exist, create new instance
    if _global_opt_config_manager is None or _global_opt_config_manager.config_dir != Path(config_dir):
        _global_opt_config_manager = OptimizationConfigManager(config_dir)
    return _global_opt_config_manager


def sync_config_with_registry(
    registry: UnifiedRegistry = None, config_dir: str = "config"
) -> None:
    """
    同步配置文件与注册表的信息

    Args:
        registry: 组件注册表，如果为None则使用全局注册表
        config_dir: 配置文件目录
    """
    if registry is None:
        from ngo.core.unified_registry import get_registry
        registry = get_registry()

    config_manager = get_global_optimization_config_manager(config_dir)
    config_manager.generate_config_from_registry(registry)

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
TorchInductor 后端实现。

该模块提供了与 PyTorch 的 TorchInductor 编译后端集成的 NGOBackend 类，
为 PyTorch 模型启用 NGO 优化。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.fx
from torch._dynamo.utils import dynamo_timed
from torch._inductor.utils import InputType

from ngo.utils.logger import get_logger
from ngo.core.config import OptimizationConfigManager, get_global_optimization_config_manager
from ngo.core.engine import OptimizationEngine
from ngo.core.unified_registry import UnifiedRegistry, get_registry, list_passes, list_patterns
from ngo.passes.base import BasePass
from ngo.patterns.base import BasePattern


@dataclass
class NGOBackendOptions:
    """NGO 后端的配置选项。"""

    enable_optimization: bool = True
    optimization_level: int = 0
    enable_profiling: bool = False
    custom_config: Dict[str, Any] = field(default_factory=dict)


class NGOBackend:
    """
    NGO 优化的 PyTorch TorchInductor 后端。

    该后端将 NGO 的优化能力与 PyTorch 的编译管道集成，
    为昇腾硬件加速提供自动图优化。
    """

    def __init__(
        self,
        options: Optional[NGOBackendOptions] = None,
        registry: Optional[UnifiedRegistry] = None,
        config_manager: Optional[OptimizationConfigManager] = None,
    ):
        """
        初始化NGO后端

        Args:
            options: 后端配置选项
            registry: 组件注册表，默认使用全局注册表
            config_manager: 配置管理器，默认使用默认配置
        """
        self._options = options or NGOBackendOptions()
        self._registry = registry or get_registry()
        self._config_manager = (
            config_manager or get_global_optimization_config_manager()
        )
        self._engine = OptimizationEngine(self._registry)
        self._logger = get_logger(f"ngo.{self.__class__.__name__}")

        # 性能跟踪
        self._compilation_count = 0
        self._total_optimization_time = 0.0
        self._cache_hit_count = 0
        
        # 应用配置到组件
        self._apply_configuration_to_components()

        # 如果启用优化，初始化优化系统
        if self._options.enable_optimization:
            try:
                self._initialize_optimization_system()
                self._logger.info("NGO后端初始化成功")
            except Exception as e:
                self._logger.error(f"初始化NGO引擎失败: {e}")
                raise

    @property
    def options(self) -> NGOBackendOptions:
        """获取后端选项"""
        return self._options

    @property
    def engine(self) -> OptimizationEngine:
        """获取优化引擎"""
        return self._engine

    @property
    def config_manager(self) -> OptimizationConfigManager:
        """获取优化配置管理器"""
        return self._config_manager

    def _initialize_optimization_system(self) -> None:
        """初始化优化系统和配置管理"""

        # 生成配置文件（如果不存在）
        if not self._config_manager.config_file.exists():
            self._logger.info("生成优化配置文件")
            self._config_manager.generate_config_from_registry(self._registry)
        else:
            self._logger.info("使用现有优化配置")

        # 记录配置摘要
        config_summary = self._config_manager.get_config_summary()
        self._logger.info(f"优化配置: {config_summary}")

    def _apply_configuration_to_components(self) -> None:
        """为所有注册的优化组件应用配置"""
        self._logger.debug("为注册组件应用配置")

        # 获取所有已注册的组件
        try:
            passes = list_passes()
            patterns = list_patterns()
            all_components = passes + patterns
        except Exception:
            all_components = []

        for component_info in all_components:
            component = self._registry.get_component(component_info.name)
            if component is None:
                self._logger.warning(f"组件 {component_info.name} 未找到")
                continue

            try:
                if hasattr(component, "config") and hasattr(component, "metadata"):
                    # 检查是否是pass
                    if isinstance(component, BasePass):
                        self._config_manager.apply_pass_config(component)
                        self._logger.debug(f"为pass应用配置: {component.metadata.name}")
                    # 检查是否是pattern
                    elif isinstance(component, BasePattern):
                        self._config_manager.apply_pattern_config(component)
                        self._logger.debug(
                            f"为pattern应用配置: {component.metadata.name}"
                        )

            except Exception as e:
                self._logger.warning(f"为组件 {component_name} 应用配置失败: {e}")

    @property
    def compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        return {
            "compilation_count": self._compilation_count,
            "total_optimization_time": self._total_optimization_time,
            "cache_hit_count": self._cache_hit_count,
            "average_optimization_time": (
                self._total_optimization_time / self._compilation_count
                if self._compilation_count > 0
                else 0.0
            ),
        }

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs
    ) -> Callable:
        """
        Compile a graph module using NGO optimization.

        Args:
            gm: Torch FX GraphModule to optimize
            example_inputs: Example input tensors for shape inference
            kwargs: Additional keyword arguments
        Returns:
            Compiled function
        """

        self._logger.debug(f"Compiling graph with {len(gm.graph.nodes)} nodes")

        start_time = time.time()
        self._compilation_count += 1

        try:
            # Add graph information to context
            self.engine.context.set_graph_module(gm)
            self.engine.context.set_data("example_inputs", example_inputs)
            self.engine.context.set_data("graph_size", len(gm.graph.nodes))

            # Add backend configuration
            self.engine.context.set_config("backend_options", self._options.__dict__)

            # 如果启用优化，执行优化
            if self._options.enable_optimization and hasattr(self._engine, "execute"):
                try:
                    optimized_gm = self._optimize_graph(gm)
                except Exception as e:
                    self._logger.warning(f"优化失败: {e}")
                    optimized_gm = gm
            else:
                optimized_gm = gm

            # Compile the optimized graph
            compiled_fn = self._compile_graph(optimized_gm, example_inputs, **kwargs)

            # Update statistics
            optimization_time = time.time() - start_time
            self._total_optimization_time += optimization_time

            self._logger.debug(f"Compilation completed in {optimization_time:.4f}s")

            return compiled_fn

        except Exception as e:
            self._logger.error(f"Compilation failed: {e}")
            raise e

    def _optimize_graph(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Optimize a graph using NGO engine.

        Args:
            gm: Graph module to optimize

        Returns:
            Optimized graph module
        """
        self._logger.debug("Starting NGO optimization")

        try:
            # Execute optimization workflow
            result = self._engine.execute(
                config={
                    "optimization_level": self._options.optimization_level,
                    "enable_profiling": self._options.enable_profiling,
                }
            )
            
            if result.success:
                return self.engine.context.graph_module
            else:
                self._logger.warning(f"Optimization failed, using original graph: {result.errors}")
                return gm

        except Exception as e:
            self._logger.error(f"Optimization failed, using original graph: {e}")
            return gm

    def _compile_graph(
        self,
        gm: torch.fx.GraphModule,  # 图模块(GraphModule)类型的参数，表示需要编译的图模块
        example_inputs: Sequence[InputType],  # 示例输入参数序列，用于指导编译过程
        **kwargs
    ) -> Callable:  # 返回一个可调用的(Callable)对象，即编译后的函数
        """
        Compile a graph module using TorchInductor.

        Args:
            gm: Graph module to compile
            example_inputs: Example input tensors

        Returns:
            Compiled function
        """
        # 导入 compile_fx
        with dynamo_timed("inductor_import", log_pt2_compile_event=True):
            from torch._inductor.compile_fx import compile_fx

        # 直接使用 compile_fx，它会处理所有预处理步骤
        options = kwargs.get("options", {})
        
        return compile_fx(gm, example_inputs, config_patches=options)

    def reset_stats(self) -> None:
        """Reset compilation statistics."""
        self._compilation_count = 0
        self._total_optimization_time = 0.0
        self._cache_hit_count = 0
        self._logger.info("Compilation statistics reset")

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        获取后端诊断信息

        Returns:
            诊断信息字典
        """
        engine_status = (
            self._engine.get_status() if self._options.enable_optimization else {}
        )

        try:
            component_count = (
                len(self._registry.get_all_components()) if self._registry else 0
            )
        except Exception:
            component_count = 0

        return {
            "backend": "NGO",
            "version": "0.1.0",
            "enabled": self._options.enable_optimization,
            "compilation_stats": self.compilation_stats,
            "engine_status": engine_status,
            "registry_components": component_count,
            "options": self._options.__dict__,
        }


def create_ngo_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs
) -> Callable:
    """
    Create a new NGO backend instance for PyTorch entry point registration.

    This function is used by PyTorch's entry point system to automatically
    register the NGO backend when the package is installed.

    Args:
        gm: Graph module to compile
        example_inputs: Example input tensors
        kwargs: Additional keyword arguments
        
    Returns:
        Compiled function
    """
    backend = NGOBackend()
    return backend(gm, example_inputs, **kwargs)

#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torchrec 版本适配器抽象基类。

设计原则：
- 只包含「已经发生过版本差异」的方法
- 新增方法使用「默认实现 + 可覆盖」而非 @abstractmethod
- 这样新增差异点时，旧适配器无需修改（OCP）
"""

from abc import ABC, abstractmethod
from dataclasses import fields
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple, Type

from torchrec.distributed.embedding_sharding import KJTListSplitsAwaitable
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import Awaitable
from torchrec.modules.embedding_configs import EmbeddingTableConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.distributed.sharding.rw_sharding import RwSparseFeaturesDist


class TorchRecVersionAdapter(ABC):
    """torchrec 版本适配器抽象基类 默认支持1.1.0版本"""

    @property
    @abstractmethod
    def version(self) -> Tuple[int, int, int]: ...

    def get_learning_rate(self, common_args: Any, optimizer_args: Any) -> float:
        """torchrec 1.1.0: learning_rate 来自 optimizer_args"""
        return optimizer_args.learning_rate

    def create_sharding_infos(
        self,
        instance: Any,
        module: Any,
        table_name_to_parameter_sharding: Any,
        prefix: str,
        fused_params: Optional[Dict],
    ) -> Dict:
        """torchrec 1.1.0: 调用模块级函数"""
        from torchrec.distributed.embeddingbag import create_sharding_infos_by_sharding

        return create_sharding_infos_by_sharding(module, table_name_to_parameter_sharding, prefix, fused_params)

    def build_args_kwargs(self, batch: Any, forward_args: Any) -> Tuple:
        """torchrec 1.1.0/1.2.0: 模块级函数"""
        from torchrec.distributed.train_pipeline.utils import _build_args_kwargs

        return _build_args_kwargs(batch, forward_args)

    def get_kernel_learning_rate(self, kernel: Any) -> float:
        """torchrec 1.1.0: kernel 尚未提供 get_learning_rate"""
        return 0.0

    def get_virtual_table_feature_num_buckets(self, instance: Any) -> Tuple[Optional[List[int]], bool]:
        """torchrec 1.1.0/1.2.0: 不支持，返回默认值"""
        return (None, False)

    # ═══════════════════════════════════════════════════════════════
    # 兼容性工具方法（兼容所有版本的参数/枚举）
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def get_output_dtensor(env: Any, fused_params: Optional[Dict]) -> bool:
        """获取 output_dtensor 参数，默认从 env 获取"""
        return getattr(env, "output_dtensor", fused_params.get("output_dtensor", False) if fused_params else False)

    @staticmethod
    def make_embedding_table_config(**kwargs: Any) -> EmbeddingTableConfig:
        """构造 EmbeddingTableConfig，自动过滤不支持的字段。

        覆盖差异点：
        - 1.5.0 新增: total_num_buckets, use_virtual_table,
                    virtual_table_eviction_policy, enable_embedding_update
        - 未来版本新增字段自动兼容
        """
        supported = {f.name for f in fields(EmbeddingTableConfig)}
        return EmbeddingTableConfig(**{k: v for k, v in kwargs.items() if k in supported})

    @staticmethod
    def make_awaitable(awaitable_type: Type[Any], **kwargs: Any) -> Any:
        """构造 Awaitable 对象，自动过滤不支持的参数。

        覆盖差异点：
        - 1.5.0 新增: module_fqn, sharding_types, resize_awaitables
        - 未来版本新增参数自动兼容
        """
        supported = set(signature(awaitable_type.__init__).parameters) - {"self"}
        return awaitable_type(**{k: v for k, v in kwargs.items() if k in supported})

    def make_kjt_list_splits_awaitable(
        self,
        awaitables: List[Awaitable[Awaitable[KeyedJaggedTensor]]],
        ctx: Any,
        module_fqn: Optional[str],
        sharding_types: List[str],
    ) -> KJTListSplitsAwaitable:
        """构造 KJTListSplitsAwaitable。

        覆盖差异点：
        - 1.5.0 新增: module_fqn, sharding_types 参数
        """
        return self.make_awaitable(
            KJTListSplitsAwaitable,
            awaitables=awaitables,
            ctx=ctx,
            module_fqn=module_fqn,
            sharding_types=sharding_types,
        )

    @staticmethod
    def filter_rw_sparse_features_dist_kwargs(**kwargs: Any) -> Dict[str, Any]:
        """过滤 RwSparseFeaturesDist 构造函数参数。

        覆盖差异点：
        - 1.5.0 新增: virtual_table_feature_num_buckets, has_uneven_virtual_tables
        """
        supported = set(signature(RwSparseFeaturesDist.__init__).parameters) - {"self"}
        return {k: v for k, v in kwargs.items() if k in supported}

    @staticmethod
    def embedding_compute_kernel_values(*names: str) -> set:
        """安全获取 EmbeddingComputeKernel 枚举值集合。

        覆盖差异点：
        - 1.5.0 新增: SSD_VIRTUAL_TABLE, DRAM_VIRTUAL_TABLE
        - 未来版本新增枚举值自动兼容
        """
        result = set()
        for name in names:
            kernel = getattr(EmbeddingComputeKernel, name, None)
            if kernel is not None:
                result.add(kernel.value)
        return result

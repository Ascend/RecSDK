#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

"""声明式版本差异表。

每个版本只声明「相对于上一版本，哪些方法的行为变了」。
新增版本 = 在 VERSION_DIFFS 末尾加一项。
API 回退 = 在 overrides 中覆盖回旧行为。

设计原则：
- 行为实现为模块级纯函数，方便独立测试
- 差异表按时间顺序排列，自文档化
- 工厂函数从基线开始依次应用差异覆盖
"""

from typing import Callable, Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# 行为实现函数（纯函数，可独立测试）
# ═══════════════════════════════════════════════════════════════


def _lr_from_common(self, common_args, optimizer_args):
    """torchrec 1.2.0+: learning_rate 来自 common_args"""
    return common_args.learning_rate


def _sharding_by_method(self, instance, module, sharding, prefix, fused_params):
    """torchrec 1.2.0+: 调用实例方法"""
    return instance.create_grouped_sharding_infos(module, sharding, prefix, fused_params)


def _build_args_method(self, batch, forward_args):
    """torchrec 1.5.0+: forward_args 实例方法"""
    if hasattr(forward_args, "build_args_kwargs"):
        return forward_args.build_args_kwargs(batch)
    raise RuntimeError("forward_args.build_args_kwargs not available in torchrec 1.5.0+")


def _lr_from_kernel_method(self, kernel):
    """torchrec 1.2.0+: kernel 提供 get_learning_rate() 方法"""
    return kernel.get_learning_rate()


def _get_virtual_table_feature_num_buckets_supported(self, instance):
    """torchrec 1.5.0+: 支持获取buckets的函数"""
    return instance._get_virtual_table_feature_num_buckets()


def check_config_new_item_v120(self, config) -> None:
    """检查v1.2.0新增的配置项"""
    if hasattr(config, "input_dim") and config.input_dim is not None:
        raise ValueError(f"The config.input_dim only support None, but got {config.input_dim}")


def check_config_new_item_v150(self, config) -> None:
    """检查v1.5.0新增的配置项"""
    # v1.2.0新增字段
    self.check_config_new_item_v120(config)

    # v1.5.0新增字段
    if hasattr(config, "total_num_buckets") and config.total_num_buckets is not None:
        raise ValueError(f"The config.total_num_buckets only support None, but got {config.total_num_buckets}")
    if hasattr(config, "use_virtual_table") and config.use_virtual_table:
        raise ValueError(f"The config.use_virtual_table only support False, but got {config.use_virtual_table}")
    if hasattr(config, "virtual_table_eviction_policy") and config.virtual_table_eviction_policy is not None:
        raise ValueError(
            f"The config.virtual_table_eviction_policy only support None, but got {config.virtual_table_eviction_policy}"
        )
    if hasattr(config, "enable_embedding_update") and config.enable_embedding_update:
        raise ValueError(
            f"The config.enable_embedding_update only support False, but got {config.enable_embedding_update}"
        )


# ═══════════════════════════════════════════════════════════════
# 版本差异表（唯一需要修改的地方）
# ═══════════════════════════════════════════════════════════════

VERSION_DIFFS: list[tuple[Tuple[int, int, int], Dict[str, Callable]]] = [
    # ── TorchRec v1.2.0 方法差异，基于 TorchRec v1.1.0 ──
    (
        (1, 2, 0),
        {
            "get_learning_rate": _lr_from_common,
            "create_sharding_infos": _sharding_by_method,
            "get_kernel_learning_rate": _lr_from_kernel_method,
            "check_embedding_config_new_item": check_config_new_item_v120,
        },
    ),
    # ── TorchRec v1.5.0 方法差异，基于 TorchRec v1.2.0，且会继承 v1.2.0 的差异方法 ──
    (
        (1, 5, 0),
        {
            "build_args_kwargs": _build_args_method,
            "get_virtual_table_feature_num_buckets": _get_virtual_table_feature_num_buckets_supported,
            "check_config_new_item_v120": check_config_new_item_v120,  # check_config_new_item_v150依赖此方法，需注册
            "check_embedding_config_new_item": check_config_new_item_v150,
        },
    ),
    # ── 未来版本在此追加 ──
    # ── 如果某版本回退了行为，直接覆盖回旧实现即可 ──
]

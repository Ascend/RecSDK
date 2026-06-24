#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


# ═══════════════════════════════════════════════════════════════
# 版本差异表（唯一需要修改的地方）
# ═══════════════════════════════════════════════════════════════

VERSION_DIFFS: list[tuple[Tuple[int, int, int], Dict[str, Callable]]] = [
    # ── 1.2.0: learning_rate 移到 common_args + sharding_infos 改为实例方法 + kernel 提供 get_learning_rate ──
    (
        (1, 2, 0),
        {
            "get_learning_rate": _lr_from_common,
            "create_sharding_infos": _sharding_by_method,
            "get_kernel_learning_rate": _lr_from_kernel_method,
        },
    ),
    # ── 1.5.0: build_args_kwargs 改为方法 + output_dtensor 优先从 env ──
    (
        (1, 5, 0),
        {
            "build_args_kwargs": _build_args_method,
            "get_virtual_table_feature_num_buckets": _get_virtual_table_feature_num_buckets_supported,
        },
    ),
    # ── 未来版本在此追加 ──
    # ── 如果某版本回退了行为，直接覆盖回旧实现即可 ──
]

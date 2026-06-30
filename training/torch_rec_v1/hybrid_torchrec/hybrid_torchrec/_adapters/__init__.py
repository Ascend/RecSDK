#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""适配器工厂：根据当前 torchrec 版本自动构建适配器。

从基线实现开始，依次应用每个版本的差异覆盖，
最终得到一个包含所有正确行为的适配器实例。

采用单例模式，确保全局只有一个适配器实例。
"""

import logging
from typing import Tuple

from hybrid_torchrec._adapters._adapter_base import TorchRecVersionAdapter
from hybrid_torchrec._adapters._version import TORCH_REC_VERSION
from hybrid_torchrec._adapters._version_diff import VERSION_DIFFS


# 单例缓存
_adapter_instance: TorchRecVersionAdapter = None


def _build_methods(target_version: Tuple[int, int, int]) -> dict:
    methods = dict()
    for ver, overrides in VERSION_DIFFS:
        if target_version >= ver:
            methods.update(overrides)
    return methods


def _create_adapter() -> TorchRecVersionAdapter:
    """创建适配器实例（单例模式）。

    首次调用时创建实例并缓存，后续调用返回同一实例。
    """
    global _adapter_instance

    if _adapter_instance is None:
        methods = _build_methods(TORCH_REC_VERSION)
        AdapterCls = type(
            f"_Adapter_{TORCH_REC_VERSION[0]}_{TORCH_REC_VERSION[1]}_{TORCH_REC_VERSION[2]}",
            (TorchRecVersionAdapter,),
            {**methods, "version": TORCH_REC_VERSION},
        )
        _adapter_instance = AdapterCls()

    return _adapter_instance


# 模块导入时创建单例实例
adapter: TorchRecVersionAdapter = _create_adapter()
logging.info("Created adapter, adapter.version: %s", adapter.version)

#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sysconfig

import torch

from hybrid_torchrec._adapters._version import TORCH_REC_VERSION
from hybrid_torchrec.modules.hash_embeddingbag import (
    HashEmbeddingBagCollection,
    HashEmbeddingBagConfig,
    HybridHashTable,  # noqa: F401
)

# Backward-compatible boolean flags: legacy code may import these names
# directly. New code should consume the version adapter (see
# hybrid_torchrec._adapters.adapter) for behavioural differences.
IS_TORCH_REC_120 = TORCH_REC_VERSION == (1, 2, 0)
IS_TORCH_REC_150 = TORCH_REC_VERSION == (1, 5, 0)

__all__ = ["HashEmbeddingBagCollection", "HashEmbeddingBagConfig"]

# 适配老版本的cust_op自定义算子和fbgemm_npu_api.so
try:
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
except FileNotFoundError:
    logging.warning("libfbgemm_npu_api.so does not exist")
except Exception as e:
    logging.warning("libfbgemm_npu_api.so failed to load: %s", e)

# 适配新版本rec_ops自定义算子包和fbgemm_ascend算子包
try:
    import fbgemm_ascend  # noqa: F401
    import rec_ops  # noqa: F401
except ModuleNotFoundError:
    logging.warning("fbgemm_ascend or rec_ops module does not exist")
except Exception as e:
    logging.warning("fbgemm_ascend or rec_ops module failed to load: %s", e)

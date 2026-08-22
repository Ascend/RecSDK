#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

# Do NOT import fbgemm_ascend here: it ships a conflicting
# split_embedding_codegen_lookup_adagrad_function (A5 impl) under torch.ops.fbgemm,
# which collides with the A2-correct NPU kernels from libfbgemm_npu_api.so.
# Load the A2 cust_op torch_plugin directly; fbgemm_gpu is already pulled in by
# torchrec when needed. If the A2 NPU lib cannot be loaded, fail loudly instead
# of silently ignoring the error so the missing-operator root cause is obvious.
try:
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
except Exception as e:  # nosec B110
    raise ImportError(
        "Failed to load A2 NPU lib libfbgemm_npu_api.so from "
        f"{sysconfig.get_path('purelib')}: {e}. Note: this path requires the "
        "cust_op A2 NPU plugin to be installed; fbgemm_ascend is intentionally "
        "not imported here to avoid the A5/A2 TORCH_LIBRARY conflict."
    ) from e

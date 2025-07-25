#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sysconfig

import torch

from hybrid_torchrec.modules.hash_embeddingbag import (
    HashEmbeddingBagCollection,
    HashEmbeddingBagConfig,
    HybridHashTable,
)

__all__ = ["HashEmbeddingBagCollection", "HashEmbeddingBagConfig"]

try:
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
except FileNotFoundError as e:
    logging.warning(f"libfbgemm_npu_api.so is not exist")
except Exception as e:
    logging.warning(f"libfbgemm_npu_api.so failed to load: {e}")

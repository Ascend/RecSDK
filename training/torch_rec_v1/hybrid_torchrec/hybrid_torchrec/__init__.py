#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

# Load the NPU operator implementations.
# - rec_cust_ops (cust_op torch_plugin) provides the mxrec-namespace lookup/backward adapters
#   and registers its A2 OPP vendors (some WITHOUT aclnn wrappers).
# - fbgemm_ascend provides the torch.ops.fbgemm NPU dispatch (incl. the
#   split_embedding_codegen_lookup_*_function_pt2 ops that need PrivateUse1, the
#   missing of which caused "weights[0] must be a CUDA tensor") and its A2 OPP vendors
#   (WITH aclnn wrappers).
# IMPORTANT: both packages prepend their own OPP vendors to ASCEND_CUSTOM_OPP_PATH on
# import, so the LAST imported package wins name collisions. fbgemm_ascend must be
# imported AFTER rec_cust_ops, otherwise rec_cust_ops' aclnn-less split_embedding_codegen_forward_*
# vendor shadows fbgemm_ascend's aclnn-backed one and the NPU kernel segfaults.
try:
    import rec_cust_ops  # noqa: F401  — cust_op torch_plugin (mxrec-namespace adapters) + ASCEND_CUSTOM_OPP_PATH
except Exception as e:  # nosec B110
    raise ImportError(
        "Failed to import rec_cust_ops (cust_op torch_plugin). It provides the mxrec-namespace "
        "lookup/backward adapters and sets ASCEND_CUSTOM_OPP_PATH."
    ) from e


try:
    import fbgemm_ascend  # noqa: F401
except Exception as e:  # nosec B110
    raise ImportError(
        "Failed to import fbgemm_ascend (NPU operator package). "
        "split_embedding_codegen_lookup_*_function_pt2 needs its PrivateUse1 dispatch."
    ) from e

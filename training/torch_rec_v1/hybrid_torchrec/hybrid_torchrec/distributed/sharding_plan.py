#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List

from torch import nn

from hybrid_torchrec.distributed.embeddingbag import HybridEmbeddingBagCollectionSharder
from hybrid_torchrec.distributed.hash_embeddingbag import HybridHashEmbeddingBagCollectionSharder
from torchrec.distributed.types import ModuleSharder, ShardingEnv


def get_default_hybrid_sharders(host_env: ShardingEnv) -> List[ModuleSharder[nn.Module]]:
    if host_env is None:
        raise ValueError("host_env should not be None")
    if host_env.process_group is None:
        raise RuntimeError("process_group should not be None")
    if host_env.process_group._get_backend_name() != "gloo":
        raise RuntimeError("Rec SDK Torch only support host dit with gloo")

    return [
        cast(ModuleSharder[nn.Module], HybridEmbeddingBagCollectionSharder(host_env)),
        cast(ModuleSharder[nn.Module], HybridHashEmbeddingBagCollectionSharder(host_env)),
    ]
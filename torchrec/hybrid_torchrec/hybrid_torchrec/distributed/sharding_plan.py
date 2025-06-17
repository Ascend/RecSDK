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
from hybrid_torchrec.distributed.hash_embeddingbag import (
    HybridHashEmbeddingBagCollectionSharder,
)
from hybrid_torchrec.distributed.embedding import HybridEmbeddingCollectionSharder
from hybrid_torchrec.distributed.hash_embedding import (
    HybridHashEmbeddingCollectionSharder,
)

from torchrec.distributed.types import ShardingEnv
from torchrec.distributed.types import (
    ModuleSharder,
)


def get_default_hybrid_sharders(
    host_env: ShardingEnv,
) -> List[ModuleSharder[nn.Module]]:
    return [
        cast(ModuleSharder[nn.Module], HybridEmbeddingBagCollectionSharder(host_env)),
        cast(
            ModuleSharder[nn.Module], HybridHashEmbeddingBagCollectionSharder(host_env)
        ),
        cast(ModuleSharder[nn.Module], HybridEmbeddingCollectionSharder(host_env)),
        cast(ModuleSharder[nn.Module], HybridHashEmbeddingCollectionSharder(host_env)),
    ]

#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchrec_embcache.distributed.configs import EmbCacheEmbeddingConfig, EmbCacheEmbeddingBagConfig
from hybrid_torchrec.constants import (
    EMBEDDINGS_DIM_ALIGNMENT,
    MAX_EMBEDDINGS_DIM,
    MAX_NUM_EMBEDDINGS,
)
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.embedding_bag import EmbCacheEmbeddingBagCollection

from torchrec import EmbeddingBagConfig, EmbeddingConfig


@pytest.mark.parametrize(
    "kwargs, err_pattern",
    [
        (
            {"name": "t", "embedding_dim": EMBEDDINGS_DIM_ALIGNMENT, "num_embeddings": 0},
            r"The num_embeddings should be in",
        ),
        (
            {
                "name": "t",
                "embedding_dim": EMBEDDINGS_DIM_ALIGNMENT,
                "num_embeddings": MAX_NUM_EMBEDDINGS + 1,
            },
            r"The num_embeddings should be in",
        ),
        (
            {"name": "t", "embedding_dim": EMBEDDINGS_DIM_ALIGNMENT - 1, "num_embeddings": 8},
            r"The embedding dim should be in",
        ),
        (
            {"name": "t", "embedding_dim": MAX_EMBEDDINGS_DIM + 1, "num_embeddings": 8},
            r"The embedding dim should be in",
        ),
        (
            {"name": "t", "embedding_dim": EMBEDDINGS_DIM_ALIGNMENT + 4, "num_embeddings": 8},
            fr"The embedding dim should be a multiple of {EMBEDDINGS_DIM_ALIGNMENT}," 
        ),
        (
            {
                "name": "t",
                "embedding_dim": EMBEDDINGS_DIM_ALIGNMENT,
                "num_embeddings": 8,
                "weight_init_min": 1.0,
                "weight_init_max": 1.0,
            },
            r"The weight_init_min should be less than weight_init_max"
        ),
        (
            {
                "name": "t",
                "embedding_dim": EMBEDDINGS_DIM_ALIGNMENT,
                "num_embeddings": 8,
                "weight_init_min": 2.0,
                "weight_init_max": 1.0,
            },
            r"The weight_init_min should be less than weight_init_max"
        ),
    ],
)
def test_embcache_embedding_config_invalid(kwargs, err_pattern):
    with pytest.raises(ValueError, match=err_pattern):
        EmbCacheEmbeddingConfig(**kwargs)
    
    with pytest.raises(ValueError, match=err_pattern):
        EmbCacheEmbeddingBagConfig(**kwargs)

    with pytest.raises(ValueError, match=err_pattern):
        config = EmbeddingConfig(init_fn=lambda *args: None, **kwargs)
        EmbCacheEmbeddingCollection([config], 1, 1, [1])
    
    with pytest.raises(ValueError, match=err_pattern):
        config = EmbeddingBagConfig(init_fn=lambda *args: None, **kwargs)
        EmbCacheEmbeddingBagCollection([config], 1, 1, [1])

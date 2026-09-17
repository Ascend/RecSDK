# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
import torch

def batch_gather_embeddings(
        rowwise_indices: torch.Tensor,
        embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        rowwise_indices: (B, N) x int, where each entry is in [0, X).
        embeddings: (B, X, D,) x float.

    Returns:
        (B, N, D,) x float, embeddings corresponding to rowwise_indices.
    """
    _, N = rowwise_indices.size()
    B, X, D = embeddings.size()
    flattened_indices = (
            rowwise_indices
            + torch.arange(
        start=0, end=B, step=1, dtype=rowwise_indices.dtype, device=rowwise_indices.device
    ).unsqueeze(1).expand(-1, N) * X
    )
    return embeddings.view(-1, D)[flattened_indices, :].reshape(rowwise_indices.size() + (D,))

#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, List

import torch
from fbgemm_gpu.split_embedding_codegen_lookup_invokers.lookup_adagrad import (
    CommonArgs,
    OptimizerArgs,
    VBEMetadata,
    Momentum,
)


@dataclass
class HybridCommonArgs:
    placeholder_autograd_tensor: torch.Tensor
    dev_weights: torch.Tensor
    host_weights: torch.Tensor
    uvm_weights: torch.Tensor
    lxu_cache_weights: torch.Tensor
    weights_placements: torch.Tensor
    weights_offsets: torch.Tensor
    D_offsets: torch.Tensor
    total_D: int
    max_D: int
    hash_size_cumsum: torch.Tensor
    total_hash_size_bits: int
    indices: torch.Tensor
    offsets: torch.Tensor
    hash_indices: torch.Tensor
    unique_indices: torch.Tensor
    unique_offset: torch.Tensor
    unique_inverse: torch.Tensor
    hash_indices2address: torch.Tensor
    pooling_mode: int
    indice_weights: Optional[torch.Tensor]
    feature_requires_grad: Optional[torch.Tensor]
    lxu_cache_locations: torch.Tensor
    uvm_cache_stats: Optional[torch.Tensor]
    output_dtype: int
    vbe_metadata: VBEMetadata
    is_experimental: bool
    use_uniq_cache_locations_bwd: bool
    use_homogeneous_placements: bool
    # Attribute `learning_rate` to adapt for torchrec 1.2.0 version.
    # In torchrec 1.1.0 version, this field has been stored in optimizer_args.
    learning_rate: float
    table_grad_accumulate_offsets: torch.Tensor
    grad_accumulate: List[torch.Tensor]
    grad_accumulate_offsets: Optional[torch.Tensor]
    use_optimize: bool


@dataclass
class HybridCommonArgsAggregation(HybridCommonArgs):
    table_offsets_multi: torch.Tensor
    indices_multi_step: torch.Tensor
    offsets_multi_step: torch.Tensor
    unique_multi_step: torch.Tensor
    unique_offset_multi_step: torch.Tensor
    unique_inverse_multi_step: torch.Tensor

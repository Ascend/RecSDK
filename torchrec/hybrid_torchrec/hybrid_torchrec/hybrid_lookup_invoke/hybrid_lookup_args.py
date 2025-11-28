#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import NamedTuple, Optional, List

import torch
from fbgemm_gpu.split_embedding_codegen_lookup_invokers.lookup_adagrad import (
    CommonArgs,
    OptimizerArgs,
    VBEMetadata,
    Momentum,
)


class HybridCommonArgs(NamedTuple):
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
    rows_per_table: Optional[torch.Tensor]
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
    table_offsets: torch.Tensor
    grad_accumulate: List[torch.Tensor]
    grad_accumulate_offsets: Optional[torch.Tensor]
    use_optimize: bool


class HybridCommonArgs_aggregation(NamedTuple):
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
    rows_per_table: Optional[torch.Tensor]
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
    table_offsets: torch.Tensor
    grad_accumulate: List[torch.Tensor]
    grad_accumulate_offsets: Optional[torch.Tensor]
    use_optimize: bool
    table_offsets_multi: torch.Tensor
    indices_multi_step: torch.Tensor
    offsets_multi_step: torch.Tensor
    unique_multi_step: torch.Tensor
    unique_offset_multi_step: torch.Tensor
    unique_inverse_multi_step: torch.Tensor

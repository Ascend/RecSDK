#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fbgemm_gpu.split_embedding_codegen_lookup_invokers.lookup_adagrad import (
    OptimizerArgs,
    Momentum,
)

from hybrid_torchrec.hybrid_lookup_invoke.hybrid_lookup_args import HybridCommonArgs


def invoke(
    common_args: HybridCommonArgs,
    optimizer_args: OptimizerArgs,
    momentum1: Momentum,
    momentum2: Momentum,
    iteration: int = 0,
) -> torch.Tensor:
    vbe_metadata = common_args.vbe_metadata

    return torch.ops.fbgemm.split_embedding_codegen_lookup_adam_function(
        # common_args
        placeholder_autograd_tensor=common_args.placeholder_autograd_tensor,
        dev_weights=common_args.dev_weights,
        uvm_weights=common_args.uvm_weights,
        lxu_cache_weights=common_args.lxu_cache_weights,
        weights_placements=common_args.weights_placements,
        weights_offsets=common_args.weights_offsets,
        D_offsets=common_args.D_offsets,
        total_D=common_args.total_D,
        max_D=common_args.max_D,
        hash_size_cumsum=common_args.hash_size_cumsum,
        rows_per_table=common_args.rows_per_table,
        total_hash_size_bits=common_args.total_hash_size_bits,
        indices=common_args.indices,
        offsets=common_args.offsets,
        hash_indices=common_args.hash_indices,
        unique_ids=common_args.unique_indices,
        unique_offsets=common_args.unique_offset,
        unique_inverse=common_args.unique_inverse,
        pooling_mode=common_args.pooling_mode,
        indice_weights=common_args.indice_weights,
        feature_requires_grad=common_args.feature_requires_grad,
        lxu_cache_locations=common_args.lxu_cache_locations,
        uvm_cache_stats=common_args.uvm_cache_stats,
        # VBE metadata
        B_offsets=vbe_metadata.B_offsets,
        vbe_output_offsets_feature_rank=vbe_metadata.output_offsets_feature_rank,
        vbe_B_offsets_rank_per_feature=vbe_metadata.B_offsets_rank_per_feature,
        max_B=vbe_metadata.max_B,
        max_B_feature_rank=vbe_metadata.max_B_feature_rank,
        vbe_output_size=vbe_metadata.output_size,
        # optimizer_args
        gradient_clipping=optimizer_args.gradient_clipping,
        max_gradient=optimizer_args.max_gradient,
        stochastic_rounding=optimizer_args.stochastic_rounding,  # if optimizer == none
        learning_rate=optimizer_args.learning_rate,
        eps=optimizer_args.eps,
        beta1=optimizer_args.beta1,
        beta2=optimizer_args.beta2,
        weight_decay=optimizer_args.weight_decay,
        # momentum1
        momentum1_dev=momentum1.dev,
        momentum1_uvm=momentum1.uvm,
        momentum1_offsets=momentum1.offsets,
        momentum1_placements=momentum1.placements,
        # momentum2
        momentum2_dev=momentum2.dev,
        momentum2_uvm=momentum2.uvm,
        momentum2_offsets=momentum2.offsets,
        momentum2_placements=momentum2.placements,
        # prev_iter
        prev_iter_dev=None,
        # iter
        iter=iteration,
        output_dtype=common_args.output_dtype,
        is_experimental=common_args.is_experimental,
        use_uniq_cache_locations_bwd=common_args.use_uniq_cache_locations_bwd,
        use_homogeneous_placements=common_args.use_homogeneous_placements,
        apply_global_weight_decay=False,
        gwd_lower_bound=0.0,
        use_optimize=common_args.use_optimize,
        # grad_accumulate
        table_offsets=common_args.table_offsets
    )

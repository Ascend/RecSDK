#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
import logging
import torch
from fbgemm_gpu.split_embedding_codegen_lookup_invokers.lookup_adagrad import (
    OptimizerArgs,
    Momentum,
)

from hybrid_torchrec.hybrid_lookup_invoke.hybrid_lookup_args import HybridCommonArgs, HybridCommonArgsAggregation
from hybrid_torchrec import IS_TORCH_REC_120


def check_unique_valid(common_args: HybridCommonArgs):
    if common_args.hash_indices is None or common_args.unique_indices is None:
        return
    hash_indices = common_args.hash_indices.to("cpu")
    offsets = common_args.offsets.to("cpu")
    batch_size = (
        common_args.offsets.shape[0] - 1
    ) // common_args.weights_offsets.shape[0]
    unique_indices = common_args.unique_indices.to("cpu")
    unique_offset = common_args.unique_offset.to("cpu")
    unique_inverse = common_args.unique_inverse.to("cpu")

    logging.info("batch size %s", batch_size)
    logging.info("unique_indices %s", unique_indices)
    logging.info("unique_offset %s", unique_offset)
    logging.info("unique_inverse %s", unique_inverse)
    result = []
    for i in range(unique_offset.shape[0] - 1):
        unique_start = unique_offset[i].item()
        logging.info("unique_start %s", unique_start)
        unique = unique_indices[unique_start:]
        batch_start = offsets[i * batch_size]
        batch_end = offsets[(i + 1) * batch_size]
        logging.info("batch_start %s", batch_start)
        logging.info("batch_end %s", batch_end)
        inverse = unique_inverse[batch_start:batch_end]
        result.append(torch.index_select(unique, dim=0, index=inverse))
    result = torch.concat(result)
    if not (result == hash_indices).all():
        raise RuntimeError("Valid unique info")


def invoke(
    common_args: HybridCommonArgs,
    optimizer_args: OptimizerArgs,
    momentum1: Momentum,
    iteration: int = 0,
    apply_global_weight_decay: bool = False,
    # only pass prev_iter_dev since prev_iter is never created on UVM
    prev_iter_dev: Optional[torch.Tensor] = None,
    gwd_lower_bound: float = 0.0,
) -> torch.Tensor:
    vbe_metadata = common_args.vbe_metadata

    # 当前版本特定
    indices = common_args.indices
    if common_args.hash_indices is not None:
        indices = common_args.hash_indices

    if common_args.host_weights.numel() > 0:
        num_offsets = common_args.D_offsets.numel() - 1
        vbe: bool = vbe_metadata.B_offsets is not None
        if vbe:
            # create offsets with fixed batch size max_b
            # not efficient but for now we just need a functional implementation for CPU
            parse_vbe_offset(common_args, num_offsets, vbe_metadata)
        else:
            offsets = common_args.offsets
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_adagrad_function_cpu(
            # common_args
            host_weights=common_args.host_weights, weights_placements=common_args.weights_placements,
            weights_offsets=common_args.weights_offsets, D_offsets=common_args.D_offsets, total_D=common_args.total_D,
            max_D=common_args.max_D, hash_size_cumsum=common_args.hash_size_cumsum,
            total_hash_size_bits=common_args.total_hash_size_bits, indices=indices, offsets=common_args.offsets,
            pooling_mode=common_args.pooling_mode, indice_weights=common_args.indice_weights,
            feature_requires_grad=common_args.feature_requires_grad,
            # optimizer_args
            gradient_clipping=optimizer_args.gradient_clipping, max_gradient=optimizer_args.max_gradient,
            stochastic_rounding=optimizer_args.stochastic_rounding,
            learning_rate=_get_lr_param(common_args, optimizer_args),
            eps=optimizer_args.eps,
            # momentum1
            momentum1_host=momentum1.host, momentum1_offsets=momentum1.offsets,
            momentum1_placements=momentum1.placements,
        )
        if vbe:
            return parse_vbe_output_offset(common_args, num_offsets, output, vbe_metadata)
        else:
            return output

    return torch.ops.fbgemm.split_embedding_codegen_lookup_adagrad_function(
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
        total_hash_size_bits=common_args.total_hash_size_bits,
        indices=indices,
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
        learning_rate=_get_lr_param(common_args, optimizer_args),
        eps=optimizer_args.eps,
        # momentum1
        momentum1_dev=momentum1.dev,
        momentum1_uvm=momentum1.uvm,
        momentum1_offsets=momentum1.offsets,
        momentum1_placements=momentum1.placements,
        # prev_iter
        prev_iter_dev=prev_iter_dev,
        # iter
        iter=iteration,
        output_dtype=common_args.output_dtype,
        is_experimental=common_args.is_experimental,
        use_uniq_cache_locations_bwd=common_args.use_uniq_cache_locations_bwd,
        use_homogeneous_placements=common_args.use_homogeneous_placements,
        apply_global_weight_decay=apply_global_weight_decay,
        gwd_lower_bound=gwd_lower_bound,
        use_optimize=common_args.use_optimize,
        # grad_accumulate
        grad_accumulate=common_args.grad_accumulate,
        grad_accumulate_offsets=common_args.grad_accumulate_offsets,
        table_grad_accumulate_offsets=common_args.table_grad_accumulate_offsets,
    )


def invoke_grad_aggregation(
        common_args: HybridCommonArgsAggregation,
        optimizer_args: OptimizerArgs,
        momentum1: Momentum,
        iteration: int = 0,
        apply_global_weight_decay: bool = False,
        # only pass prev_iter_dev since prev_iter is never created on UVM
        prev_iter_dev: Optional[torch.Tensor] = None,
        gwd_lower_bound: float = 0.0,
) -> torch.Tensor:
    vbe_metadata = common_args.vbe_metadata

    indices = common_args.indices
    if common_args.hash_indices is not None:
        indices = common_args.hash_indices

    if common_args.host_weights.numel() > 0:
        num_offsets = common_args.D_offsets.numel() - 1
        vbe: bool = vbe_metadata.B_offsets is not None
        if vbe:
            # create offsets with fixed batch size max_b
            # not efficient but for now we just need a functional implementation for CPU
            max_b = vbe_metadata.max_B
            offsets = torch.empty([num_offsets * max_b + 1], dtype=common_args.offsets.dtype,
                                  device=common_args.offsets.device)
            for t in range(num_offsets):
                b_offsets = vbe_metadata.B_offsets
                if not isinstance(b_offsets, torch.Tensor):
                    raise TypeError("b_offsets must be a torch.Tensor")
                begin = b_offsets[t]
                end = b_offsets[t + 1]
                offsets[t * max_b: t * max_b + end - begin] = common_args.offsets[begin: end]
                offsets[t * max_b + end - begin: (t + 1) * max_b] = common_args.offsets[end]
            offsets[-1] = common_args.offsets[-1]
        else:
            offsets = common_args.offsets
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_adagrad_function_cpu(
            # common_args
            host_weights=common_args.host_weights,
            weights_placements=common_args.weights_placements,
            weights_offsets=common_args.weights_offsets,
            D_offsets=common_args.D_offsets,
            total_D=common_args.total_D,
            max_D=common_args.max_D,
            hash_size_cumsum=common_args.hash_size_cumsum,
            total_hash_size_bits=common_args.total_hash_size_bits,
            indices=indices,
            offsets=common_args.offsets,
            pooling_mode=common_args.pooling_mode,
            indice_weights=common_args.indice_weights,
            feature_requires_grad=common_args.feature_requires_grad,
            # optimizer_args
            gradient_clipping=optimizer_args.gradient_clipping,
            max_gradient=optimizer_args.max_gradient,
            stochastic_rounding=optimizer_args.stochastic_rounding,
            learning_rate=_get_lr_param(common_args, optimizer_args),
            eps=optimizer_args.eps,
            # momentum1
            momentum1_host=momentum1.host,
            momentum1_offsets=momentum1.offsets,
            momentum1_placements=momentum1.placements,

        )
        if vbe:
            output_new = torch.empty([vbe_metadata.output_size], dtype=output.dtype, device=output.device)
            b_offsets_rank_per_feature = vbe_metadata.B_offsets_rank_per_feature
            if not isinstance(b_offsets_rank_per_feature, torch.Tensor):
                raise TypeError("b_offsets_rank_per_feature must be a torch.Tensor")
            output_offsets_feature_rank = vbe_metadata.output_offsets_feature_rank
            if not isinstance(output_offsets_feature_rank, torch.Tensor):
                raise TypeError("output_offsets_feature_rank must be a torch.Tensor")
            num_features = b_offsets_rank_per_feature.size(1) - 1
            for r in range(num_features):
                d_offset = 0
                for t in range(num_offsets):
                    o_begin = output_offsets_feature_rank[r * num_offsets + t].item()
                    o_end = output_offsets_feature_rank[r * num_offsets + t + 1].item()
                    dim = common_args.D_offsets[t + 1].item() - common_args.D_offsets[t].item()
                    b_begin = b_offsets_rank_per_feature[t][r].item()
                    b_end = b_offsets_rank_per_feature[t][r + 1].item()
                    if o_end - o_begin != (b_end - b_begin) * dim:
                        raise ValueError("Assertion failed: o_end - o_begin != (b_end - b_begin) * dim")
                    output_new[o_begin: o_end] = output[b_begin: b_end, d_offset: d_offset + dim].flatten()
                    d_offset += dim
            return output_new
        else:
            return output

    result = torch.ops.fbgemm.split_embedding_codegen_lookup_adagrad_function_grad_aggregation(
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
        total_hash_size_bits=common_args.total_hash_size_bits,
        indices=indices,
        offsets=common_args.offsets,
        indices_multi_step=common_args.indices_multi_step,
        offsets_multi_step=common_args.offsets_multi_step,
        hash_indices=common_args.hash_indices,
        unique_ids=common_args.unique_indices,
        unique_offsets=common_args.unique_offset,
        unique_inverse=common_args.unique_inverse,
        unique_multi_step=common_args.unique_multi_step,
        unique_offset_multi_step=common_args.unique_offset_multi_step,
        unique_inverse_multi_step=common_args.unique_inverse_multi_step,
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
        learning_rate=_get_lr_param(common_args, optimizer_args),
        eps=optimizer_args.eps,
        # momentum1
        momentum1_dev=momentum1.dev,
        momentum1_uvm=momentum1.uvm,
        momentum1_offsets=momentum1.offsets,
        momentum1_placements=momentum1.placements,
        # prev_iter
        prev_iter_dev=prev_iter_dev,
        # iter
        iter=iteration,
        output_dtype=common_args.output_dtype,
        is_experimental=common_args.is_experimental,
        use_uniq_cache_locations_bwd=common_args.use_uniq_cache_locations_bwd,
        use_homogeneous_placements=common_args.use_homogeneous_placements,
        apply_global_weight_decay=apply_global_weight_decay,
        gwd_lower_bound=gwd_lower_bound,
        # grad_accumulate
        grad_accumulate=common_args.grad_accumulate,
        grad_accumulate_offsets=common_args.grad_accumulate_offsets,
        use_optimize=common_args.use_optimize,
        table_grad_accumulate_offsets=common_args.table_grad_accumulate_offsets,
        table_offsets_multi=common_args.table_offsets_multi
    )
    return result


def _get_lr_param(common_args, optimizer_args):
    return common_args.learning_rate if IS_TORCH_REC_120 else optimizer_args.learning_rate


def parse_vbe_output_offset(common_args, num_offsets, output, vbe_metadata):
    output_new = torch.empty([vbe_metadata.output_size], dtype=output.dtype, device=output.device)
    b_offsets_rank_per_feature = vbe_metadata.B_offsets_rank_per_feature
    if not isinstance(b_offsets_rank_per_feature, torch.Tensor):
        raise TypeError("b_offsets_rank_per_feature must be a torch.Tensor")
    output_offsets_feature_rank = vbe_metadata.output_offsets_feature_rank
    if not isinstance(output_offsets_feature_rank, torch.Tensor):
        raise TypeError("output_offsets_feature_rank must be a torch.Tensor")
    num_features = b_offsets_rank_per_feature.size(1) - 1
    for r in range(num_features):
        d_offset = 0
        for t in range(num_offsets):
            o_begin = output_offsets_feature_rank[r * num_offsets + t].item()
            o_end = output_offsets_feature_rank[r * num_offsets + t + 1].item()
            dim = common_args.D_offsets[t + 1].item() - common_args.D_offsets[t].item()
            b_begin = b_offsets_rank_per_feature[t][r].item()
            b_end = b_offsets_rank_per_feature[t][r + 1].item()
            if o_end - o_begin != (b_end - b_begin) * dim:
                raise ValueError("Assertion failed: o_end - o_begin != (b_end - b_begin) * dim")
            output_new[o_begin: o_end] = output[b_begin: b_end, d_offset: d_offset + dim].flatten()
            d_offset += dim
    return output_new


def parse_vbe_offset(common_args, num_offsets, vbe_metadata):
    max_b = vbe_metadata.max_B
    offsets = torch.empty([num_offsets * max_b + 1], dtype=common_args.offsets.dtype,
                          device=common_args.offsets.device)
    for t in range(num_offsets):
        b_offsets = vbe_metadata.B_offsets
        if not isinstance(b_offsets, torch.Tensor):
            raise TypeError("b_offsets must be a torch.Tensor")
        begin = b_offsets[t]
        end = b_offsets[t + 1]
        offsets[t * max_b: t * max_b + end - begin] = common_args.offsets[begin: end]
        offsets[t * max_b + end - begin: (t + 1) * max_b] = common_args.offsets[end]
    offsets[-1] = common_args.offsets[-1]

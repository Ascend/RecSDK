#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
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

import torch
import torch.distributed as dist


class DistributedEmbeddingFunction(torch.autograd.Function):
    """Custom autograd function for distributed embedding lookup.

    This implementation shards the embedding table across multiple devices and
    performs distributed lookup using all-to-all communication.
    """

    @staticmethod
    def forward(ctx, expected_lookup_ids, local_embedding, shard_start, shard_end):
        """Forward pass for distributed embedding lookup.

        Args:
            ctx: Context object to save information for backward pass
            expected_lookup_ids: Tensor containing input IDs
            local_embedding: Tensor containing local embedding weights
            shard_start: Starting index of the local embedding shard
            shard_end: Ending index of the local embedding shard

        Returns:
            Tensor containing embedding vectors for the input IDs
        """
        # Get distributed training information
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{dist.get_rank()}")
        elif torch.npu.is_available():
            device = torch.device(f"npu:{dist.get_rank()}")
        else:
            device = "cpu"

        # Extract input parameters
        embedding_dim = local_embedding.size(-1)

        # Save information for backward pass
        ctx.embedding_weight_shape = local_embedding.shape
        ctx.embedding_dim = embedding_dim

        # Special case: single process (no distribution needed)
        if world_size == 1:
            ctx.input_ids = expected_lookup_ids
            lookup_res = torch.embedding(local_embedding, expected_lookup_ids)
            return lookup_res

        # Step 1: Deduplicate input IDs to reduce communication volume
        expected_lookup_unique_ids, expected_lookup_inverse_indices, _ = torch.unique(
            expected_lookup_ids.flatten(), return_inverse=True, return_counts=True
        )

        # Step 2: Calculate how many IDs each rank needs to process
        # This tensor will store how many IDs this rank needs to send to each other rank
        expected_lookup_unique_ids_dist_counts = torch.zeros(
            world_size, dtype=torch.int64, device=device
        )

        # Determine which IDs belong to which rank's shard
        for r in range(world_size):
            start = r * ((shard_end - shard_start) * world_size) // world_size
            end = (r + 1) * ((shard_end - shard_start) * world_size) // world_size
            mask = (expected_lookup_unique_ids >= start) & (
                expected_lookup_unique_ids < end
            )
            expected_lookup_unique_ids_dist_counts[r] = mask.sum()

        # Step 3: Exchange count information with all ranks
        # This tensor will store how many IDs this rank will receive from each other rank
        real_lookup_ids_dist_counts = torch.zeros_like(
            expected_lookup_unique_ids_dist_counts
        )
        dist.all_to_all_single(
            real_lookup_ids_dist_counts,
            expected_lookup_unique_ids_dist_counts,
        )

        # Step 4: Prepare buffer for receiving IDs from other ranks
        real_lookup_global_ids = torch.zeros(
            real_lookup_ids_dist_counts.sum(), device=device, dtype=torch.int64
        )

        # Step 5: Exchange actual IDs with all ranks
        dist.all_to_all_single(
            real_lookup_global_ids,
            expected_lookup_unique_ids,
            output_split_sizes=real_lookup_ids_dist_counts.tolist(),
            input_split_sizes=expected_lookup_unique_ids_dist_counts.tolist(),
        )

        # Step 6: Convert global IDs to local indices within this rank's shard
        real_lookup_local_ids = real_lookup_global_ids - shard_start

        # Step 7: Perform local embedding lookup
        real_lookup_res = torch.embedding(local_embedding, real_lookup_local_ids)

        # Step 8: Prepare buffer for receiving embedding vectors from other ranks
        expected_lookup_unique_res = torch.zeros(
            len(expected_lookup_unique_ids) * embedding_dim,
            device=device,
        )

        # Step 9: Exchange embedding vectors with all ranks
        dist.all_to_all_single(
            expected_lookup_unique_res,
            real_lookup_res.flatten(),
            output_split_sizes=(
                expected_lookup_unique_ids_dist_counts * embedding_dim
            ).tolist(),
            input_split_sizes=(real_lookup_ids_dist_counts * embedding_dim).tolist(),
        )

        # Step 10: Reshape the received embedding vectors
        expected_lookup_unique_res = expected_lookup_unique_res.reshape(
            -1, embedding_dim
        )

        # Step 11: Restore original order using inverse indices
        expected_lookup_res = torch.index_select(
            expected_lookup_unique_res, 0, expected_lookup_inverse_indices
        )

        # Save information for backward pass
        ctx.expected_lookup_unique_ids = expected_lookup_unique_ids
        ctx.expected_lookup_inverse_indices = expected_lookup_inverse_indices
        ctx.real_lookup_local_ids = real_lookup_local_ids
        ctx.expected_lookup_unique_ids_dist_counts = (
            expected_lookup_unique_ids_dist_counts
        )
        ctx.real_lookup_ids_dist_counts = real_lookup_ids_dist_counts
        return expected_lookup_res.reshape(*expected_lookup_ids.shape, -1)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for distributed embedding lookup.

        Args:
            ctx: Context object with saved information from forward pass
            grad_output: Gradient of the loss with respect to the forward output

        Returns:
            Tuple of gradients for each input to forward:
            (None, embedding_grad, None, None, None)
            where embedding_grad is the gradient for local_embedding
        """
        # Retrieve saved information from context
        embedding_weight_shape = ctx.embedding_weight_shape
        embedding_dim = ctx.embedding_dim

        # Initialize gradient tensor for embedding weights
        embedding_grad = torch.zeros(
            embedding_weight_shape, device=grad_output.device, dtype=grad_output.dtype
        )

        # Reshape gradient to match embedding dimensions
        flat_grad_output = grad_output.reshape(-1, embedding_dim)

        # Special case: single process (no distribution needed)
        if dist.get_world_size() == 1:
            input_ids = ctx.input_ids
            # Accumulate gradients at the positions specified by input_ids
            embedding_grad.index_add_(0, input_ids.flatten(), flat_grad_output)
            # Return gradients for each input to forward
            # None for input_ids (IntTensor)
            # embedding_grad for local_embedding
            # None for shard_start, shard_end (integers, not tensors)
            return None, embedding_grad, None, None

        # Retrieve saved information for distributed backward
        expected_lookup_unique_ids = ctx.expected_lookup_unique_ids
        expected_lookup_inverse_indices = ctx.expected_lookup_inverse_indices

        # Step 1: Aggregate gradients for unique IDs
        expected_lookup_embedding_grad = torch.zeros(
            len(expected_lookup_unique_ids),
            embedding_dim,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        expected_lookup_embedding_grad.index_add_(
            0, expected_lookup_inverse_indices.flatten(), flat_grad_output
        )

        # Retrieve count information for all-to-all communication
        expected_lookup_unique_ids_dist_counts = (
            ctx.expected_lookup_unique_ids_dist_counts
        )
        real_lookup_ids_dist_counts = ctx.real_lookup_ids_dist_counts

        # Step 2: Prepare buffer for receiving gradients from other ranks
        real_lookup_embedding_grad = torch.zeros(
            sum(real_lookup_ids_dist_counts) * embedding_dim,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )

        # Step 3: Exchange gradients with all ranks (reverse of forward pass)
        dist.all_to_all_single(
            real_lookup_embedding_grad,
            expected_lookup_embedding_grad.flatten(),
            output_split_sizes=(real_lookup_ids_dist_counts * embedding_dim).tolist(),
            input_split_sizes=(
                expected_lookup_unique_ids_dist_counts * embedding_dim
            ).tolist(),
        )

        # Step 4: Reshape received gradients
        real_lookup_local_ids = ctx.real_lookup_local_ids
        real_lookup_embedding_grad = real_lookup_embedding_grad.reshape(
            -1, embedding_dim
        )

        # Step 5: Accumulate gradients at the positions specified by local_ids
        embedding_grad.index_add_(
            0, real_lookup_local_ids.flatten(), real_lookup_embedding_grad
        )

        # Return gradients for each input to forward
        # None for input_ids (IntTensor)
        # embedding_grad for local_embedding
        # None for shard_start, shard_end (integers, not tensors)
        return None, embedding_grad, None, None

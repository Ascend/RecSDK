#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext


class HybridSequenceShardingContext(SequenceShardingContext):

    def __init__(
        self,
        # Fields of EmbeddingShardingContext
        batch_size_per_rank: Optional[List[int]] = None,
        batch_size_per_rank_per_feature: Optional[List[List[int]]] = None,
        batch_size_per_feature_pre_a2a: Optional[List[int]] = None,
        variable_batch_per_feature: bool = False,
        # Fields of SequenceShardingContext
        features_before_input_dist: Optional[KeyedJaggedTensor] = None,
        input_splits: Optional[List[int]] = None,
        output_splits: Optional[List[int]] = None,
        sparse_features_recat: Optional[torch.Tensor] = None,
        unbucketize_permute_tensor: Optional[torch.Tensor] = None,
        lengths_after_input_dist: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            batch_size_per_rank,
            batch_size_per_rank_per_feature,
            batch_size_per_feature_pre_a2a,
            variable_batch_per_feature,
            features_before_input_dist,
            input_splits,
            output_splits,
            sparse_features_recat,
            unbucketize_permute_tensor,
            lengths_after_input_dist,
        )


    def to(self, device: torch.device, non_blocking: bool = False) -> "HybridSequenceShardingContext":
        new_context = HybridSequenceShardingContext(
            self.batch_size_per_rank,
            self.batch_size_per_rank_per_feature,
            self.batch_size_per_feature_pre_a2a,
            self.variable_batch_per_feature,
            self.features_before_input_dist,
            self.input_splits,
            self.output_splits,
            self.sparse_features_recat,
            self.unbucketize_permute_tensor,
            self.lengths_after_input_dist,
        )
        
        if self.features_before_input_dist is not None:
            new_context.features_before_input_dist = self.features_before_input_dist.to(device=device,
                                                                                        non_blocking=non_blocking)
            
        if self.sparse_features_recat is not None:
            new_context.sparse_features_recat = self.sparse_features_recat.to(device=device, non_blocking=non_blocking)

        if self.unbucketize_permute_tensor is not None:
            new_context.unbucketize_permute_tensor = self.unbucketize_permute_tensor.to(device=device,
                                                                                        non_blocking=non_blocking)

        if self.lengths_after_input_dist is not None:
            new_context.lengths_after_input_dist = self.lengths_after_input_dist.to(device=device,
                                                                                    non_blocking=non_blocking)

        return new_context

    def pin_memory(self):
        if self.features_before_input_dist is not None:
            self.features_before_input_dist = self.features_before_input_dist.pin_memory()

        if self.sparse_features_recat is not None:
            self.sparse_features_recat = self.sparse_features_recat.pin_memory()      

        if self.unbucketize_permute_tensor is not None:
            self.unbucketize_permute_tensor = self.unbucketize_permute_tensor.pin_memory()

        if self.lengths_after_input_dist is not None:
            self.lengths_after_input_dist = self.lengths_after_input_dist.pin_memory()            
        return self  
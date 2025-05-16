#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
from typing import List

import torch
from torch.autograd.profiler import record_function


torch.ops.load_library(os.path.join(os.path.dirname(__file__), "libhybrid_cpp.so"))


class HashMapBase(torch.nn.Module):
    def forward(self, ids: torch.Tensor, high_precison: bool) -> tuple[torch.Tensor]:
        pass


class IdsMapper(HashMapBase):
    """
    This class is primarily used for managing global ids.
    Its core functionality is to convert global ids into indices,
    which represent offsets in the embedding table stored on the NPU device.
    Additionally, it provides functionality for evicting ids,
    exporting all ids along with their corresponding indices,
    retrieving timestamps for the ids, and returning the current count
    of ids stored in the IdsMapper.
    """

    def __init__(self, n):
        super().__init__()
        self.ids_mapper = torch.classes.hybrid.IdsMapper(n)
        self.n = n

    def forward(self, ids: torch.Tensor, high_precison: bool):
        with record_function("## ids2indices ##"):
            result, unique, unique_inverse = self.ids_mapper.ids2indices_unique(
                ids, high_precison
            )
            return result, unique, unique_inverse

    def ids2indices_unique_out(
        self,
        ids: torch.Tensor,
        hash_indices: torch.Tensor,
        offset: torch.Tensor,
        unique: torch.Tensor,
        unique_inverse: torch.Tensor,
        unique_offset: List[int],
        tensor_i: int,
    ):
        self.ids_mapper.ids2indices_unique_out(
            ids, hash_indices, offset, unique, unique_inverse, unique_offset, tensor_i
        )

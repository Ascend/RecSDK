#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Dict, List, Tuple

import torch
from torchrec.sparse.jagged_tensor import (
    JaggedTensor,
    KeyedJaggedTensor,
)


class JaggedTensorWithCount(JaggedTensor):
    _fields = [
        "_counts"
    ]

    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        counts: Optional[torch.Tensor] = None,
    ) -> None:
        if counts is not None and values.size() != counts.size():
            raise ValueError(f"counts size must same with values, but got timestamp size:{counts.size()},"
                             f" values size:{values.size()}.")

        super().__init__(values, weights, lengths, offsets)

        # values中每个ids出现次数，分桶去重时会进行计算，input_dist all2all会做集合通信，post dist input时做count记录
        self._counts = counts

    @property
    def counts(self):
        return self._counts


class KeyedJaggedTensorWithCount(KeyedJaggedTensor):
    _fields = [
        "_counts"
    ]

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        # Below exposed to ensure torch.script-able
        stride_per_key: Optional[List[int]] = None,
        length_per_key: Optional[List[int]] = None,
        lengths_offset_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
        jt_dict: Optional[Dict[str, JaggedTensor]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            keys,
            values,
            weights,
            lengths,
            offsets,
            stride,
            stride_per_key_per_rank,
            stride_per_key,
            length_per_key,
            lengths_offset_per_key,
            offset_per_key,
            index_per_key,
            jt_dict,
            inverse_indices
        )

        self._counts: torch.Tensor = counts

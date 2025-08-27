#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Dict, List, Tuple

import torch

from torch.autograd.profiler import record_function
from torchrec.sparse.jagged_tensor import (
    _permute_tensor_by_segments,
    _sum_by_splits,
    JaggedTensor,
    KeyedJaggedTensor,
)
from torchrec.pt2.checks import is_non_strict_exporting
from .extended_jagged_tensor import ExtendedJaggedTensor, KeyedExtendedJaggedTensor


class JaggedTensorWithCount(ExtendedJaggedTensor):
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
        super().__init__(
            values=values,
            extra=counts,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            extra_field_name="counts"
        )
        # values中每个ids出现次数，分桶去重时会进行计算，input_dist all2all会做集合通信，post dist input时做count记录
        self._counts = counts

    @property
    def counts(self):
        return self._counts


class KeyedJaggedTensorWithCount(KeyedExtendedJaggedTensor):
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
            keys=keys,
            values=values,
            extra=counts,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=stride_per_key,
            length_per_key=length_per_key,
            lengths_offset_per_key=lengths_offset_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
            inverse_indices=inverse_indices,
            extra_field_name="counts"
        )
        self._counts: torch.Tensor = counts

    @property
    def counts(self) -> torch.Tensor:
        return self._counts

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensorWithCount]) -> "KeyedJaggedTensorWithCount":
        """
        Constructs a KeyedJaggedTensorWithCount from a dictionary of JaggedTensorWithCounts.
        Automatically calls `kjt.sync()` on newly created KJT.

        Args:
            jt_dict (Dict[str, JaggedTensorWithCount]): dictionary of JaggedTensorWithCounts.

        Returns:
            KeyedJaggedTensorWithCount: constructed KeyedJaggedTensorWithCount.
        """
        return KeyedJaggedTensorWithCount.from_jt_dict_base(jt_dict, "counts")

    def split(self, segments: List[int]) -> List["KeyedJaggedTensorWithCount"]:
        split_list: List[KeyedJaggedTensorWithCount] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            keys: List[str] = self._keys[start:end]

            stride, stride_per_key_per_rank = (
                (None, self.stride_per_key_per_rank()[start:end])
                if self.variable_stride_per_key()
                else (self._stride, None)
            )
            if segment == len(self._keys):
                # no torch slicing required
                split_list.append(
                    KeyedJaggedTensorWithCount(
                        keys=self._keys,
                        values=self._values,
                        counts=self._counts,
                        weights=self.weights_or_none(),
                        lengths=self._lengths,
                        offsets=self._offsets,
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=self._length_per_key,
                        offset_per_key=self._offset_per_key,
                        index_per_key=self._index_per_key,
                        jt_dict=self._jt_dict,
                    )
                )
            elif segment == 0:
                empty_int_list: List[int] = torch.jit.annotate(List[int], [])
                split_list.append(
                    KeyedJaggedTensorWithCount(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        counts=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._counts.dtype,
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else torch.tensor(
                                empty_int_list,
                                device=self.device(),
                                dtype=self.weights().dtype,
                            )
                        ),
                        lengths=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        offsets=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=None,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                    )
                )
            else:
                split_length_per_key = _length_per_key[start:end]
                split_list.append(
                    KeyedJaggedTensorWithCount(
                        keys=keys,
                        values=self._values[start_offset:end_offset],
                        counts=(
                            self._counts[start_offset:end_offset]
                            if self._counts is not None
                            else None
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else self.weights()[start_offset:end_offset]
                        ),
                        lengths=self.lengths()[
                            self.lengths_offset_per_key()[
                                start
                            ]: self.lengths_offset_per_key()[end]
                        ],
                        offsets=None,
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=split_length_per_key,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                    )
                )
            start = end
            start_offset = end_offset
        return split_list

    def permute(
        self,
        permute_order: List[int],
        permuted_length_per_key: List[int],
    ) -> "KeyedJaggedTensorWithCount":
        permuted_length_per_key_sum = sum(permuted_length_per_key)
        # 避免直接访问受保护的成员
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            # 使用公共API替代受保护成员的访问
            if permuted_length_per_key_sum <= 0:
                raise ValueError("permuted_length_per_key_sum needs to be greater than 0")

        with record_function("KeyedJaggedTensorWithCount.permute"):
            permuted_values = _permute_tensor_by_segments(
                self._values,
                self._offsets,
                permute_order,
                permuted_length_per_key,
            )
            permuted_counts = _permute_tensor_by_segments(
                self._counts,
                self._offsets,
                permute_order,
                permuted_length_per_key,
            )
            permuted_lengths = _sum_by_splits(
                torch.ones_like(self._values),
                self._offsets,
                permute_order,
                permuted_length_per_key,
            )
            permuted_offsets = torch.cumsum(
                torch.cat([torch.tensor([0]), permuted_lengths]), dim=0
            )

            return KeyedJaggedTensorWithCount(
                keys=self._keys,
                values=permuted_values,
                counts=permuted_counts,
                weights=None,
                lengths=permuted_lengths,
                offsets=permuted_offsets,
                stride=self._stride,
                stride_per_key_per_rank=self._stride_per_key_per_rank,
                length_per_key=permuted_length_per_key,
                offset_per_key=None,
                index_per_key=self._index_per_key,
                jt_dict=None,
            )

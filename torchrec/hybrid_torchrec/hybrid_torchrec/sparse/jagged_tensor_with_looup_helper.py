#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Dict, List, Optional, Tuple

import torch

from torchrec.sparse.jagged_tensor import (
    JaggedTensor,
    KeyedJaggedTensor,
    _jagged_tensor_string,
)


class KeyedJaggedTensorWithLookHelper(KeyedJaggedTensor):

    _fields = [
        "_values",
        "_hash_indices",
        "_unique_offset",
        "_unique_indices",
        "_unique_ids",
        "_unique_inverse",
        "_weights",
        "_lengths",
        "_offsets",
    ]

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        hash_indices: torch.Tensor = None,
        unique_indices: torch.Tensor = None,
        unique_offset: torch.Tensor = None,
        unique_offset_host: torch.Tensor = None,
        unique_offset_list_single: torch.Tensor = None,
        unique_ids: torch.Tensor = None,
        unique_inverse: torch.Tensor = None,
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

        self._hash_indices = hash_indices
        self._unique_offset = unique_offset
        self._unique_ids = unique_ids
        self._unique_indices = unique_indices
        self._unique_inverse = unique_inverse
        self._unique_offset_host = unique_offset_host
        self._unique_offset_list_single = unique_offset_list_single

    def __str__(self) -> str:
        if len(self._keys) == 0 or self._offsets is None and self._lengths is None:
            return "KeyedJaggedTensor()\n"
        offsets = self.offsets()

        return (
            "KeyedJaggedTensor({\n"
            + ",\n".join(
                [
                    "    "
                    + _jagged_tensor_string(
                        self._keys[index],
                        self._values,
                        self._weights,
                        offsets,
                        sum(self.stride_per_key()[:index]),
                        sum(self.stride_per_key()[: index + 1]),
                    )
                    for index in range(len(self._keys))
                ]
            )
            + "\n})\n"
        )

    @property
    def hash_indices(self) -> torch.Tensor:
        return self._hash_indices

    @property
    def unique_indices(self) -> torch.Tensor:
        return self._unique_indices
    
    @unique_indices.setter
    def unique_indices(self, value: torch.Tensor) -> None:
        self._unique_indices = value

    @property
    def unique_offset(self) -> torch.Tensor:
        return self._unique_offset

    @property
    def unique_offset_host(self) -> torch.Tensor:
        if (self._unique_offset_host is None):
            logging.warning("There is no unique_offset_host, must d2h")
            return self._unique_offset.cpu().tolist()
        else:
            return self._unique_offset_host
    
    @property
    def unique_ids(self) -> torch.Tensor:
        return self._unique_ids

    @property
    def unique_inverse(self) -> torch.Tensor:
        return self._unique_inverse
    
    @staticmethod
    def from_offsets_sync(
        keys: List[str],
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        return NotImplemented

    @staticmethod
    def from_lengths_sync(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        return NotImplemented

    @staticmethod
    def concat(
        kjt_list: List["KeyedJaggedTensor"],
    ) -> "KeyedJaggedTensor":
        return NotImplemented

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "KeyedJaggedTensor":
        return NotImplemented

    @staticmethod
    def empty_like(kjt: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        return NotImplemented

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor":
        return NotImplemented

    @staticmethod
    def dist_init(
        keys: List[str],
        tensors: List[torch.Tensor],
        variable_stride_per_key: bool,
        num_workers: int,
        recat: Optional[torch.Tensor],
        stride_per_rank: Optional[List[int]],
        stagger: int = 1,
    ) -> "KeyedJaggedTensor":
        return NotImplemented

    def split(self, segments: List[int]) -> List["KeyedJaggedTensorWithLookHelper"]:
        split_list: List[KeyedJaggedTensorWithLookHelper] = []
        start = 0
        start_offset = 0
        start_unique_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            end_unique_offset = self.unique_offset_host[end]
            keys: List[str] = self._keys[start:end]
            stride, stride_per_key_per_rank = ((None, self.stride_per_key_per_rank()[start:end])
                                               if self.variable_stride_per_key() else (self._stride, None))
            if segment == len(self._keys):
                # no torch slicing required
                self.split_with_segment_equal_keys_length(split_list, stride, stride_per_key_per_rank)
            elif segment == 0:
                self.split_with_segment_zero(keys, split_list, stride, stride_per_key_per_rank)
            else:
                split_length_per_key = _length_per_key[start:end]
                if start_unique_offset == end_unique_offset:
                    raise RuntimeError(
                        "start_unique_offset == end_unique_offset, it caused by spliting keys on the same table")
                split_list.append(KeyedJaggedTensorWithLookHelper(
                    keys=keys, values=self._values[start_offset:end_offset],
                    hash_indices=(self._hash_indices[start_offset:end_offset]
                                  if self._hash_indices is not None else None),
                    unique_indices=(self._unique_indices[start_unique_offset:end_unique_offset]
                                    if self._hash_indices is not None else None),
                    unique_offset=(self._unique_offset[start:end] - self._unique_offset[start]
                                   if self._unique_offset is not None else None),
                    unique_ids=(self._unique_ids[start_unique_offset:end_unique_offset]
                                if self._unique_ids is not None else None),
                    unique_inverse=(self._unique_inverse[start_offset:end_offset]
                                    if self._unique_inverse is not None else None),
                    weights=(None if self.weights_or_none() is None else self.weights()[start_offset:end_offset]),
                    lengths=self.lengths()[self.lengths_offset_per_key()[start]: self.lengths_offset_per_key()[end]],
                    offsets=None, stride=stride, stride_per_key_per_rank=stride_per_key_per_rank,
                    length_per_key=split_length_per_key, offset_per_key=None, index_per_key=None, jt_dict=None,)
                )
            start = end
            start_offset = end_offset
            start_unique_offset = end_unique_offset
        return split_list


    def split_with_segment_zero(self, keys, split_list, stride, stride_per_key_per_rank):
        empty_int_list: List[int] = torch.jit.annotate(List[int], [])
        split_list.append(
            KeyedJaggedTensorWithLookHelper(
                keys=keys,
                values=torch.tensor(
                    empty_int_list,
                    device=self.device(),
                    dtype=self._values.dtype,
                ),
                hash_indices=torch.tensor(
                    empty_int_list,
                    device=self.device(),
                    dtype=self._values.dtype,
                ),
                unique_indices=torch.tensor(
                    empty_int_list,
                    device=self.device(),
                    dtype=self._values.dtype,
                ),
                unique_offset=torch.tensor(
                    empty_int_list,
                    device=self.device(),
                    dtype=self._values.dtype,
                ),
                unique_ids=torch.tensor(
                    empty_int_list,
                    device=torch.device('cpu'),
                    dtype=self._values.dtype,
                ),
                unique_inverse=torch.tensor(
                    empty_int_list,
                    device=self.device(),
                    dtype=self._values.dtype,
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

    def split_with_segment_equal_keys_length(self, split_list, stride, stride_per_key_per_rank):
        split_list.append(
            KeyedJaggedTensorWithLookHelper(
                keys=self._keys,
                values=self._values,
                hash_indices=self._hash_indices,
                unique_indices=self.unique_indices,
                unique_offset=self._unique_offset,
                unique_ids=self._unique_ids,
                unique_inverse=self._unique_inverse,
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

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensor":
        return NotImplemented

    def to_dict(self) -> Dict[str, JaggedTensor]:
        return NotImplemented

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
        self._values.record_stream(stream)
        if self._hash_indices is not None:
            self._hash_indices.record_stream(stream)
        if self._unique_indices is not None:
            self._unique_indices.record_stream(stream)
        if self._unique_offset is not None:
            self._unique_offset.record_stream(stream)
        if self._unique_inverse is not None:
            self._unique_inverse.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            weights.record_stream(stream)
        if lengths is not None:
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            lengths.record_stream(stream)
        if offsets is not None:
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            offsets.record_stream(stream)

    @staticmethod
    def to_device_non_blocking(var, device, non_blocking):
        return var.to(device, non_blocking=non_blocking) if var is not None else None

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "KeyedJaggedTensor":

        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank) if self.variable_stride_per_key() else (self._stride, None))
        length_per_key = self._length_per_key
        offset_per_key = self._offset_per_key
        index_per_key = self._index_per_key
        jt_dict = self._jt_dict

        return KeyedJaggedTensorWithLookHelper(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            hash_indices=self.to_device_non_blocking(self._hash_indices, device, non_blocking),
            unique_indices=self.to_device_non_blocking(self._unique_indices, device, non_blocking),
            unique_offset=self.to_device_non_blocking(self._unique_offset, device, non_blocking),
            unique_offset_host=self.unique_offset_host,
            unique_ids=self._unique_ids,
            unique_inverse=self.to_device_non_blocking(self._unique_inverse, device, non_blocking),
            weights=self.to_device_non_blocking(weights, device, non_blocking),
            lengths=self.to_device_non_blocking(lengths, device, non_blocking),
            offsets=self.to_device_non_blocking(offsets, device, non_blocking),
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
        )



    def pin_memory(self) -> "KeyedJaggedTensorWithLookHelper":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )

        return KeyedJaggedTensorWithLookHelper(
            keys=self._keys,
            values=self._values.pin_memory(),
            hash_indices=(
                self._hash_indices.pin_memory()
                if self._hash_indices is not None
                else None
            ),
            unique_indices=(
                self._unique_indices.pin_memory()
                if self._unique_indices is not None
                else None
            ),
            unique_offset=(
                self._unique_offset.pin_memory()
                if self._unique_offset is not None
                else None
            ),
            unique_offset_host=self.unique_offset_host,
            unique_ids=self._unique_ids,
            unique_inverse=(
                self._unique_inverse.pin_memory()
                if self._unique_inverse is not None
                else None
            ),
            weights=weights.pin_memory() if weights is not None else None,
            lengths=lengths.pin_memory() if lengths is not None else None,
            offsets=offsets.pin_memory() if offsets is not None else None,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
            jt_dict=None,
        )

    def dist_labels(self) -> List[str]:
        labels = ["lengths", "values"]
        if self.variable_stride_per_key():
            labels.append("strides")
        if self.weights_or_none() is not None:
            labels.append("weights")
        return labels

    def dist_splits(self, key_splits: List[int]) -> List[List[int]]:
        return NotImplemented

    def dist_tensors(self) -> List[torch.Tensor]:
        return NotImplemented

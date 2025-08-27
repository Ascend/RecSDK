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
    _pin_and_move,
    _permute_tensor_by_segments,
)
from torchrec.pt2.checks import is_torchdynamo_compiling, is_non_strict_exporting
from hybrid_torchrec.sparse.extended_jagged_tensor import ExtendedJaggedTensor, KeyedExtendedJaggedTensor


class JaggedTensorWithTimestamp(ExtendedJaggedTensor):
    _fields = ["_timestamps"]

    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            values=values,
            extra=timestamps,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            extra_field_name="timestamps"
        )
        # 和values值对应的时间戳，size需和values相同, 仅在input dist前使用
        self._timestamps = timestamps

    @property
    def timestamps(self):
        return self._timestamps


class KeyedJaggedTensorWithTimestamp(KeyedExtendedJaggedTensor):
    _fields = ["_timestamps"]

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
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
            extra=timestamps,
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
            extra_field_name="timestamps"
        )
        self._timestamps: torch.Tensor = timestamps

    @property
    def timestamps(self) -> torch.Tensor:
        return self._timestamps

    @staticmethod
    def from_jt_dict(
        jt_dict: Dict[str, JaggedTensorWithTimestamp],
    ) -> "KeyedJaggedTensorWithTimestamp":
        """
        Constructs a KeyedJaggedTensor from a dictionary of JaggedTensorWithTimestamps.
        Automatically calls `kjt.sync()` on newly created KJT.

        Args:
            jt_dict (Dict[str, JaggedTensor]): dictionary of JaggedTensors.

        Returns:
            KeyedJaggedTensorWithTimestamp: constructed KeyedJaggedTensorWithTimestamp.
        """
        return KeyedJaggedTensorWithTimestamp.from_jt_dict_base(jt_dict, "timestamps")

    def split(self, segments: List[int]) -> List["KeyedJaggedTensorWithTimestamp"]:
        split_list: List[KeyedJaggedTensorWithTimestamp] = []
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
                    KeyedJaggedTensorWithTimestamp(
                        keys=self._keys,
                        values=self._values,
                        timestamps=self._timestamps,
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
                    KeyedJaggedTensorWithTimestamp(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        timestamps=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._timestamps.dtype,
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
                    KeyedJaggedTensorWithTimestamp(
                        keys=keys,
                        values=self._values[start_offset:end_offset],
                        timestamps=(
                            self._timestamps[start_offset:end_offset]
                            if self._timestamps is not None
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
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensorWithTimestamp":
        """
        Permutes the KeyedJaggedTensorWithTimestamp.

        Args:
            indices (List[int]): list of indices.
            indices_tensor (Optional[torch.Tensor]): tensor of indices.

        Returns:
            KeyedJaggedTensorWithTimestamp: permuted KeyedJaggedTensorWithTimestamp.
        """
        if indices_tensor is None:
            indices_tensor = torch.tensor(
                indices, dtype=torch.int, device=self.device()
            )

        length_per_key = self.length_per_key()
        permuted_keys: List[str] = []
        permuted_stride_per_key_per_rank: List[List[int]] = []
        permuted_length_per_key: List[int] = []
        permuted_length_per_key_sum = 0
        for index in indices:
            key = self.keys()[index]
            permuted_keys.append(key)
            permuted_length_per_key.append(length_per_key[index])
            if self.variable_stride_per_key():
                permuted_stride_per_key_per_rank.append(
                    self.stride_per_key_per_rank()[index]
                )

        permuted_length_per_key_sum = sum(permuted_length_per_key)
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            # 使用公共API替代受保护成员的访问
            if permuted_length_per_key_sum <= 0:
                raise ValueError("permuted_length_per_key_sum needs to be greater than 0")

        if self.variable_stride_per_key():
            length_per_key_tensor = _pin_and_move(
                torch.tensor(self.length_per_key()), self.device()
            )
            stride_per_key_tensor = _pin_and_move(
                torch.tensor(self.stride_per_key()), self.device()
            )
            permuted_lengths, _ = _permute_tensor_by_segments(
                self.lengths(),
                stride_per_key_tensor,
                indices_tensor,
                None,
            )
            permuted_values, permuted_weights = _permute_tensor_by_segments(
                self.values(),
                length_per_key_tensor,
                indices_tensor,
                self.weights_or_none(),
            )
            permuted_timestamps, _ = _permute_tensor_by_segments(
                self.timestamps,
                length_per_key_tensor,
                indices_tensor,
                self.weights_or_none(),
            )
        elif is_torchdynamo_compiling() and not torch.jit.is_scripting():
            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                indices_tensor,
                self.lengths(),
                self.values(),
                self.stride(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
            _, permuted_timestamps, _ = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                indices_tensor,
                self.lengths(),
                self.timestamps,
                self.stride(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        else:
            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.values(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
            _, permuted_timestamps, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.timestamps,
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        stride_per_key_per_rank = (
            permuted_stride_per_key_per_rank if self.variable_stride_per_key() else None
        )
        kjt = KeyedJaggedTensorWithTimestamp(
            keys=permuted_keys,
            values=permuted_values,
            timestamps=permuted_timestamps,
            weights=permuted_weights,
            lengths=permuted_lengths.view(-1),
            offsets=None,
            stride=self._stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=None,
            length_per_key=permuted_length_per_key if len(permuted_keys) > 0 else None,
            lengths_offset_per_key=None,
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )
        return kjt

    def pin_memory(self) -> "KeyedJaggedTensorWithTimestamp":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )

        return KeyedJaggedTensorWithTimestamp(
            keys=self._keys,
            values=self._values.pin_memory(),
            timestamps=(
                self._timestamps.pin_memory() if self._timestamps is not None else None
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

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "KeyedJaggedTensorWithTimestamp":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        length_per_key = self._length_per_key
        offset_per_key = self._offset_per_key
        index_per_key = self._index_per_key
        jt_dict = self._jt_dict

        return KeyedJaggedTensorWithTimestamp(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            timestamps=(
                self._timestamps.to(device, non_blocking=non_blocking)
                if self._timestamps is not None
                else None
            ),
            weights=(
                weights.to(device, non_blocking=non_blocking)
                if weights is not None
                else None
            ),
            lengths=(
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            offsets=(
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
        )

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        super().record_stream(stream)
        if self._timestamps is not None:
            self._timestamps.record_stream(stream)

    def to_dict(self) -> Dict[str, JaggedTensor]:
        # invoke base class's method, and will discard timestamp data.
        return super().to_dict()

    def dist_splits(self, key_splits: List[int]) -> List[List[int]]:
        return NotImplemented

    def dist_tensors(self) -> List[torch.Tensor]:
        return NotImplemented

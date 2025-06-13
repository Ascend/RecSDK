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
    _pin_and_move,
    _permute_tensor_by_segments,
    _sum_by_splits,
    JaggedTensor,
    KeyedJaggedTensor,
)
from torchrec.pt2.checks import is_torchdynamo_compiling, is_non_strict_exporting


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
        kjt_keys = list(jt_dict.keys())
        kjt_vals_list: List[torch.Tensor] = []
        kjt_counts_list: List[torch.Tensor] = []
        kjt_lens_list: List[torch.Tensor] = []
        kjt_weights_list: List[torch.Tensor] = []
        stride_per_key: List[int] = []
        for jt in jt_dict.values():
            stride_per_key.append(len(jt.lengths()))
            kjt_vals_list.append(jt.values())
            kjt_counts_list.append(jt.counts)
            kjt_lens_list.append(jt.lengths())
            weight = jt.weights_or_none()
            if weight is not None:
                kjt_weights_list.append(weight)
        kjt_vals = torch.concat(kjt_vals_list)
        kjt_lens = torch.concat(kjt_lens_list)

        # handle custom attribute: counts
        kjt_counts = (
            torch.concat(kjt_counts_list) if len(kjt_counts_list) > 0 else None
        )

        kjt_weights = (
            torch.concat(kjt_weights_list) if len(kjt_weights_list) > 0 else None
        )
        kjt_stride, kjt_stride_per_key_per_rank = (
            (stride_per_key[0], None)
            if all(s == stride_per_key[0] for s in stride_per_key)
            else (None, [[stride] for stride in stride_per_key])
        )
        kjt = KeyedJaggedTensorWithCount(
            keys=kjt_keys,
            values=kjt_vals,
            counts=kjt_counts,
            weights=kjt_weights,
            lengths=kjt_lens,
            stride=kjt_stride,
            stride_per_key_per_rank=kjt_stride_per_key_per_rank,
        ).sync()
        return kjt

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
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensorWithCount":
        """
        Permutes the KeyedJaggedTensorWithCount.

        Args:
            indices (List[int]): list of indices.
            indices_tensor (Optional[torch.Tensor]): tensor of indices.

        Returns:
            KeyedJaggedTensorWithCount: permuted KeyedJaggedTensorWithCount.
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
            torch._check_is_size(permuted_length_per_key_sum)
            torch._check(permuted_length_per_key_sum != -1)
            torch._check(permuted_length_per_key_sum != 0)

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
            permuted_counts, _ = _permute_tensor_by_segments(
                self.counts,
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
            _, permuted_counts, _ = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                indices_tensor,
                self.lengths(),
                self.counts,
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
            _, permuted_counts, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.counts,
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        stride_per_key_per_rank = (
            permuted_stride_per_key_per_rank if self.variable_stride_per_key() else None
        )
        kjt = KeyedJaggedTensorWithCount(
            keys=permuted_keys,
            values=permuted_values,
            counts=permuted_counts,
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

    def pin_memory(self) -> "KeyedJaggedTensorWithCount":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )

        return KeyedJaggedTensorWithCount(
            keys=self._keys,
            values=self._values.pin_memory(),
            counts=(
                self._counts.pin_memory()
                if self._counts is not None
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

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "KeyedJaggedTensorWithCount":
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

        return KeyedJaggedTensorWithCount(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            counts=(
                self._counts.to(device, non_blocking=non_blocking)
                if self._counts is not None
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
        if self._counts is not None:
            self._counts.record_stream(stream)

    def to_dict(self) -> Dict[str, JaggedTensor]:
        # invoke base class's method, and will discard timestamp data.
        return super().to_dict()

    def dist_labels(self) -> List[str]:
        labels = ["lengths", "values"]
        if self.variable_stride_per_key():
            labels.append("strides")
        if self.weights_or_none() is not None:
            labels.append("weights")
        if self._counts is not None:
            labels.append("counts")
        return labels

    def dist_splits(self, key_splits: List[int]) -> List[List[int]]:
        batch_size_per_split = _sum_by_splits(self.stride_per_key(), key_splits)
        length_per_split = _sum_by_splits(self.length_per_key(), key_splits)
        splits = [batch_size_per_split, length_per_split]
        if self.variable_stride_per_key():
            splits.append(key_splits)
        if self.weights_or_none() is not None:
            splits.append(length_per_split)
        if self._counts is not None:
            splits.append(length_per_split)
        return splits

    def dist_tensors(self) -> List[torch.Tensor]:
        tensors = [self.lengths(), self.values()]
        if self.variable_stride_per_key():
            strides = _pin_and_move(torch.tensor(self.stride_per_key()), self.device())
            tensors.append(strides)
        if self.weights_or_none() is not None:
            tensors.append(self.weights())
        if self._counts is not None:
            tensors.append(self._counts)
        return tensors

    @staticmethod
    def dist_init(
        keys: List[str],
        tensors: List[torch.Tensor],
        variable_stride_per_key: bool,
        num_workers: int,
        recat: Optional[torch.Tensor],
        stride_per_rank: Optional[List[int]],
        stagger: int = 1,
    ) -> "KeyedJaggedTensorWithCount":
        # The original largest length is 4, there is an extra counts params, the biggest length is 5.
        if len(tensors) not in [2, 3, 4, 5]:
            raise RuntimeError(f"tensors length must in [2, 3, 4, 5] but got:{len(tensors)}")
        lengths = tensors[0]
        values = tensors[1]
        stride_per_rank_per_key = tensors[2] if variable_stride_per_key else None

        # 仅当local unique且有表开启准入时，会使用KeyedJaggedTensorWithCount做all2all
        # 此时会固定在tensors列表末尾传递counts数据
        weights = (
            tensors[-2]
            if (variable_stride_per_key and len(tensors) == 5)
               or (not variable_stride_per_key and len(tensors) == 4)
            else None
        )
        counts = tensors[-1]

        if variable_stride_per_key:
            stride_per_key_per_rank_tensor: torch.Tensor = stride_per_rank_per_key.view(
                num_workers, len(keys)
            ).T.cpu()

            strides_cumsum: torch.Tensor = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(stride_per_rank_per_key)
            ).cpu()

            cumsum_lengths = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

            n = strides_cumsum.size(0)
            strides_cumsum_from_1 = torch.narrow(
                strides_cumsum, dim=0, start=1, length=n - 1
            )
            strides_cumsum_to_minus_1 = torch.narrow(
                strides_cumsum, dim=0, start=0, length=n - 1
            )
            length_per_key_tensor = (
                cumsum_lengths[strides_cumsum_from_1]
                - cumsum_lengths[strides_cumsum_to_minus_1]
            )

            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    new_lengths, _ = _permute_tensor_by_segments(
                        lengths,
                        stride_per_rank_per_key,
                        torch.jit._unwrap_optional(recat),
                        None,
                    )
                    new_values, new_weights = _permute_tensor_by_segments(
                        values,
                        length_per_key_tensor,
                        torch.jit._unwrap_optional(recat),
                        weights,
                    )
                    if counts is not None:
                        new_counts, _ = _permute_tensor_by_segments(
                            counts,
                            length_per_key_tensor,
                            torch.jit._unwrap_optional(recat),
                            None,
                        )

            stride_per_key_per_rank = torch.jit.annotate(
                List[List[int]], stride_per_key_per_rank_tensor.tolist()
            )

            if not stride_per_key_per_rank:
                stride_per_key_per_rank = [[0]] * len(keys)
            if stagger > 1:
                stride_per_key_per_rank_stagger: List[List[int]] = []
                local_world_size = num_workers // stagger
                for i in range(len(keys)):
                    stride_per_rank_stagger: List[int] = []
                    for j in range(local_world_size):
                        stride_per_rank_stagger.extend(
                            stride_per_key_per_rank[i][j::local_world_size]
                        )
                    stride_per_key_per_rank_stagger.append(stride_per_rank_stagger)
                stride_per_key_per_rank = stride_per_key_per_rank_stagger

            kjt = KeyedJaggedTensorWithCount(
                keys=keys,
                values=new_values,
                counts=new_counts,
                weights=new_weights,
                lengths=lengths,
                stride_per_key_per_rank=stride_per_key_per_rank,
            )
            return kjt.sync()
        else:
            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    stride = stride_per_rank[0]

                    single_batch_per_rank = True
                    new_counts = None
                    if not is_torchdynamo_compiling():
                        single_batch_per_rank = all(
                            s == stride for s in stride_per_rank
                        )
                    if (
                        single_batch_per_rank
                        and is_torchdynamo_compiling()
                        and not torch.jit.is_scripting()
                    ):
                        (
                            new_lengths,
                            new_values,
                            new_weights,
                        ) = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                            torch.jit._unwrap_optional(recat),
                            lengths,
                            values,
                            stride,
                            weights,
                            values.numel(),
                        )
                        if counts is not None:
                            _, new_counts, _ = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                                torch.jit._unwrap_optional(recat),
                                lengths,
                                counts,
                                stride,
                                None,
                                counts.numel(),
                            )
                    elif single_batch_per_rank:
                        (
                            new_lengths,
                            new_values,
                            new_weights,
                        ) = torch.ops.fbgemm.permute_2D_sparse_data(
                            torch.jit._unwrap_optional(recat),
                            lengths.view(-1, stride),
                            values,
                            weights,
                            values.numel(),
                        )
                        if counts is not None:
                            _, new_counts, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                                torch.jit._unwrap_optional(recat),
                                lengths.view(-1, stride),
                                counts,
                                None,
                                counts.numel(),
                            )
                        new_lengths = new_lengths.view(-1)
                    else:  # variable batch size per rank
                        (
                            new_lengths,
                            new_values,
                            new_weights,
                        ) = torch.ops.fbgemm.permute_1D_sparse_data(
                            torch.jit._unwrap_optional(recat),
                            lengths.view(-1),
                            values,
                            weights,
                            values.numel(),
                        )
                        if counts is not None:
                            _, new_counts, _ = torch.ops.fbgemm.permute_1D_sparse_data(
                                torch.jit._unwrap_optional(recat),
                                lengths.view(-1),
                                counts,
                                None,
                                counts.numel(),
                            )
                else:
                    new_lengths = lengths
                    new_values = values
                    new_weights = weights
                    new_counts = counts
            kjt = KeyedJaggedTensorWithCount(
                keys=keys,
                values=new_values,
                counts=new_counts,
                weights=new_weights,
                lengths=new_lengths,
                stride=sum(stride_per_rank),
            )
            return kjt.sync()



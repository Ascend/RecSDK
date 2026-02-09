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
)
from torchrec.pt2.checks import is_torchdynamo_compiling

from .extended_jagged_tensor import ExtendedJaggedTensor, KeyedExtendedJaggedTensor


class JaggedTensorWithCount(ExtendedJaggedTensor):
    """带有计数信息的JaggedTensor"""
    
    _fields = "_counts"

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
        )
        # values中每个ids出现次数，分桶去重时会进行计算，input_dist all2all会做集合通信，post dist input时做count记录
        self._counts = counts

    @property
    def counts(self) -> Optional[torch.Tensor]:
        return self._extra


class KeyedJaggedTensorWithCount(KeyedExtendedJaggedTensor[JaggedTensorWithCount]):
    """带有计数信息的KeyedJaggedTensor"""
    
    _fields = "_counts"

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
        extra: Optional[torch.Tensor] = None,
    ) -> None:
        if extra is not None and counts is None:
            counts = extra
            
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
            field_tensors={"_counts": counts} if counts is not None else {}
        )
        self._counts: Optional[torch.Tensor] = counts

    @property
    def counts(self) -> Optional[torch.Tensor]:
        return self._extra

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensorWithCount]) -> "KeyedJaggedTensorWithCount":
        """
        从JaggedTensorWithCount字典构造KeyedJaggedTensorWithCount
        """
        # 创建一个实例用于调用_construct_from_jt_dict方法
        dummy_instance = KeyedJaggedTensorWithCount(
            keys=[],
            values=torch.tensor([]),
            counts=None
        )
        
        # 使用_construct_from_jt_dict方法创建实例
        return dummy_instance._construct_from_jt_dict(
            jt_dict,
            KeyedJaggedTensorWithCount,
            lambda jt: jt.counts
        )

    def split(self, segments: List[int]) -> List["KeyedJaggedTensorWithCount"]:
        return self.split_extend(segments, KeyedJaggedTensorWithCount)

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensorWithCount":
        return self.permute_extend(
            indices, 
            indices_tensor, 
            KeyedJaggedTensorWithCount,
            _permute_tensor_by_segments  # 使用特定的permute函数
        )

    def pin_memory(self) -> "KeyedJaggedTensorWithCount":
        return self.pin_memory_extend(KeyedJaggedTensorWithCount)

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "KeyedJaggedTensorWithCount":
        return self.to_base(device, non_blocking, KeyedJaggedTensorWithCount)

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
            count_dim = self._counts.shape[-1]
            splits.append([count_dim * x for x in length_per_split])
            if count_dim != 1:
                self._counts = self._counts.view(-1)
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
        lengths, values, stride_per_rank_per_key, weights = unpack_tensors(tensors, variable_stride_per_key)
        counts = tensors[-1]
        count_dim = int(len(counts) / torch.sum(lengths).item()) if counts is not None else 0
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
                            length_per_key_tensor * count_dim,
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
                                lengths * count_dim,
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
                                lengths.view(-1, stride) * count_dim,
                                counts,
                                None,
                                counts.numel(),
                            )
                        new_lengths = new_lengths.view(-1)
                        new_counts = new_counts.view(-1) if new_counts is not None else None
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
                                lengths.view(-1) * count_dim,
                                counts,
                                None,
                                counts.numel(),
                            )
                            new_counts = new_counts.view(-1, count_dim)
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


def unpack_tensors(tensors: List[torch.Tensor], variable_stride_per_key: bool):
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
    return lengths, values, stride_per_rank_per_key, weights
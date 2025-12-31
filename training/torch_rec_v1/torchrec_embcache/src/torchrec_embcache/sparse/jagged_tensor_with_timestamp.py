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
    _permute_tensor_by_segments,
)
from hybrid_torchrec.sparse.extended_jagged_tensor import (
    ExtendedJaggedTensor, 
    KeyedExtendedJaggedTensor, 
    _validate_tensor_param)


def _validate_timestamps_shape(timestamps: Optional[torch.Tensor], values: torch.Tensor) -> None:
    """验证timestamps与values的形状一致性"""
    _validate_tensor_param(timestamps, "timestamps")
    _validate_tensor_param(values, "values")
    if timestamps is not None:
        if timestamps.shape != values.shape:
            raise ValueError(f"timestamps shape {timestamps.shape} must match values shape {values.shape}")


class JaggedTensorWithTimestamp(ExtendedJaggedTensor):
    """带有时间戳信息的JaggedTensor"""

    _fields = "_timestamps"

    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> None:
        # 校验timestamps与values的形状一致性
        _validate_timestamps_shape(timestamps, values)
        
        super().__init__(
            values=values,
            extra=timestamps,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
        )
        # 和values值对应的时间戳，size需和values相同, 仅在input dist前使用
        self._timestamps = timestamps

    @property
    def timestamps(self) -> Optional[torch.Tensor]:
        return self._extra


class KeyedJaggedTensorWithTimestamp(
    KeyedExtendedJaggedTensor[JaggedTensorWithTimestamp]
):
    """带有时间戳信息的KeyedJaggedTensor"""

    _fields = "_timestamps"

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
        # 为兼容重构后的基类添加extra参数
        extra: Optional[torch.Tensor] = None,
    ) -> None:
        # 校验timestamps与values的形状一致性
        _validate_timestamps_shape(timestamps, values)
        
        # 处理来自基类的extra参数
        if extra is not None and timestamps is None:
            timestamps = extra

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
            field_tensors={"_timestamps": timestamps} if timestamps is not None else {},
        )
        self._timestamps: Optional[torch.Tensor] = timestamps

    @property
    def timestamps(self) -> Optional[torch.Tensor]:
        return self._extra

    @staticmethod
    def from_jt_dict(
        jt_dict: Dict[str, JaggedTensorWithTimestamp],
    ) -> "KeyedJaggedTensorWithTimestamp":
        """
        从JaggedTensorWithTimestamp字典构造KeyedJaggedTensorWithTimestamp
        """
        # 创建一个实例用于调用_construct_from_jt_dict方法
        dummy_instance = KeyedJaggedTensorWithTimestamp(
            keys=[], values=torch.tensor([]), timestamps=None
        )

        # 使用_construct_from_jt_dict方法创建实例
        return dummy_instance._construct_from_jt_dict(
            jt_dict, KeyedJaggedTensorWithTimestamp, lambda jt: jt.timestamps
        )

    def split(self, segments: List[int]) -> List["KeyedJaggedTensorWithTimestamp"]:
        return self.split_extend(segments, KeyedJaggedTensorWithTimestamp)

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensorWithTimestamp":
        return self.permute_extend(
            indices,
            indices_tensor,
            KeyedJaggedTensorWithTimestamp,
            _permute_tensor_by_segments,  # 使用特定的permute函数
        )

    def pin_memory(self) -> "KeyedJaggedTensorWithTimestamp":
        return self.pin_memory_extend(KeyedJaggedTensorWithTimestamp)

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "KeyedJaggedTensorWithTimestamp":
        return self.to_base(device, non_blocking, KeyedJaggedTensorWithTimestamp)

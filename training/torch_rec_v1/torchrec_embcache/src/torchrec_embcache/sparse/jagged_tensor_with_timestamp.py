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
from hybrid_torchrec.sparse.extended_jagged_tensor import ExtendedJaggedTensor, KeyedExtendedJaggedTensor


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
        # 参数类型校验
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"values must be torch.Tensor, but got {type(values)}")
        if weights is not None and not isinstance(weights, torch.Tensor):
            raise TypeError(f"weights must be torch.Tensor or None, but got {type(weights)}")
        if lengths is not None and not isinstance(lengths, torch.Tensor):
            raise TypeError(f"lengths must be torch.Tensor or None, but got {type(lengths)}")
        if offsets is not None and not isinstance(offsets, torch.Tensor):
            raise TypeError(f"offsets must be torch.Tensor or None, but got {type(offsets)}")
        if timestamps is not None and not isinstance(timestamps, torch.Tensor):
            raise TypeError(f"timestamps must be torch.Tensor or None, but got {type(timestamps)}")
        
        # 校验timestamps与values的形状一致性
        if timestamps is not None:
            if timestamps.shape != values.shape:
                raise ValueError(f"timestamps shape {timestamps.shape} must match values shape {values.shape}")
        
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
        # 参数类型校验
        if not isinstance(keys, list):
            raise TypeError(f"keys must be List[str], but got {type(keys)}")
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("All elements in keys must be strings")
        
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"values must be torch.Tensor, but got {type(values)}")
        
        if timestamps is not None and not isinstance(timestamps, torch.Tensor):
            raise TypeError(f"timestamps must be torch.Tensor or None, but got {type(timestamps)}")
        
        if weights is not None and not isinstance(weights, torch.Tensor):
            raise TypeError(f"weights must be torch.Tensor or None, but got {type(weights)}")
        
        if lengths is not None and not isinstance(lengths, torch.Tensor):
            raise TypeError(f"lengths must be torch.Tensor or None, but got {type(lengths)}")
        
        if offsets is not None and not isinstance(offsets, torch.Tensor):
            raise TypeError(f"offsets must be torch.Tensor or None, but got {type(offsets)}")
        
        if stride is not None and not isinstance(stride, int):
            raise TypeError(f"stride must be int or None, but got {type(stride)}")
        
        if stride_per_key_per_rank is not None:
            if not isinstance(stride_per_key_per_rank, list):
                raise TypeError(
                    f"stride_per_key_per_rank must be List[List[int]] or None, "
                    f"but got {type(stride_per_key_per_rank)}"
                )
            if not all(isinstance(inner_list, list) for inner_list in stride_per_key_per_rank):
                raise TypeError("All elements in stride_per_key_per_rank must be lists")
            if not all(
                isinstance(item, int) 
                for inner_list in stride_per_key_per_rank 
                for item in inner_list
            ):
                raise TypeError("All elements in stride_per_key_per_rank inner lists must be integers")
        
        self.check_list(stride_per_key)
        self.check_list(length_per_key)
        self.check_list(lengths_offset_per_key)
        self.check_list(offset_per_key)
        
        if index_per_key is not None and not isinstance(index_per_key, dict):
            raise TypeError(
                f"index_per_key must be Dict[str, int] or None, but got {type(index_per_key)}"
            )
        if (
            index_per_key is not None 
            and not all(isinstance(k, str) and isinstance(v, int) for k, v in index_per_key.items())
        ):
            raise TypeError("All keys in index_per_key must be strings and all values must be integers")
        
        if jt_dict is not None and not isinstance(jt_dict, dict):
            raise TypeError(
                f"jt_dict must be Dict[str, JaggedTensor] or None, but got {type(jt_dict)}"
            )
        
        if inverse_indices is not None:
            if not isinstance(inverse_indices, tuple):
                raise TypeError(
                    f"inverse_indices must be Tuple[List[str], torch.Tensor] or None, "
                    f"but got {type(inverse_indices)}"
                )
            if len(inverse_indices) != 2:
                raise ValueError("inverse_indices must be a tuple of length 2")
            if not isinstance(inverse_indices[0], list) or not all(
                isinstance(item, str) for item in inverse_indices[0]
            ):
                raise TypeError("First element of inverse_indices must be List[str]")
            if not isinstance(inverse_indices[1], torch.Tensor):
                raise TypeError("Second element of inverse_indices must be torch.Tensor")
        
        if extra is not None and not isinstance(extra, torch.Tensor):
            raise TypeError(f"extra must be torch.Tensor or None, but got {type(extra)}")
        
        # 校验timestamps与values的形状一致性
        if timestamps is not None:
            if timestamps.shape != values.shape:
                raise ValueError(f"timestamps shape {timestamps.shape} must match values shape {values.shape}")
        
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

    def check_list(self, key_list):
        if key_list is not None and not isinstance(key_list, list):
            raise TypeError(f"stride_per_key must be List[int] or None, but got {type(key_list)}")
        if key_list is not None and not all(isinstance(item, int) for item in key_list):
            raise TypeError("All elements in stride_per_key must be integers")

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

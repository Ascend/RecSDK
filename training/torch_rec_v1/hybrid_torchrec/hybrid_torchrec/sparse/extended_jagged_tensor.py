#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, List, Tuple, TypeVar, Generic, Callable

import torch
from torchrec.sparse.jagged_tensor import (
    _pin_and_move,
    _permute_tensor_by_segments,
    JaggedTensor,
    KeyedJaggedTensor,
)
from torchrec.pt2.checks import is_torchdynamo_compiling, is_non_strict_exporting


T = TypeVar('T', bound='ExtendedJaggedTensor')
KT = TypeVar('KT', bound='KeyedExtendedJaggedTensor')


class ExtendedJaggedTensor(JaggedTensor):
    """扩展的JaggedTensor基类，用于处理带有额外字段的JaggedTensor"""
    
    # 子类需要定义_fields属性，例如_fields = "_counts"
    _fields: str = ""
    
    def __init__(
        self,
        values: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(values, weights, lengths, offsets)
        self._extra = extra
        
        # 动态设置字段属性
        if self._fields and extra is not None:
            setattr(self, self._fields, extra)

    @property
    def extra(self) -> Optional[torch.Tensor]:
        return self._extra


class KeyedExtendedJaggedTensor(KeyedJaggedTensor, Generic[T]):
    """扩展的KeyedJaggedTensor基类，用于处理带有额外字段的KeyedJaggedTensor"""

    # 子类需要定义_fields属性，例如_fields = "_counts"
    _fields: str = ""

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
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
        field_tensors: Optional[Dict[str, torch.Tensor]] = None
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
        self._extra: Optional[torch.Tensor] = extra

        # 设置字段张量
        field_tensors = field_tensors or {}

        # 验证字段名是否匹配 _fields
        for field in field_tensors:
            if field != self._fields:
                raise ValueError(f"Field '{field}' not match declared field '{self._fields}'")

        # 动态设置字段属性
        if self._fields:
            tensor = field_tensors.get(self._fields, extra)
            if tensor is not None and values.size() != tensor.size():
                raise ValueError(
                    f"Field '{self._fields}' size must match values size, "
                    f"but got tensor size: {tensor.size()}, values size: {values.size()}"
                )
            setattr(self, self._fields, tensor)

    @property
    def extra(self) -> Optional[torch.Tensor]:
        return self._extra

    def split_extend(self, segments: List[int], constructor: Callable[..., KT]) -> List[KT]:
        """通用的split方法，子类需要提供构造函数"""
        split_list: List[KT] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        if len(segments) > len(_length_per_key):
            raise ValueError("len of segments should be < len(keys)")

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
                    constructor(
                        keys=self._keys,
                        values=self._values,
                        extra=self._extra,
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
                # 创建额外字段的空张量
                extra_tensor = None
                if self._extra is not None:
                    extra_tensor = torch.tensor(
                        empty_int_list,
                        device=self.device(),
                        dtype=self._extra.dtype,
                    )

                split_list.append(
                    constructor(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        extra=extra_tensor,
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
                # 处理额外字段的切片
                sliced_extra = None
                if self._extra is not None:
                    sliced_extra = self._extra[start_offset:end_offset]

                split_list.append(
                    constructor(
                        keys=keys,
                        values=self._values[start_offset:end_offset],
                        extra=sliced_extra,
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

    def permute_extend(
        self, 
        indices: List[int], 
        indices_tensor: Optional[torch.Tensor],
        constructor: Callable[..., KT],
        extra_permute_func: Optional[Callable] = None
    ) -> KT:
        """
        通用的permute方法
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

            # 处理额外字段
            if extra_permute_func:
                permuted_extra, _ = extra_permute_func(
                    self.extra,
                    length_per_key_tensor,
                    indices_tensor,
                    self.weights_or_none(),
                )
            else:
                permuted_extra, _ = _permute_tensor_by_segments(
                    self.extra,
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

            if self.extra is not None:
                _, permuted_extra, _ = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                    indices_tensor,
                    self.lengths(),
                    self.extra,
                    self.stride(),
                    self.weights_or_none(),
                    permuted_length_per_key_sum,
                )
            else:
                permuted_extra = None

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

            if self.extra is not None:
                _, permuted_extra, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                    indices_tensor,
                    self.lengths().view(len(self._keys), -1),
                    self.extra,
                    self.weights_or_none(),
                    permuted_length_per_key_sum,
                )
            else:
                permuted_extra = None

        stride_per_key_per_rank = (
            permuted_stride_per_key_per_rank if self.variable_stride_per_key() else None
        )

        result = constructor(
            keys=permuted_keys,
            values=permuted_values,
            extra=permuted_extra,
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
        return result

    def pin_memory_extend(self, constructor: Callable[..., KT]) -> KT:
        """通用的pin_memory方法"""
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )

        # 处理额外字段的pin_memory
        pinned_extra = None
        if self._extra is not None:
            pinned_extra = self._extra.pin_memory()

        return constructor(
            keys=self._keys,
            values=self._values.pin_memory(),
            extra=pinned_extra,
            weights=weights.pin_memory() if weights is not None else None,
            lengths=lengths.pin_memory() if lengths is not None else None,
            offsets=offsets.pin_memory() if offsets is not None else None,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
        )

    def _construct_from_jt_dict(
        self, 
        jt_dict: Dict[str, T],
        constructor: Callable[..., KT],
        extra_field_getter: Callable[[T], Optional[torch.Tensor]]
    ) -> KT:
        """
        通用的from_jt_dict实现
        """
        kjt_keys = list(jt_dict.keys())
        kjt_vals_list: List[torch.Tensor] = []
        kjt_extra_list: List[torch.Tensor] = []
        kjt_lens_list: List[torch.Tensor] = []
        kjt_weights_list: List[torch.Tensor] = []
        stride_per_key: List[int] = []

        for jt in jt_dict.values():
            stride_per_key.append(len(jt.lengths()))
            kjt_vals_list.append(jt.values())
            kjt_extra_list.append(extra_field_getter(jt))
            kjt_lens_list.append(jt.lengths())
            weight = jt.weights_or_none()
            if weight is not None:
                kjt_weights_list.append(weight)

        kjt_vals = torch.concat(kjt_vals_list)
        kjt_lens = torch.concat(kjt_lens_list)

        # 处理额外字段
        kjt_extra = (
            torch.concat(kjt_extra_list) 
            if len(kjt_extra_list) > 0 and all(t is not None for t in kjt_extra_list) 
            else None
        )

        kjt_weights = (
            torch.concat(kjt_weights_list) if len(kjt_weights_list) > 0 else None
        )

        kjt_stride, kjt_stride_per_key_per_rank = (
            (stride_per_key[0], None)
            if all(s == stride_per_key[0] for s in stride_per_key)
            else (None, [[stride] for stride in stride_per_key])
        )

        kjt = constructor(
            keys=kjt_keys,
            values=kjt_vals,
            extra=kjt_extra,
            weights=kjt_weights,
            lengths=kjt_lens,
            stride=kjt_stride,
            stride_per_key_per_rank=kjt_stride_per_key_per_rank,
        ).sync()
        return kjt

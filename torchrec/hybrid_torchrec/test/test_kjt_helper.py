#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
import pytest
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
    KeyedJaggedTensor,
)

TEST_NUM = 100
IDS_RANGE_TIMES = 10


@pytest.mark.parametrize("table_num", [3])
@pytest.mark.parametrize("feature_names", [[2, 3, 4]])
@pytest.mark.parametrize("input_size", [1000])
def test_unique_split(table_num, feature_names, input_size):
    """Test ids2indices with sequential numbers"""
    keys = [f"feat{str(i)}" for i in range(sum(feature_names))]
    total_values = []
    length = [input_size for i in range(sum(feature_names))]

    for table_id in range(table_num):
        for _ in range(feature_names[table_id]):
            value = torch.randint(0, input_size, (input_size,))
            total_values.append(value)
    kjt = KeyedJaggedTensor(
        keys=keys,
        values=torch.cat(total_values).reshape(-1),
        lengths=torch.Tensor(length).long(),
    )
    kjt_list = kjt.split(feature_names)
    unique_indices = []
    unique_inverse = []
    unique_offset = []
    start = 0
    for kjt in kjt_list:
        unique_indice, inverse = torch.unique(kjt.values(), return_inverse=True)
        unique_indices.append(unique_indice)
        unique_inverse.append(inverse)
        unique_offset.extend([start] * len(kjt.keys()))
        start += unique_indice.shape[0]

    unique_offset.append(start)
    kjt_helper = KeyedJaggedTensorWithLookHelper(
        keys=keys,
        values=torch.concat(total_values),
        lengths=torch.Tensor(length).long(),
        hash_indices=torch.concat(total_values),
        unique_indices=torch.concat(unique_indices),
        unique_offset=torch.Tensor(unique_offset).long(),
        unique_inverse=torch.concat(unique_inverse),
    )

    kjt_helper_list = kjt_helper.split(feature_names)
    for kjt in kjt_helper_list:
        kjt: KeyedJaggedTensorWithLookHelper
        unique_results = []
        gloden = kjt.values()
        for ind, unique_offset in enumerate(kjt.unique_offset):
            unique = kjt.unique_indices[unique_offset:]
            unique_inverse = kjt.unique_inverse[
                kjt.offsets()[ind]: kjt.offsets()[ind + 1]
            ]
            unique_results.append(
                torch.index_select(unique, dim=0, index=unique_inverse)
            )
        assert (
            gloden == torch.concat(unique_results)
        ).all(), "kjt_helper split can not inverse unique"

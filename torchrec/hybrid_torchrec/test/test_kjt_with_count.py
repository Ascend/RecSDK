#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch
from hybrid_torchrec.sparse import (
    JaggedTensorWithCount,
    KeyedJaggedTensorWithCount)

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.parametrize("table_num", [3])
@pytest.mark.parametrize("feature_names", [[1, 1, 2]])
@pytest.mark.parametrize("input_size", [10])
def test_kjt_with_count(table_num, feature_names, input_size):
    input_dict = {}
    feature_len = sum(feature_names)
    for ind in range(feature_len - 1, -1, -1):  # feature 逆序，用于后面验证permute和split
        name = f"feat{ind}"
        id_range = input_size
        ids = torch.randint(0, id_range, (input_size,))
        lengths = torch.ones(input_size).long()
        counts = torch.clone(ids)  # 生成和key相同的counts信息
        input_dict[name] = JaggedTensorWithCount(values=ids, lengths=lengths, counts=counts)

    kjt_with_count = KeyedJaggedTensorWithCount.from_jt_dict(input_dict)
    logging.info("kjt_with_count:%s", kjt_with_count)

    # permute
    feature_names_for_sharding = [f"feat{ind}" for ind in range(feature_len)]
    input_feature_names = kjt_with_count.keys()
    features_order_index = []
    for f in feature_names_for_sharding:
        features_order_index.append(input_feature_names.index(f))
    kjt_permuted = kjt_with_count.permute(features_order_index)
    logging.info("kjt_with_count keys after permute:%s", kjt_permuted.keys())

    offset_per_key = kjt_permuted.offset_per_key()
    for i in range(len(offset_per_key) - 1):
        start = offset_per_key[i]
        end = offset_per_key[i + 1]
        keys = kjt_permuted.values()[start:end]
        key_with_count = kjt_permuted.counts[start:end]
        assert torch.all(key_with_count == keys), "key_with_count is not equal after kjt permute."

    # split 比较分割后的count是否符合预期
    feature_splits = [1, 1, 2]
    kjt_list = kjt_permuted.split(feature_splits)
    for index, kjt in enumerate(kjt_list):
        key_with_counts = kjt.counts
        keys = kjt.values()
        assert torch.all(key_with_counts == keys), "key_with_counts is not equals with keys after kjt split."


if __name__ == '__main__':
    test_kjt_with_count(3, [1, 1, 2], 10)

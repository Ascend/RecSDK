#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import pytest
import torch

from torchrec_embcache.sparse.jagged_tensor_with_timestamp import (
    JaggedTensorWithTimestamp,
    KeyedJaggedTensorWithTimestamp,
)

TEST_NUM = 100
IDS_RANGE_TIMES = 10

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("table_num", [3])
@pytest.mark.parametrize("feature_names", [[1, 1, 2]])
@pytest.mark.parametrize("input_size", [10])
def test_unique_split(table_num, feature_names, input_size):
    input_dict = {}
    feature_len = sum(feature_names)
    specific_time = time.struct_time((2023, 5, 5, 16, 33, 20, 0, 0, 0))  # 1683275600  -> feat3, feat2
    specific_time1 = time.struct_time((2023, 5, 5, 15, 33, 20, 0, 0, 0))  # 1683272000  -> feat1
    specific_time2 = time.struct_time((2023, 5, 5, 14, 33, 20, 0, 0, 0))  # 1683268400  -> feat0
    timestamp0 = int(time.mktime(specific_time))
    timestamp1 = int(time.mktime(specific_time1))
    timestamp2 = int(time.mktime(specific_time2))
    timestamp_list = [timestamp0, timestamp0, timestamp1, timestamp2]

    for ind in range(feature_len - 1, -1, -1):  # feature 逆序
        name = f"feat{ind}"
        id_range = input_size
        ids = torch.randint(0, id_range, (input_size,))
        lengths = torch.ones(input_size).long()
        timestamps = torch.full(ids.size(), timestamp_list[abs(ind - len(timestamp_list) + 1)], dtype=torch.int64)
        input_dict[name] = JaggedTensorWithTimestamp(values=ids, lengths=lengths, timestamps=timestamps)

    kjt_with_time = KeyedJaggedTensorWithTimestamp.from_jt_dict(input_dict)
    logging.info("kjt_with_time keys:%s", kjt_with_time.keys())

    # permute
    feature_names_for_sharding = [f"feat{ind}" for ind in range(feature_len)]  # 对应模型sharding的 feature name
    input_feature_names = kjt_with_time.keys()  # 输入数据input 的feature names

    # 传入索引列表，和索引张量，进行转置
    features_order_index = []
    for f in feature_names_for_sharding:
        features_order_index.append(input_feature_names.index(f))
    kjt_permuted = kjt_with_time.permute(features_order_index)
    logging.info("kjt_with_time keys after permute:%s", kjt_permuted.keys())

    timestamp_list_with_permuted = list(reversed(timestamp_list))
    offset_per_key = kjt_permuted.offset_per_key()
    for i in range(len(offset_per_key) - 1):
        start = offset_per_key[i]
        end = offset_per_key[i + 1]
        timestamp_per_key = kjt_permuted.timestamps[start:end]
        assert torch.all(timestamp_per_key == timestamp_list_with_permuted[i]), \
            "timestamp_per_key is not expected after kjt permute."

    # split 比较分割后的timestamp是否符合预期
    feature_splits = [1, 1, 2]
    kjt_list = kjt_permuted.split(feature_splits)
    timestamp_list = [timestamp2, timestamp1, timestamp0]
    for index, kjt in enumerate(kjt_list):
        kjt_timestamp = kjt.timestamps
        assert torch.all(kjt_timestamp == timestamp_list[index]), "kjt_timestamp is not expected after kjt split."


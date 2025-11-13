#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable

import numpy as np
import tensorflow as tf

from config import Config


def generate_dataset(batch_number: int = 100) -> tf.compat.v1.data.Dataset:
    """Generates a TensorFlow dataset using a data generator.

    The dataset is created using a generator function and specifies the output types and shapes
    for the dataset elements. If the rank size is greater than 1, the dataset is sharded accordingly.

    Args:
        batch_number (int): The number of batches to generate. Default is 100.

    Returns:
        tf.compat.v1.data.Dataset: A TensorFlow dataset object.

    """
    dataset = tf.compat.v1.data.Dataset.from_generator(
        generator=_get_data_generator(batch_number),
        output_types={
            Config.item_ids: Config.key_type,
            Config.user_ids: Config.key_type,
            Config.label_0: Config.label_type,
            Config.label_1: Config.label_type,
        },
        output_shapes={
            Config.item_ids: tf.TensorShape([Config.batch_size, Config.item_feat_cnt]),
            Config.user_ids: tf.TensorShape([Config.batch_size, Config.user_feat_cnt]),
            Config.label_0: tf.TensorShape([Config.batch_size]),
            Config.label_1: tf.TensorShape([Config.batch_size]),
        },
    )

    if Config.rank_size > 1:
        dataset = dataset.shard(Config.rank_size, Config.rank_id)

    return dataset


def _get_data_generator(batch_number: int) -> Callable:
    def _data_generator():
        i = 0
        while i < batch_number:
            item_ids = np.random.randint(0, Config.item_range, (Config.batch_size, Config.item_feat_cnt))
            user_ids = np.random.randint(0, Config.user_range, (Config.batch_size, Config.user_feat_cnt))
            label_0 = np.random.randint(0, 2, (Config.batch_size,))
            label_1 = np.random.randint(0, 2, (Config.batch_size,))

            yield {
                Config.item_ids: item_ids,
                Config.user_ids: user_ids,
                Config.label_0: label_0,
                Config.label_1: label_1,
            }
            i += 1

    return _data_generator

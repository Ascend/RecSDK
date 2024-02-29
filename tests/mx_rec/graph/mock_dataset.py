#!/usr/bin/env python3
# coding: UTF-8
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

from typing import Dict, Iterable

import numpy as np
import tensorflow as tf


def gen_mock_dataset(batch_num: int = 100, batch_size: int = 4096) -> tf.compat.v1.data.Dataset:
    def data_generator() -> Iterable[Dict[str, np.ndarray]]:
        i = 0
        while i < batch_num:
            mock_ids = np.random.randint(low=0, high=100, size=(batch_size, 8))
            mock_labels = np.random.randint(low=0, high=100, size=(batch_size, 1))
            mock_timestamp = np.random.randint(low=0, high=100, size=(batch_size, 1))
            yield {"mock_ids": mock_ids, "mock_labels": mock_labels, "mock_timestamp": mock_timestamp}
            i += 1

    dataset = tf.compat.v1.data.Dataset.from_generator(
        generator=data_generator,
        output_types={"mock_ids": tf.int64, "mock_labels": tf.int32, "mock_timestamp": tf.int32},
        output_shapes={
            "mock_ids": tf.TensorShape([batch_size, 8]),
            "mock_labels": tf.TensorShape([batch_size, 1]),
            "mock_timestamp": tf.TensorShape([batch_size, 1]),
        },
    )
    return dataset

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os

import tensorflow as tf

from mx_rec.core.asc.helper import get_asc_insert_func

USE_PIPELINE_TEST = False
USE_STATIC = False
USE_EXPANSION = False


class InputConfig:
    def __init__(self, feature_spec_list, rank_id, local_rank_id, rank_size, data_path, file_pattern, 
                 total_batch_size, num_epochs=1, perform_shuffle=False, training=True):
        self.feature_spec_list = feature_spec_list
        self.rank_id = rank_id
        self.local_rank_id = local_rank_id
        self.rank_size = rank_size
        self.data_path = data_path
        self.file_pattern = file_pattern
        self.total_batch_size = total_batch_size
        self.num_epochs = num_epochs
        self.perform_shuffle = perform_shuffle
        self.training = training


def input_fn_tfrecord(input_config: InputConfig):
    feature_spec_list = input_config.feature_spec_list
    rank_id = input_config.rank_id
    local_rank_id = input_config.local_rank_id
    rank_size = input_config.rank_size
    data_path = input_config.data_path
    file_pattern = input_config.file_pattern
    total_batch_size = input_config.total_batch_size
    num_epochs = input_config.num_epochs
    perform_shuffle = input_config.perform_shuffle
    training = input_config.training

    line_per_sample = 1024 * 8
    total_batch_size = int(total_batch_size / line_per_sample)
    num_parallel = 8

    def extract_fn(data_record):
        features = {
            'label': tf.FixedLenFeature(shape=(line_per_sample,), dtype=tf.float32),
            'feat_ids': tf.FixedLenFeature(shape=(128 * line_per_sample,), dtype=tf.int64)
        }
        sample = tf.parse_single_example(data_record, features)
        return sample

    def reshape_fn(batch):
        batch['label'] = tf.reshape(batch['label'], [-1, ])
        batch['feat_ids'] = tf.reshape(batch['feat_ids'], [-1, 128])
        return batch

    all_files = os.listdir(data_path)
    files = [os.path.join(data_path, f) for f in all_files if f.startswith(file_pattern)]
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel)
    batch_size = total_batch_size // rank_size
    dataset = dataset.shard(rank_size, rank_id)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(extract_fn, num_parallel_calls=num_parallel).batch(batch_size,
                                                                             drop_remainder=True)
    dataset = dataset.map(reshape_fn, num_parallel_calls=num_parallel)
    insert_fn = get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=True, dump_graph=False)
    dataset = dataset.map(insert_fn)
    dataset = dataset.prefetch(int(100))
    return dataset
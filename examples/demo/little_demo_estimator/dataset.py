#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import tensorflow as tf

from random_data_generator import get_data_generator, get_large_scale_data_generator
from mx_rec.util.initialize import get_rank_size, get_rank_id, get_host_pipeline_ops


def generate_dataset(cfg, use_timestamp=False, batch_number=100):
    dataset = tf.compat.v1.data.Dataset.from_generator(
        generator=get_data_generator(cfg, batch_number=batch_number),
        output_types={"item_ids": cfg.key_type,
                      "user_ids": cfg.key_type,
                      "category_ids": cfg.key_type,
                      "label_0": cfg.label_type,
                      "label_1": cfg.label_type},
        output_shapes={"item_ids": tf.TensorShape([cfg.batch_size, cfg.item_feat_cnt]),
                       "user_ids": tf.TensorShape([cfg.batch_size, cfg.user_feat_cnt]),
                       "category_ids": tf.TensorShape([cfg.batch_size, cfg.category_feat_cnt]),
                       "label_0": tf.TensorShape([cfg.batch_size]),
                       "label_1": tf.TensorShape([cfg.batch_size])})
    if use_timestamp:
        dataset = dataset.map(add_timestamp_func)

    rank_size = get_rank_size()
    rank_id = get_rank_id()
    if rank_size > 1:
        dataset = dataset.shard(rank_size, rank_id)

    return dataset


def add_timestamp_func(batch):
    host_pipeline_ops = get_host_pipeline_ops()
    timestamp = host_pipeline_ops.return_timestamp(tf.cast(batch['label_0'], tf.int64))
    batch["timestamp"] = timestamp
    return batch


def generate_large_scale_data(cfg):
    key_type_list = [cfg.key_type for _ in range(cfg.lookup_count)]
    output_type_dict = dict(zip(cfg.tensor_name_list, key_type_list))
    output_type_dict["label_0"] = cfg.label_type
    output_type_dict["label_1"] = cfg.label_type

    tensor_shape_list = [tf.TensorShape([cfg.batch_size]) for _ in range(cfg.lookup_count)]
    output_shape_dict = dict(zip(cfg.tensor_name_list, tensor_shape_list))
    output_shape_dict["label_0"] = tf.TensorShape([cfg.batch_size])
    output_shape_dict["label_1"] = tf.TensorShape([cfg.batch_size])

    dataset = tf.data.Dataset.from_generator(generator=get_large_scale_data_generator(cfg),
                                             output_types=output_type_dict,
                                             output_shapes=output_shape_dict)
    rank_size = get_rank_size()
    rank_id = get_rank_id()
    if rank_size > 1:
        dataset = dataset.shard(rank_size, rank_id)

    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    return batch, iterator

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter


class Config:
    """
    配置类
    """

    def __init__(self, batch_size=32, batch_number=100):
        self.batch_size = batch_size
        self.batch_number = batch_number
        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.float32
        self.item_range = 16
        self.item_feat_cnt = 8


def __get_data_generator(cfg: Config) -> Callable:
    """
    生成数据迭代器

    Args:
        cfg: 配置类实例

    Returns: 数据迭代器fn

    """

    def data_generator():
        i = 0
        while i < cfg.batch_number:
            item_ids = np.random.randint(0, cfg.item_range, (cfg.batch_size, cfg.item_feat_cnt))
            label_0 = np.random.randint(0, 2, (cfg.batch_size,))

            yield {"item_ids": item_ids,
                   "label_0": label_0}
            i += 1

    return data_generator


def generate_dataset(cfg: Config) -> DatasetV1Adapter:
    """
    生成dataset

    Args:
        cfg: 配置类实例

    Returns: dataset

    """

    dataset = tf.compat.v1.data.Dataset.from_generator(
        generator=__get_data_generator(cfg),
        output_types={"item_ids": cfg.key_type,
                      "label_0": cfg.label_type},
        output_shapes={"item_ids": tf.TensorShape([cfg.batch_size, cfg.item_feat_cnt]),
                       "label_0": tf.TensorShape([cfg.batch_size])})
    return dataset

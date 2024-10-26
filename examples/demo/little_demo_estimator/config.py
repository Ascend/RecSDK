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

import os
import math

import tensorflow as tf
from enum import Enum
from mx_rec.util.communication.hccl_ops import get_rank_size

class CacheModeEnum(Enum):
    HBM = "HBM"
    DDR = "DDR"
    SSD = "SSD"


class Config:
    def __init__(self, mode="simple", task_name="default"):
        self.task_name = task_name
        if mode == "simple":
            self.generate_simple_config()
        else:
            self.generate_large_scale_config()

    def generate_simple_config(self):
        try:
            use_dp = bool(int(os.getenv("USE_DP", 0)))
        except ValueError as err:
            raise ValueError(
                "please correctly config USE_DP only 0 or 1 is supported."
            ) from err

        self.batch_numbers = 8192
        self.batch_size = 4096

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.float32

        self.item_range = 80000 * get_rank_size() if not use_dp else 80000
        self.user_range = 200000 * get_rank_size() if not use_dp else 200000
        self.category_range = 5000 * get_rank_size() if not use_dp else 5000
        self.item_feat_cnt = 16
        self.user_feat_cnt = 8
        self.category_feat_cnt = 3
        self.access_threshold = 100
        self.eviction_threshold = 60

        rank_size = get_rank_size()
        coefficient = 1.1
        self.item_send_cnt = (
            min(
                int(self.batch_size * self.item_feat_cnt * coefficient),
                math.ceil(self.item_range / rank_size),
            )
            if not use_dp
            else self.item_range
        )
        self.item_vocab_size = (
            max(self.item_send_cnt * rank_size * rank_size, self.item_range)
            if not use_dp
            else max(self.item_send_cnt * rank_size, self.item_range)
        )
        self.user_send_cnt = (
            min(
                int(self.batch_size * self.user_feat_cnt * coefficient),
                math.ceil(self.user_range / rank_size),
            )
            if not use_dp
            else self.user_range
        )
        self.user_vocab_size = (
            max(self.user_send_cnt * rank_size * rank_size, self.user_range)
            if not use_dp
            else max(self.user_send_cnt * rank_size, self.user_range)
        )
        self.category_send_cnt = (
            min(
                int(self.batch_size * self.category_feat_cnt * coefficient),
                math.ceil(self.category_range / rank_size),
            )
            if not use_dp
            else self.category_range
        )

        self.user_hashtable_dim = 32
        self.user_hashtable_threshold = 1
        self.item_hashtable_dim = 8
        self.item_hashtable_threshold = 1

        self.learning_rate = 0.01

    def generate_large_scale_config(self):
        self.lookup_count = 40
        self.tensor_name_list = [
            "sparse_tensor_%d" % i for i in range(self.lookup_count)
        ]
        self.hashtable_name_list = [
            "hashtable_%d" % i for i in range(self.lookup_count)
        ]
        self.batch_size = 9600

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.float32

        self.vocabulary_size = 500000
        self.feat_cnt = 1

        rank_size = get_rank_size()
        coefficient = 1.1
        self.send_cnt = min(
            int(self.batch_size * self.feat_cnt * coefficient),
            math.ceil(self.vocabulary_size / rank_size),
        )

        self.hashtable_dim = 8
        self.learning_rate = 0.01

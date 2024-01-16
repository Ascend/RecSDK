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

import os


class SparseEmbeddingMock:
    """
    sparse embedding mock module
    """

    def __init__(self, host_vocab_size=0):
        self.is_save = True
        self.table_name = "test_table"
        self.slice_device_vocabulary_size = 10
        self.scalar_emb_size = 4
        self.host_vocabulary_size = host_vocab_size
        self.optimizer = dict()
        self.use_dynamic_expansion = False

    def set_optimizer(self, key, state_dict):
        if key in self.optimizer:
            raise ValueError(f"Optimizer {key} has been set for hash table {self.table_name}")

        self.optimizer[key] = state_dict

#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.


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
        self.use_feature_mapping = None
        self.optimizer = dict()
        self.use_dynamic_expansion = False

    def set_optimizer(self, key, state_dict):
        if key in self.optimizer:
            raise ValueError(f"Optimizer {key} has been set for hash table {self.table_name}")

        self.optimizer[key] = state_dict

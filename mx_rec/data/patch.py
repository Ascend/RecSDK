#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from tensorflow.python.data.ops.dataset_ops import DatasetV2, DatasetV1Adapter

from mx_rec.data.dataset import EosDataset


def patch_for_dataset_eos_map():
    """
    给DatasetV2类增加eos_map方法.
    Returns: None
    """

    def eos_map_fn(self, librec, channel_id):
        return DatasetV1Adapter(EosDataset(self, librec, channel_id))

    DatasetV2.eos_map = eos_map_fn

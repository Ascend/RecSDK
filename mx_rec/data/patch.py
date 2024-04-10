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

from tensorflow.python.data.ops.dataset_ops import DatasetV2, DatasetV1Adapter

from mx_rec.data.dataset import EosDataset


def patch_for_dataset_eos_map():
    """
    给DatasetV2类增加eos_map方法.
    Returns: None
    """

    def eos_map_fn(self, librec, channel_id, max_train_steps=-1, max_eval_steps=-1):
        return DatasetV1Adapter(EosDataset(self, librec, channel_id, max_train_steps, max_eval_steps))

    DatasetV2.eos_map = eos_map_fn

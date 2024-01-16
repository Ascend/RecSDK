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

from tensorflow.python.data.ops.dataset_ops import get_legacy_output_types, get_legacy_output_classes, \
    get_legacy_output_shapes, UnaryDataset
from tensorflow.python.data.util import structure
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


class EosDataset(UnaryDataset):
    """用于发送end_of_sequence的dataset."""

    def __init__(self, input_dataset, librec, channel_id):
        self._input_dataset = input_dataset
        output_types = get_legacy_output_types(input_dataset)
        output_classes = get_legacy_output_classes(input_dataset)
        input_shapes = get_legacy_output_shapes(self._input_dataset)
        output_shapes = input_shapes

        self._structure = structure.convert_legacy_structure(
            output_types, output_shapes, output_classes)
        channel_id = ops.convert_to_tensor(channel_id, dtype=dtypes.int32, name="channel_id")
        self._input_datasets = [input_dataset]
        variant_tensor = librec.eos_dataset(
            input_dataset=input_dataset._variant_tensor, channel_id=channel_id,
            output_shapes=self._flat_shapes, output_types=self._flat_types)
        super(EosDataset, self).__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._structure

    def _inputs(self):
        return self._input_datasets

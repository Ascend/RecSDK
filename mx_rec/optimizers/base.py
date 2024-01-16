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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from tensorflow.python.framework import ops
from tensorflow.python.training.optimizer import _TensorProcessor

from mx_rec.util.log import logger


class CustomizedOptimizer:

    name_counter = defaultdict(int)

    def __init__(self):
        self.unique_name = ""
        self.base_name = ""

    def initialize_slots(self, var, table_instance):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def insert_slot(self, slot, named_slots_key, slot_name):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def get_slot_init_values(self):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def _get_name(self, name="CustomizedOptimizer"):
        if name in CustomizedOptimizer.name_counter:
            CustomizedOptimizer.name_counter[name] += 1
            count = CustomizedOptimizer.name_counter.get(name)

        else:
            count = CustomizedOptimizer.name_counter[name]
        self.unique_name = name + "_" + str(count)
        self.base_name = name


def custom_update_op(self, opt, grad):
    if isinstance(grad, ops.Tensor):
        update_op = opt._apply_sparse(grad, self._v)
        return update_op
    else:
        raise RuntimeError("Only support g with type Tensor.")


def patch_for_optimizer():
    _TensorProcessor.update_op = custom_update_op
    logger.debug("update_op in Class optimizer._TensorProcessor has been patched.")
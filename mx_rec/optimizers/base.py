# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2025. All rights reserved.
# Description: build script.
# Author: MindX SDK

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import logging
from tensorflow.python.framework import ops
from tensorflow.python.training.optimizer import _TensorProcessor


class CustomizedOptimizer:

    name_counter = defaultdict(int)

    def __init__(self):
        self.unique_name = ""
        self.base_name = ""

    def __get_name__(self, name="CustomizedOptimizer"):
        if name in CustomizedOptimizer.name_counter:
            CustomizedOptimizer.name_counter[name] += 1
            count = CustomizedOptimizer.name_counter.get(name)

        else:
            count = CustomizedOptimizer.name_counter[name]
        self.unique_name = name + "_" + str(count)
        self.base_name = name

    def initialize_slots(self, var, table_instance):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def insert_slot(self, slot, named_slots_key, slot_name):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def get_slot_init_values(self):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")


def my_update_op(self, opt, grad):
    logging.debug("tf.compat.v1.training.optimizer._TensorProcessor has been patched, update_op.")
    if isinstance(grad, ops.Tensor):
        logging.debug(">>>>Enter update_op ops.Tensor")
        update_op = opt._apply_sparse(grad, self._v)  # pylint: disable=protected-access
        return update_op
    else:
        raise RuntimeError("Only support g with type Tensor.")


def patch_for_optimizer():
    _TensorProcessor.update_op = my_update_op
    logging.debug("Class tf.compat.v1.training.optimizer._TensorProcessor has been patched.")
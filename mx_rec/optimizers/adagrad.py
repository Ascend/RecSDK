#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adagrad, training_ops
from tensorflow.python.training import slot_creator

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import get_table_instance
from mx_rec.util.variable import remove_saving_var, check_param_type


def create_hash_optimizer(learning_rate=0.001,
                          initial_accumulator_value=0.9,
                          use_locking=False,
                          name="Adagrad"):
    """
    Create an instance of adagrad hash optimizer
    :param learning_rate: A `Tensor` or a floating point value.  The learning rate.
    :param initial_accumulator_value:  A floating point value. Starting value for the accumulators, must be positive.
    :param use_locking: If `True` use locks for update operations.
    :param name: Optional name prefix for the operations created when applying gradients.  Defaults to "Adagrad".
    :return: adagrad hash optimizer instance
    """
    return CustomizedAdagrad(learning_rate=learning_rate,
                             initial_accumulator_value=initial_accumulator_value,
                             use_locking=use_locking,
                             name=name)


class CustomizedAdagrad(adagrad.AdagradOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self,
                 learning_rate,
                 initial_accumulator_value,
                 use_locking=False,
                 name="Adagrad"):
        self.optimizer_type = "Adagrad"
        super(CustomizedAdagrad, self)._get_name(name=name)
        super(CustomizedAdagrad, self).__init__(learning_rate=learning_rate,
                                                initial_accumulator_value=initial_accumulator_value,
                                                use_locking=use_locking,
                                                name=self.unique_name)

        self._check_input_param()

    def initialize_slots(self, var, table_instance):
        # Create slots for the first and second moments.
        def creat_one_single_slot(var, op_name):
            new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
            # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
            return new_slot_variable

        accumulator = creat_one_single_slot(var, self._name + "/" + "accumulator")
        remove_saving_var(accumulator)
        named_slot_key = (var.op.graph, var.op.name)
        table_instance = get_table_instance(var)
        if self._name in table_instance.optimizer:
            raise EnvironmentError(f"Sparse optimizer named {self._name} has exists.")

        table_instance.set_optimizer(self._name, {"accumulator": accumulator})
        return [{"slot": accumulator, "named_slot_key": named_slot_key, "slot_name": "acc", "optimizer": self}]

    def insert_slot(self, slot, named_slots_key, slot_name):
        named_slots = self._slot_dict(slot_name)
        if named_slots_key in named_slots:
            raise EnvironmentError(f"named_slots_key should be global unique, but it has been in use now, "
                                   f"please double check.")

        named_slots[named_slots_key] = slot

    def get_slot_init_values(self):
        # return state value list of adagrad that needs to initialize in ASC DDR.
        initial_accumulator_value = 0.0
        return [initial_accumulator_value]

    def _check_input_param(self):
        check_param_type("learning_rate", self._learning_rate, (tf.Tensor, float))
        check_param_type("initial_accumulator_value", self._initial_accumulator_value, (tf.Tensor, float))
        check_param_type("use_locking", self._use_locking, bool)

    def _create_slots(self, var_list):
        logging.debug(" Start _create_slots")
        for var in var_list:
            dtype = var.dtype.base_dtype
            if var.get_shape().is_fully_defined():
                init = init_ops.constant_initializer(self._initial_accumulator_value,
                                                     dtype=dtype)
            else:
                init = self._init_constant_op(var, dtype)

            acc_state_name = self._name + "/" + "accumulator"
            self._get_or_make_slot_with_initializer(var, init, var.get_shape(), dtype,
                                                    "acc", acc_state_name)

    def _apply_sparse(self, grad, var):
        acc = self.get_slot(var, "acc")
        return training_ops.sparse_apply_adagrad(
            var, acc, math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad.values,
            grad.indices,
            use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices):
        acc = self.get_slot(var, "acc")
        return training_ops.resource_sparse_apply_adagrad(
            var.handle, acc.handle, math_ops.cast(self._learning_rate_tensor, grad.dtype),
            grad, indices, use_locking=self._use_locking)

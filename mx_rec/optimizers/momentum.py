#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops
from tensorflow.python.training import momentum
from tensorflow.python.training import slot_creator

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import get_table_instance
from mx_rec.util.variable import remove_saving_var, check_and_get_config_via_var, check_param_type, check_param_range


def create_hash_optimizer(learning_rate_input=0.001, mom=0.9, enable_locking=False, optimizer_name="momentum",
                          enable_nesterov=False):
    """
    Create an instance of hash optimizer
    :param learning_rate_input: A `Tensor` or a floating point value.  The learning rate.
    :param mom: A `Tensor` or a floating point value.  The momentum.
    :param enable_locking: If `True` use locks for update operations.
    :param optimizer_name: Optional name prefix for the operations created when applying gradients.
    Defaults to "Momentum".
    :param enable_nesterov: If `True` use Nesterov Momentum. See (Sutskever et al., 2013). This implementation always
    computes gradients at the value of the variable(s) passed to the optimizer. Using Nesterov Momentum makes the
    variable(s) track the values called `theta_t + mu*v_t` in the paper. This implementation is an approximation of
    the original formula, valid for high values of momentum. It will compute the "adjusted gradient" in NAG by
    assuming that the new gradient will be estimated by the current average gradient plus the product of momentum and
     the change in the average gradient.
    :return: momentum hash optimizer instance
    """
    return CustomizedMomentum(learning_rate=learning_rate_input,
                              momentum_var=mom,
                              use_locking=enable_locking,
                              name=optimizer_name,
                              use_nesterov=enable_nesterov)


class CustomizedMomentum(momentum.MomentumOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self,
                 learning_rate,
                 momentum_var,
                 use_locking=False,
                 name="Momentum",
                 use_nesterov=False):
        self.optimizer_type = "Momentum"
        super(CustomizedMomentum, self).__get_name__(name=name)
        super(CustomizedMomentum, self).__init__(learning_rate=learning_rate,
                                                 momentum=momentum_var,
                                                 use_locking=use_locking,
                                                 name=self.unique_name,
                                                 use_nesterov=use_nesterov)

        self._check_input_param()

    def _check_input_param(self):
        check_param_type("learning_rate", self._learning_rate, (tf.Tensor, float))
        check_param_type("momentum", self._momentum, (tf.Tensor, float))
        check_param_type("use_locking", self._use_locking, bool)
        check_param_type("use_nesterov", self._use_nesterov, bool)

        check_param_range("momentum", self._momentum, 0.0, 1.0)

    def _create_slots(self, var_list):
        logging.debug(" Start _create_slots")
        m_state_name = self._name + "/" + "momentum"
        for var in var_list:
            table_instance = check_and_get_config_via_var(var, self.optimizer_type)
            momentum_slot = self._zeros_slot(var, "m", m_state_name)

            remove_saving_var(momentum_slot)
            if self._name not in table_instance.optimizer:
                table_instance.set_optimizer(self._name, {"momentum": momentum_slot})
        logging.debug(" End  _create_slots")

    def _apply_sparse(self, grad, var):
        mom = self.get_slot(var, "m")
        return training_ops.sparse_apply_momentum(
            var, mom, math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad.values, grad.indices, math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov).op

    def _resource_apply_sparse(self, grad, var, indices):
        mom = self.get_slot(var, "m")
        return training_ops.resource_sparse_apply_momentum(
            var.handle, mom.handle, math_ops.cast(self._learning_rate_tensor, grad.dtype),
            grad, indices, math_ops.cast(self._momentum_tensor, grad.dtype),
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov)

    def initialize_slots(self, var, table_instance):
        # Create slots for the first and second moments.
        def creat_one_single_slot(var, op_name):
            new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
            # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
            return new_slot_variable

        momentum_slot = creat_one_single_slot(var, self._name + "/" + "momentum")
        remove_saving_var(momentum_slot)
        named_slot_key = (var.op.graph, var.op.name)
        table_instance = get_table_instance(var)
        if self._name in table_instance.optimizer:
            raise EnvironmentError(f"Sparse optimizer named {self._name} has exists.")

        table_instance.set_optimizer(self._name, {"momentum": momentum_slot})
        return [{"slot": momentum_slot, "named_slot_key": named_slot_key, "slot_name": "m", "optimizer": self}]

    def insert_slot(self, slot, named_slots_key, slot_name):
        named_slots = self._slot_dict(slot_name)
        if named_slots_key in named_slots:
            raise EnvironmentError(f"named_slots_key should be global unique, but it has been in use now, "
                                   f"please double check.")

        named_slots[named_slots_key] = slot

    def get_slot_init_values(self):
        # return state value list of momentum that needs to initialize in ASC DDR.
        initial_momentum_value = 0.0
        return [initial_momentum_value]

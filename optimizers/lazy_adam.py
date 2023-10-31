#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adam
from tensorflow.python.training import slot_creator

from optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import get_table_instance, insert_removing_var_list
from mx_rec.util.variable import check_and_get_config_via_var


def create_hash_optimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="LazyAdam"):
    """
    Args:
        learning_rate: learning rate
        beta1:
        beta2:
        epsilon:
        name:

    Returns: a customized optimizer instance
    """
    return CustomizedLazyAdam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, name=name)


class CustomizedLazyAdam(adam.AdamOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="LazyAdam"):
        self.optimizer_type = "LazyAdam"
        super(CustomizedLazyAdam, self)._get_name(name=name)
        super(CustomizedLazyAdam, self).__init__(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                 epsilon=epsilon, use_locking=use_locking, name=self.unique_name)

    def initialize_slots(self, var, table_instance):
        # Create slots for the first and second moments.
        def creat_one_single_slot(var, op_name):
            new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
            # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
            return new_slot_variable

        momentum = creat_one_single_slot(var, self._name + "/" + "momentum")
        velocity = creat_one_single_slot(var, self._name + "/" + "velocity")
        insert_removing_var_list(momentum.name)
        insert_removing_var_list(velocity.name)
        named_slot_key = (var.op.graph, var.op.name)
        table_instance = get_table_instance(var)
        if self._name in table_instance.optimizer:
            raise EnvironmentError(f"Sparse optimizer named {self._name} has exists.")

        table_instance.set_optimizer(self._name, {"momentum": momentum, "velocity": velocity})
        return [{"slot": momentum, "named_slot_key": named_slot_key, "slot_name": "m", "optimizer": self},
                {"slot": velocity, "named_slot_key": named_slot_key, "slot_name": "v", "optimizer": self}]

    def insert_slot(self, slot, named_slots_key, slot_name):
        named_slots = self._slot_dict(slot_name)
        if named_slots_key in named_slots:
            raise EnvironmentError(f"named_slots_key should be global unique, but it has been in use now, "
                                   f"please double check.")

        named_slots[named_slots_key] = slot

    def get_slot_init_values(self):
        # return state value list of adam that needs to initialize in ASC DDR.
        initial_momentum_value = 0.0
        initial_velocity_value = 0.0
        return [initial_momentum_value, initial_velocity_value]

    def _apply_sparse_duplicate_indices(self, grad, var):
        #  _apply_sparse_duplicate_indices method include tf.unique and unsorted_segment_sum operations which may
        #  introduce dynamic shape problem, if encounter that, please de-annotation the method below.
        return self._apply_sparse(grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return self._resource_apply_sparse(grad, handle, indices)

    def _apply_dense(self, grad, var):
        raise NotImplementedError("You are using a wrong type of variable.")

    def _cast_to_base_type(self, var):
        var_type = var.dtype.base_dtype
        temp_lr = math_ops.cast(self._lr_t, var_type)
        temp_b1 = math_ops.cast(self._beta1_t, var_type)
        temp_b2 = math_ops.cast(self._beta2_t, var_type)
        temp_epsilon = math_ops.cast(self._epsilon_t, var_type)
        temp = {
            'temp_lr': temp_lr,
            'temp_b1': temp_b1,
            'temp_b2': temp_b2,
            'temp_epsilon': temp_epsilon,
        }
        return temp

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._apply_sparse_shared(
            grad,
            handle,
            indices,
            self._resource_scatter_nd_add)

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: tf.compat.v1.scatter_nd_add(x, i, v))

    def _apply_sparse_shared(self, grad, var, indices, scatter_nd_add):
        power_b1, power_b2 = self._get_beta_accumulators()
        power_b1 = math_ops.cast(power_b1, var.dtype.base_dtype)
        power_b2 = math_ops.cast(power_b2, var.dtype.base_dtype)
        temp = self._cast_to_base_type(var)
        temp_lr = temp.get("temp_lr")
        temp_b1 = temp.get("temp_b1")
        temp_b2 = temp.get("temp_b2")
        temp_epsilon = temp.get("temp_epsilon")
        learning_rate = tf.divide(temp_lr * math_ops.sqrt(1 - power_b2), (1 - power_b1))

        abs_indices = tf.math.maximum(indices, 0)
        nd_indices = tf.expand_dims(indices, 1)

        momentum = self.get_slot(var, "m")
        old_m_slice = tf.gather(momentum, abs_indices)
        m_t_slice = temp_b1 * old_m_slice + (1 - temp_b1) * grad
        m_update_op = scatter_nd_add(momentum, nd_indices, m_t_slice - old_m_slice)

        velocity = self.get_slot(var, "v")
        old_v_slice = tf.gather(velocity, abs_indices)
        v_t_slice = temp_b2 * old_v_slice + (1 - temp_b2) * math_ops.square(grad)
        v_update_op = scatter_nd_add(velocity, nd_indices, v_t_slice - old_v_slice)

        denominator_slice = math_ops.sqrt(v_t_slice) + temp_epsilon
        var_update_op = scatter_nd_add(var, nd_indices, tf.divide(-learning_rate * m_t_slice, denominator_slice))
        return control_flow_ops.group(m_update_op, v_update_op, var_update_op)

    def _resource_scatter_nd_add(self, x, i, v):
        with ops.control_dependencies([
            gen_state_ops.resource_scatter_nd_add(x.handle, i, v)]):
            return x.value()

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        # Create slots for the first and second moments.
        m_state_name = self._name + "/" + "momentum"
        v_state_name = self._name + "/" + "velocity"
        for each_var in var_list:
            table_instance = check_and_get_config_via_var(each_var, self.optimizer_type)

            momentum = self._zeros_slot(each_var, "m", m_state_name)
            velocity = self._zeros_slot(each_var, "v", v_state_name)
            # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
            insert_removing_var_list(momentum.name)
            insert_removing_var_list(velocity.name)

            if self._name not in table_instance.optimizer:
                table_instance.set_optimizer(self._name, {"momentum": momentum, "velocity": velocity})

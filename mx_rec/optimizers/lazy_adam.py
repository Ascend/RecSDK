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

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adam

from mx_rec.optimizers.base import CustomizedOptimizer, control_update_op_decorator
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.validator.validator import para_checker_decorator, StringValidator, FloatValidator, ClassValidator


@para_checker_decorator(check_option_list=[
    ("learning_rate", FloatValidator, {"min_value": 0.0, "max_value": 10.0}, ["check_value"]),
    ("beta1", FloatValidator, {"min_value": 0.0, "max_value": 1.0}, ["check_value_for_open_interval"]),
    ("beta2", FloatValidator, {"min_value": 0.0, "max_value": 1.0}, ["check_value"]),
    ("epsilon", FloatValidator, {"min_value": 0.0, "max_value": 1.0}, ["check_value_for_left_open_interval"]),
    ("name", StringValidator, {"min_len": 1, "max_len": 200}, ["check_string_length"]),
    ("use_fusion_optim", ClassValidator, {"classes": (bool,)}),
])
def create_hash_optimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="LazyAdam",
                          use_fusion_optim=False):
    """
    Args:
        learning_rate: learning rate
        beta1:
        beta2:
        epsilon:
        name:
        use_fusion_optim: if use fused optimizer
    Returns: a customized optimizer instance
    """
    if ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError("dynamic expansion mode is not compatible with the optimizer, please config dynamic "
                         "expansion mode and optimizer correctly")
    optimizer = CustomizedLazyAdam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, name=name,
                                   use_fusion_optim=use_fusion_optim)
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer
    return optimizer


class CustomizedLazyAdam(adam.AdamOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="LazyAdam",
                 use_fusion_optim=False):
        self.optimizer_type = "LazyAdam"
        self.optim_param_list = ["momentum", "velocity"]
        self.config_instance = ConfigInitializer.get_instance()
        self.use_fusion_optim = use_fusion_optim
        if self.use_fusion_optim:
            self._custom_initial_beta1 = beta1
            self._custom_initial_beta2 = beta2
            self._custom_initial_epsilon = epsilon
        super(CustomizedLazyAdam, self)._get_name(name=name)
        super(CustomizedLazyAdam, self).__init__(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                 epsilon=epsilon, use_locking=use_locking, name=self.unique_name)
        self._slot_num = 2
        self._derivative = 2

    def get_slot_init_values(self):
        # return state value list of adam that needs to initialize in ASC DDR.
        initial_momentum_value = 0.0
        initial_velocity_value = 0.0
        return [initial_momentum_value, initial_velocity_value]

    def _apply_sparse_duplicate_indices(self, grad, var):
        #  _apply_sparse_duplicate_indices method include tf.unique and unsorted_segment_sum operations which may
        #  introduce dynamic shape problem, if encounter that, please de-annotation the method below.
        unique_local_grad, unique_keys = self.sum_same_id_gradients(grad=grad.values, var=var, is_expansion=False)
        gradient_no_duplicate_indices = ops.IndexedSlices(
            indices=unique_keys,
            values=unique_local_grad,
            dense_shape=grad.dense_shape)
        return self._apply_sparse(gradient_no_duplicate_indices, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        unique_local_grad, unique_keys = self.sum_same_id_gradients(grad=grad, var=handle, is_expansion=False)
        return self._resource_apply_sparse(unique_local_grad, handle, unique_keys)

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

    @control_update_op_decorator
    def _resource_apply_sparse(self, grad, handle, indices):
        return self._apply_sparse_shared(
            grad,
            handle,
            indices,
            self._resource_scatter_nd_add)

    @control_update_op_decorator
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

        if self.use_fusion_optim:
            nd_indices = tf.expand_dims(indices, 1)
            slot_m = self.get_slot(var, "m")
            slot_v = self.get_slot(var, "v")
            output_m, output_v, output_var = \
                import_host_pipeline_ops().lazy_adam(grad, nd_indices, slot_m, slot_v, var, learning_rate,
                                                     self._custom_initial_beta1, self._custom_initial_beta2,
                                                     self._custom_initial_epsilon)
            return control_flow_ops.group(output_m, output_v, output_var)

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

        denominator_slice = math_ops.sqrt(v_t_slice + temp_epsilon)
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
            momentum = self._zeros_slot(each_var, "m", m_state_name)
            velocity = self._zeros_slot(each_var, "v", v_state_name)
            # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
            self.config_instance.sparse_embed_config.insert_removing_var_list(momentum.name)
            self.config_instance.sparse_embed_config.insert_removing_var_list(velocity.name)

            table_instance = self.config_instance.sparse_embed_config.get_table_instance(each_var)
            ConfigInitializer.get_instance().optimizer_config.set_optimizer_for_table(table_instance.table_name,
                                                                                      self.optimizer_type,
                                                                                      {"momentum": momentum,
                                                                                       "velocity": velocity})

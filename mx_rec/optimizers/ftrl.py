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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.training import ftrl
from tensorflow.python.training import slot_creator

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.variable import check_and_get_config_via_var
from mx_rec.constants.constants import MAX_INT32
from mx_rec.validator.validator import para_checker_decorator, ClassValidator, NumValidator, StringValidator, \
    FloatValidator


@para_checker_decorator(check_option_list=[
    ("learning_rate", FloatValidator, {"min_value": -MAX_INT32, "max_value": MAX_INT32}, ["check_value"]),
    ("initial_accumulator_value", FloatValidator, {"min_value": 0, "max_value": 1e4}, ["check_value"]),
    ("learning_rate_power", FloatValidator, {"min_value": -MAX_INT32, "max_value": 0}, ["check_value"]),
    ("l1_regularization_strength", FloatValidator, {"min_value": 0, "max_value": 1e4}, ["check_value"]),
    ("l2_regularization_strength", FloatValidator, {"min_value": 0, "max_value": 1e4}, ["check_value"]),
    ("l2_shrinkage_regularization_strength", FloatValidator, {"min_value": 0, "max_value": 1e4}, ["check_value"]),
    ("use_locking", ClassValidator, {"classes": (bool,)}),
    ("name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"]),
    ("accum_name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"]),
    ("linear_name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"])
])
def create_hash_optimizer(learning_rate, use_locking=False, name="Ftrl", **kwargs):
    if ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError("dynamic expansion mode is not compatible with the optimizer, please config dynamic "
                         "expansion mode and optimizer correctly")
    optimizer = CustomizedFtrl(learning_rate=learning_rate, use_locking=use_locking, name=name, **kwargs)
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer
    return optimizer


class CustomizedFtrl(ftrl.FtrlOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, use_locking=False, name="Ftrl", **kwargs):
        self.optimizer_type = "ftrl"
        self.optim_param_list = ["accum", "linear"]
        super(CustomizedFtrl, self)._get_name(name=name)
        super(CustomizedFtrl, self).__init__(
            learning_rate=learning_rate,
            learning_rate_power=kwargs.get("learning_rate_power", -0.5),
            initial_accumulator_value=kwargs.get("initial_accumulator_value", 0.1),
            l1_regularization_strength=kwargs.get("l1_regularization_strength", 0.0),
            l2_regularization_strength=kwargs.get("l2_regularization_strength", 0.0),
            use_locking=use_locking,
            name=self.unique_name,
            accum_name=kwargs.get("accum_name", None),
            linear_name=kwargs.get("linear_name", None),
            l2_shrinkage_regularization_strength=kwargs.get("l2_shrinkage_regularization_strength", 0.0)
        )

    def initialize_slots(self, var, table_instance):
        val = constant_op.constant(
            self._initial_accumulator_value, dtype=var.dtype, shape=var.get_shape())

        accum = slot_creator.create_slot(var, val, self._name + "/" + "accum")
        linear = slot_creator.create_zeros_slot(var, self._name + "/" + "linear")
        ConfigInitializer.get_instance().sparse_embed_config.insert_removing_var_list(accum.name)
        ConfigInitializer.get_instance().sparse_embed_config.insert_removing_var_list(linear.name)
        named_slot_key = (var.op.graph, var.op.name)
        table_instance = self.config_instance.sparse_embed_config.get_table_instance(var)
        ConfigInitializer.get_instance().optimizer_config.set_optimize_for_table(table_instance.table_name, self._name,
                                                                        {"accum": accum, "linear": linear})
        return [{"slot": accum, "named_slot_key": named_slot_key, "slot_name": "accum", "optimizer": self},
                {"slot": linear, "named_slot_key": named_slot_key, "slot_name": "linear", "optimizer": self}]

    def insert_slot(self, slot, named_slots_key, slot_name):
        named_slots = self._slot_dict(slot_name)
        if named_slots_key in named_slots:
            raise EnvironmentError(f"named_slots_key should be global unique, but it has been in use now, "
                                   f"please double check.")

        named_slots[named_slots_key] = slot

    def get_slot_init_values(self):
        # return state value list of ftrl that needs to initialize in ASC DDR.
        initial_linear_value = 0.0
        return [self._initial_accumulator_value, initial_linear_value]

    def _apply_sparse_duplicate_indices(self, grad, var):
        return self._apply_sparse(grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return self._resource_apply_sparse(grad, handle, indices)

    def _resource_apply_sparse(self, grad, handle, indices):
        if self._l2_shrinkage_regularization_strength <= 0.0:
            return self._apply_sparse_shared(
                grad,
                handle,
                indices,
                self._resource_scatter_nd_update)
        else:
            return self._apply_sparse_shared_v2(
                grad,
                handle,
                indices,
                self._resource_scatter_nd_update)

    def _apply_sparse(self, grad, var):
        if self._l2_shrinkage_regularization_strength <= 0.0:
            return self._apply_sparse_shared(
                grad.values,
                var,
                grad.indices,
                lambda x, i, v: tf.compat.v1.scatter_nd_update(x, i, v))
        else:
            return self._apply_sparse_shared_v2(
                grad.values,
                var,
                grad.indices,
                lambda x, i, v: tf.compat.v1.scatter_nd_update(x, i, v))

    def _apply_sparse_shared(self, grad, var, indices, scatter_nd_update):
        accum = self.get_slot(var, "accum")
        linear = self.get_slot(var, "linear")
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        l1 = math_ops.cast(self._l1_regularization_strength_tensor, var.dtype.base_dtype)

        if tf.__version__.startswith("1"):
            l2 = math_ops.cast(self._l2_regularization_strength_tensor, var.dtype.base_dtype)
        else:
            l2 = math_ops.cast(self._adjusted_l2_regularization_strength_tensor, var.dtype.base_dtype)
        lr_power = math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype)

        abs_indices = tf.math.maximum(indices, 0)
        nd_indices = tf.expand_dims(indices, 1)
        accum_old = tf.gather(accum, abs_indices)
        linear_old = tf.gather(linear, abs_indices)
        var_old = tf.gather(var, abs_indices)

        accum_update = accum_old + tf.multiply(grad, grad)
        with tf.control_dependencies([accum_update]):
            accum_update_op = scatter_nd_update(accum, nd_indices, accum_update)

        sigma = math_ops.pow(accum_update, -lr_power) - math_ops.pow(accum_old, -lr_power)
        sigma = tf.divide(sigma, lr)

        linear_update = linear_old + grad + tf.multiply(sigma, var_old)
        with tf.control_dependencies([linear_update]):
            linear_update_op = scatter_nd_update(linear, nd_indices, linear_update)

        quadratic = tf.divide(1.0, math_ops.pow(accum_update, lr_power) * lr) + 2 * l2

        var_new = tf.math.sign(linear_update) * l1 - linear_update
        var_new = tf.divide(var_new, quadratic)
        mask = math_ops.cast(tf.math.greater(tf.abs(linear_update), l1), var.dtype.base_dtype)

        var_update = tf.multiply(var_new, mask)

        var_update_op = scatter_nd_update(var, nd_indices, var_update)

        return control_flow_ops.group(accum_update_op, linear_update_op, var_update_op)

    def _apply_sparse_shared_v2(self, grad, var, indices, scatter_nd_update):
        accum = self.get_slot(var, "accum")
        linear = self.get_slot(var, "linear")
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        l1 = math_ops.cast(self._l1_regularization_strength_tensor, var.dtype.base_dtype)
        if tf.__version__.startswith("1"):
            l2 = math_ops.cast(self._l2_regularization_strength_tensor, var.dtype.base_dtype)
        else:
            l2 = math_ops.cast(self._adjusted_l2_regularization_strength_tensor, var.dtype.base_dtype)
        lr_power = math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype)
        l2_shrinkage = math_ops.cast(self._l2_shrinkage_regularization_strength_tensor, var.dtype.base_dtype)

        abs_indices = tf.math.maximum(indices, 0)
        nd_indices = tf.expand_dims(indices, 1)
        accum_old = tf.gather(accum, abs_indices)
        linear_old = tf.gather(linear, abs_indices)
        var_old = tf.gather(var, abs_indices)

        grad_with_shrinkage = grad + 2 * l2_shrinkage * var_old

        accum_update = accum_old + tf.multiply(grad, grad)
        with tf.control_dependencies([accum_update]):
            accum_update_op = scatter_nd_update(accum, nd_indices, accum_update)

        sigma = math_ops.pow(accum_update, -lr_power) - math_ops.pow(accum_old, -lr_power)
        sigma = tf.divide(sigma, lr)

        with tf.control_dependencies([grad_with_shrinkage]):
            linear_update = linear_old + grad_with_shrinkage - tf.multiply(sigma, var_old)
            with tf.control_dependencies([linear_update]):
                linear_update_op = scatter_nd_update(linear, nd_indices, linear_update)

        quadratic = tf.divide(1.0, math_ops.pow(accum_update, lr_power) * lr) + 2 * l2

        var_new = tf.math.sign(linear_update) * l1 - linear_update
        var_new = tf.divide(var_new, quadratic)
        mask = math_ops.cast(tf.math.greater(tf.abs(linear_update), l1), var.dtype.base_dtype)

        var_update = tf.multiply(var_new, mask)

        var_update_op = scatter_nd_update(var, nd_indices, var_update)

        return control_flow_ops.group(accum_update_op, linear_update_op, var_update_op)

    def _resource_scatter_nd_update(self, x, i, v):
        with ops.control_dependencies([
            gen_state_ops.resource_scatter_nd_update(x.handle, i, v)]):
            return x.value()

    def _create_slots(self, var_list):

        # Create slots for the first and second moments.
        accum_state_name = self._name + "/" + "accum"
        linear_state_name = self._name + "/" + "linear"
        for each_var in var_list:
            with ops.colocate_with(each_var):
                val = constant_op.constant(
                    self._initial_accumulator_value, dtype=each_var.dtype, shape=each_var.get_shape())

                table_instance = check_and_get_config_via_var(each_var, self.optimizer_type)

                accum = self._get_or_make_slot(each_var, val, "accum", accum_state_name)
                linear = self._zeros_slot(each_var, "linear", linear_state_name)
                # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
                ConfigInitializer.get_instance().sparse_embed_config.insert_removing_var_list(accum.name)
                ConfigInitializer.get_instance().sparse_embed_config.insert_removing_var_list(linear.name)
                table_instance = self.config_instance.sparse_embed_config.get_table_instance(each_var)
                ConfigInitializer.get_instance().optimizer_config.set_optimize_for_table(table_instance.table_name,
                                                                                         self._name,
                                                                                         {"accum": accum,
                                                                                          "linear": linear})

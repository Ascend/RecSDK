#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
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
from mx_rec.util.initialize import get_table_instance, insert_removing_var_list
from mx_rec.util.variable import check_and_get_config_via_var, check_param_type, check_param_range


def create_hash_optimizer(learning_rate, use_locking=False, name="Ftrl", **kwargs):

    return CustomizedFtrl(learning_rate=learning_rate, use_locking=use_locking, name=name, **kwargs)


class CustomizedFtrl(ftrl.FtrlOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, use_locking=False, name="Ftrl", **kwargs):
        self.optimizer_type = "ftrl"
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

        param_name_list = ["initial_accumulator_value", "l1_regularization_strength",
                            "l2_regularization_strength", "l2_shrinkage_regularization_strength"]

        def _check_param_type_range(param_name_list):
            for name in param_name_list:
                if kwargs.get(name, None):
                    check_param_type(name, kwargs.get(name), (int, float))
                    check_param_range(name, kwargs.get(name), 0, 1e4)

        if kwargs.get("accum_name", None):
            check_param_type("accum_name", kwargs.get("accum_name"), str)

        if kwargs.get("linear_name", None):
            check_param_type("linear_name", kwargs.get("linear_name"), str)

        check_param_type("use_locking", use_locking, bool)

        _check_param_type_range(param_name_list)

    def initialize_slots(self, var, table_instance):
        val = constant_op.constant(
            self._initial_accumulator_value, dtype=var.dtype, shape=var.get_shape())

        accum = slot_creator.create_slot(var, val, self._name + "/" + "accum")
        linear = slot_creator.create_zeros_slot(var, self._name + "/" + "linear")
        insert_removing_var_list(accum.name)
        insert_removing_var_list(linear.name)
        named_slot_key = (var.op.graph, var.op.name)
        table_instance = get_table_instance(var)
        if self._name in table_instance.optimizer:
            raise EnvironmentError(f"Sparse optimizer named {self._name} has exists.")

        table_instance.set_optimizer(self._name, {"accum": accum, "linear": linear})
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
        logging.debug(f"######### _apply_sparse_duplicate_indices {var}")
        return self._apply_sparse(grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        logging.debug(f"######### _resource_apply_sparse_duplicate_indices {indices}")
        return self._resource_apply_sparse(grad, handle, indices)

    def _resource_apply_sparse(self, grad, handle, indices):
        logging.debug("Enter _resource_apply_sparse")
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
        logging.debug("Enter _apply_sparse")
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
        logging.debug("Enter _apply_sparse_shared")
        accum = self.get_slot(var, "accum")
        linear = self.get_slot(var, "linear")
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        l1 = math_ops.cast(self._l1_regularization_strength_tensor, var.dtype.base_dtype)
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
        logging.debug("Enter _apply_sparse_shared_v2")
        accum = self.get_slot(var, "accum")
        linear = self.get_slot(var, "linear")
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        l1 = math_ops.cast(self._l1_regularization_strength_tensor, var.dtype.base_dtype)
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
        logging.debug(" Enter _create_slots")

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
                insert_removing_var_list(accum.name)
                insert_removing_var_list(linear.name)

                if self._name not in table_instance.optimizer:
                    table_instance.set_optimizer(self._name, {"accum": accum, "linear": linear})

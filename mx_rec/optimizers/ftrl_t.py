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
from tensorflow.python.training import optimizer
from tensorflow.python.training import slot_creator

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import get_table_instance, insert_removing_var_list
from mx_rec.util.variable import check_and_get_config_via_var


def create_hash_optimizer(learning_rate, use_locking=False, name="Ftrl_t", **kwargs):

    return CustomizedFtrlT(learning_rate=learning_rate, use_locking=use_locking, name=name, **kwargs)


class CustomizedFtrlT(optimizer.Optimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, use_locking=False, name="Ftrl_t", **kwargs):
        self.optimizer_type = "ftrl"
        super(CustomizedFtrlT, self)._get_name(name=name)

        self._learning_rate = learning_rate
        self._alpha = kwargs.get("alpha", 0.06)
        self._beta = kwargs.get("beta", 1.0)
        self._lambda1 = kwargs.get("lambda1", 0.0)
        self._lambda2 = kwargs.get("lambda2", 0.0)
        self._epsilon = kwargs.get("epsilon", 0.0)
        self._grad_factor = kwargs.get("grad_factor", 0.0)
        self._z_name = kwargs.get("z_name", None)
        self._n_name = kwargs.get("n_name", None)
        self._g_name = kwargs.get("g_name", None)
        self._learning_rate_tensor = None
        self._alpha_tensor = None
        self._beta_tensor = None
        self._lambda1_tensor = None
        self._lambda2_tensor = None
        self._epsilon_tensor = None
        self._grad_factor_tensor = None
        super(CustomizedFtrlT, self).__init__(use_locking, self.unique_name)

    def initialize_slots(self, var, table_instance):
        z = slot_creator.create_zeros_slot(var, self._name + "/" + "z")
        n = slot_creator.create_zeros_slot(var, self._name + "/" + "n")
        g = slot_creator.create_zeros_slot(var, self._name + "/" + "g")
        w = slot_creator.create_zeros_slot(var, self._name + "/" + "w")
        insert_removing_var_list(z.name)
        insert_removing_var_list(n.name)
        insert_removing_var_list(g.name)
        insert_removing_var_list(w.name)
        named_slot_key = (var.op.graph, var.op.name)
        table_instance = get_table_instance(var)
        if self._name in table_instance.optimizer:
            raise EnvironmentError(f"Sparse optimizer named {self._name} has exists.")

        table_instance.set_optimizer(self._name, {"z": z, "n": n, "g": g, "w": w})
        return [{"slot": z, "named_slot_key": named_slot_key, "slot_name": "z", "optimizer": self},
                {"slot": n, "named_slot_key": named_slot_key, "slot_name": "n", "optimizer": self},
                {"slot": g, "named_slot_key": named_slot_key, "slot_name": "g", "optimizer": self},
                {"slot": w, "named_slot_key": named_slot_key, "slot_name": "w", "optimizer": self}]

    def insert_slot(self, slot, named_slots_key, slot_name):
        named_slots = self._slot_dict(slot_name)
        if named_slots_key in named_slots:
            raise EnvironmentError(f"named_slots_key should be global unique, but it has been in use now, "
                                   f"please double check.")

        named_slots[named_slots_key] = slot

    def get_slot_init_values(self):
        initial_z_value = 0.0
        initial_n_value = 0.0
        initial_g_value = 0.0
        initial_w_value = 0.0
        return [initial_z_value, initial_n_value, initial_g_value, initial_w_value]

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(
            self._learning_rate, name="learning_rate")
        self._alpha_tensor = ops.convert_to_tensor(self._alpha, name="alpha")
        self._beta_tensor = ops.convert_to_tensor(self._beta, name="beta")
        self._lambda1_tensor = ops.convert_to_tensor(self._lambda1, name="lambda1")
        self._lambda2_tensor = ops.convert_to_tensor(self._lambda2, name="lambda2")
        self._epsilon_tensor = ops.convert_to_tensor(self._epsilon, name="epsilon")
        self._grad_factor_tensor = ops.convert_to_tensor(self._grad_factor, name="grad_factor")

    def _apply_sparse_duplicate_indices(self, grad, var):
        return self._apply_sparse(grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return self._resource_apply_sparse(grad, handle, indices)

    def _resource_apply_sparse(self, grad, handle, indices):
        if self._lambda1 > 1e-10:
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
        if self._lambda1 > 1e-10:
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
        z = self.get_slot(var, "z")
        n = self.get_slot(var, "n")
        g = self.get_slot(var, "g")
        w = self.get_slot(var, "w")
        alpha = math_ops.cast(self._alpha_tensor, var.dtype.base_dtype)
        beta = math_ops.cast(self._beta_tensor, var.dtype.base_dtype)
        lambda1 = math_ops.cast(self._lambda1_tensor, var.dtype.base_dtype)
        lambda2 = math_ops.cast(self._lambda2_tensor, var.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)
        grad_factor = math_ops.cast(self._grad_factor_tensor, var.dtype.base_dtype)

        abs_indices = tf.math.maximum(indices, 0)
        nd_indices = tf.expand_dims(indices, 1)
        with tf.control_dependencies([grad]):
            z_old = tf.gather(z, abs_indices)
            n_old = tf.gather(n, abs_indices)
            g_old = tf.gather(g, abs_indices)
            var_old = tf.gather(w, abs_indices)

        g_new = grad_factor * g_old + (1.0 - grad_factor) * grad
        with tf.control_dependencies([g_new]):
            g_update = scatter_nd_update(g, nd_indices, g_new)

        rho = tf.divide(tf.sqrt(n_old + tf.square(g_new)) - tf.sqrt(n_old), alpha)
        z_new = (1.0 - epsilon) * z_old + g_new - tf.multiply(rho, var_old)
        with tf.control_dependencies([z_new]):
            z_update = scatter_nd_update(z, nd_indices, z_new)

        n_new = (1.0 - epsilon) * n_old + tf.square(g_new)
        with tf.control_dependencies([n_new]):
            n_update = scatter_nd_update(n, nd_indices, n_new)

        denominator = tf.divide((beta + tf.sqrt(n_new)), alpha) + lambda2
        numerator = lambda1 * tf.sign(z_new) - z_new
        mask = math_ops.cast(tf.math.greater(tf.abs(z_new), lambda1), var.dtype.base_dtype)
        var_new = tf.multiply(mask, tf.divide(numerator, denominator))
        with tf.control_dependencies([var_new]):
            w_update = scatter_nd_update(w, nd_indices, var_new)
            var_update = scatter_nd_update(var, nd_indices, var_new)

        return control_flow_ops.group(g_update, z_update, n_update, w_update, var_update)

    def _apply_sparse_shared_v2(self, grad, var, indices, scatter_nd_update):
        z = self.get_slot(var, "z")
        n = self.get_slot(var, "n")
        g = self.get_slot(var, "g")
        w = self.get_slot(var, "w")
        alpha = math_ops.cast(self._alpha_tensor, var.dtype.base_dtype)
        beta = math_ops.cast(self._beta_tensor, var.dtype.base_dtype)
        lambda2 = math_ops.cast(self._lambda2_tensor, var.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)
        grad_factor = math_ops.cast(self._grad_factor_tensor, var.dtype.base_dtype)

        abs_indices = tf.math.maximum(indices, 0)
        nd_indices = tf.expand_dims(indices, 1)
        with tf.control_dependencies([grad]):
            z_old = tf.gather(z, abs_indices)
            n_old = tf.gather(n, abs_indices)
            g_old = tf.gather(g, abs_indices)
            var_old = tf.gather(w, abs_indices)

        g_new = grad_factor * g_old + (1.0 - grad_factor) * grad
        with tf.control_dependencies([g_new]):
            g_update = scatter_nd_update(g, nd_indices, g_new)

        rho = tf.divide(tf.sqrt(n_old + tf.square(g_new)) - tf.sqrt(n_old), alpha)
        z_new = (1.0 - epsilon) * z_old + g_new - tf.multiply(rho, var_old)
        with tf.control_dependencies([z_new]):
            z_update = scatter_nd_update(z, nd_indices, z_new)

        n_new = (1.0 - epsilon) * n_old + tf.square(g_new)
        with tf.control_dependencies([n_new]):
            n_update = scatter_nd_update(n, nd_indices, n_new)

        denominator = tf.divide((beta + tf.sqrt(n_new)), alpha) + lambda2
        var_new = tf.divide(-1.0 * z_new, denominator)
        with tf.control_dependencies([var_new]):
            w_update = scatter_nd_update(w, nd_indices, var_new)
            var_update = scatter_nd_update(var, nd_indices, var_new)

        return control_flow_ops.group(g_update, z_update, n_update, w_update, var_update)

    def _resource_scatter_nd_update(self, x, i, v):
        with ops.control_dependencies([
            gen_state_ops.resource_scatter_nd_update(x.handle, i, v)]):
            return x.value()

    def _create_slots(self, var_list):

        # Create slots for the first and second moments.
        z_state_name = self._name + "/" + "z"
        n_state_name = self._name + "/" + "n"
        g_state_name = self._name + "/" + "g"
        w_state_name = self._name + "/" + "w"
        for each_var in var_list:
            with ops.colocate_with(each_var):
                table_instance = check_and_get_config_via_var(each_var, self.optimizer_type)

                z = self._zeros_slot(each_var, "z", z_state_name)
                n = self._zeros_slot(each_var, "n", n_state_name)
                g = self._zeros_slot(each_var, "g", g_state_name)
                w = self._zeros_slot(each_var, "w", w_state_name)
                # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
                insert_removing_var_list(z.name)
                insert_removing_var_list(n.name)
                insert_removing_var_list(g.name)
                insert_removing_var_list(w.name)

                if self._name not in table_instance.optimizer:
                    table_instance.set_optimizer(self._name, {"z": z, "n": n, "g": g, "w": w})

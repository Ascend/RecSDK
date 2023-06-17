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


def create_ftrl_dense_optimizer(learning_rate, use_locking=False, name="Ftrl_t_dense", **kwargs):

    return CustomizedFtrlTZ(learning_rate=learning_rate, use_locking=use_locking, name=name, **kwargs)


class CustomizedFtrlTZ(optimizer.Optimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, use_locking=False, name="Ftrl_t_dense", **kwargs):
        self.optimizer_type = "ftrl"
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
        super(CustomizedFtrlTZ, self).__init__(use_locking, name)
        logging.debug("CustomizedFtrlTZ __init__ ok")

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(
            self._learning_rate, name="learning_rate")
        self._alpha_tensor = ops.convert_to_tensor(self._alpha, name="alpha")
        self._beta_tensor = ops.convert_to_tensor(self._beta, name="beta")
        self._lambda1_tensor = ops.convert_to_tensor(self._lambda1, name="lambda1")
        self._lambda2_tensor = ops.convert_to_tensor(self._lambda2, name="lambda2")
        self._epsilon_tensor = ops.convert_to_tensor(self._epsilon, name="epsilon")
        self._grad_factor_tensor = ops.convert_to_tensor(self._grad_factor, name="grad_factor")

    def _resource_apply_dense(self, grad, handle):
        if self._lambda1 > 1e-10:
            return self._apply_dense_shared(
                grad,
                handle)
        else:
            return self._apply_dense_shared_v2(
                grad,
                handle)

    def _apply_dense(self, grad, var):
        if self._lambda1 > 1e-10:
            return self._apply_dense_shared(
                grad,
                var)
        else:
            return self._apply_dense_shared_v2(
                grad,
                var)

    def _apply_dense_shared(self, grad, var):
        logging.debug("Enter _apply_dense_shared")
        z_var = self.get_slot(var, "z")
        n_var = self.get_slot(var, "n")
        g_var = self.get_slot(var, "g")
        w_var = self.get_slot(var, "w")
        alpha = math_ops.cast(self._alpha_tensor, var.dtype.base_dtype)
        beta = math_ops.cast(self._beta_tensor, var.dtype.base_dtype)
        lambda1 = math_ops.cast(self._lambda1_tensor, var.dtype.base_dtype)
        lambda2 = math_ops.cast(self._lambda2_tensor, var.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)
        grad_factor = math_ops.cast(self._grad_factor_tensor, var.dtype.base_dtype)

        z_old = tf.identity(z_var)
        n_old = tf.identity(n_var)
        g_old = tf.identity(g_var)
        var_old = tf.identity(w_var)

        g_new = grad_factor * g_old + (1.0 - grad_factor) * grad
        with tf.control_dependencies([g_new]):
            g_update = tf.compat.v1.assign(g_var, g_new)

        rho = tf.divide(tf.sqrt(n_old + tf.square(g_new)) - tf.sqrt(n_old), alpha)
        z_new = (1.0 - epsilon) * z_old + g_new - tf.multiply(rho, var_old)
        with tf.control_dependencies([z_new]):
            z_update = tf.compat.v1.assign(z_var, z_new)

        n_new = (1.0 - epsilon) * n_old + tf.square(g_new)
        with tf.control_dependencies([n_new]):
            n_update = tf.compat.v1.assign(n_var, n_new)

        denominator = tf.divide((beta + tf.sqrt(n_new)), alpha) + lambda2
        numerator = lambda1 * tf.sign(z_new) - z_new
        mask = math_ops.cast(tf.math.greater(tf.abs(z_new), lambda1), var.dtype.base_dtype)
        var_new = tf.multiply(mask, tf.divide(numerator, denominator))
        with tf.control_dependencies([var_new]):
            w_update = tf.compat.v1.assign(w_var, var_new)
            var_updata = tf.compat.v1.assign(var, var_new)

        return control_flow_ops.group(g_update, z_update, n_update, w_update, var_updata)

    def _apply_dense_shared_v2(self, grad, var):
        logging.debug("Enter _apply_dense_shared_v2")
        z_var = self.get_slot(var, "z")
        n_var = self.get_slot(var, "n")
        g_var = self.get_slot(var, "g")
        w_var = self.get_slot(var, "w")
        alpha = math_ops.cast(self._alpha_tensor, var.dtype.base_dtype)
        beta = math_ops.cast(self._beta_tensor, var.dtype.base_dtype)
        lambda2 = math_ops.cast(self._lambda2_tensor, var.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)
        grad_factor = math_ops.cast(self._grad_factor_tensor, var.dtype.base_dtype)

        z_old = tf.identity(z_var)
        n_old = tf.identity(n_var)
        g_old = tf.identity(g_var)
        var_old = tf.identity(w_var)

        g_new = grad_factor * g_old + (1.0 - grad_factor) * grad
        with tf.control_dependencies([g_new]):
            g_updata = tf.compat.v1.assign(g_var, g_new)

        rho = tf.divide(tf.sqrt(n_old + tf.square(g_new)) - tf.sqrt(n_old), alpha)
        z_new = (1.0 - epsilon) * z_old + g_new - tf.multiply(rho, var_old)
        with tf.control_dependencies([z_new]):
            z_updata = tf.compat.v1.assign(z_var, z_new)

        n_new = (1.0 - epsilon) * n_old + tf.square(g_new)
        with tf.control_dependencies([n_new]):
            n_updata = tf.compat.v1.assign(n_var, n_new)

        denominator = tf.divide((beta + tf.sqrt(n_new)), alpha) + lambda2
        var_new = tf.divide(-1.0 * z_new, denominator)
        with tf.control_dependencies([var_new]):
            w_updata = tf.compat.v1.assign(w_var, var_new)
            var_updata = tf.compat.v1.assign(var, var_new)

        return control_flow_ops.group(g_updata, z_updata, n_updata, w_updata, var_updata)

    def _resource_scatter_nd_update(self, x_input, i_input, v_input):
        with ops.control_dependencies([
            gen_state_ops.resource_scatter_nd_update(x_input.handle, i_input, v_input)]):
            return x_input.value()

    def _create_slots(self, var_list):
        logging.debug(" Enter _create_slots")

        # Create slots for the first and second moments.
        z_state_name = self._name + "/" + "z"
        n_state_name = self._name + "/" + "n"
        g_state_name = self._name + "/" + "g"
        w_state_name = self._name + "/" + "w"
        for each_var in var_list:
            with ops.colocate_with(each_var):
                z_zero = self._zeros_slot(each_var, "z", z_state_name)
                n_zero = self._zeros_slot(each_var, "n", n_state_name)
                g_zero = self._zeros_slot(each_var, "g", g_state_name)
                w_zero = self._zeros_slot(each_var, "w", w_state_name)
                # make sure sparse optimizer statements will not be saved and restored within tf checkpoint.
                insert_removing_var_list(z_zero.name)
                insert_removing_var_list(n_zero.name)
                insert_removing_var_list(g_zero.name)
                insert_removing_var_list(w_zero.name)



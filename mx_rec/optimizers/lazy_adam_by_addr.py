#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import logging
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.eager import context
from tensorflow.python.framework import indexed_slices

from mx_rec.util.initialize import get_host_pipeline_ops, insert_optimizer
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.variable import check_param_type, check_param_range


def create_hash_optimizer_by_address(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                     name="LazyAdamByAddress"):
    """
    Args:
        learning_rate: learning rate
        beta1:
        beta2:
        epsilon:
        name:

    Returns: a customized optimizer instance
    """

    optimizer_by_addr = CustomizedLazyAdamByAddress(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                    epsilon=epsilon, name=name)
    insert_optimizer(optimizer_by_addr)
    return optimizer_by_addr


class CustomizedLazyAdamByAddress(optimizer.Optimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False,
                 name="LazyAdamByAddress"):
        self.optimizer_type = "LazyAdamByAddress"
        super(CustomizedLazyAdamByAddress, self).__init__(use_locking, name)

        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._non_slot_dict = {}
        self._slot_num = 2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._check_input_param()

    def _check_input_param(self):
        check_param_type("beta1", self._beta1, (int, float))
        check_param_range("beta1", self._beta1, 0, 1)

        check_param_type("beta2", self._beta2, (int, float))
        check_param_range("beta2", self._beta2, 0, 1)

        check_param_type("epsilon", self._epsilon, (int, float))
        check_param_range("epsilon", self._epsilon, 0, 1)

        check_param_type("use_locking", self._use_locking, bool)

    @property
    def slot_num(self):
        return self._slot_num

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, addr_list):
        first_addr = addr_list[0]
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_addr)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_addr)

    def _create_non_slot_variable(self, initial_value, name, colocate_with):
        """Add an extra variable, not associated with a slot."""
        # Recommendation: Use OptimizerV2 if your optimizer uses non-slot variables.
        eager = context.executing_eagerly()
        graph = None if eager else tf.get_default_graph()

        key = (name, graph)
        var = self._non_slot_dict.get(key, None)
        if var is None:
            distribution_strategy = distribute_ctx.get_strategy()
            with distribution_strategy.extended.colocate_vars_with(colocate_with):
                var = variable_scope.variable(
                    initial_value, name=name, trainable=False,
                    use_resource=resource_variable_ops.is_resource_variable(
                        colocate_with))
            self._non_slot_dict[key] = var
        return var

    def _prepare(self):
        learn_rate = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)

        self._lr_t = ops.convert_to_tensor(learn_rate, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)

    def get_slot_init_values(self):
        # return state value list of adam that needs to initialize in ASC DDR.
        initial_momentum_value = 0.0
        initial_velocity_value = 0.0
        return [initial_momentum_value, initial_velocity_value]

    def _apply_dense(self, grad, var):
        logging.debug(">>>>Enter _apply_dense")
        raise NotImplementedError("You are using a wrong type of variable.")

    def _cast_to_base_type(self, var):
        var_type = var.dtype.base_dtype
        temp_lr = math_ops.cast(self._lr_t, var_type)
        temp_b1 = math_ops.cast(self._beta1_t, var_type)
        temp_b2 = math_ops.cast(self._beta2_t, var_type)
        temp_epsilon = math_ops.cast(self._epsilon_t, var_type)
        return temp_lr, temp_b1, temp_b2, temp_epsilon

    def _apply_sparse(self, grad, addr):
        logging.debug(">>>> Enter _apply_sparse Lazy_adam by addr")
        return self._apply_sparse_shared(
            grad,
            addr)

    def _apply_sparse_shared(self, grad, addr):
        power_b1, power_b2 = self._get_beta_accumulators()
        power_b1 = math_ops.cast(power_b1, grad.dtype.base_dtype)
        power_b2 = math_ops.cast(power_b2, grad.dtype.base_dtype)
        temp_lr, temp_b1, temp_b2, temp_epsilon = self._cast_to_base_type(grad)
        learning_rate = tf.divide(temp_lr * math_ops.sqrt(1 - power_b2), (1 - power_b1))

        host_pipeline_ops = get_host_pipeline_ops()
        dim = grad.shape.as_list()[-1]
        combined_tensor = \
            host_pipeline_ops.embedding_lookup_by_address(addr, embedding_dim=3 * dim, embedding_type=1)

        split_length = [dim] + [dim] + [dim]
        split_tensors = tf.split(combined_tensor, split_length, axis=1)

        old_m_slice = split_tensors[1]
        m_t_slice = temp_b1 * old_m_slice + (1 - temp_b1) * grad

        old_v_slice = split_tensors[2]
        v_t_slice = temp_b2 * old_v_slice + (1 - temp_b2) * math_ops.square(grad)

        denominator_slice = math_ops.sqrt(v_t_slice) + temp_epsilon
        update_list = [tf.divide(-learning_rate * m_t_slice, denominator_slice)] + [m_t_slice - old_m_slice] + \
                      [v_t_slice - old_v_slice]
        update_tensor = tf.concat(update_list, axis=1)
        var_update_op = host_pipeline_ops.embedding_update_by_address(addr, update_tensor, update_type=0)
        var_update_op = tf.identity(var_update_op, name="identity_var_update_op")

        return var_update_op

    def _convert_grads_and_addrs(self, grads_and_vars):
        converted_grads_and_addrs = []
        for grad, addr in grads_and_vars:
            if grad is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    grad = ops.convert_to_tensor_or_indexed_slices(grad)
                except TypeError as error:
                    raise TypeError("Gradient must be convertible to a Tensor or IndexedSlices, or None") from error
                if not isinstance(grad, (ops.Tensor, indexed_slices.IndexedSlices)):
                    raise TypeError("Gradient must be a Tensor, IndexedSlices, or None")
            processor = _get_processor(addr)
            converted_grads_and_addrs.append((grad, addr, processor))
        return converted_grads_and_addrs

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        # No DistributionStrategy case.
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")

        converted_grads_and_addrs = tuple(self._convert_grads_and_addrs(grads_and_vars))
        addr_list = [a for g, a, _ in converted_grads_and_addrs if g is not None]
        if not addr_list:
            raise ValueError("No gradients provided for any address: %s." %
                             ([str(a) for _, a, _ in converted_grads_and_addrs],))
        with ops.init_scope():
            self._create_slots(addr_list)
        update_ops = []
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            for grad, addr, processor in converted_grads_and_addrs:
                if grad is None:
                    continue
                if (context.executing_eagerly() or
                        resource_variable_ops.is_resource_variable(addr)
                        and not addr._in_graph_mode):  # pylint: disable=protected-access
                    scope_name = ""
                else:
                    scope_name = addr.op.name
                with ops.name_scope(
                        f"update_{scope_name}"), ops.colocate_with(addr):
                    update_ops.append(processor.update_op(self, grad))

            apply_updates = self._finish(update_ops, name)

            if not context.executing_eagerly():
                if isinstance(apply_updates, ops.Tensor):
                    logging.debug(">>>>Enter ops.Tensor")
                    apply_updates = apply_updates.op
                train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
                if apply_updates not in train_op:
                    logging.debug(">>>>Enter apply_updates not in train_op")
                    train_op.append(apply_updates)
            else:
                raise RuntimeError("eager wrong.")

            return apply_updates


def get_filtered_grad_fn(grad_fn):
    def filtered_grad_fn(*args, **kwargs):
        return [(g, a) for g, a in grad_fn(*args, **kwargs) if g is not None]

    return filtered_grad_fn


class _OptimizableAddr(metaclass=abc.ABCMeta):
    """Interface for abstracting over addresses in the optimizers."""

    @abc.abstractmethod
    def target(self):
        """Returns the optimization target for this address."""
        raise NotImplementedError("Calling an abstract method.")

    @abc.abstractmethod
    def update_op(self, opt, grad):
        """Returns the update ops for updating the address."""
        raise NotImplementedError("Calling an abstract method.")


class _TensorByAddressProcessor(_OptimizableAddr):
    """Processor for Tensor filled with addresses."""

    def __init__(self, addr):
        self._a = addr

    def target(self):
        return self._a

    def __str__(self):
        return "<_TensorByAddressProcessor(%s)>" % self._a

    def update_op(self, opt, grad):
        if isinstance(grad, ops.Tensor):
            logging.debug(">>>>Enter update_op ops.Tensor")
            update_op = opt._apply_sparse(grad, self._a)  # pylint: disable=protected-access
            return update_op
        else:
            raise RuntimeError("Only support g with type Tensor.")


def _get_processor(addr):
    """The processor of v."""
    if isinstance(addr, ops.Tensor):
        logging.debug(">>>>Enter _get_processor tensor")
        return _TensorByAddressProcessor(addr)
    raise NotImplementedError("Trying to optimize unsupported type ", addr)

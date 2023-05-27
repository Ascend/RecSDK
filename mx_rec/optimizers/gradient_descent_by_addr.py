#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import logging
from collections import defaultdict

from tensorflow.python.framework import ops, indexed_slices
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.eager import context
from tensorflow.python.ops import resource_variable_ops

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import get_host_pipeline_ops, insert_optimizer


def create_hash_optimizer_by_addr(learning_rate, weight_decay=0.0001, use_locking=False, name="GradientDescentByAddr"):
    optimizer_by_addr = CustomizedGradientDescentByAddr(learning_rate=learning_rate,
                                                        weight_decay=weight_decay,
                                                        use_locking=use_locking,
                                                        name=name)
    insert_optimizer(optimizer_by_addr)
    return optimizer_by_addr


class CustomizedGradientDescentByAddr(optimizer.Optimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, weight_decay, use_locking=False, name="GradientDescentByAddr"):
        self.optimizer_type = "gradient_descent_by_addr"
        self.weight_decay = weight_decay
        super(CustomizedGradientDescentByAddr, self).__init__(use_locking, name)

        self._learning_rate = learning_rate
        self._learning_rate_tensor = None
        self._slot_num = 0

    @property
    def slot_num(self):
        return self._slot_num

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
                        "update_" + scope_name), ops.colocate_with(addr):
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

    def get_slot_init_values(self):
        return []

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

    def _prepare(self):
        learning_rate = self._call_if_callable(self._learning_rate)
        self._learning_rate_tensor = ops.convert_to_tensor(
            learning_rate, name="learning_rate")

    def _apply_sparse(self, grad, addr):
        logging.debug(">>>> Enter _apply_sparse SGD by addr")
        host_pipeline_ops = get_host_pipeline_ops()
        dim = grad.shape.as_list()[-1]
        if self.weight_decay is None:
            nd_value = grad * math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype)
        else:
            lookup_tensor = \
                host_pipeline_ops.embedding_lookup_by_address(addr, embedding_dim=dim, embedding_type=1)
            nd_value = (grad + math_ops.cast(self.weight_decay, grad.dtype.base_dtype) * lookup_tensor) * math_ops.cast(
                self._learning_rate_tensor, grad.dtype.base_dtype)
        var_update_op = host_pipeline_ops.embedding_update_by_address(addr, -nd_value, update_type=0)

        return var_update_op

    def _apply_dense(self, grad, var):
        logging.debug(">>>> Enter _apply_dense")
        raise NotImplementedError("You are using a wrong type of variable.")


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


def _get_processor(addr):
    """The processor of v."""
    if isinstance(addr, ops.Tensor):
        logging.debug(">>>>Enter _get_processor tensor")
        return _TensorByAddressProcessor(addr)
    raise NotImplementedError("Trying to optimize unsupported type ", addr)


class _TensorByAddressProcessor(_OptimizableAddr):
    """Processor for Tensor filled with addresses."""

    def __init__(self, addr):
        self._a = addr

    def __str__(self):
        return "<_TensorByAddressProcessor(%s)>" % self._a

    def target(self):
        return self._a

    def update_op(self, opt, grad):
        if isinstance(grad, ops.Tensor):
            logging.debug(">>>>Enter update_op ops.Tensor")
            update_op = opt._apply_sparse(grad, self._a)  # pylint: disable=protected-access
            return update_op
        else:
            raise RuntimeError("Only support g with type Tensor.")

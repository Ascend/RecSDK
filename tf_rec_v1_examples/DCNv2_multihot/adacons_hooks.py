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

import tensorflow as tf
from npu_bridge.hccl import hccl_ops


def compute_alphas(u_i, norm_g_i, size, device):
    """
    Computes the alpha scaling factor for each device based on its gradient's norm and
    the global size of the distributed system.

    Args:
        u_i (tf.Tensor): The dot product of the gradient sum and the gradient itself.
        norm_g_i (tf.Tensor): The sum of squared gradients, representing the gradient norm.
        size (int): The total number of devices participating in the computation.
        device (int): The ID of the current device, used to select the appropriate alpha.

    Returns:
        tf.Tensor: The alpha scaling factor for the current device.

    Example:
    ```python
    # Assuming u_i and norm_g_i are computed tensors
    u_i = tf.constant(5.0, dtype=tf.float32)
    norm_g_i = tf.constant(4.0, dtype=tf.float32)
    rank_size = 8  # Total number of devices
    device_id = 2  # Current device ID

    alpha = compute_alphas(u_i, norm_g_i, rank_size, device_id)
    print(alpha)
    ```
    This example will return the alpha value for the current device.
    """
    # Calculate alpha_i by normalizing with respect to gradient norm and the number of devices
    alpha_i = u_i / size / (tf.sqrt(norm_g_i) + 1e-8)

    # Stack alpha_i and the square root of the gradient norm for collective operations
    combined_a_ngi = tf.stack([alpha_i, tf.sqrt(norm_g_i)])

    # Perform All-gather across devices to gather alpha and norm_g_i values
    combined_a_ngi_list = hccl_ops.allgather(
        tensor=combined_a_ngi,
        rank_size=size
    )

    # Reshape the result to have shape [size, 2], where each row corresponds to a device
    combined_a_ngi_list = tf.reshape(combined_a_ngi_list, [-1, 2])

    # Split the gathered data into alphas and norm_g_i components
    alphas = combined_a_ngi_list[:, 0]
    all_norm_g_i = combined_a_ngi_list[:, 1]

    # Ensure that alphas are non-negative by shifting if any are negative
    alphas = tf.cond(tf.reduce_min(alphas) < 0,
                     lambda: alphas - tf.reduce_min(alphas),
                     lambda: alphas)

    # Normalize alphas so that their sum is 1
    alphas = alphas / (tf.reduce_sum(alphas) + 1e-8)

    # Further scale alphas using the gradient norms (betas)
    alphas = alphas / (all_norm_g_i + 1e-8)

    # Select the alpha for the current device. If only one device, return the whole alphas tensor.
    alpha_i = alphas[device] if size > 1 else alphas

    return alpha_i


def adacons_hooks(gradients, rank_size, device_id):
    """
    Custom hook to modify gradients for distributed training using the Adacons method.
    It adjusts the gradients by multiplying them with a computed scaling factor (alpha_i)
    and synchronizes them across devices using All-reduce operations.

    Args:
        gradients (list of tuple[tf.Tensor, tf.Variable]): List of gradient-variable pairs.
        rank_size (int): The total number of devices in the distributed setup.
        device_id (int): The device ID of the current device.

    Returns:
        list of tuple[tf.Tensor, tf.Variable]: A list of tuples, each containing the final processed gradient
        and its corresponding variable.

    Example:
    ```python
    # Assuming gradients is a list of (gradient, variable) tuples
    gradients = [(tf.random.uniform([10]), tf.Variable(0.1)) for _ in range(4)]  # 4 devices
    rank_size = 4  # Total number of devices
    device_id = 1  # Current device ID

    # Process gradients and apply Adacons scaling
    updated_gradients = adacons_hooks(gradients, rank_size, device_id)
    for grad, var in updated_gradients:
        print(grad)
        print(var)
    ```

    This example will show how to process gradients and apply the scaling factors.
    """
    futures = []

    def process_gradient(g_i):
        """
        Process a single gradient by applying the Adacons method and synchronizing it across devices.

        Args:
            g_i (tf.Tensor): The gradient tensor to be processed.

        Returns:
            tf.Tensor: The processed gradient after applying scaling and synchronization.
        """
        # Clone the gradient tensor to prevent modifying the original during the process
        g_bar = tf.identity(g_i)

        # Perform All-reduce to sum the gradients from all devices
        g_sum_val = hccl_ops.allreduce(tensor=g_bar, reduction='sum', fusion=2, fusion_id=11)

        # Compute the dot product of the summed gradients and the current gradient
        u_i = tf.reduce_sum(g_sum_val * g_i)

        # Compute the norm (sum of squares) of the current gradient
        norm_g_i = tf.reduce_sum(tf.square(g_i))

        # Compute the alpha_i for the current device
        alpha_i = compute_alphas(u_i, norm_g_i, rank_size, device_id)

        # Scale the gradient by the computed alpha_i
        g_i *= alpha_i

        # Perform All-reduce again to synchronize the scaled gradients across all devices
        final_gi = hccl_ops.allreduce(tensor=g_i, reduction='sum', fusion=2, fusion_id=22)

        return final_gi

    # Process each gradient in the list of gradients
    for grad, var in gradients:
        final_process_gi = process_gradient(grad)
        futures.append((final_process_gi, var))

    return futures

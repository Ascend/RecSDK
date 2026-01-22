#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

from typing import Tuple

import tensorflow as tf


class LearningRateScheduler:
    """LR Scheduler combining Polynomial Decay with Warmup at the beginning."""

    def __init__(
        self, base_lr_dense: float, base_lr_sparse: float, warmup_steps: int, decay_start_steps: int, decay_steps: int
    ):
        self._warmup_total_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self._decay_start_total_steps = tf.constant(decay_start_steps, dtype=tf.int32)
        self._decay_total_steps = tf.constant(decay_steps)
        self._decay_end_total_steps = decay_start_steps + decay_steps
        self._poly_power = 2.0
        self._base_lr_dense = base_lr_dense
        self._base_lr_sparse = base_lr_sparse
        self._sparse_after_decay = tf.cast(1 / self._decay_total_steps, tf.float32)
        self._lr_factor_constant = tf.constant(1, dtype=tf.float32)

    def calc(self, global_step: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor]:
        # Used for the warmup stage.
        warmup_step_ratio = tf.cast(1 / self._warmup_total_steps, tf.float32)
        lr_warmup_factor = 1.0 - tf.cast(self._warmup_total_steps - global_step, tf.float32) * warmup_step_ratio
        lr_warmup_factor = tf.cast(lr_warmup_factor, tf.float32)

        # Used for the decay stage.
        lr_decay_factor = (self._decay_end_total_steps - global_step) / self._decay_start_total_steps
        lr_decay_factor = tf.math.pow(lr_decay_factor, self._poly_power)
        lr_decay_factor = tf.cast(lr_decay_factor, tf.float32)

        lr_decay_factor = tf.where(
            global_step < self._decay_end_total_steps,
            lr_decay_factor,
            self._sparse_after_decay
        )

        poly_schedule = tf.where(
            global_step < self._decay_start_total_steps,
            self._lr_factor_constant,
            lr_decay_factor
        )

        lr_factor = tf.where(
            global_step < self._warmup_total_steps,
            lr_warmup_factor,
            poly_schedule
        )

        lr_sparse = self._base_lr_sparse * lr_factor
        lr_dense = self._base_lr_dense * lr_factor

        return lr_dense, lr_sparse

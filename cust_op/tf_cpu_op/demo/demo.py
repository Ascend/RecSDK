#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('op', None, 'operator')
tf.app.flags.DEFINE_integer('minval', 0, 'min val')
tf.app.flags.DEFINE_integer('maxval', 10000, 'max val')
tf.app.flags.DEFINE_integer('iters', 5000, 'op iters')
tf.app.flags.DEFINE_integer('row', 1, 'maxtrix row')
tf.app.flags.DEFINE_integer('col', 4000, 'maxtrix col')
FLAGS = tf.app.flags.FLAGS

MAX_COL = 100000
MAX_ITERS = 10000
MIN_VAL = -32768
MAX_VAL = 32767


class TensorflowBenchmark(tf.test.Benchmark):
    def __init__(self):
        super(TensorflowBenchmark, self).__init__()

    def benchmark_less(self, shape: list):
        iters = FLAGS.iters
        minval = FLAGS.minval
        maxval = FLAGS.maxval
        with tf.compat.v1.Session() as sess:
            matrix1 = tf.random.uniform([shape[0], shape[1]], minval=minval, maxval=maxval, dtype=tf.int64)
            matrix2 = tf.random.uniform([shape[0], shape[1]], minval=minval, maxval=maxval, dtype=tf.int64)

            product = tf.less(matrix1, matrix2)
            _ = tf.test.Benchmark().run_op_benchmark(
                sess=sess,
                op_or_tensor=product,
                burn_iters=10,
                min_iters=iters,
                store_trace=False
            )

    def benchmark_greater(self, shape: list):
        iters = FLAGS.iters
        minval = FLAGS.minval
        maxval = FLAGS.maxval
        with tf.compat.v1.Session() as sess:
            matrix1 = tf.random.uniform([shape[0], shape[1]], minval=minval, maxval=maxval, dtype=tf.int64)
            matrix2 = tf.random.uniform([shape[0], shape[1]], minval=minval, maxval=maxval, dtype=tf.int64)

            product = tf.greater(matrix1, matrix2)
            _ = tf.test.Benchmark().run_op_benchmark(
                sess=sess,
                op_or_tensor=product,
                burn_iters=10,
                min_iters=iters,
                store_trace=False
            )

    def benchmark_floor_mod(self, shape: list):
        iters = FLAGS.iters
        minval = FLAGS.minval
        maxval = FLAGS.maxval
        with tf.compat.v1.Session() as sess:
            matrix1 = tf.random.uniform([shape[0], shape[1]], minval=minval, maxval=maxval, dtype=tf.float32)
            matrix2 = tf.random.uniform([shape[0], shape[1]], minval=minval, maxval=maxval, dtype=tf.float32)

            product = tf.math.floormod(matrix1, matrix2, name='FloorMod')
            _ = tf.test.Benchmark().run_op_benchmark(
                sess=sess,
                op_or_tensor=product,
                burn_iters=10,
                min_iters=iters,
                store_trace=False
            )

    def benchmark_select(self, shape: list):
        iters = FLAGS.iters
        minval = FLAGS.minval
        maxval = FLAGS.maxval
        with tf.compat.v1.Session() as sess:
            matrix1 = tf.random.uniform([1, shape[1]], minval=minval, maxval=maxval, dtype=tf.int64)
            matrix2 = tf.random.uniform([1, shape[1]], minval=minval, maxval=maxval, dtype=tf.int64)

            random_tensor = tf.random.uniform([1, shape[1]], minval=0.0, maxval=2.0)
            cond = tf.cast(tf.floor(random_tensor), tf.bool)

            product = tf.compat.v1.where(cond, matrix1, matrix2)
            _ = tf.test.Benchmark().run_op_benchmark(
                sess=sess,
                op_or_tensor=product,
                burn_iters=10,
                min_iters=iters,
                store_trace=False
            )


def run_op():
    shape = [FLAGS.row, FLAGS.col]
    if FLAGS.op == 'Less':
        benchmark.benchmark_less(shape)
    elif FLAGS.op == 'Greater':
        benchmark.benchmark_greater(shape)
    elif FLAGS.op == 'FloorMod':
        benchmark.benchmark_floor_mod(shape)
    elif FLAGS.op == 'Select':
        benchmark.benchmark_select(shape)
    else:
        raise Exception(f"operator {FLAGS.op} is not found")


def go(_):
    if FLAGS.iters < 1 or FLAGS.iters > MAX_ITERS:
        raise Exception(f"iters should in [1,{MAX_ITERS}],but get {FLAGS.iters}")
    if FLAGS.minval < MIN_VAL or FLAGS.minval > MAX_VAL:
        raise Exception(f"minval should in [{MIN_VAL},{MAX_VAL}],but get {FLAGS.minval}")
    if FLAGS.maxval < MIN_VAL or FLAGS.maxval > MAX_VAL:
        raise Exception(f"maxval should in [{MIN_VAL},{MAX_VAL}],but get {FLAGS.maxval}")
    if FLAGS.maxval <= FLAGS.minval:
        raise Exception(f"minval({FLAGS.minval}) shoud less than maxval({FLAGS.maxval})")
    if FLAGS.row != 1:
        raise Exception(f"row shold be 1,but get {FLAGS.row}")
    if FLAGS.col < 1 or FLAGS.col > MAX_COL:
        raise Exception(f"col should in [1,{MAX_COL}],but get ({FLAGS.col})")

    run_op()


if __name__ == '__main__':
    benchmark = TensorflowBenchmark()
    # 接收参数
    tf.app.run(go)

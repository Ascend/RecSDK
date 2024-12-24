import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('op', None, 'operator')
tf.app.flags.DEFINE_integer('minval', 0, 'min val')
tf.app.flags.DEFINE_integer('maxval', 10000, 'max val')
tf.app.flags.DEFINE_integer('iters', 5000, 'op iters')
tf.app.flags.DEFINE_integer('row', 1, 'maxtrix row')
tf.app.flags.DEFINE_integer('col', 4000, 'maxtrix col')
FLAGS = tf.app.flags.FLAGS


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


def go(_):
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
        raise Exception(f"给定算子不存在: {FLAGS.op}")


if __name__ == '__main__':
    benchmark = TensorflowBenchmark()
    # 接收参数
    tf.app.run(go)

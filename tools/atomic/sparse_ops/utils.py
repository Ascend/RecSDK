"""
utils
"""
from __future__ import absolute_import
import tensorflow as tf
from mpi4py import rc

tf.get_logger().setLevel("ERROR")
rc.initialize = False  # if = True, The Init is done when "from mpi4py import MPI" is called


def ops():
    """
    返回emb相关的算子
    """
    return tf.load_op_library("libcust_ops.so")


def dataset_ops():
    """
    返回emb相关的算子
    """
    return tf.load_op_library("libasc_dataset_ops.so")

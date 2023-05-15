# coding: UTF-8
import os
import logging
import tensorflow as tf

from mx_rec.util.constants import HOST_PIPELINE_OPS_LIB_PATH


def import_host_pipeline_ops():
    host_pipeline_ops_lib_path = os.getenv(HOST_PIPELINE_OPS_LIB_PATH)
    if host_pipeline_ops_lib_path:
        logging.debug(f"Using the HOST_PIPELINE_OPS_LIB_PATH '{host_pipeline_ops_lib_path}' to get ops lib.")
        return tf.load_op_library(host_pipeline_ops_lib_path)
    else:
        mx_rec_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
        so_path = os.path.join(mx_rec_dir, 'mx_rec/libasc/libasc_ops.so')
        logging.debug(f"Using the DEFAULT PATH '{so_path}' to get ops lib.")
        return tf.load_op_library(so_path)

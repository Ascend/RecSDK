# coding: UTF-8
import logging
import math
import tensorflow as tf

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from mx_rec.util.initialize import get_rank_size


class Config:
    def __init__(self, mode="simple", task_name="default"):
        self.task_name = task_name
        if mode == "simple":
            self.generate_simple_config()
        else:
            self.generate_large_scale_config()

    def generate_simple_config(self):
        self.batch_number = 8192
        self.batch_size = 4096

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.float32

        self.item_range = 80000
        self.user_range = 200000
        self.category_range = 5000
        self.item_feat_cnt = 16
        self.user_feat_cnt = 8
        self.category_feat_cnt = 3
        self.access_threshold = 100
        self.eviction_threshold = 60

        rank_size = get_rank_size()
        coefficient = 1.1
        if rank_size != 0:
            max_ui_send_cnt = max(self.item_feat_cnt, self.user_feat_cnt)
            max_ui_range = max(self.item_range, self.user_range)
            self.item_send_cnt = min(int(self.batch_size * self.item_feat_cnt * coefficient),
                                     math.ceil(self.item_range / rank_size))
            self.item_vocab_size = max(self.item_send_cnt * rank_size * rank_size, self.item_range)
            self.user_send_cnt = min(int(self.batch_size * max_ui_send_cnt * coefficient),
                                     math.ceil(max_ui_range / rank_size))
            self.user_vocab_size = max(self.user_send_cnt * rank_size * rank_size, self.user_range)
            self.category_send_cnt = min(int(self.batch_size * self.category_feat_cnt * coefficient),
                                         math.ceil(self.category_range / rank_size))
        else:
            raise ZeroDivisionError("rank size must be an integer which is greater value zero.")

        self.user_hashtable_dim = 32
        self.user_hashtable_threshold = 1
        self.item_hashtable_dim = 8
        self.item_hashtable_threshold = 1

        self.learning_rate = 0.01

    def generate_large_scale_config(self):
        self.lookup_count = 40
        self.tensor_name_list = ["sparse_tensor_%d" % i for i in range(self.lookup_count)]
        self.hashtable_name_list = ["hashtable_%d" % i for i in range(self.lookup_count)]
        self.batch_size = 9600

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.float32

        self.vocabulary_size = 500000
        self.feat_cnt = 1

        rank_size = get_rank_size()
        coefficient = 1.1
        if rank_size != 0:
            self.send_cnt = min(int(self.batch_size * self.feat_cnt * coefficient),
                                math.ceil(self.vocabulary_size / rank_size))
        else:
            raise ZeroDivisionError("rank size must be an integer which is greater value zero.")

        self.hashtable_dim = 8
        self.learning_rate = 0.01


def sess_config(dump_data=False, dump_path="./dump_output", dump_steps="0|1|2"):
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                              log_device_placement=False)

    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = False
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = 1
    custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes("level0:pairwise;level1:pairwise")
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = 1
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_impl_mode.ini")

    if dump_data:
        """
            To see the details, please refer to the descriptions at official web site
        """
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_steps)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    return session_config

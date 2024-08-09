# coding: UTF-8
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
import math
import os

import tensorflow as tf
from mx_rec.util.communication.hccl_ops import get_rank_size
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from utils import (LOCAL_RANK_ID, PRECISION_CHECK_PATH, PRECISION_DUMP_STEP,
                   RANK_ZERO, PrecisionDumpInfo)

GLOBAL_RANDOM_SEED = 128
try:
    USE_DYNAMIC = bool(int(os.getenv("USE_DYNAMIC", 0)))
    USE_DYNAMIC_EXPANSION = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
    USE_MULTI_LOOKUP = bool(int(os.getenv("USE_MULTI_LOOKUP", 1)))
    MODIFY_GRAPH_FLAG = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
    USE_TIMESTAMP = bool(int(os.getenv("USE_TIMESTAMP", 0)))
    USE_ONE_SHOT = bool(int(os.getenv("USE_ONE_SHOT", 0)))
    USE_DETERMINISTIC = bool(int(os.getenv("USE_DETERMINISTIC", 0)))
    MULTI_LOOKUP_TIMES = int(os.getenv("MULTI_LOOKUP_TIMES", 2))
    PRECISION_CHECK = bool(int(os.getenv("PRECISION_CHECK", 0)))
    USE_DP = bool(int(os.getenv("USE_DP", 0)))
except ValueError as err:
    raise ValueError("please correctly config USE_DYNAMIC or USE_DYNAMIC_EXPANSION or "
                     "USE_MULTI_LOOKUP or USE_MODIFY_GRAPH or USE_TIMESTAMP or USE_ONE_SHOT or USE_DETERMINISTIC"
                     "or USE_DP only 0 or 1 is supported.") from err


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

        self.item_range = 80000 * get_rank_size() if not USE_DP else 80000
        self.user_range = 200000 * get_rank_size() if not USE_DP else 200000
        self.category_range = 5000 * get_rank_size() if not USE_DP else 5000
        self.item_feat_cnt = 16
        self.user_feat_cnt = 8
        self.category_feat_cnt = 3
        self.access_threshold = 2
        self.eviction_threshold = 2

        rank_size = get_rank_size()
        coefficient = 1.1
        if rank_size != 0:
            max_ui_send_cnt = max(self.item_feat_cnt, self.user_feat_cnt)
            max_ui_range = max(self.item_range, self.user_range)
            self.item_send_cnt = min(int(self.batch_size * self.item_feat_cnt * coefficient),
                                     math.ceil(self.item_range / rank_size)) if not USE_DP else self.item_range
            self.item_vocab_size = max(self.item_send_cnt * rank_size * rank_size, self.item_range) if not USE_DP \
                else max(self.item_send_cnt * rank_size, self.item_range)
            self.user_send_cnt = min(int(self.batch_size * max_ui_send_cnt * coefficient),
                                     math.ceil(max_ui_range / rank_size)) if not USE_DP else self.user_range
            self.user_vocab_size = max(self.user_send_cnt * rank_size * rank_size, self.user_range) if not USE_DP \
                else max(self.user_send_cnt * rank_size, self.user_range)
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


def construct_basic_config(npu_custom_op):
    npu_custom_op.parameter_map["mix_compile_mode"].b = False
    npu_custom_op.parameter_map["use_off_line"].b = True
    npu_custom_op.parameter_map["min_group_size"].b = 1
    npu_custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes("level0:pairwise;level1:pairwise")
    npu_custom_op.parameter_map["enable_data_pre_proc"].b = True
    npu_custom_op.parameter_map["iterations_per_loop"].i = 1
    npu_custom_op.parameter_map["hcom_parallel"].b = False
    npu_custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_impl_mode.ini")
    npu_custom_op.parameter_map["op_execute_timeout"].i = 2000
    npu_custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")


def construct_deterministic_config(npu_custom_op):
    npu_custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
    npu_custom_op.parameter_map["deterministic"].i = 1


def construct_op_dump_config(npu_custom_op):
    npu_custom_op.parameter_map["enable_dump"].b = True

    dump_path = os.path.join(PRECISION_CHECK_PATH, "04dump_op")

    if not os.path.exists(dump_path):
        os.makedirs(dump_path, mode=0o750, exist_ok=True)

    dump_step = "|".join([str(step_num - 1) for step_num in PRECISION_DUMP_STEP])
    dump_mode = "all"

    npu_custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
    npu_custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_step)
    npu_custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes(dump_mode)

    table_list = ["user_table", "item_table"]

    dump_op_info = {"table_list": table_list,
                    "dump_mode": dump_mode}
    PrecisionDumpInfo.add_item("dump_op_info", dump_op_info)

    dump_op_list = []

    dump_emb_op_info = {}
    for table_name in table_list:
        look_up_by_id_offset = [f"{table_name}//{table_name}_lookup/gather_for_id_offsets"]

        update_by_id_offset = [f"LazyAdam_0/update_{table_name}/ScatterNdAdd_2",
                               f"LazyAdam_0/update_{table_name}/ScatterNdAdd",
                               f"LazyAdam_0/update_{table_name}/ScatterNdAdd_1"]
        lookup_table_byaddress = [f"{table_name}//{table_name}_lookup/EmbeddingLookupByAddress"]
        update_grad_byaddress = [f"LazyAdamByAddress_0/update_{table_name}//{table_name}_lookup/"
                                 f"id_offsets/{table_name}/GetNext/EmbeddingUpdateByAddress"]

        if USE_DYNAMIC_EXPANSION:
            table_dump_op_name_list = lookup_table_byaddress + update_grad_byaddress
            dump_emb_op_info[table_name] = {"lookup_table": lookup_table_byaddress,
                                            "update_grad":update_grad_byaddress}
        else:
            table_dump_op_name_list = look_up_by_id_offset + update_by_id_offset
            dump_emb_op_info[table_name] = {"lookup_table": look_up_by_id_offset,
                                            "update_grad":update_by_id_offset}

        dump_op_list.extend(table_dump_op_name_list)

    dump_ops_string = " ".join(dump_op_list)
    PrecisionDumpInfo.add_item("dump_emb_op_info", dump_emb_op_info)
    npu_custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes(dump_ops_string)


def construct_npu_sess_config(dump_data=False):
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                              log_device_placement=False)

    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    construct_basic_config(custom_op)
    if USE_DETERMINISTIC:
        construct_deterministic_config(custom_op)
    if dump_data:
        construct_op_dump_config(custom_op)

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return session_config

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

import argparse
import logging
import os
import stat

import tensorflow as tf

from graph_partition import GraphPartitioner

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path", type=str, default="./")
    parser.add_argument("--tags_name", type=str, default="serve")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--output_filename", type=str, default="config.cfg")
    args = parser.parse_args()

    signature_def = "serving_default"

    # 模型配置
    embedding_lookup_op_type = ["Sum"]
    heavy_load_ops = ["MatMul"]  # 必须下沉的算子（暂时没用到）
    use_whole_graph = False
    partition_to_first_heavy_load = False
    #########################################################

    output_filepath = os.path.join(args.output_path, args.output_filename)
    output_filepath = os.path.realpath(output_filepath)
    model_path = os.path.realpath(args.model_path)

    with tf.compat.v1.Session() as sess:
        meta_graph = tf.compat.v1.saved_model.loader.load(
            sess, [args.tags_name], model_path
        )
        ops = sess.graph.get_operations()
        graph_partitioner = GraphPartitioner()

        graph_partitioner.graph = sess.graph
        graph_partitioner.signature_def = meta_graph.signature_def.get(signature_def)
        graph_partitioner.set_embedding_lookup_op_type(embedding_lookup_op_type)

        inputs, outputs = graph_partitioner.get_sub_graph()

    res_string = "[[" + inputs + "," + outputs + "]]"

    with os.fdopen(os.open("template.cfg", os.O_RDONLY)) as ori_cfg:
        template = ori_cfg.read()

    output = template.replace("#value@in_out_pair#", res_string)
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    # create and write new cfg
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
    with os.fdopen(os.open(output_filepath, flags, mode), "w") as file:
        file.write(output)
    logging.info("Generate %s success.", output_filepath)

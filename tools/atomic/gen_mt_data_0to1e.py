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

import logging
import os
import sys
import random
import numpy as np
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)

np.random.seed(0)

LINE_PER_SAMPLE = 10000
SAMPLES_NUM = 10000 * 800 
sparse_feat_list = ['feat_ids']
sparse_feat_len = [100]

NUM = 0

hot_zhanbi = sys.argv[1:][0]
hot_zhanbi = float(hot_zhanbi)/10
logging.info("hot_ratio = %f", hot_zhanbi)

tfpath = os.path.join("/home/insert/data", str(hot_zhanbi))
if not os.path.exists(tfpath):
    os.mkdir(tfpath)
    
tfpath = os.path.join("/home/insert/data", str(hot_zhanbi), "tf")

part1 = np.array(random.sample(range(0, 2), 1)) 


def write_records(writer, line_cnt, file_cnt):
    features = {
        'label': tf.train.Feature(
            float_list=tf.train.FloatList(value=np.random.randint(2, size=LINE_PER_SAMPLE).tolist()))
    }

    count = 0
    for _, sparse_feat in enumerate(sparse_feat_list):
        np.random.seed(count)
        # global num
        logging.info("===sparse=%s", sparse_feat)
        part2 = np.array(random.sample(
            range(0 + 100 * LINE_PER_SAMPLE * (10 * file_cnt + line_cnt), 
                  100 * LINE_PER_SAMPLE * (10 * file_cnt + line_cnt + 1)), 
            int(100 * LINE_PER_SAMPLE * (1 - hot_zhanbi))
        ))
        features[sparse_feat] = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=part1.astype(np.int64).tolist() * int(100 * LINE_PER_SAMPLE * hot_zhanbi) + 
                      part2.astype(np.int64).tolist()
            )
        )

        count += 1
    features = tf.train.Features(feature=features)
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())


def gen_tfrecords(tf_path):
    file_cnt = 0
    line_per_file = 10
    line_cnt = 0
    writer = tf.python_io.TFRecordWriter(f"{tf_path}_{file_cnt}.tfrecord")
    sample_cnt = 0
    while True:
        write_records(writer, line_cnt, file_cnt)
        line_cnt += 1
        sample_cnt += LINE_PER_SAMPLE
        logging.info(f">>>>>>>>>>>>count {sample_cnt} end.")
        if sample_cnt == SAMPLES_NUM:
            break
        if line_cnt == line_per_file:
            file_cnt += 1
            line_cnt = 0
            writer.close()
            writer = tf.python_io.TFRecordWriter(f"{tf_path}_{file_cnt}.tfrecord")
    writer.close()


if __name__ == '__main__':
    gen_tfrecords(tf_path=tfpath)

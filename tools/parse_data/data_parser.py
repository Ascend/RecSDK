#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
# Description:
# Author: MindX SDK
# Create: 2023-06-28

# -----------------------------------------ReadMe  Begin--------------------------------------------
# 1. 功能描述
# 本工具用于单测tensorflow数据解析阶段耗时，便于分析数据解析阶段是不是整个pipeline的瓶颈？堵塞了pipeline的流畅运行？
# 2. 注意事项
# 数据解析逻辑主要包含在make_dataset函数中，本函数缺省使用criteo数据集。如果需要测试其他数据集的解析耗时，可根据需要重新定义make_dataset；
# 3. 绑核
# 为了模拟真实场景，bind_cpu默认模拟了80核cpu、8worker平均分配核；如果worker数目不同、真实cpu核数不同，可根据需要重新定义bind_cpu函数；
# 4. 启动执行
# 4.1 单worker执行： python3 data_parser.py
# 4.2 多worker执行： bash run.sh data_parser.py
# -----------------------------------------ReadMe  End--------------------------------------------

import os
import sys
import time

import logging
import psutil

import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)


def make_dataset(data_path, batch_size=102400, line_per_sample=1024):
    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.FixedLenFeature(shape=(line_per_sample,), dtype=tf.int64),
            'sparse_feature': tf.FixedLenFeature(shape=(26 * line_per_sample,), dtype=tf.int64),
            'dense_feature': tf.FixedLenFeature(shape=(13 * line_per_sample,), dtype=tf.float32),
        }
        sample = tf.parse_single_example(data_record, features)
        return sample

    def feat_cast(feat):
        for name, tensor in feat.items():
            if tensor.dtype == tf.int64:
                feat[name] = tf.cast(tensor, tf.int32)
        return feat

    def reshape_fn(batch):
        batch['label'] = tf.reshape(batch['label'], [-1, 1])
        batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 13])
        batch['dense_feature'] = tf.math.log(batch['dense_feature'] + 3.0)
        batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 26])
        return batch

    file_list = sorted([os.path.join(data_path, file) for file in os.listdir(data_path)])
    dataset = tf.data.TFRecordDataset(file_list, num_parallel_reads=4)

    num_parallel = 8
    dataset = dataset.map(extract_fn, num_parallel_calls=num_parallel)

    line_cnt = batch_size // line_per_sample
    dataset = dataset.batch(line_cnt, drop_remainder=True)

    dataset = dataset.map(feat_cast, num_parallel_calls=num_parallel)
    dataset = dataset.map(reshape_fn, num_parallel_calls=num_parallel)

    dataset = dataset.prefetch(10)
    return dataset


def bind_cpu(rank_id):
    process = psutil.Process()
    cpu_kernels = {
        0: 0,
        1: 10,
        2: 40,
        3: 50,
        4: 20,
        5: 30,
        6: 60,
        7: 70
    }
    try:
        process.cpu_affinity([cpu_kernels.get(rank_id) + x for x in range(10)])
    except IndexError:
        logging.error("error cpu bind info, skipped.")


if __name__ == '__main__':
    RANK_ID = 0
    if (len(sys.argv) > 1):
        RANK_ID = int(sys.argv[1])
    bind_cpu(RANK_ID)

    DATA_PATH = "/media/mxRec/data/criteo_tfrecord_small/train"
    train_dataset = make_dataset(DATA_PATH)
    iterator = train_dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    input_data = []
    for example in next_batch:
        input_data.append(next_batch[example])

    COUNT = 0
    TOTAL_TIME = 0.0

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                start_time = time.time()
                result = sess.run(input_data[0])
                end_time = time.time()

                COUNT += 1

                if COUNT > 1:
                    TOTAL_TIME += end_time - start_time
                logging.info("StepId:%d, StepTimeCost(ms):%f", COUNT, (end_time - start_time))
            except tf.errors.OutOfRangeError as e:
                logging.error("End of Training Dataset")
                break
    logging.info("StepTimeCost avg(ms):%f", TOTAL_TIME / (COUNT - 1))
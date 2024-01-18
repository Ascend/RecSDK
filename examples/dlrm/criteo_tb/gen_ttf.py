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

import os
import pickle
import collections
import logging
import argparse
from multiprocessing import Process
import numpy as np
import time
from tqdm import tqdm
from glob import glob
from collections import Counter, OrderedDict
import sys

import tensorflow as tf


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info',
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = logging.FileHandler(filename=filename)  # 往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)

    def info(self, *args):
        if len(args) == 1:
            self.logger.info(*args)
        else:
            self.logger.info([*args])


class CriteoStatsDict():
    def __init__(self):
        self.field_size = 39  # value_1-13;  cat_1-26;
        self.val_cols = ["val_{}".format(i + 1) for i in range(13)]
        self.cat_cols = ["cat_{}".format(i + 1) for i in range(26)]

        #
        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}
        self.global_idx_range_dict = {col: 0 for col in self.cat_cols}  # cat_0: v_0, cat_1: v_1, ...
        self.hist_map = {col: set() for col in self.cat_cols}
        #
        self.hash_bucket = 40000000
        self.dense_bias = 1
        self.hush_bucket_offset = [self.hash_bucket * i for i in range(26)]  # voc_size=1040000000
        self.slot_size_array = [39884407, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532952,
                                2953546, 403346, 10, 2208, 11938, 155, 4, 976, 14, 39979772, 25641295,
                                39664985, 585935, 12972, 108, 36]
        self.offset_size_list = np.cumsum([0] + self.slot_size_array[:-1])


    def stats_cats(self, cat_list):
        def map_cat_count(i, cat):
            capped_value = int(cat, 16) % self.hash_bucket if cat else self.hash_bucket

            key_col = self.cat_cols[i]
            if capped_value not in self.hist_map[key_col]:
                self.hist_map[key_col].add(capped_value)

        for i, cat in enumerate(cat_list):
            map_cat_count(i, cat)

    #
    def save_dict(self, output_path, hist_map, prefix=""):
        with open(os.path.join(output_path, "{}hist_map.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(hist_map, file_wrt)

    #
    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}hist_map.pkl".format(prefix)), "rb") as file_wrt:
            self.hist_map = pickle.load(file_wrt)


    def map_cat2id(self, denses, cats):
        # dense-features
        # dense missing value, the minimum in dense-feature is 1.0
        dense_list = [int(d) + self.dense_bias if d else self.dense_bias for d in denses]


        # cat-features
        # process missing value
        cat_list = []
        def map_cat_count(i, cat):
            capped_value = int(cat, 16) % self.hash_bucket if cat else self.hash_bucket

            key_col = self.cat_cols[i]
            if capped_value in self.hist_map[key_col]:
                cat_list.append(self.hist_map[key_col][capped_value])
            else:
                print(f"error: {key_col}, {cat}, {capped_value}")


        for i, cat in enumerate(cats):
            map_cat_count(i, cat)
        # cat preprocess: mod and offset
        cat_list = [0 if cat < 0 else cat % (self.slot_size_array[idx] + 1) for idx, cat in enumerate(cat_list)]
        cat_list = [cat + offset for cat, offset in zip(cat_list, self.offset_size_list)]


        return dense_list, cat_list

def statsdata_multiprocess(process_num, process_id, data_file_path, output_path, criteo_stats):
    start_time = time.time()
    with open(data_file_path, encoding="utf-8") as file_in:
        errorline_list = []
        count = 0
        for i, line in enumerate(file_in):
            if i % process_num != process_id:
                continue
            count += 1
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != 40:
                errorline_list.append(count)
                print("line: {}".format(line))
                continue
            if count % 1000000 == 0:
                print("Have handle {}w lines.".format(count // 10000))
            cats = items[14:]
            criteo_stats.stats_cats(cats)
    criteo_stats.save_dict(output_path)
    print('statsdata time cost: {:.2f}s'.format(time.time() - start_time))


def get_unique_id_multiprocess(process_num, process_id, data_file_path, output_path, criteo_stats):
    if os.path.exists(os.path.join(output_path, "unique_id.pkl")):
        return
    start_time = time.time()
    cat_sets = [OrderedDict() for col in criteo_stats.cat_cols]
    cat_global_id_nums = [0 for col in criteo_stats.cat_cols]
    hash_bucket = criteo_stats.hash_bucket
    line_num = 0
    with open(data_file_path, encoding="utf-8") as file_in:
        errorline_list = []

        for i, line in enumerate(file_in):
            line_num += 1
    start_line = process_id * ((line_num + process_num) // process_num)
    end_line = (process_id + 1) * ((line_num + process_num) // process_num)
    with open(data_file_path, encoding="utf-8") as file_in:
        errorline_list = []
        count = 0
        for i, line in enumerate(file_in):
            if i < start_line or i >= end_line:
                continue
            count += 1
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != 40:
                errorline_list.append(count)
                print("line: {}".format(line))
                continue
            if count % 10000 == 0:
                print("Have handle {}w lines.".format(count // 10000))
                sys.stdout.flush()
            cats = items[14:]
            # criteo_stats.stats_cats(cats)
            # def map_cat_count(i, cat):
            for k, cat in enumerate(cats):
                # map_cat_count(i, cat)
                capped_value = int(cat, 16) % hash_bucket if cat else hash_bucket
                # if capped_value not in self.hist_map[key_col]:
                if capped_value not in cat_sets:
                    cat_sets[k][capped_value] = cat_global_id_nums[k]
                    cat_global_id_nums[k] += 1
    with open(os.path.join(output_path, "unique_id.pkl"), "wb") as file_wrt:
        pickle.dump(cat_sets, file_wrt)
    print('statsdata time cost: {:.2f}s'.format(time.time() - start_time))


def merge_stats_count(stats_dir, criteo_stats):
    if os.path.exists(f'{stats_dir}/hist_map.pkl'):
        return
    stats_sub_dirs = sorted(glob(f'{stats_dir}/*[0-9]'))
    with open(f'{stats_sub_dirs[0]}/unique_id.pkl', 'rb') as f:
        all_hist_map = pickle.load(f)

    for i in tqdm(range(1, len(stats_sub_dirs))):
        with open(f'{stats_sub_dirs[i]}/unique_id.pkl', 'rb') as f:
            others_count = pickle.load(f)
        for k, _ in enumerate(criteo_stats.cat_cols):
            all_count_1, others_count_1 = all_hist_map[k], others_count[k]
            all_count_1.update(others_count_1)
            all_hist_map[k] = all_count_1
    hist_map = {}
    for i, col in enumerate(criteo_stats.cat_cols):
        hist_map[col] = dict(zip(list(all_hist_map[i].keys()), range(len(all_hist_map[i]))))

    criteo_stats.save_dict(stats_dir, hist_map)


def mkdir_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def make_example(label_list, dense_feat_list, sparse_feat_list):
    dense_feature = np.array(dense_feat_list, dtype=np.float32).reshape(-1)
    sparse_feature = np.array(sparse_feat_list, dtype=np.int64).reshape(-1)
    label = np.array(label_list, dtype=np.int64).reshape(-1)
    feature_dict = {"dense_feature": tf.train.Feature(float_list=tf.train.FloatList(value=dense_feature)),
                "sparse_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=sparse_feature)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example

def convert_input2tfrd_multiprocess(process_num, process_id, in_file_path, output_path, criteo_stats, line_per_sample=1024,
                            part_rows=2000000, mode="train_"):
    start_time = time.time()
    print("----------" * 10 + "\n" * 2)

    part_number = 0
    file_name = output_path + "part_{:0>8d}.tfrecord"

    file_writer = tf.python_io.TFRecordWriter(file_name.format(part_number))
    sample_count = 0
    part_count = 0
    line_num = 0
    with open(in_file_path, encoding="utf-8") as file_in:
        errorline_list = []

        for i, line in tqdm(enumerate(file_in)):
            line_num += 1
    print(f'line_num: {line_num}')
    start_line = process_id * ((line_num + process_num) // process_num)
    end_line = (process_id + 1) * ((line_num + process_num) // process_num)
    dense_res_list = []
    cat_res_list = []
    label_res_list = []
    with open(in_file_path, encoding="utf-8") as file_in:
        total_count = 0
        part_number = 0
        for i, line in enumerate(file_in):
            if i < start_line or i >= end_line:
                continue

            total_count += 1
            if total_count % 10000 == 0:
                print("Have handle {}w tfrecords.".format(total_count // 10000))
                sys.stdout.flush()
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != 40:
                continue
            label = int(items[0])
            values = items[1:14]
            cats = items[14:]
            assert len(values) == 13, "values.size： {}".format(len(values))
            assert len(cats) == 26, "cats.size： {}".format(len(cats))
            val_list, cat_list = criteo_stats.map_cat2id(values, cats)
            dense_res_list.append(val_list)
            cat_res_list.append(cat_list)
            label_res_list.append(label)
            sample_count += 1
            if sample_count % line_per_sample == 0 and sample_count > 0:
                ex = make_example(label_res_list, dense_res_list, cat_res_list)
                serialized = ex.SerializeToString()
                file_writer.write(serialized)
                part_count += line_per_sample
                sample_count = 0
                dense_res_list = []
                cat_res_list = []
                label_res_list = []
                if part_count >= part_rows:
                    part_number += 1
                    file_writer.close()
                    file_writer = tf.python_io.TFRecordWriter(file_name.format(part_number))
                    part_count = 0

        if sample_count > 0:
            file_writer.close()
            part_number += 1

    print('convert_input2tfrd time cost: {:.2f}s'.format(time.time() - start_time))
    return part_number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get and Process datasets')
    parser.add_argument('--train_data_dir', default="train",
                        help='day_0, ..., day_22 file path')
    parser.add_argument('--test_data_dir', default="test",
                        help='day_23 file path')
    parser.add_argument('--tf_base_dir', default="tf_base_dir",
                        help='tfrecord saved base path. Disk occupation better over 720G.')
    parser.add_argument('--stats_process_num', default=72, type=int,
                        help='prcoess num of stats')
    parser.add_argument('--train_process_num', default=69, type=int,
                        help='prcoess num of train tfrecord generation')
    parser.add_argument('--test_process_num', default=24, type=int,
                        help='prcoess num of test tfrecord generation')
    args, _ = parser.parse_known_args()
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    criteo_stats = CriteoStatsDict()

    base_path = "./"
    train_data_files = sorted(glob(f'{train_data_dir}/*'))
    test_data_files = sorted(glob(f'{test_data_dir}/*'))
    data_files = train_data_files + test_data_files
    print("train data files: ", train_data_files)
    print("test data files: ", test_data_files)
    print("data files: ", data_files)
    process_num = args.stats_process_num
    if True:
        processs = []
        for process_id in range(process_num):
            sub_process_num = process_num // len(data_files)
            data_file = data_files[process_id//sub_process_num]
            stats_output_path = base_path + f"/stats_dict_mp/{process_id:02}/"
            mkdir_path(stats_output_path)
            p = Process(target=get_unique_id_multiprocess, args=(
                sub_process_num, process_id % sub_process_num, data_file, stats_output_path, criteo_stats))
            processs.append(p)
        for p in processs:
            p.start()
        for p in processs:
            p.join()
        merge_stats_count(base_path + f"/stats_dict_mp/", criteo_stats)

    print("----------" * 10)
    stats_output_path = base_path + f"/stats_dict_mp/"
    criteo_stats.load_dict(dict_path=stats_output_path, prefix="")

    spe_num = 1024
    tf_base_dir = args.tf_base_dir

    # gen_train tfrecords
    dataset_mode = "train"
    save_tfrecord_path = os.path.join(tf_base_dir, "tfrecord", dataset_mode)
    mkdir_path(save_tfrecord_path)
    processs = []
    process_num = args.train_process_num
    assert process_num % len(train_data_files) == 0, print(
        f'process_num {process_num} must exact div length of data_files {len(data_files)}')

    for process_id in range(process_num):
        sub_process_num = process_num // len(train_data_files)
        data_file = train_data_files[process_id // sub_process_num]
        output_path = f'{save_tfrecord_path}/{process_id:04}_'
        p = Process(target=convert_input2tfrd_multiprocess, args=(sub_process_num, process_id%sub_process_num, data_file, output_path,
                                                     criteo_stats, spe_num,
                                                     5000000))
        processs.append(p)
    for p in processs:
        p.start()
    for p in processs:
        p.join()

    # gen_test tfrecords
    dataset_mode = "test"
    save_tfrecord_path = os.path.join(tf_base_dir, "tfrecord", dataset_mode)
    mkdir_path(save_tfrecord_path)
    processs = []
    process_num = args.test_process_num
    assert process_num % len(test_data_files) == 0, print(
        f'process_num {process_num} must exact div length of data_files {len(data_files)}')

    for process_id in range(process_num):
        sub_process_num = process_num // len(test_data_files)
        data_file = test_data_files[process_id // sub_process_num]
        output_path = f'{save_tfrecord_path}/{process_id:04}_'
        p = Process(target=convert_input2tfrd_multiprocess, args=(sub_process_num, process_id%sub_process_num, data_file, output_path,
                                                     criteo_stats, spe_num,
                                                     5000000))

        processs.append(p)
    for p in processs:
        p.start()
    for p in processs:
        p.join()
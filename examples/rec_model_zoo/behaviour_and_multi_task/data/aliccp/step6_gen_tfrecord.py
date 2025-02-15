import math
import os
import glob
import stat
import json
import argparse
from multiprocessing import Pool

import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length


def chunkify_file(filepath, output_file_path, chunk_size):
    chunks = []
    file_end = os.path.getsize(filepath)
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    fd = os.open(filepath, flags, modes)
    with os.fdopen(fd, "rb") as f:
        chunk_end = f.tell()
        count = 0
        while True:
            chunk_start = chunk_end
            f.seek(f.tell() + chunk_size * 1024 * 1024, os.SEEK_SET)
            f.readline()  # make this chunk line aligned
            chunk_end = min(f.tell(), file_end)
            chunks.append((chunk_start, chunk_end - chunk_start, filepath, output_file_path,
                           filepath.replace(".csv", "_max_length.json"), count))
            count += 1
            if chunk_end >= file_end:
                break
    return chunks


def gen_tfrecords(chunk_data):
    chunk_start, chunk_size, input_file_path, output_file_path, max_length_file_path, task_index = chunk_data[0], \
    chunk_data[1], chunk_data[2], chunk_data[3], chunk_data[4], chunk_data[5]
    out_file = output_file_path + ".{:05d}".format(task_index)
    fields = ["101", "109_14", "110_14", "127_14", "150_14", "121", "122", "124", "125", "126", "127", "128", "129",
              "205", "206", "207", "210", "216", "508", "509", "702", "853", "301"]
    multi_hot_fields = set(["109_14", "110_14", "127_14", "150_14", "210", "853"])
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    max_length_file_info = json.load(os.fdopen(os.open(max_length_file_path, flags, modes), "r"))

    flags = os.O_RDONLY
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    tfrecord_out = tf.io.TFRecordWriter(out_file)
    with os.fdopen(os.open(input_file_path, flags, modes), "rb") as fi:
        fi.seek(chunk_start)
        chunk = fi.read(chunk_size)
        if chunk[-1] != 10:
            raise ValueError(
                f"ends not with \\n, but ascii code {chunk[-1]} {chunk_start}, {chunk_size}, "
                f"{input_file_path}, {output_file_path}, {task_index}")
        lines = str(chunk, encoding="utf-8").strip().split("\n")
    for line in lines:
        cells = line.strip().split(',')
        # index, y, z, ids
        _, y, z = cells[:3]
        field_values = cells[3:]
        if len(field_values) != len(fields):
            raise(f"{line} values not enough, {chunk_start}, {chunk_size}, "
                      f"{input_file_path}, {output_file_path}, {task_index}")

        feature = {
            "y": tf.train.Feature(float_list=tf.train.FloatList(value=[float(y)])),
            "z": tf.train.Feature(float_list=tf.train.FloatList(value=[float(z)]))
        }

        one_hot_value_list = []
        for index, field in enumerate(fields):
            field_value_string = field_values[index]
            if field not in multi_hot_fields:
                if "#" in field_value_string:
                    raise ValueError(f"{field} contains multi hot values, {chunk_start}, {chunk_size}, "
                                     f"{input_file_path}, {output_file_path}, {task_index}")
                one_hot_value_list.append(int(field_value_string))
            else:
                field_value = list(map(lambda s: int(s), field_value_string.split("#")))
                if args.padding:
                    feature.update({field: tf.train.Feature(int64_list=tf.train.Int64List(
                        value=np.array(field_value + [-1] * (max_length_file_info[field] - len(field_value)),
                                       dtype=np.int64)))})
                else:
                    feature.update({field: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.array(field_value, dtype=np.int64)))})
        feature.update({"one_hot_fields": tf.train.Feature(
            int64_list=tf.train.Int64List(value=np.array(one_hot_value_list, dtype=np.int64)))})

        # serialized to Example
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        tfrecord_out.write(serialized)
    tfrecord_out.close()


def gen_tfrecords_chunk(chunk_data):
    gen_tfrecords(chunk_data)


if not os.path.exists("./aliccp_out"):
    os.mkdir("./aliccp_out")
file_list = glob.glob(os.path.join(".", "data_*.csv"))
tasks = []
for file in file_list:
    part = os.path.basename(file).split("_")[1].split(".")[0]
    output_folder = os.path.join("./aliccp_out", part)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(file) + ".tfrecord")
    tasks += chunkify_file(file, output_path, 400)
with Pool(processes=args.proc) as pool:
    result = list(pool.imap(gen_tfrecords_chunk, tasks))

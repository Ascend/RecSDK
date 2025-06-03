#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
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
import glob
import stat
import json
import argparse
from multiprocessing import Pool

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length

CHUNK_SIZE = 400  # 读取偏移M,400*1024*1024。


class TorchDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()


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


def gen_torch_dataset(chunk_data):
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
    with os.fdopen(os.open(input_file_path, flags, modes), "rb") as fi:
        fi.seek(chunk_start)
        chunk = fi.read(chunk_size)
        if chunk[-1] != ord('\n'):
            raise ValueError(
                f"ends not with \\n, but ascii code {chunk[-1]} {chunk_start}, {chunk_size}, "
                f"{input_file_path}, {output_file_path}, {task_index}")
        lines = str(chunk, encoding="utf-8").strip().split("\n")

    samples = []
    for line in lines:
        cells = line.strip().split(',')
        # index, y, z, ids
        _, y, z = cells[:3]
        field_values = cells[3:]
        if len(field_values) != len(fields):
            raise ValueError(f"{line} values not enough, {chunk_start}, {chunk_size}, "
                      f"{input_file_path}, {output_file_path}, {task_index}")

        feature = {
            "y": torch.tensor([y], dtype=torch.float32),
            "z": torch.tensor([z], dtype=torch.float32)
        }

        for index, field in enumerate(fields):
            field_value_string = field_values[index]
            if field not in multi_hot_fields:
                if "#" in field_value_string:
                    raise ValueError(f"{field} contains multi hot values, {chunk_start}, {chunk_size}, "
                                     f"{input_file_path}, {output_file_path}, {task_index}")

                feature.update({field: torch.tensor(int(field_value_string), dtype=torch.int64)})
            else:
                field_value = list(map(lambda s: int(s), field_value_string.split("#")))
                if args.padding:
                    padded = field_values + [-1] * (max_length_file_info[field] - len(field_value))
                    feature.update({field: torch.tensor(padded, dtype=torch.int64)})

                else:
                    feature.update({field: torch.tensor(field_value, dtype=torch.int64)})

        samples.append(feature)
    with h5py.File(out_file, 'w') as hf:
        hf.create_dataset('y', data=torch.stack([s['y'] for s in samples]).numpy())
        hf.create_dataset('z', data=torch.stack([s['z'] for s in samples]).numpy())
        for field in fields:
            hf.create_dataset(field, data=torch.stack([s[field] for s in samples]).numpy())


def gen_torch_dataset_chunk(chunk_data):
    gen_torch_dataset(chunk_data)


if not os.path.exists("./aliccp_out"):
    os.mkdir("./aliccp_out")
file_list = glob.glob(os.path.join(".", "data_*.csv"))
tasks = []
for file in file_list:
    part = os.path.basename(file).split("_")[1].split(".")[0]
    output_folder = os.path.join("./aliccp_out", part)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(file) + ".pth")
    tasks += chunkify_file(file, output_path, CHUNK_SIZE)
with Pool(processes=args.proc) as pool:
    result = list(pool.imap(gen_torch_dataset_chunk, tasks))

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

import os
import stat
import argparse
import math

import numpy as np

parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length
np.random.seed(2024)


def iter_count(file_name):
    from itertools import takewhile, repeat
    buffer = 1024 * 1024
    flags_ = os.O_RDONLY
    modes_ = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    fd = os.open(file_name, flags_, modes_)
    with os.fdopen(fd, 'r') as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count("\n") for buf in buf_gen)


def random_true_false(length: int, num_of_true: int):
    return np.random.permutation(length) < num_of_true


if __name__ == "__main__":
    line_count = iter_count("./sample_skeleton_test_parsed.csv")
    random_arr = random_true_false(line_count, int(line_count * 0.5))

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH

    with os.fdopen(os.open("sample_skeleton_test_splitted_parsed.csv", flags, modes), 'w') as testfile:
        with  os.fdopen(os.open("sample_skeleton_val_splitted_parsed.csv", flags, modes), 'w') as valfile:
            flags = os.O_RDONLY
            modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            fd = os.open("sample_skeleton_test_parsed.csv", flags, modes)
            with os.fdopen(fd, 'r') as f:
                p = 0
                while True:
                    lines = f.readlines(int(1e7))
                    if len(lines) == 0:
                        break
                    for line in lines:
                        if random_arr[p]:
                            testfile.write(line)
                        else:
                            valfile.write(line)
                        p += 1

# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import os


def write_data(file_name, x, y, dup):
    clear_file(file_name)
    length = 200
    interval = 100000
    count = 0
    ids = []
    all_ids = []
    for j in range(0, int(x / length)):
        for i in range(0, length):
            ids.append(random.randrange(0 + interval * j, interval * (j + 1)))
        if len(ids) == 200:
            for k in range(0, dup):
                count = int(count) + len(ids)
                for val in ids:
                    all_ids.append(val)
            ids = []

    if int(len(ids)) > 0:
        for val in ids:
            all_ids.append(val)

    ids = []
    my_set = set()
    for j in range(0, int(y / length)):
        for i in range(0, length):
            ids.append(random.randrange(0 + interval * (int(x / length) + j), interval * (int(x / length) + j + 1)))
            count = count + 1
            all_ids.append(ids[i])
        if len(ids) == 200:
            ids = []
    if int(len(ids)) > 0:
        for val in ids:
            all_ids.append(val)

    random.shuffle(all_ids)

    ids = []
    for j in range(0, len(all_ids)):
        ids.append(all_ids[j])
        if len(ids) % 200 == 0:
            write_file(ids, file_name)
            ids = []

    write_file(ids, file_name)

    for val in all_ids:
        my_set.add(val)

    print("count: ", count, "all_ids len:", len(all_ids), " set size: ", len(my_set))


def main():
    # 300w id去重率20%
    # 6x + y =300
    # x + y = 60
    # x = 48 y =12
    write_data('data20.txt', 48*10000, 12*10000, 6)

    # 300w id去重率30%
    # 6x + y =300
    # x + y = 90
    # x = 42 y =48
    write_data('data30.txt', 42*10000, 48*10000, 6)

    # 300w id去重率40%
    # 6x + y =300
    # x + y = 120
    # x = 36 y =84
    write_data('data40.txt', 36*10000, 84*10000, 6)


def write_file(ids, file_name):
    w = ""
    for id in ids:
        w += str(id) + ", "
    f = open(file_name, 'a')
    f.write(w + "\n")
    f.close()


def clear_file(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r+") as f:
            f.truncate(0)


if __name__ == '__main__':
    main()

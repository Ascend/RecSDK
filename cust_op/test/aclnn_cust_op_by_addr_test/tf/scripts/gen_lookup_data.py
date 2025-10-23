#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

import numpy as np


def gen_golden_data_simple():
    emb_size = 100000
    addr_num = 21632
    dim = 8

    offset = np.random.randint(0, emb_size, size=addr_num).astype(np.int64)
    emb = np.array([i * 0.01 for i in range(emb_size * dim)]).astype(np.float32)
    offset.tofile("./input/input_lookup_x.bin")
    emb.tofile("./input/input_lookup_emb.bin")
    for i in range(addr_num):
        start = offset[i]
        if i == 0:
            golden = emb[start * dim:start * dim + dim]
        else:
            golden = np.concatenate((golden, emb[start * dim:start * dim + dim]), axis=0)

    golden = golden.astype(np.float32)
    golden.tofile("./output/lookup_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

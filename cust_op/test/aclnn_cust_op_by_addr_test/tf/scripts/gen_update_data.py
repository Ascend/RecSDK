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
    # embedding_update_by_address算子,grad搬运需要是不同随机小数，否则在算子侧搬运grad数据时，无法准确验证搬运的grad是否正确
    grad = np.random.uniform(1, 10, size=[addr_num * dim]).astype(np.float32)

    offset.tofile("./input/input_update_x.bin")
    grad.tofile("./input/input_update_y.bin")
    emb.tofile("./input/input_update_emb.bin")
    for i in range(addr_num):
        start = offset[i]
        emb[start * dim:start * dim + dim] += grad[i * dim:i * dim + dim]

    golden = emb.astype(np.float32)
    golden.tofile("./output/update_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
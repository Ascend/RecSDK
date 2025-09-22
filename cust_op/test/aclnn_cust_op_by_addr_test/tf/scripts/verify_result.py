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

import os
import logging
import sys

import numpy as np
# 1e-6为了保证更大批次数据，数据准确
loss = 1e-6
minimum = 10e-10
logging.basicConfig(level=logging.INFO)


def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float32)
    golden = np.fromfile(golden, dtype=np.float32)
    if real_result.shape != golden.shape:
        logging.error("shape error")
        sys.exit(-1)
    diff = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    result_atol = np.less_equal(diff, loss)  # 绝对误差大于loss的为False
    result_rtol = np.less_equal(diff / np.add(deno, minimum), loss)  # 相对误差大于loss的为False
    if not result_rtol.all() and not result_atol.all():
        if np.sum(~result_rtol) > real_result.size * loss and \
                np.sum(~result_atol) > real_result.size * loss:  # 误差允许为1000000个里出现一个
            logging.error("result error")
            sys.exit(-1)
    logging.info("test pass")
    return True


if __name__ == '__main__':
    try:
        verify_result(sys.argv[1], sys.argv[2])
    except Exception as e:
        logging.error(f"an error occurred during verification:{e}", exc_info=True)

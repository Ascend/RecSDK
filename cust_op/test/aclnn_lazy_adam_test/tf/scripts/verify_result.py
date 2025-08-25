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
import numpy as np

_LOSS_THRESHOLD = 1e-6  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过万分之一
_MINIMUM = 10e-10

logging.getLogger().setLevel(logging.INFO)


def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float32)  # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.float32)  # 从bin文件读取预期运算结果
    result = np.abs(real_result - golden)  # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, _LOSS_THRESHOLD)  # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, _MINIMUM), _LOSS_THRESHOLD)  # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        # 误差超出预期时返回打印错误，返回对比失败
        if np.sum(result_rtol == False) > real_result.size * _LOSS_THRESHOLD \
                and np.sum(result_atol == False) > real_result.size * _LOSS_THRESHOLD:
            logging.error("[ERROR] output verify result error.")
            return False
    logging.info("output verify pass.")
    return True


if __name__ == '__main__':
    logging.info("start verify outputM.")
    verify_result("output/outputM.bin", "output/goldenOutputM.bin")
    logging.info("start verify outputV.")
    verify_result("output/outputV.bin", "output/goldenOutputV.bin")
    logging.info("start verify outputVar.")
    verify_result("output/outputVar.bin", "output/goldenOutputVar.bin")

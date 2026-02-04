#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import time
import logging

import torch
import pandas as pd

import dynamic_emb_extensions as demb


logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


def main():
    cur_device = 0
    torch.npu.set_device(cur_device)

    # warmup
    logging.info("warmup........")
    warmup_iter_num = 10
    for _ in range(warmup_iter_num):
        demb.device_timestamp()
        torch.npu.synchronize()

    # test
    logging.info("test........")
    repeats = 10000
    start_time = time.time()
    for _ in range(repeats):
        demb.device_timestamp()
        torch.npu.synchronize()
    end_time = time.time()
    time_cost = (end_time - start_time) * 1000000 / repeats

    logging.info(f"call device_timestamp average time cost: {time_cost:.6f} us")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import sys
from datetime import datetime

import pytz


def setup_logging(rank):
    this_time = str(
        datetime.now(tz=pytz.timezone("PRC")).strftime(
            "%m_%d_%H_%M_%S",
        )
    )
    format_mess = logging.Formatter(
        fmt=f"[rank{rank}][%(levelname)s][%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    file_handler = logging.FileHandler(
        f"test_rank{rank}_{this_time}.log", encoding="utf-8"
    )
    file_handler.setFormatter(format_mess)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def setup_main_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="[MAIN][%(levelname)s][%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"test_main_{datetime.now(tz=pytz.timezone('PRC')).strftime('%m_%d_%H_%M_%S')}.log"),
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ],
        force=True
    )

#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd
import sys
import logging
import numpy as np

LOSS = 1e-3
MINIMUM = 10e-10

logging.getLogger().setLevel(logging.INFO)


def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float32)
    golden = np.fromfile(golden, dtype=np.float32)
    real_result = real_result[:golden.size]
    logging.info(real_result[:32])
    logging.info(golden[:32])
    result = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    result_atol = np.less_equal(result, LOSS)
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), LOSS)
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * LOSS and \
            np.sum(result_atol == False) > real_result.size * LOSS:
            logging.error("[ERROR] result error")
            return False
    logging.info("test pass")
    return True

if __name__ == '__main__':
    logging.info("=============================grad query============")
    verify_result(sys.argv[1], sys.argv[4])
    logging.info("=============================grad key============")
    verify_result(sys.argv[2], sys.argv[5])
    logging.info("=============================grad value============")
    verify_result(sys.argv[3], sys.argv[6])
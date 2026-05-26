#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import logging
import sys
import sysconfig
import time
import torch
from test_concat_2d_jagged_tensor import gen_data, gen_offset, concat_jagged_tensor_golden

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:0"


def setup_logging():
    # 清除已有的 handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,  # 明确指定输出到 stdout
    )


setup_logging()
logger = logging.getLogger(__name__)


def test_performance_concat_jagged():
    logger.info("Testing concat_2d_jagged performance...")

    # 测试参数组合
    input_shape = [500, 50000]
    input_cols = [128, 256, 512]
    input_dtypes = [torch.float16, torch.float32, torch.bfloat16, torch.int32]

    for input_col in input_cols:
        for input_dtype in input_dtypes:
            logger.info("Testing with input_shape=%s, dim=%s, input_dtype=%s", input_shape, input_col, input_dtype)

            total_time = 0.0
            jt_num = 2
            values = gen_data(jt_num, input_shape, input_col, input_dtype)
            # 生成40组数据并测试
            for i in range(40):
                # 生成数据
                offsets, max_seqlens = gen_offset(jt_num, input_shape, seq_len=50)

                # 将数据移到NPU
                valuesA = values[0].to(torch.device(DEVICE))
                valuesB = values[1].to(torch.device(DEVICE))
                offsetA = offsets[0].to(torch.device(DEVICE))
                offsetB = offsets[1].to(torch.device(DEVICE))

                # 前10组预热
                if i < 10:
                    torch.ops.mxrec.concat_2d_jagged(max_seqlens, valuesA, valuesB, offsetA, offsetB)
                    continue

                torch.npu.synchronize()
                # 计时
                start_time = time.time()
                torch.ops.mxrec.concat_2d_jagged(max_seqlens, valuesA, valuesB, offsetA, offsetB)
                torch.npu.synchronize()
                end_time = time.time()

                current_time = (end_time - start_time) * 1000  # 转换为毫秒
                total_time += current_time

            average_time = total_time / 30
            logger.info("Average time: %.4f ms", average_time)
            logger.info("Total time: %.4f ms", total_time)


def test_performance_split_jagged():
    logger.info("Testing split_2d_jagged performance...")

    # 测试参数组合
    output_shape = [500, 50000]
    input_cols = [128, 256, 512]
    input_dtypes = [torch.float16, torch.float32, torch.bfloat16, torch.int32]

    for input_col in input_cols:
        for input_dtype in input_dtypes:
            logger.info("Testing with output_shape=%s, dim=%s, input_dtype=%s", output_shape, input_col, input_dtype)

            total_time = 0.0
            jt_num = 2
            values = gen_data(jt_num, output_shape, input_col, input_dtype)
            # 生成40组数据并测试
            for i in range(40):
                # 生成数据, seq_len固定为50
                offsets, max_seqlens = gen_offset(jt_num, output_shape, seq_len=50)
                concated_tensor = concat_jagged_tensor_golden(values[0], values[1], offsets[0], offsets[1], max_seqlens)

                # 将数据移到NPU
                input_tensor = concated_tensor.to(torch.device(DEVICE))
                offsetA = offsets[0].to(torch.device(DEVICE))
                offsetB = offsets[1].to(torch.device(DEVICE))

                # 前10组预热
                if i < 10:
                    torch.ops.mxrec.split_2d_jagged(input_tensor, max_seqlens, offsetA, offsetB)
                    continue

                torch.npu.synchronize()
                # 计时
                start_time = time.time()
                torch.ops.mxrec.split_2d_jagged(input_tensor, max_seqlens, offsetA, offsetB)
                torch.npu.synchronize()
                end_time = time.time()

                current_time = (end_time - start_time) * 1000  # 转换为毫秒
                total_time += current_time

            average_time = total_time / 30
            logger.info("Average time: %.4f ms", average_time)
            logger.info("Total time: %.4f ms", total_time)


if __name__ == "__main__":
    test_performance_concat_jagged()
    test_performance_split_jagged()

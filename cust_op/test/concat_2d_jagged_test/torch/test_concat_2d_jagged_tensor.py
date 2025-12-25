#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import random
import sysconfig
import pytest
import numpy as np
import torch
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:0"

debug = False
if debug:
    seed = 42
    random.seed(seed)


def concat_jagged_tensor_golden(value_a, value_b, offset_a, offset_b, max_seqlens):
    valuesList = [value_a, value_b]
    offsetList = [offset_a, offset_b]
    output = []
    offsetListLen = len(offsetList)
    offsetLen = len(offsetList[0])
    for j in range(offsetLen - 1):
        for i in range(offsetListLen):
            for k in range(offsetList[i][j], offsetList[i][j + 1]):
                output.append(valuesList[i][k])
    return torch.stack(output)


def split_jagged_tensor_golden(input_values, offset_a, offset_b):
    offsetList = [offset_a, offset_b]
    offsetLen = len(offsetList[0])
    offsetcnt = len(offsetList)
    index = 0
    outputs = [[] for _ in range(offsetcnt)]
    for i in range(offsetLen - 1):
        for j in range(offsetcnt):
            slice_size = offsetList[j][i + 1] - offsetList[j][i]
            outputs[j].append(input_values[index:index + slice_size])
            index += slice_size
    return [torch.vstack(array_lst) for array_lst in outputs]


def random_seq_len(num: int, min_len: int = 2, max_len: int = None):
    # check
    if num < 1:
        raise ValueError("num 必须大于等于 1 否则无法生成递增序列")
    if max_len is None:
        max_len = num + 1
    min_len = max(min_len, 2)
    max_len = min(max_len, num + 1)
    if min_len > max_len:
        raise ValueError(f"min_len({min_len}) 不能大于 max_len({max_len})")
    seq_len = random.randint(min_len, max_len)
    return seq_len


def generate_increasing_sequence(num: int, seq_len: int):
    """
    生成从0开始，严格递增到num的随机数组
    return:随机递增序列
    """
    # 随机选择中间元素
    middle_candidates = list(range(1, num))
    selected_middle = random.sample(middle_candidates, k=seq_len - 2)
    # 拼接并排序（确保递增）
    result = [0] + sorted(selected_middle) + [num]
    result_np = np.array(result, dtype=np.int32)
    return result_np


def gen_data(jt_num, inputs_shape, input_col, input_dtype):
    input_values = []
    for i in range(jt_num):
        input_value = np.random.uniform(1, 100, [inputs_shape[i], input_col])
        input_value = torch.from_numpy(input_value).to(input_dtype)
        input_values.append(input_value)
    offsets = []
    seq_len = random_seq_len(num=min(inputs_shape), min_len=2, max_len=100)
    for i in range(0, jt_num):
        offset = generate_increasing_sequence(num=inputs_shape[i], seq_len=seq_len)
        offset = torch.from_numpy(offset).to(torch.int64)
        offsets.append(offset)

    seqlens = []
    for i in range(jt_num):
        for j in range(len(offsets[i]) - 1):
            seqlens.append(offsets[i][j + 1] - offsets[i][j])

    max_seqlens = max(seqlens)
    return input_values, offsets, max_seqlens


@pytest.mark.parametrize("input_shape", [[100, 200], [1000, 100000]])
@pytest.mark.parametrize("input_col", [128, 256, 512])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_concat_jagged_tensor(input_shape, input_col, input_dtype):
    jt_num = 2
    values, offsets, max_seqlens = gen_data(jt_num, input_shape, input_col, input_dtype)

    golden_output = concat_jagged_tensor_golden(values[0],
                                                values[1],
                                                offsets[0],
                                                offsets[1],
                                                max_seqlens)

    valuesA = values[0].to(torch.device(DEVICE))
    valuesB = values[1].to(torch.device(DEVICE))
    offsetA = offsets[0].to(torch.device(DEVICE))
    offsetB = offsets[1].to(torch.device(DEVICE))

    test_ouput = torch.ops.mxrec.concat_2d_jagged(max_seqlens, valuesA, valuesB, offsetA, offsetB)
    golden_output_cpu = golden_output.to(torch.device('cpu'))
    test_ouput_cpu = test_ouput.to(torch.device('cpu'))
    assert torch.allclose(golden_output_cpu, test_ouput_cpu, rtol=1e-04, atol=1e-04), "gloden and result is not closed"



@pytest.mark.parametrize("output_shape", [[100, 200], [1000, 100000]])
@pytest.mark.parametrize("input_col", [128, 256, 512])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_split_jagged_tensor(output_shape, input_col, input_dtype):
    jt_num = 2
    values, offsets, max_seqlens = gen_data(jt_num, output_shape, input_col, input_dtype)
    concated_tensor = concat_jagged_tensor_golden(values[0],
                                                  values[1],
                                                  offsets[0],
                                                  offsets[1],
                                                  max_seqlens)

    gold_value_a, gold_value_b = split_jagged_tensor_golden(concated_tensor, offsets[0], offsets[1])

    input_tensor = concated_tensor.to(torch.device(DEVICE))
    offsetA = offsets[0].to(torch.device(DEVICE))
    offsetB = offsets[1].to(torch.device(DEVICE))
    test_value_a, test_value_b = torch.ops.mxrec.split_2d_jagged(input_tensor, max_seqlens, offsetA, offsetB)

    test_value_a_cpu = test_value_a.to(torch.device('cpu'))
    test_value_b_cpu = test_value_b.to(torch.device('cpu'))

    assert torch.allclose(gold_value_a, test_value_a_cpu, rtol=1e-04, atol=1e-04), "gloden and result is not closed"
    assert torch.allclose(gold_value_b, test_value_b_cpu, rtol=1e-04, atol=1e-04), "gloden and result is not closed"

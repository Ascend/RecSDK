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
            outputs[j].append(input_values[index : index + slice_size])
            index += slice_size
    return [torch.vstack(array_lst) for array_lst in outputs]


def concat_jagged_tensor_golden_with_prefix(value_a, value_b, offset_a, offset_b, n_prefix_from_right):
    """
    Golden implementation for concat with nPrefixFromRight parameter.
    Output structure: [B_prefix] + [A] + [B_remaining]
    """
    offsetList = [offset_a, offset_b]
    offsetLen = len(offsetList[0])
    output = []

    for j in range(offsetLen - 1):
        a_start, a_end = offsetList[0][j], offsetList[0][j + 1]
        b_start, b_end = offsetList[1][j], offsetList[1][j + 1]

        b_len = b_end - b_start
        prefix_size = min(n_prefix_from_right, b_len)

        # B prefix: first n elements from B
        for k in range(b_start, min(b_start + prefix_size, b_end)):
            output.append(value_b[k])

        # A: all elements from A
        for k in range(a_start, a_end):
            output.append(value_a[k])

        # B remaining: remaining elements from B
        for k in range(b_start + prefix_size, b_end):
            output.append(value_b[k])

    return torch.stack(output)


def split_jagged_tensor_golden_with_prefix(input_values, offset_a, offset_b, n_prefix_to_right):
    """
    Golden implementation for split with nPrefixFromRight parameter.
    Reverse operation of concat_jagged_tensor_golden_with_prefix.

    Input: [B_prefix] + [A] + [B_remaining]
    Output: A, B (where B = B_prefix + B_remaining)
    """
    offsetList = [offset_a, offset_b]
    offsetLen = len(offsetList[0])

    output_a = []
    output_b = []

    index = 0
    for j in range(offsetLen - 1):
        a_start, a_end = offsetList[0][j], offsetList[0][j + 1]
        b_start, b_end = offsetList[1][j], offsetList[1][j + 1]

        a_len = a_end - a_start
        b_len = b_end - b_start

        prefix_size = min(n_prefix_to_right, b_len)
        remaining_b_size = b_len - prefix_size

        # B prefix: first n elements go to B
        for _ in range(prefix_size):
            output_b.append(input_values[index])
            index += 1

        # A: next a_len elements go to A
        for _ in range(a_len):
            output_a.append(input_values[index])
            index += 1

        # B remaining: last (b_len - prefix_size) elements go to B
        for _ in range(remaining_b_size):
            output_b.append(input_values[index])
            index += 1

    return torch.vstack(output_a), torch.vstack(output_b)


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
        if input_dtype == torch.int32:
            input_value = np.random.randint(1, 100, [inputs_shape[i], input_col])
            input_value = torch.from_numpy(input_value).to(input_dtype)
        else:
            input_value = np.random.uniform(1, 100, [inputs_shape[i], input_col])
            input_value = torch.from_numpy(input_value).to(input_dtype)
        input_values.append(input_value)
    return input_values


def gen_offset(jt_num, inputs_shape, seq_len=None):
    offsets = []
    # 功能用例使用随机shape,性能用例使用确定shape.
    if not seq_len:
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
    return offsets, max_seqlens


@pytest.mark.parametrize("input_shape", [[100, 200], [1000, 100000]])
@pytest.mark.parametrize("input_col", [128, 256, 512])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32, torch.bfloat16, torch.int32])
def test_concat_jagged_tensor(input_shape, input_col, input_dtype):
    jt_num = 2
    values = gen_data(jt_num, input_shape, input_col, input_dtype)
    offsets, max_seqlens = gen_offset(jt_num, input_shape)

    golden_output = concat_jagged_tensor_golden(values[0], values[1], offsets[0], offsets[1], max_seqlens)

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
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32, torch.bfloat16, torch.int32])
def test_split_jagged_tensor(output_shape, input_col, input_dtype):
    jt_num = 2
    values = gen_data(jt_num, output_shape, input_col, input_dtype)
    offsets, max_seqlens = gen_offset(jt_num, output_shape)
    concated_tensor = concat_jagged_tensor_golden(values[0], values[1], offsets[0], offsets[1], max_seqlens)

    gold_value_a, gold_value_b = split_jagged_tensor_golden(concated_tensor, offsets[0], offsets[1])

    input_tensor = concated_tensor.to(torch.device(DEVICE))
    offsetA = offsets[0].to(torch.device(DEVICE))
    offsetB = offsets[1].to(torch.device(DEVICE))
    test_value_a, test_value_b = torch.ops.mxrec.split_2d_jagged(input_tensor, max_seqlens, offsetA, offsetB)

    test_value_a_cpu = test_value_a.to(torch.device('cpu'))
    test_value_b_cpu = test_value_b.to(torch.device('cpu'))

    assert torch.allclose(gold_value_a, test_value_a_cpu, rtol=1e-04, atol=1e-04), "gloden and result is not closed"
    assert torch.allclose(gold_value_b, test_value_b_cpu, rtol=1e-04, atol=1e-04), "gloden and result is not closed"


@pytest.mark.parametrize("input_shape", [[100, 200], [500, 100000]])
@pytest.mark.parametrize("input_col", [128, 256])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_prefix", [1, 5, 10])
def test_concat_jagged_tensor_with_prefix(input_shape, input_col, input_dtype, n_prefix):
    """Test concat_2d_jagged with nPrefixFromRight != 0"""
    jt_num = 2

    values = gen_data(jt_num, input_shape, input_col, input_dtype)
    offsets, max_seqlens = gen_offset(jt_num, input_shape)

    golden_output = concat_jagged_tensor_golden_with_prefix(values[0], values[1], offsets[0], offsets[1], n_prefix)

    valuesA = values[0].to(torch.device(DEVICE))
    valuesB = values[1].to(torch.device(DEVICE))
    offsetA = offsets[0].to(torch.device(DEVICE))
    offsetB = offsets[1].to(torch.device(DEVICE))

    test_output = torch.ops.mxrec.concat_2d_jagged(max_seqlens, valuesA, valuesB, offsetA, offsetB, False, n_prefix)

    golden_output_cpu = golden_output.to(torch.device('cpu'))
    test_output_cpu = test_output.to(torch.device('cpu'))

    assert torch.allclose(golden_output_cpu, test_output_cpu, rtol=1e-04, atol=1e-04), (
        "golden and result is not closed for concat with nPrefixFromRight"
    )


@pytest.mark.parametrize("output_shape", [[100, 200], [500, 1000]])
@pytest.mark.parametrize("input_col", [128, 256])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_prefix", [1, 5, 10])
def test_split_jagged_tensor_with_prefix(output_shape, input_col, input_dtype, n_prefix):
    """Test split_2d_jagged with nPrefixFromRight != 0"""
    jt_num = 2

    values = gen_data(jt_num, output_shape, input_col, input_dtype)
    offsets, max_seqlens = gen_offset(jt_num, output_shape)

    # First concat with prefix to get input for split
    concated_tensor = concat_jagged_tensor_golden_with_prefix(values[0], values[1], offsets[0], offsets[1], n_prefix)

    # Golden split
    gold_value_a, gold_value_b = split_jagged_tensor_golden_with_prefix(
        concated_tensor, offsets[0], offsets[1], n_prefix
    )

    # Test split on NPU
    input_tensor = concated_tensor.to(torch.device(DEVICE))
    offsetA = offsets[0].to(torch.device(DEVICE))
    offsetB = offsets[1].to(torch.device(DEVICE))

    test_value_a, test_value_b = torch.ops.mxrec.split_2d_jagged(
        input_tensor, max_seqlens, offsetA, offsetB, 0, n_prefix
    )

    test_value_a_cpu = test_value_a.to(torch.device('cpu'))
    test_value_b_cpu = test_value_b.to(torch.device('cpu'))

    assert torch.allclose(gold_value_a, test_value_a_cpu, rtol=1e-04, atol=1e-04), (
        "golden and result A is not closed for split with nPrefixFromRight"
    )
    assert torch.allclose(gold_value_b, test_value_b_cpu, rtol=1e-04, atol=1e-04), (
        "golden and result B is not closed for split with nPrefixFromRight"
    )

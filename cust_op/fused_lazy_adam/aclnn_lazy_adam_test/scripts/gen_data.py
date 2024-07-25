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

import copy
import os
import numpy as np

# 获取项目路径
_CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
_PROJECT_PATH = os.path.dirname(_CURRENT_PATH)
_INPUT_PATH = os.path.join(_PROJECT_PATH, "./input")
_OUTPUT_PATH = os.path.join(_PROJECT_PATH, "./output")

_DIM_0 = 2000000
_DIM_1 = 564096
_DIM_2 = 32


def _gather(input_data, indices):
    out = np.zeros((len(indices), input_data.shape[1]))
    for i, index_ in enumerate(indices):
        # 跳过index小于0的数据
        if index_[0] < 0:
            continue
        out[i] = input_data[index_[0]]
    return out


def _scatter_nd_update(momentum, indices, update_value):
    out = copy.deepcopy(momentum)
    for i, index_ in enumerate(indices):
        if index_[0] < 0:
            continue
        else:
            out[index_[0]] = update_value[i]
    return out


def _scatter_nd_add(momentum, indices, update_value):
    out = copy.deepcopy(momentum)
    for i, index_ in enumerate(indices):
        if index_[0] < 0:
            continue
        else:
            out[indices[i][0]] = out[index_[0]] + update_value[i]
    return out


def _gen_input_data():
    range_start = 1
    range_end = 2

    dtype_chose = np.float32
    shape0 = (_DIM_0, _DIM_2)
    indices_shape = (_DIM_1, 1)
    grad_shape = (_DIM_1, _DIM_2)

    input_var = np.random.uniform(range_start, range_end, size=shape0).astype(dtype_chose)  # shape [2000000,32]
    input_m = np.random.uniform(range_start, range_end, size=shape0).astype(dtype_chose)  # shape [2000000,32]
    input_v = np.random.uniform(range_start, range_end, size=shape0).astype(dtype_chose)  # shape [2000000,32]

    # indices shape [564096,1]
    indices = np.random.permutation(np.arange(_DIM_0)).astype(np.int32)[:indices_shape[0]].reshape(-1, 1)
    # gradient shape [564096,32]
    gradient = np.random.uniform(range_start, range_end, size=grad_shape).astype(dtype_chose)

    if not os.path.exists(_INPUT_PATH):
        os.makedirs(_INPUT_PATH)
    indices.tofile(os.path.join(_INPUT_PATH, "indices.bin"))
    gradient.tofile(os.path.join(_INPUT_PATH, "gradient.bin"))
    input_m.tofile(os.path.join(_INPUT_PATH, "inputM.bin"))
    input_v.tofile(os.path.join(_INPUT_PATH, "inputV.bin"))
    input_var.tofile(os.path.join(_INPUT_PATH, "inputVar.bin"))


def _gen_golden_data():
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    epsilon = 1e-7

    lr = np.array(lr).astype(np.float32)
    beta1 = np.array(beta1).astype(np.float32)
    beta2 = np.array(beta2).astype(np.float32)
    epsilon = np.array(epsilon).astype(np.float32)

    lr.tofile(os.path.join(_INPUT_PATH, "learningRate.bin"))

    indices = np.fromfile(os.path.join(_INPUT_PATH, "indices.bin"), dtype=np.int32).reshape(
        (_DIM_1, 1))  # shape (564096,1)
    gradient = np.fromfile(os.path.join(_INPUT_PATH, "gradient.bin"), dtype=np.float32).reshape(
        (_DIM_1, _DIM_2))  # shape (564096,32)
    input_m = np.fromfile(os.path.join(_INPUT_PATH, "inputM.bin"), dtype=np.float32).reshape(
        (_DIM_0, _DIM_2))  # shape (2000000,32)
    input_v = np.fromfile(os.path.join(_INPUT_PATH, "inputV.bin"), dtype=np.float32).reshape(
        (_DIM_0, _DIM_2))  # shape (2000000,32)
    input_var = np.fromfile(os.path.join(_INPUT_PATH, "inputVar.bin"), dtype=np.float32).reshape(
        (_DIM_0, _DIM_2))  # shape (2000000,32)

    old_m_slice = _gather(input_m, indices)  # shape(564096,32)
    old_m_slice = np.array(old_m_slice).astype(np.float32)  #
    update_m = beta1 * old_m_slice + (1 - beta1) * gradient
    out_m = _scatter_nd_update(input_m, indices, update_m)

    old_v_slice = _gather(input_v, indices)
    old_v_slice = np.array(old_v_slice).astype(np.float32)
    update_v = beta2 * old_v_slice + (1 - beta2) * np.square(gradient)
    out_v = _scatter_nd_update(input_v, indices, update_v)

    denominator_slice = np.sqrt(np.abs(update_v)) + epsilon
    update_var = np.divide(-lr * update_m, denominator_slice)
    out_var = _scatter_nd_add(input_var, indices, update_var)

    return out_m, out_v, out_var


def _gen_input_and_golden_data():
    # 产生输入数据
    _gen_input_data()

    # 产生真值数据
    out_m, out_v, out_var = _gen_golden_data()
    if not os.path.exists(_OUTPUT_PATH):
        os.makedirs(_OUTPUT_PATH)
    out_m.tofile(os.path.join(_OUTPUT_PATH, "goldenOutputM.bin"))
    out_v.tofile(os.path.join(_OUTPUT_PATH, "goldenOutputV.bin"))
    out_var.tofile(os.path.join(_OUTPUT_PATH, "goldenOutputVar.bin"))


if __name__ == "__main__":
    _gen_input_and_golden_data()

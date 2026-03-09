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
import logging
import torch
from torch import Tensor

MARE_L1 = 5
MERE_L1 = 1.5
RMSE_L1 = 1.5


def allclose(a, b, atol, ratio):
    if a.shape != b.shape:
        raise Exception("The shape of a and b must be same.")
    diff = torch.abs(a.cpu() - b.cpu()) > atol
    diff_count = torch.sum(diff)
    diff_ratio = diff_count / a.numel()
    return diff_ratio < ratio


def compute_mare(actual: Tensor, golden: Tensor):
    """
    计算最大相对误差
    """
    if actual.shape != golden.shape:
        raise ValueError(f"actual shape {actual.shape} != golden shape {golden.shape}")
    # 计算相对误差，添加防止溢出的处理
    diff = torch.abs(actual - golden)
    denominator = torch.abs(golden) + 1e-7
    rel_error = torch.where(denominator > 1e-7, diff / denominator, diff)
    return rel_error.max().item()


def compute_mere(actual: Tensor, golden: Tensor):
    """
    计算平均相对误差
    """
    if actual.shape != golden.shape:
        raise ValueError(f"actual shape {actual.shape} != golden shape {golden.shape}")
    # 计算相对误差，添加防止溢出的处理
    diff = torch.abs(actual - golden)
    denominator = torch.abs(golden) + 1e-7
    rel_error = torch.where(denominator > 1e-7, diff / denominator, diff)
    return rel_error.mean().item()


def compute_rmse(actual: Tensor, golden: Tensor):
    """
    计算均方根误差
    """
    if actual.shape != golden.shape:
        raise ValueError(f"actual shape {actual.shape} != golden shape {golden.shape}")
    squared_error = (actual - golden).pow(2)
    mse = torch.mean(squared_error)
    return torch.sqrt(mse).item()


def compare_data_with_double_pole(tensor_msg: str, actual_fused: Tensor, actual_npu: Tensor, golden: Tensor):
    """
    双标杆对比
    Args:
        tensor_msg: 待比较tensor描述信息
        actual_fused: NPU融合算子计算结果
        actual_npu: NPU小算子计算结果
        golden: CPU 高精度计算结果
    """
    if actual_fused.device.type != golden.device.type:
        actual_fused = actual_fused.to(golden.device)
    if actual_npu.device.type != golden.device.type:
        actual_npu = actual_npu.to(golden.device)
    actual_fused = actual_fused.float()
    actual_npu = actual_npu.float()
    golden = golden.float()

    mare_fused = compute_mare(actual_fused, golden)
    mare_npu = compute_mare(actual_npu, golden)
    mere_fused = compute_mere(actual_fused, golden)
    mere_npu = compute_mere(actual_npu, golden)
    rmse_fused = compute_rmse(actual_fused, golden)
    rmse_npu = compute_rmse(actual_npu, golden)

    print_msg = (f"{tensor_msg}, mare_fused: {mare_fused}, mare_npu: {mare_npu}, mere_fused: {mere_fused},"
                 f" mere_npu: {mere_npu}, rmse_fused: {rmse_fused}, rmse_npu: {rmse_npu};")
    assert mare_fused / mare_npu <= MARE_L1 if mare_npu != 0.0 else abs(mare_fused - mare_npu) < 1e-6, \
        f"{print_msg} mare error ratio does not meet the requirement"
    assert mere_fused / mere_npu <= MERE_L1 if mere_npu != 0.0 else abs(mere_fused - mere_npu) < 1e-6, \
        f"{print_msg} mere error ratio does not meet the requirement"
    assert rmse_fused / rmse_npu <= RMSE_L1 if rmse_npu != 0.0 else (rmse_fused - rmse_npu) < 1e-6, \
        f"{print_msg} rmse error ratio does not meet the requirement"

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
import itertools
import logging
import sysconfig
import sys
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))
from utils import compare_data_with_double_pole

is_ascend_950 = False
is_gpu = torch.cuda.is_available()
if not is_gpu:
    import torch_npu
    torch.npu.set_device(0)
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
    is_ascend_950 = str(torch_npu.npu.get_device_name()).strip().startswith("Ascend950PR")

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(321)
np.random.seed(42)


DIM0_MAX = int(1e6)
SUPPORT_DIM1_LIST = [512, 1024]
SUPPORT_DTYPES = [torch.float16, torch.bfloat16]
if not is_ascend_950:
    SUPPORT_DTYPES.append(torch.float32)

eps = 1e-5

# (dim0, dim1, eps, dropout_ratio, dtype, weight_and_bias_dim) 组成的列表，weight_and_bias_dim可选
INVALID_PARAMS = [
    (0, 100, 1e-5, 0.3, torch.bfloat16),  # 不支持的dim0
    (0, DIM0_MAX + 1, 1e-5, 0.3, torch.bfloat16),  # 不支持的dim0
    (1, 100, 1e-5, 0.3, torch.bfloat16),  # 不支持的dim1
    (1, 512, 1e-5, -0.2, torch.bfloat16),  # 不支持的dropout_ratio
    (1, 512, 1e-5, 1.3, torch.bfloat16),  # 不支持的dropout_ratio2
    (1, 512, 0.0, 1.3, torch.bfloat16),  # 不支持的eps
    (1, 512, 1e-11, 1.3, torch.bfloat16),  # 不支持的eps3
    (1, 512, 1e-1, 1.3, torch.bfloat16),  # 不支持的eps3
    (1, 512, 1e-5, 1.3, torch.float64),  # 不支持的dtype
    (1, 512, 1e-5, 0.3, torch.bfloat16, 1024),  # weight bias dim和x dim1不一致
]

cpu_device = torch.device("cpu")
accelerator_device = torch.device("cuda") if is_gpu else torch.device("npu")


def device_synchronize():
    if is_gpu:
        torch.cuda.synchronize()
    else:
        torch_npu.npu.synchronize()


def norm_multiply_dropout_pytorch(x, u, gamma, bias, convert_to_high_level_type=True):
    high_level_type = torch.float64 if x.dtype == torch.float32 else torch.float32
    ori_dtype = x.dtype
    if convert_to_high_level_type:
        x = x.to(high_level_type)
        u = u.to(high_level_type)
        gamma = gamma.to(high_level_type)
        bias = bias.to(high_level_type)
    ln_ret = F.layer_norm(x, normalized_shape=[x.shape[-1]], weight=gamma, bias=bias, eps=eps)
    ln_out = ln_ret * u
    if convert_to_high_level_type:
        ln_out = ln_out.to(ori_dtype)

    return ln_out


def norm_multiply_dropout_by_device(x_fused, u_fused, g_fused, b_fused):
    if is_gpu:
        return norm_multiply_dropout_pytorch(x_fused, u_fused, g_fused, b_fused)
    else:
        return torch.ops.mxrec.norm_multiply_dropout(x_fused, u_fused, g_fused, b_fused, eps, 0.0)[0]


def generate_input_tensor(dim0, dim1, dtype, weight_bias_dim=None):
    torch.manual_seed(321)
    # weight_bias_dim参数用于构造错误场景，构造weight和bias的dim和dim1不一致
    # 正常场景下生成数据不用传weight_bias_dim参数
    if not weight_bias_dim:
        weight_bias_dim = dim1

    x_data = torch.rand((dim0, dim1), dtype=dtype).uniform_(-1, 1)
    u_data = torch.rand((dim0, dim1), dtype=dtype).uniform_(-1, 1)
    g_data = torch.rand((weight_bias_dim,), dtype=dtype).uniform_(-1, 1)
    b_data = torch.rand((weight_bias_dim,), dtype=dtype).uniform_(-1, 1)
    dy_data = torch.rand((dim0, dim1), dtype=dtype).uniform_(-1, 1)
    return x_data, u_data, g_data, b_data, dy_data


def pairwise_sum_2d_recursive(x, dim=-1):
    """
    沿指定维度递归成对求和
    x: [B, C]
    return: [B, 1]
    """
    if x.shape[dim] == 1:
        return x

    # 沿dim维度分成两半
    mid = x.shape[dim] // 2

    if dim == -1 or dim == 1:
        left = x[:, :mid]
        right = x[:, mid:]
    else:
        left = x[:mid, :]
        right = x[mid:, :]

    # 递归求和后相加
    return pairwise_sum_2d_recursive(left, dim) + pairwise_sum_2d_recursive(right, dim)


def ln_mul_fwd(x: Tensor, u: Tensor, gamma: Tensor, beta: Tensor):
    high_level_type = torch.float64 if x.dtype == torch.float32 else torch.float32
    ori_dtype = x.dtype
    x = x.to(high_level_type)
    u = u.to(high_level_type)
    gamma = gamma.to(high_level_type)
    beta = beta.to(high_level_type)

    # 求和计算
    mean = two_level_sum_with_grouped(x)
    var = two_level_sum_with_grouped((x - mean).pow(2))

    std = torch.sqrt(var + eps)
    x_hat = (x - mean) / std  # (..., C)
    y_ln = x_hat * gamma + beta  # (..., C)
    y = y_ln * u  # (..., C)

    y = y.to(ori_dtype)
    return y, mean, var


def pairwise_sum_dim0_iterative(x):
    """
    迭代层级 pairwise 归约沿 dim=0
    x: [B, C]
    return: [C]
    """
    dim0, dim1 = x.shape

    # 填充到2的幂次
    next_pow2 = 2 ** (dim0 - 1).bit_length() if dim0 > 1 else 1
    if dim0 != next_pow2:
        pad_size = next_pow2 - dim0
        pad = torch.zeros(pad_size, dim1, dtype=x.dtype, device=x.device)
        data = torch.cat([x, pad], dim=0)  # [next_pow2, C]
    else:
        data = x.clone()

    # 层级归约: [N, C] -> [N/2, C] -> ... -> [1, C] -> [C]
    while data.shape[0] > 1:
        new_dim0 = data.shape[0]
        # 重塑为 [N/2, 2, C] 后在 dim=1 求和
        data = data.view(new_dim0 // 2, 2, dim1).sum(dim=1)  # [N/2, C]

    return data[0]  # [C]


def two_level_sum_with_grouped(x, dim: int = -1):
    """
    分段求和
    """
    group_size = 64
    if dim < 0:
        dim = x.ndim + dim

    n = x.shape[dim]
    if n < group_size:
        return x.sum(dim=dim)

    pad_len = (group_size - n % group_size) % group_size
    num_full_segments = (n + pad_len) // group_size
    if pad_len > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len

        padding = torch.full(pad_shape, 0.0, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, padding], dim=dim)
    else:
        x_padded = x

    new_shape = list(x_padded.shape)
    new_shape[dim:dim + 1] = [num_full_segments, group_size]
    x_reshaped = x_padded.view(new_shape)

    level1_sum = x_reshaped.sum(dim=dim + 1)
    return level1_sum.sum(dim=dim, keepdim=True)


def ln_mul_bwd(inputs: tuple[Tensor, Tensor, Tensor, Tensor], dy: Tensor, mean: Tensor, var: Tensor):
    x, u, gamma, beta = inputs
    high_level_type = torch.float64 if x.dtype == torch.float32 else torch.float32
    ori_dtype = dy.dtype
    dy = dy.to(high_level_type)
    gamma = gamma.to(high_level_type)
    beta = beta.to(high_level_type)

    std = torch.sqrt(var + eps)
    x_hat = (x - mean) / std
    dim1 = x_hat.shape[-1]
    denominator = 1 / std  # (B, 1)

    d_ln_out = dy * u
    dg_pre = x_hat * d_ln_out
    dg = pairwise_sum_dim0_iterative(dg_pre)  # (C, 1)
    db = pairwise_sum_dim0_iterative(dy * u)  # (C, 1)

    reduce_dx_hat = pairwise_sum_2d_recursive(dy * u * gamma)
    reduce_x_dx = pairwise_sum_2d_recursive(dy * u * gamma * x_hat)

    dx = (dy * u * gamma - (reduce_dx_hat + x_hat * reduce_x_dx) / dim1) * denominator
    du = dy * (gamma * x_hat + beta)
    dx, du, dg, db = [t.to(ori_dtype) for t in (dx, du, dg, db)]
    return dx, du, dg, db


def get_npu_fused_op_ret(input_data_list, dy):
    x_fused, u_fused, g_fused, b_fused = \
        [t.to(accelerator_device).contiguous().requires_grad_() for t in input_data_list]
    y_fused = norm_multiply_dropout_by_device(x_fused, u_fused, g_fused, b_fused)
    dy_npu = dy.to(accelerator_device).contiguous()
    y_fused.backward(dy_npu)
    device_synchronize()
    return y_fused.cpu(), x_fused.grad.cpu(), u_fused.grad.cpu(), g_fused.grad.cpu(), b_fused.grad.cpu()


def get_npu_small_op_ret(input_data_list, dy):
    x_npu, u_npu, g_npu, b_npu = [t.to(accelerator_device).contiguous().requires_grad_() for t in input_data_list]
    y_npu = norm_multiply_dropout_pytorch(x_npu, u_npu, g_npu, b_npu, convert_to_high_level_type=False)
    dy_npu = dy.to(accelerator_device).contiguous()
    y_npu.backward(dy_npu)
    device_synchronize()
    return y_npu.cpu(), u_npu.grad.cpu(), u_npu.grad.cpu(), g_npu.grad.cpu(), b_npu.grad.cpu()


@pytest.mark.parametrize("dim0", [1, 1000, 2345, 4096 * 4, 262144, 927750, DIM0_MAX])
@pytest.mark.parametrize("dim1", SUPPORT_DIM1_LIST)
@pytest.mark.parametrize("dtype", SUPPORT_DTYPES)
def test_norm_multiply_double_pole(dim0: int, dim1: int, dtype):
    # 使用双标杆，计算对比精度
    logging.info(f"===test case info: dim0:{dim0}, dim1:{dim1}, dtype:{dtype}, dropout_ratio:{0.0}===")
    x, u, g, b, dy = generate_input_tensor(dim0, dim1, dtype)

    # =====  pytorch小算子执行 =====
    x_pt, u_pt, g_pt, b_pt = [t.clone().contiguous().requires_grad_() for t in [x, u, g, b]]
    y_golden, mean1, var1 = ln_mul_fwd(x_pt, u_pt, g_pt, b_pt)
    dy_pt = dy.clone().contiguous()
    inputs = (x_pt, u_pt, g_pt, b_pt)
    x_pt_grad, u_pt_grad, g_pt_grad, b_pt_grad = ln_mul_bwd(inputs, dy_pt, mean1, var1)
    device_synchronize()

    input_data_list = (x, u, g, b)

    # ===== NPU融合算子执行 =====
    y_fused, x_fused_grad, u_fused_grad, g_fused_grad, b_fused_grad = get_npu_fused_op_ret(input_data_list, dy)

    # ===== NPU小算子执行 =====
    y_npu, x_npu_grad, u_npu_grad, g_npu_grad, b_npu_grad = get_npu_small_op_ret(input_data_list, dy)

    # 双标杆对比较前反向结果
    msg = f"params-{dim0}_{dim1}_{dtype}"
    compare_data_with_double_pole(f"{msg}, y", y_fused, y_npu, y_golden)
    compare_data_with_double_pole(f"{msg}, x_grad", x_fused_grad, x_npu_grad, x_pt_grad)
    compare_data_with_double_pole(f"{msg}, u_grad", u_fused_grad, u_npu_grad, u_pt_grad)
    compare_data_with_double_pole(f"{msg}, g_grad", g_fused_grad, g_npu_grad, g_pt_grad)
    compare_data_with_double_pole(f"{msg}, b_grad", b_fused_grad, b_npu_grad, b_pt_grad)
    logging.info("compare end.")


@dataclass
class ExecuteConfig:
    dim0: int
    dim1: int
    eps: float
    dropout_ratio: float
    dtype: torch.dtype
    weight_and_bias_dim: int = None


params = {
    "dim0": [1, 1000, 2345, 4096 * 4, 262144, 927750, DIM0_MAX],
    "dim1": SUPPORT_DIM1_LIST,
    "eps": [1e-5],
    "dropout_ratio": [0.3],
    "dtype": SUPPORT_DTYPES,
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_norm_multiply_dropout(config: ExecuteConfig):
    dim0 = config.dim0
    dim1 = config.dim1
    epsilon = config.eps
    dropout_ratio = config.dropout_ratio
    dtype = config.dtype
    weight_and_bias_dim = config.weight_and_bias_dim
    logging.info(f"===test case info: dim0:{dim0}, dim1:{dim1}, eps:{epsilon},"
                 f" dropout_ratio:{dropout_ratio}, dtype:{dtype}===")
    # 执行带dropout的场景 由于dropout具有随机性，仅验证算子功能正常执行
    x, u, g, b, dy = generate_input_tensor(dim0, dim1, dtype, weight_bias_dim=weight_and_bias_dim)
    x_fused, u_fused, g_fused, b_fused = [t.to(accelerator_device).contiguous().requires_grad_() for t in [x, u, g, b]]
    dy_npu = dy.to(accelerator_device).contiguous()
    y = torch.ops.mxrec.norm_multiply_dropout(x_fused, u_fused, g_fused, b_fused, epsilon, dropout_ratio)[0]
    y.backward(dy_npu)
    device_synchronize()


@pytest.mark.parametrize("invalid_params", INVALID_PARAMS)
def test_invalid_params(invalid_params: tuple):
    with pytest.raises(RuntimeError):
        config = ExecuteConfig(*invalid_params)
        test_norm_multiply_dropout(config)


if __name__ == '__main__':
    pytest.main([__file__, "-sv"])

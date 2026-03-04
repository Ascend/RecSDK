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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))
from utils import compare_data_with_double_pole

is_gpu = torch.cuda.is_available()
if not is_gpu:
    import torch_npu
    torch.npu.set_device(0)
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(321)
np.random.seed(42)


DIM0_MAX = int(1e6)
SUPPORT_DIM1_LIST = [512, 1024]
SUPPORT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

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

torch.npu.set_device(0)

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
    ln_out = u * ln_ret
    if convert_to_high_level_type:
        ln_out = ln_out.to(ori_dtype)

    return ln_out


def norm_multiply_dropout_by_device(x_fused, u_fused, g_fused, b_fused):
    if is_gpu:
        return norm_multiply_dropout_pytorch(x_fused, u_fused, g_fused, b_fused)
    else:
        return torch.ops.mxrec.norm_multiply_dropout(x_fused, u_fused, g_fused, b_fused, eps, 0.0)[0]


@pytest.mark.parametrize("dim0", [1, 1000, 2345, 4096 * 4, 262144, 927750, DIM0_MAX])
@pytest.mark.parametrize("dim1", SUPPORT_DIM1_LIST)
@pytest.mark.parametrize("dtype", SUPPORT_DTYPES)
def test_norm_multiply_double_pole(dim0: int, dim1: int, dtype):
    # 使用双标杆，计算对比精度
    logging.info(f"===test case info: dim0:{dim0}, dim1:{dim1}, dtype:{dtype}, dropout_ratio:{0.0}===")
    # 由于dropout具有随机性，和golden比较数据时需设置dropout_ratio为0
    low = -1.0
    high = 1.0
    b_np, dy_np, g_np, u_np, x_np = generate_np_data(dim0, dim1, high, low)

    # =====  pytorch小算子执行 =====
    golden_device = cpu_device
    x_pt = torch.tensor(x_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    u_pt = torch.tensor(u_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    g_pt = torch.tensor(g_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    b_pt = torch.tensor(b_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    y_golden = norm_multiply_dropout_pytorch(x_pt, u_pt, g_pt, b_pt)
    dy_pt = torch.tensor(dy_np, dtype=dtype, device=golden_device).contiguous()
    y_golden.backward(dy_pt)
    device_synchronize()

    # ===== NPU融合算子执行 =====
    x_fused = torch.tensor(x_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    u_fused = torch.tensor(u_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    g_fused = torch.tensor(g_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    b_fused = torch.tensor(b_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    y_fused = norm_multiply_dropout_by_device(x_fused, u_fused, g_fused, b_fused)
    dy_fused = torch.tensor(dy_np, dtype=dtype, device=accelerator_device).contiguous()
    y_fused.backward(dy_fused)
    device_synchronize()

    # ===== NPU小算子执行 =====
    x_npu = torch.tensor(x_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    u_npu = torch.tensor(u_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    g_npu = torch.tensor(g_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    b_npu = torch.tensor(b_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_(True)
    y_npu = norm_multiply_dropout_pytorch(x_npu, u_npu, g_npu, b_npu, convert_to_high_level_type=False)
    dy_fused = torch.tensor(dy_np, dtype=dtype, device=accelerator_device).contiguous()
    y_npu.backward(dy_fused)
    device_synchronize()

    # 双标杆对比较前反向结果
    compare_data_with_double_pole("y", y_fused, y_npu, y_golden)
    compare_data_with_double_pole("x grad", x_fused.grad, x_npu.grad, x_pt.grad)
    compare_data_with_double_pole("u grad", u_fused.grad, u_npu.grad, u_pt.grad)
    compare_data_with_double_pole("g grad", g_fused.grad, g_npu.grad, g_pt.grad)
    compare_data_with_double_pole("b grad", b_fused.grad, b_npu.grad, b_pt.grad)
    logging.info("compare end.")


def generate_np_data(dim0, dim1, high, low, weight_bias_dim=None):
    # weight_bias_dim参数用于构造错误场景，构造weight和bias的dim和dim1不一致
    # 正常场景下生成数据不用传weight_bias_dim参数
    if not weight_bias_dim:
        weight_bias_dim = dim1

    x_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))
    u_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))
    g_np = np.random.uniform(low=low, high=high, size=(weight_bias_dim,))
    b_np = np.random.uniform(low=low, high=high, size=(weight_bias_dim,))
    dy_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))
    return b_np, dy_np, g_np, u_np, x_np


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
    low = -1.0
    high = 1.0
    b_np, dy_np, g_np, u_np, x_np = generate_np_data(dim0, dim1, high, low, weight_bias_dim=weight_and_bias_dim)
    dy_fused = torch.tensor(dy_np, dtype=dtype, device=accelerator_device).contiguous()
    x_fused = torch.tensor(x_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_()
    u_fused = torch.tensor(u_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_()
    g_fused = torch.tensor(g_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_()
    b_fused = torch.tensor(b_np, dtype=dtype, device=accelerator_device).contiguous().requires_grad_()
    y = torch.ops.mxrec.norm_multiply_dropout(x_fused, u_fused, g_fused, b_fused, epsilon, dropout_ratio)[0]
    y.backward(dy_fused)
    device_synchronize()


@pytest.mark.parametrize("invalid_params", INVALID_PARAMS)
def test_invalid_params(invalid_params: tuple):
    with pytest.raises(RuntimeError):
        config = ExecuteConfig(*invalid_params)
        test_norm_multiply_dropout(config)


if __name__ == '__main__':
    pytest.main([__file__, "-sv"])

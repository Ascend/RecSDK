#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from pathlib import Path
import pytest
import torch
import torch_npu
import numpy as np

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent /
                           "framework/torch_plugin/torch_library/ln_mul/build/libln_mul.so"))


def get_golden(x: np.ndarray, u: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """
    x,u   : [A, R]  float32
    gamma : [R]     同一 dtype
    beta  : [R]     同一 dtype
    return: y       float32
    """
    x_f = x.astype(np.float32)
    u_f = u.astype(np.float32)
    gamma_f = gamma.astype(np.float32)
    beta_f = beta.astype(np.float32)

    mean = x_f.mean(axis=1, keepdims=True)          # [A, 1]
    var = x_f.var(axis=1, keepdims=True, ddof=0)    # [A, 1]
    rstd = 1.0 / np.sqrt(var + eps)                 # [A, 1]

    y = (x_f - mean) * rstd * gamma_f + beta_f      # broadcasting [A, R]
    y = y * u_f
    return y.astype(x.dtype)


def get_op(x: torch.Tensor, u: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    y = torch.ops.mxrec.ln_mul(x, u, gamma, beta)
    return y.cpu().numpy()


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("a", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("r", [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def test_ln_mul(dtype, a, r):
    torch.npu.set_device(0)
    # 构造随机输入
    x = torch.randn(a, r, device="npu", dtype=dtype)
    u = torch.randn(a, r, device="npu", dtype=dtype)
    gamma = torch.randn(r, device="npu", dtype=dtype)
    beta = torch.randn(r, device="npu", dtype=dtype)

    # golden
    y_golden = get_golden(x.cpu().numpy(), u.cpu().numpy(), gamma.cpu().numpy(), beta.cpu().numpy())

    # 自定义算子
    y_op = get_op(x, u, gamma, beta)

    # 断言
    rtol = 1e-3 if dtype == torch.float16 else 1e-4
    atol = 1e-4 if dtype == torch.float32 else 1e-3
    np.testing.assert_allclose(y_op, y_golden, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
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
import sysconfig

import pytest

import torch
import torch.nn.functional as F
import torch_npu
import numpy as np

logging.getLogger().setLevel(logging.INFO)
torch.npu.set_device(0)
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
torch.manual_seed(321)
np.random.seed(42)

_PRECISION_ERROR_RANGE = {
    torch.float32: 1e-4,
    torch.int64: 1e-4,
    torch.float16: 1e-3,
    torch.bfloat16: 5e-3,
    torch.int32: 1e-4
}

cpu_device = torch.device("cpu")
npu_device = torch.device("npu")
GRAD_ALLOW_MAX_DIFF = 0.0625


def compare_data_equal(key, npu_tensor, golden_tensor, precision=None, allow_max_diff=None):
    if precision is None:
        precision = _PRECISION_ERROR_RANGE[npu_tensor.dtype]
    if npu_tensor.device != golden_tensor.device:
        npu_tensor = npu_tensor.cpu()
    is_equal = torch.allclose(npu_tensor, golden_tensor, rtol=precision, atol=precision, equal_nan=False)
    if not is_equal:
        logging.error(
            f"Error, key:{key} is not equal in tensor1 and tensor2 by precision:{precision},"
            f" shape1:{npu_tensor.shape}, shape2:{golden_tensor.shape}")
        logging.info("Error detail info, key:{}, tensor1:{}, tensor2:{}".format(key, npu_tensor, golden_tensor))
        if allow_max_diff is not None:
            max_diff = torch.abs(npu_tensor - golden_tensor).max().item()
            if max_diff <= allow_max_diff:
                logging.warning(f"OK, key:{key} is equal with precision:{precision} in condition 'allow_max_diff':"
                                f"{allow_max_diff}, shape:{npu_tensor.shape}")
                return
        raise RuntimeError(f"Error, key:{key} is not equal in tensor1 and tensor2 by precision:{precision},"
                           f" shape1:{npu_tensor.shape}, shape2:{golden_tensor.shape}")
    else:
        logging.info(f"OK, key:{key} is equal with precision:{precision}. shape:{npu_tensor.shape}")


def norm_multiply_dropout_pytorch(x, u, gamma, bias, eps):
    ori_dtype = x.dtype
    x = x.to(torch.float32)
    u = u.to(torch.float32)
    gamma = gamma.to(torch.float32)
    bias = bias.to(torch.float32)

    ln_ret = F.layer_norm(x, normalized_shape=[x.shape[-1]], weight=gamma, bias=bias, eps=eps)
    ln_out = u * ln_ret
    ln_out = F.dropout(
        ln_out,
        p=0.0,
        training=True,
    )
    ln_out = ln_out.to(ori_dtype)

    return ln_out


@pytest.mark.parametrize("dim0", [262144, 4096 * 4])
@pytest.mark.parametrize("dim1", [512])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_norm_multiply_dropout(dim0, dim1, dtype):
    eps = 1e-5
    dropout_ratio = 0.0  # 测试时仅支持dropout_ratio为0
    # 生成数据
    low = -1.0
    high = 1.0

    x_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))
    u_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))
    g_np = np.random.uniform(low=low, high=high, size=(dim1,))
    b_np = np.random.uniform(low=low, high=high, size=(dim1,))
    dy_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))

    # =====  pytorch小算子执行  ====
    golden_device = cpu_device
    x_pt = torch.tensor(x_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    u_pt = torch.tensor(u_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    g_pt = torch.tensor(g_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    b_pt = torch.tensor(b_np, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    y1 = norm_multiply_dropout_pytorch(x_pt, u_pt, g_pt, b_pt, eps)
    dy_pt = torch.tensor(dy_np, dtype=dtype, device=golden_device).contiguous()
    y1.backward(dy_pt)
    torch_npu.npu.synchronize()

    # ===== 融合算子执行 ====
    x_fused = torch.tensor(x_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
    u_fused = torch.tensor(u_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
    g_fused = torch.tensor(g_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
    b_fused = torch.tensor(b_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
    y2, _, _ = torch.ops.mxrec.norm_multiply_dropout(x_fused, u_fused, g_fused, b_fused, eps, dropout_ratio)
    dy_fused = torch.tensor(dy_np, dtype=dtype, device=npu_device).contiguous()
    y2.backward(dy_fused)
    torch_npu.npu.synchronize()

    # 比较前反向结果
    compare_data_equal("y ret", y2.cpu(), y1.cpu())
    compare_data_equal("x grad", x_fused.grad, x_pt.grad)
    compare_data_equal("u grad", u_fused.grad, u_pt.grad)
    compare_data_equal("g grad", g_fused.grad, g_pt.grad, allow_max_diff=GRAD_ALLOW_MAX_DIFF)
    compare_data_equal("b grad", b_fused.grad, b_pt.grad, allow_max_diff=GRAD_ALLOW_MAX_DIFF)
    logging.info(" compare end.")


if __name__ == '__main__':
    test_norm_multiply_dropout(262144, 512, torch.bfloat16)

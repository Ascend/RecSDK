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
import torch
import torch_npu
import pytest

import numpy as np
import torch.nn.functional as F

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(321)
np.random.seed(42)

_PRECISION_ERROR_RANGE = {
    torch.float32: 1e-4,
    torch.int64: 1e-4,
    torch.float16: 1e-3,
    torch.bfloat16: 5e-3,
    torch.int32: 1e-4
}

npu_device = torch.device("npu")

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def compare_data_equal(key, npu_tensor, cpu_tensor, precision=None, is_print=True):
    if precision is None:
        precision = _PRECISION_ERROR_RANGE[npu_tensor.dtype]
    if npu_tensor.device != cpu_tensor.device:
        npu_tensor = npu_tensor.cpu()
    is_equal = torch.allclose(npu_tensor, cpu_tensor, rtol=precision, atol=precision, equal_nan=False)
    if not is_equal:
        logging.info(
            f"Error, key:{key} is not equal in tensor1 and tensor2 by precision:{precision},"
            f" shape1:{npu_tensor.shape}, shape2:{cpu_tensor.shape}")
        logging.info("Error detail info, key:{}, tensor1:{}, tensor2:{}".format(key, npu_tensor, cpu_tensor))
        raise RuntimeError(f"Error, key:{key} is not equal in tensor1 and tensor2 by precision:{precision},"
                           f" shape1:{npu_tensor.shape}, shape2:{cpu_tensor.shape}")
    else:
        logging.info(f"OK, key:{key} is equal with precision:{precision}. shape:{npu_tensor.shape}")


def linear_silu_split_by_pytorch(x_pt, uvqk_pt, split_args):
    mm_ret = torch.mm(x_pt, uvqk_pt)
    qkvu = F.silu(mm_ret)
    u, v, q, k = torch.split(qkvu, split_args, dim=-1)
    return u, v, q, k


@pytest.mark.parametrize("m", [262144, 8192])
@pytest.mark.parametrize("n", [128, 64])
@pytest.mark.parametrize("k", [4096, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_linear_silu_split_fused_op(m, n, k, dtype):
    low = -1.0
    high = 1.0
    x = np.random.uniform(low=low, high=high, size=(m, n))  # [262144, 128] [m, n]
    uvqk = np.random.uniform(low=low, high=high, size=(n, k))  # [128, 4096]  [n, k]
    split_args = [uvqk.shape[-1] // 4] * 4  # [k // 4] * 4
    logging.info("m:%d, n:%d, k:%d, x shape:%s, uvqk shape:%s, dtype:%s",
                 m, n, k, x.shape, uvqk.shape, dtype)

    # 1 pytorch小算子调用， 前向加反向
    golden_device = npu_device
    x_pt = torch.tensor(x, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    uvqk_pt = torch.tensor(uvqk, dtype=dtype, device=golden_device).contiguous().requires_grad_(True)
    u_golden, v_golden, q_golden, k_golden = linear_silu_split_by_pytorch(x_pt, uvqk_pt, split_args)
    logging.info("pytorch exec, u_golden shape: %s", u_golden.shape)
    loss_pt = torch.sum(torch.concat([q_golden, k_golden, v_golden, u_golden], dim=-1))
    loss_pt.backward()
    torch_npu.npu.synchronize()
    x_pt_grad = x_pt.grad.to(torch.device("cpu"))
    uvqk_pt_grad = uvqk_pt.grad.to(torch.device("cpu"))
    torch_npu.npu.synchronize()

    # 2 fused op 调用 前反向
    x_fused = torch.tensor(x, dtype=dtype, device=torch.device("npu")).contiguous().requires_grad_(True)
    uvqk_fused = torch.tensor(uvqk, dtype=dtype, device=torch.device("npu")).contiguous().requires_grad_(True)
    weight_t = uvqk_fused.T
    bias_fused = (torch.full((k,), 0.0, device=x_fused.device, dtype=torch.float32)
                  .contiguous().requires_grad_())
    logging.info("=== x_fused.shape:%s, uvqk_fused.shape:%s", x_fused.shape, uvqk_fused.shape)
    # 融合算子接收转置后的输入
    result = torch.ops.mxrec.linear_silu_split_uvqk(x_fused, weight_t, bias_fused, split_args)
    # 融合算子返回结果 user, value, query, key, linear_output
    u_fused = result[0]
    v_fused = result[1]
    q_fused = result[2]
    k_fused = result[3]
    logging.info("fused op exec, u_golden shape:%s", u_fused.shape)
    loss_fused = torch.sum(torch.concat([q_fused, k_fused, v_fused, u_fused], dim=-1))
    loss_fused.backward()
    x_fused_grad = x_fused.grad.to(torch.device("cpu"))
    uvqk_fused_grad = uvqk_fused.grad.to(torch.device("cpu"))
    torch_npu.npu.synchronize()

    # 比较前向结果
    compare_data_equal(" forward ret, q_golden", q_fused, q_golden)
    compare_data_equal(" forward ret, k", k_fused, k_golden)
    compare_data_equal(" forward ret, v_golden", v_fused, v_golden)
    compare_data_equal(" forward ret, u_golden", u_fused, u_golden)

    # 比较反向结果
    compare_data_equal(" backward ret, x grad", x_fused_grad.reshape(x_pt_grad.shape), x_pt_grad)
    compare_data_equal(" backward ret, uvqk grad", uvqk_fused_grad.reshape(uvqk_pt_grad.shape), uvqk_pt_grad)

    logging.info("test fused op end.")


if __name__ == "__main__":
    test_linear_silu_split_fused_op(262144, 128, 4096, torch.bfloat16)

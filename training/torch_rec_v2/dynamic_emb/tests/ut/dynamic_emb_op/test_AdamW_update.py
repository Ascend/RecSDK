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

import pytest
import torch

from dynamic_emb_extensions import (
    DynamicEmbDataType,
    dynamic_emb_AdamW_with_pointer,
)


class OptimizerParams:
    '''优化器参数
        lr: 学习率
        beta1: 一阶衰减系数
        beta2: 二阶衰减系数
        eps: 小常数，防止除零
        weight_decay: 权重衰减系数
    '''
    def __init__(self, lr, beta1, beta2, eps, weight_decay):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay


@pytest.fixture
def optimizer_params(lr, beta1, beta2, eps, weight_decay):
    return OptimizerParams(
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算每行首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    # 取出二维张量的维度
    n, m = x_2d.size()
    # 计算每个元素的字节大小
    elem_size = x_2d.element_size()
    # 计算每行的字节偏移量
    row_stride = m * elem_size
    # 计算指针数组
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", [1, 1024, 4096, 8192, 10240, 102400])
@pytest.mark.parametrize("embedding_dim", [8, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("weight_decay", [0.0, 0.001, 0.01, 0.1])
@pytest.mark.parametrize("iter_num", [10, 100, 1000, 10000])
def test_dynamic_emb_AdamW_with_pointer(device, batch_size, embedding_dim, optimizer_params, iter_num): 
    """
    1. 使用 PyTorch 模拟生成符合真实分布的 m, v, params。
    2. 将状态同步给 Custom 算子。
    3. 执行单步更新对比。
    """
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = optimizer_params.lr
    beta1 = optimizer_params.beta1
    beta2 = optimizer_params.beta2
    eps = optimizer_params.eps
    weight_decay = optimizer_params.weight_decay

    params_init = torch.randn(batch_size, embedding_dim, device=f'npu:{device}')
    param_torch = torch.nn.Parameter(params_init.clone())
    optimizer_torch = torch.optim.AdamW([param_torch], 
                                        lr=lr, 
                                        betas=(beta1, beta2), 
                                        eps=eps, 
                                        weight_decay=weight_decay)
    for _ in range(iter_num - 1):
        # 模拟随机梯度
        dummy_grad = torch.randn_like(param_torch)
        optimizer_torch.zero_grad()
        param_torch.grad = dummy_grad
        optimizer_torch.step()
    # 提取参数
    current_params = param_torch.data.clone()
    # 提取优化器状态 (m, v); 如果是第0步，state可能是空的，需要手动初始化为0
    if len(optimizer_torch.state) == 0:
        m_init = torch.zeros_like(current_params)
        v_init = torch.zeros_like(current_params)
    else:
        state = optimizer_torch.state[param_torch]
        m_init = state['exp_avg'].clone()
        v_init = state['exp_avg_sq'].clone()

    values = torch.cat([current_params, m_init, v_init], dim=1).contiguous()
    val_pointers = get_dim_pointers_optimized(values)

    # 生成一个新的测试梯度
    test_grad = torch.randn_like(current_params)
    
    dynamic_emb_AdamW_with_pointer(
        test_grad, 
        val_pointers, 
        DynamicEmbDataType.Float32, 
        embedding_dim * 2,
        lr, beta1, beta2, eps, weight_decay, 
        iter_num
    )
    torch.npu.synchronize()

    optimizer_torch.zero_grad()
    param_torch.grad = test_grad.clone()
    optimizer_torch.step()
    torch.npu.synchronize()
    
    # 提取 Custom 结果
    custom_param_result = values[:, :embedding_dim]
    # 提取 PyTorch 结果
    torch_param_result = param_torch.data

    torch.testing.assert_close(custom_param_result, torch_param_result, rtol=1e-5, atol=1e-5)

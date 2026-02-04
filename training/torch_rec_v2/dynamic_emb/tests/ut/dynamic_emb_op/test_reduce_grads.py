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
import logging
import random

import pytest
import torch

import dynamic_emb_extensions


# 计算预期值
def reduce_grads(keys, grads):
    # unqiue
    unique_key, inverse, _ = torch.unique(keys, sorted=False, return_inverse=True, return_counts=True)

    # reduce
    tmp = torch.zeros(unique_key.size(0), grads.size(1))
    unique_grad = tmp.scatter_reduce(0, inverse.unsqueeze(1).expand_as(grads), grads, reduce='sum')

    return unique_key, unique_grad, inverse


@pytest.mark.parametrize('device', ['npu:0'])
def test_reduce_grads_case1(device):
    torch.npu.set_device(0)

    # 构造数据
    k2v = {1: [1.0, 1.1, 1.2], 2: [2.0, 2.2, 2.3], 3: [3.3, 3.4, 3.5]}
    ks = [1, 2, 2, 3, 3, 3]
    vs = [k2v.get(i, [0, 0, 0]) for i in ks]

    org_key = torch.tensor(ks, dtype=torch.int64)
    org_grad = torch.tensor(vs, dtype=torch.float32)

    # 构造前向结果
    exp_unique_key, exp_unique_grad, inverse = reduce_grads(org_key, org_grad)

    org_grad_npu = org_grad.to(device)
    inverse_npu = inverse.to(device)
    exp_unique_key_npu = exp_unique_key.to(device)

    ret_unique_key, ret_unique_grad = dynamic_emb_extensions.reduce_grads(org_grad_npu, exp_unique_key_npu, inverse_npu)
    ret_unique_key = ret_unique_key.to('cpu')
    ret_unique_grad = ret_unique_grad.to('cpu')

    # 校验
    assert torch.equal(ret_unique_key, exp_unique_key)
    assert torch.allclose(ret_unique_grad, exp_unique_grad)


@pytest.mark.parametrize('device', ['npu:0'])
def test_reduce_grads_case2(device):
    torch.npu.set_device(0)

    # 构造数据
    dim = 16
    k2v = {}
    valid_keys = [1, 2, 3]
    for k in valid_keys:
        k2v[k] = [random.random() for i in range(dim)]
    ks = [1, 2, 2, 3, 3, 3]
    vs = [k2v.get(i, [0, 0, 0]) for i in ks]

    org_key = torch.tensor(ks, dtype=torch.int64)
    org_grad = torch.tensor(vs, dtype=torch.float32)

    # 构造前向结果
    exp_unique_key, exp_unique_grad, inverse = reduce_grads(org_key, org_grad)

    org_grad_npu = org_grad.to(device)
    inverse_npu = inverse.to(device)
    exp_unique_key_npu = exp_unique_key.to(device)

    ret_unique_key, ret_unique_grad = dynamic_emb_extensions.reduce_grads(org_grad_npu, exp_unique_key_npu, inverse_npu)
    ret_unique_key = ret_unique_key.to('cpu')
    ret_unique_grad = ret_unique_grad.to('cpu')

    # 校验
    assert torch.equal(ret_unique_key, exp_unique_key)
    assert torch.allclose(ret_unique_grad, exp_unique_grad)


# 随机值
@pytest.mark.parametrize('device', ['npu:0'])
@pytest.mark.parametrize('length', [10, 100, 1000])
@pytest.mark.parametrize('dim', [8, 128])
def test_reduce_grads_case3(length, device, dim):
    torch.npu.set_device(0)

    k2v = {}
    valid_keys = [1, 2, 3]
    for k in valid_keys:
        k2v[k] = [random.random() for i in range(dim)]

    ks = [random.choice(valid_keys) for i in range(length)]
    vs = [k2v.get(i, [0, 0, 0]) for i in ks]

    org_key = torch.tensor(ks, dtype=torch.int64)
    org_grad = torch.tensor(vs, dtype=torch.float32)

    # 构造前向结果
    exp_unique_key, exp_unique_grad, inverse = reduce_grads(org_key, org_grad)

    org_grad_npu = org_grad.to(device)
    inverse_npu = inverse.to(device)
    exp_unique_key_npu = exp_unique_key.to(device)

    ret_unique_key, ret_unique_grad = dynamic_emb_extensions.reduce_grads(org_grad_npu, exp_unique_key_npu, inverse_npu)
    ret_unique_key = ret_unique_key.to('cpu')
    ret_unique_grad = ret_unique_grad.to('cpu')

    # 校验
    assert torch.equal(ret_unique_key, exp_unique_key)
    assert torch.allclose(ret_unique_grad, exp_unique_grad)
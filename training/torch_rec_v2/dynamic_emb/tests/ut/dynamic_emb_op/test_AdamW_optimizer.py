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
"""AdamW dynamic embedding optimizer update tests."""

import pytest
import torch

from optimizer_update_test_utils import (
    OptimizerParams,
    run_adamw_fused_hybrid_test,
    run_adamw_fused_test,
    run_adamw_pointer_hybrid_test,
    run_adamw_pointer_test,
    run_adamw_table_test,
)

# ---------------------------------------------------------------------------
# AdamW parametrization
# ---------------------------------------------------------------------------
_ADAMW_POINTER_BATCH = [1, 1024, 4096, 8192, 10240, 102400]
_ADAMW_POINTER_DIM = [8, 64, 128, 256, 512, 1024, 31, 1023]
_ADAMW_TABLE_BATCH = [1, 1024, 4096, 102400]
_ADAMW_TABLE_DIM = [64, 128, 31, 1023]
_ADAMW_FUSED_BATCH = [1, 1024, 4096, 8192, 102400]
_ADAMW_FUSED_DIM = [8, 64, 128, 256, 31, 1023]
_ADAMW_EPS_FP32_BF16 = [1e-8, 0.1]
_ADAMW_EPS_FP16 = [1e-4, 0.1]
_ADAMW_GRAD_TYPE_EPS = [
    *((grad_type, eps) for grad_type in (torch.float32, torch.bfloat16) for eps in _ADAMW_EPS_FP32_BF16),
    *((grad_type, eps) for grad_type in (torch.float16,) for eps in _ADAMW_EPS_FP16),
]
_ADAMW_HYBRID_GRAD_TYPE_EPS = [(torch.float32, eps) for eps in _ADAMW_EPS_FP32_BF16]


@pytest.fixture(name="optimizer_params")
def _optimizer_params(lr, beta1, beta2, eps, weight_decay):
    return OptimizerParams(
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", _ADAMW_POINTER_BATCH)
@pytest.mark.parametrize("embedding_dim", _ADAMW_POINTER_DIM)
@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("weight_decay", [0.0, 0.001, 0.01, 0.1])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(("grad_type", "eps"), _ADAMW_GRAD_TYPE_EPS)
def test_dynamic_emb_AdamW_with_pointer(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type):
    run_adamw_pointer_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", _ADAMW_TABLE_BATCH)
@pytest.mark.parametrize("embedding_dim", _ADAMW_TABLE_DIM)
@pytest.mark.parametrize("lr", [0.001, 0.01])
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(("grad_type", "eps"), _ADAMW_GRAD_TYPE_EPS)
def test_dynamic_emb_AdamW_with_table(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type):
    run_adamw_table_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", _ADAMW_FUSED_BATCH)
@pytest.mark.parametrize("embedding_dim", _ADAMW_FUSED_DIM)
@pytest.mark.parametrize("lr", [0.001, 0.01])
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(("grad_type", "eps"), _ADAMW_GRAD_TYPE_EPS)
def test_dynamic_emb_AdamW_fused(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type):
    run_adamw_fused_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", _ADAMW_POINTER_BATCH)
@pytest.mark.parametrize("embedding_dim", _ADAMW_POINTER_DIM)
@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("weight_decay", [0.0, 0.001, 0.01, 0.1])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(("grad_type", "eps"), _ADAMW_HYBRID_GRAD_TYPE_EPS)
def test_dynamic_emb_AdamW_with_pointer_hybrid(
    device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type
):
    run_adamw_pointer_hybrid_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", _ADAMW_FUSED_BATCH)
@pytest.mark.parametrize("embedding_dim", _ADAMW_FUSED_DIM)
@pytest.mark.parametrize("lr", [0.001, 0.01])
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(("grad_type", "eps"), _ADAMW_HYBRID_GRAD_TYPE_EPS)
def test_dynamic_emb_AdamW_fused_hybrid(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type):
    run_adamw_fused_hybrid_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type)

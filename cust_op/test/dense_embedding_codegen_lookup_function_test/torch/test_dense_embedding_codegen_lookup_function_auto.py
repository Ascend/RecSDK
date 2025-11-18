#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
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
import os
from math import log
from multiprocessing import pool
import random
import sysconfig
import torch
import pytest
import torch.nn as nn
import fbgemm_gpu
from torchrec.distributed.batched_embedding_kernel import DenseTableBatchedEmbeddingBagsCodegen
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
# 确保日志配置使用英文输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s:%(lineno)d in %(funcName)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE_ID = "npu:0"
RANDOM_SEED = 42
RUN_ITERS = 100

# 加载NPU库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def set_random_seed(seed):
    """
    设置随机数种子，保证随机的可重复性
    
    Args:
        seed (int): 随机数种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 为了确保在不同硬件和PyTorch版本上的一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_and_weights(num_embeddings, dim, indices_num):
    indices = torch.randint(0, num_embeddings, (indices_num,))
    offsets = torch.tensor([0, indices_num])
    weights = torch.randn(num_embeddings, dim)
    logging.info(f"Data prepared: indices={indices.shape}, offsets={offsets.shape}, weights={weights.shape}")
    return indices, offsets, weights


def compute_batched_embedding_model(indices, offsets, weights, device):
    num_embeddings, embedding_dim = weights.shape
    embedding_table = DenseTableBatchedEmbeddingBagsCodegen(
        [(num_embeddings, embedding_dim)],
        use_cpu=False,
        pooling_mode=PoolingMode.NONE,
    )
    embedding_table.weights.data.copy_(weights.reshape(-1))
    embedding_table.weights.requires_grad = True
    embedding_table.weights.retain_grad()
    logger.info(f"Device: {device}")
    embedding_table = embedding_table.to(device)
    weights = weights.to(device)
    indices = indices.to(device)
    offsets = offsets.to(device)
    torch.npu.synchronize()

    run_iters = RUN_ITERS
    for _ in range(run_iters):
        fwd_out = embedding_table(indices, offsets)
        loss = fwd_out.sum()
        loss.backward()
        torch.npu.synchronize()

    return loss, embedding_table.weights.grad.view(-1, embedding_dim)


def compute_nn_embedding_model(indices, weights, device):
    num_embeddings, embedding_dim = weights.shape
    embedding_table = nn.Embedding(num_embeddings, embedding_dim)
    embedding_table.weight.data.copy_(weights.reshape(-1, embedding_dim))
    embedding_table.weight.requires_grad = True
    embedding_table.weight.retain_grad()
    logger.info(f"Device: {device}")
    embedding_table = embedding_table.to(device)
    weights = weights.to(device)
    indices = indices.to(device)
    torch.npu.synchronize()

    run_iters = RUN_ITERS
    for _ in range(run_iters):
        fwd_out = embedding_table(indices)
        loss = fwd_out.sum()
        loss.backward()
        torch.npu.synchronize()

    return loss, embedding_table.weight.grad.view(-1, embedding_dim)


def compare_outputs(cpu_output, npu_output, name):
    # 验证输出形状一致
    if cpu_output.shape != npu_output.shape:
        raise AssertionError(f"{name} shape mismatch: CPU {cpu_output.shape} vs NPU {npu_output.shape}")

    # 验证输出结果一致
    if not torch.allclose(cpu_output.cpu(), npu_output.cpu(), rtol=1e-4, atol=1e-4):
        raise AssertionError(f"{name} CPU and NPU outputs are inconsistent")
    
    return True


@pytest.mark.parametrize("num_embeddings", [51, 10001])
@pytest.mark.parametrize("embedding_dim", [16, 128])
@pytest.mark.parametrize("indices_num", [1000, 5000])
def test_dense_embedding_codegen_lookup_function_auto(num_embeddings, embedding_dim, indices_num):
    set_random_seed(RANDOM_SEED)
    indices, offsets, weights = prepare_data_and_weights(num_embeddings, embedding_dim, indices_num)

    loss, grad = compute_batched_embedding_model(indices, offsets, weights, DEVICE_ID)
    loss_gold, grad_gold = compute_nn_embedding_model(indices, weights, "cpu")
 
    assert compare_outputs(loss, loss_gold, "loss")
    assert compare_outputs(grad, grad_gold, "grad")

    logging.info("test passed")

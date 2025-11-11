#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import os
import random
import sysconfig
from collections import namedtuple

import fbgemm_gpu
import numpy as np
import pytest
import torch

# 定义反向传播参数命名元组
BackwardParams = namedtuple('BackwardParams', [
    'weights', 'weights_offsets', 'indices', 'offsets', 'max_d', 'grad_output'
])

# 配置日志格式，包含文件名、函数名和行号
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s:%(lineno)d in %(funcName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE_ID = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def backward_cpu(params):
    """
    CPU反向传播实现 - 直接实现反向逻辑，不依赖前向过程

    Args:
        params: BackwardParams命名元组，包含以下字段:
            weights: dev_weights tensor [num_embeddings, embedding_dim]
            weights_offsets: weights_offsets tensor [num_tables] 或 None（单表）
            indices: indices tensor [total_indices]
            offsets: offsets tensor [num_tables * batch_size + 1]
            max_d: maximum embedding dimension
            grad_output: gradient from upper layer [total_indices, max_d]

    Returns:
        gradient tensor with same shape as weights
    """
    # 解包参数
    weights, weights_offsets, indices, offsets, max_d, grad_output = params
    
    # 创建梯度张量，初始化为0，与NPU版本保持一致
    grad_weights = torch.zeros_like(weights)

    # 获取嵌入维度
    embedding_dim = weights.size(1) if weights.dim() > 1 else 1

    # 计算表数量
    num_tables = len(weights_offsets) if weights_offsets is not None else 1

    # 处理每个表
    for i in range(num_tables):
        # 计算当前表的索引范围
        indices_start = int(offsets[i].item()) if hasattr(offsets[i], "item") else int(offsets[i])
        indices_end = int(offsets[i + 1].item()) if hasattr(offsets[i + 1], "item") else int(offsets[i + 1])

        # 获取当前表的权重偏移量，需要除以嵌入维度得到实际的索引偏移
        weight_offset = int(weights_offsets[i].item()) // embedding_dim if weights_offsets is not None else 0

        # 计算当前表的大小（以嵌入向量为单位）
        if weights_offsets is not None:
            if i < num_tables - 1:
                table_size = (int(weights_offsets[i + 1].item()) - int(weights_offsets[i].item())) // embedding_dim
            else:
                table_size = (weights.size(0) * embedding_dim - int(weights_offsets[i].item())) // embedding_dim
        else:
            table_size = weights.size(0)

        # 处理当前表的所有索引
        for j in range(indices_start, indices_end):
            if j < len(indices):
                idx = indices[j]
                actual_idx = int(idx.item()) + weight_offset if hasattr(idx, "item") else int(idx) + weight_offset

                # 检查索引是否超出总权重范围
                if actual_idx >= weights.size(0):
                    logging.warning(f"索引 {actual_idx} 超出权重张量范围 {weights.size(0)}")

                # 检查索引是否超出当前表范围
                raw_idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
                if raw_idx >= table_size:
                    logging.warning(f"索引 {raw_idx} 超出表 {i} 的范围 {table_size}")

                # 将grad_output的梯度累加到对应位置的grad_weights中
                if actual_idx < weights.size(0):
                    grad_weights[actual_idx, :max_d] += grad_output[j, :]

    return grad_weights


def backward_npu(params):
    """
    NPU反向传播实现 - 直接调用反向算子，不调用前向过程

    Args:
        params: BackwardParams命名元组，包含以下字段:
            weights: dev_weights tensor [num_embeddings, embedding_dim]
            weights_offsets: weights_offsets tensor [num_tables] 或 None（单表）
            indices: indices tensor [total_indices]
            offsets: offsets tensor [num_tables * batch_size + 1]
            max_d: maximum embedding dimension
            grad_output: gradient from upper layer [total_indices, max_d]

    Returns:
        gradient tensor with same shape as weights
    """
    # 解包参数
    weights, weights_offsets, indices, offsets, max_d, grad_output = params
    
    torch.npu.set_device(DEVICE_ID)

    # Move tensors to NPU
    weights_npu = weights.to(DEVICE_ID)
    indices_npu = indices.to(DEVICE_ID)
    offsets_npu = offsets.to(DEVICE_ID)
    grad_output_npu = grad_output.to(DEVICE_ID)

    # 处理权重偏移量
    if weights_offsets is not None:
        weights_offsets_npu = weights_offsets.to(DEVICE_ID)
    else:
        # 单表情况，创建默认偏移量
        weights_offsets_npu = torch.tensor([0], dtype=torch.int64).to(DEVICE_ID)

    dev_weights = weights_npu.contiguous().view(-1)

    # 直接调用反向传播函数，只传递实际需要的核心参数
    result = torch.ops.mxrec.dense_embedding_codegen_lookup_function_grad(
        devWeights=dev_weights,
        grad=grad_output_npu,
        weightsOffsets=weights_offsets_npu,
        dOffsets=torch.tensor([0, max_d], dtype=torch.int32).to(DEVICE_ID),
        totalD=max_d,
        maxD=max_d,
        hashSizeCumsum=torch.tensor([0, weights.size(0)], dtype=indices.dtype).to(DEVICE_ID),
        totalHashSizeBits=0,
        indices=indices_npu,
        offsets=offsets_npu,
        poolingMode=0,
        indiceWeightsOptional=None,
        featureRequiresGrad=None,
        outputDtypeOptional=0,
        bOffsetOptional=None,
        vbeOutputOffsetsFeatureRankOptional=None,
        vbeBOffsetsRankPerFeatureOptional=None,
        maxB=0,
        maxBFeatureRank=0,
        vbeOutputSize=0,
    )

    # 返回梯度，需要恢复为原始形状
    # 注意：反向传播算子返回的是一个tensor_list，其中第一个元素是梯度累计结果
    return result[0].view_as(weights).detach().cpu()


def generate_test_data(num_embeddings, embedding_dim, batch_size, max_seq_len, num_tables=1):
    """
    生成测试数据

    Args:
        num_embeddings: 总的嵌入向量数量
        embedding_dim: 嵌入维度
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        num_tables: 表数量(默认为1表示单表)

    Returns:
        weights, weights_offsets, indices, offsets, max_d
    """
    # 创建权重矩阵
    weights = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32)

    # 初始化索引和偏移量
    indices_list = []
    offsets = [0]

    total_tables = max(1, num_tables)

    # 为每个表和每个批次生成数据
    for table_id in range(total_tables):
        # 计算当前表的索引范围
        table_start = (table_id * num_embeddings) // total_tables
        table_end = ((table_id + 1) * num_embeddings) // total_tables
        table_end = min(table_end, num_embeddings)

        # 确保索引不会超过当前表的范围
        table_size = table_end - table_start
        if table_size <= 0:
            table_size = max(1, num_embeddings // total_tables)

        # 确保table_size不超过实际可用的嵌入数量
        table_size = min(table_size, num_embeddings - table_start)

        table_total_indices = 0
        # 为当前表的每个批次生成索引
        for batch_idx in range(batch_size):
            seq_len = min(max_seq_len, max(1, random.randint(1, max_seq_len)))
            # 在当前表的范围内生成随机索引，确保不超过表的大小
            # indices表示的就是表内索引，不是全局索引
            if table_size > 0:
                # 保证索引在当前表的范围内[0, table_size)
                batch_indices = torch.randint(0, max(1, table_size), (seq_len,), dtype=torch.int64)
                indices_list.append(batch_indices)
                table_total_indices += len(batch_indices)

        # 更新偏移量 - 每个表一个偏移量值
        offsets.append(offsets[-1] + table_total_indices)

    # 合并所有表的索引
    indices = torch.cat(indices_list) if indices_list else torch.tensor([], dtype=torch.int64)
    offsets = torch.tensor(offsets, dtype=torch.int64)

    # 创建权重偏移量 - 以元素为单位而不是以嵌入向量为单位
    weights_offsets = torch.tensor(
        [(i * num_embeddings) // num_tables * embedding_dim for i in range(num_tables)],
        dtype=torch.int64,
    )

    # 如果有多个表，需要调整最终的weights大小以适应所有表
    if num_tables > 1:
        final_num_embeddings = max([(i + 1) * num_embeddings // num_tables for i in range(num_tables)])
        if final_num_embeddings < num_embeddings:
            # 扩展weights张量以包含所有可能的索引
            padding_size = num_embeddings - final_num_embeddings
            padding = torch.zeros(padding_size, embedding_dim, dtype=weights.dtype)
            weights = torch.cat([weights, padding], dim=0)

    max_d = embedding_dim

    return weights, weights_offsets, indices, offsets, max_d


@pytest.mark.parametrize("num_embeddings", [51, 10001, 100001])
@pytest.mark.parametrize("embedding_dim", [16, 128])
@pytest.mark.parametrize("batch_size", [1, 8, 128])
@pytest.mark.parametrize("max_seq_len", [50, 200])
@pytest.mark.parametrize("num_tables", [1, 5])
def test_dense_embedding_codegen_lookup_function_backward(
    num_embeddings, embedding_dim, batch_size, max_seq_len, num_tables
):
    """
    参数化测试反向传播函数
    """
    # 生成测试数据
    weights, weights_offsets, indices, offsets, max_d = generate_test_data(
        num_embeddings, embedding_dim, batch_size, max_seq_len, num_tables
    )

    # 生成grad_output，确保它是2D张量
    total_indices = indices.size(0)
    grad_output = torch.randn(total_indices, max_d, dtype=torch.float32)

    # 打印测试信息
    logger.info(
        f"测试参数: num_embeddings={num_embeddings}, embedding_dim={embedding_dim}, "
        f"batch_size={batch_size}, max_seq_len={max_seq_len}, num_tables={num_tables}"
    )

    try:
        # 使用具名元组封装参数
        cpu_params = BackwardParams(
            weights=weights,
            weights_offsets=weights_offsets,
            indices=indices,
            offsets=offsets,
            max_d=max_d,
            grad_output=grad_output
        )
        
        # 调用CPU版本
        cpu_grad = backward_cpu(cpu_params)

        # 调用NPU版本
        npu_grad = backward_npu(cpu_params)

        # 验证输出形状一致
        if cpu_grad.shape != npu_grad.shape:
            logger.error(f"输出形状不匹配: CPU {cpu_grad.shape} vs NPU {npu_grad.shape}")
            raise AssertionError(f"输出形状不匹配: CPU {cpu_grad.shape} vs NPU {npu_grad.shape}")

        # 验证输出结果一致
        if not torch.allclose(cpu_grad, npu_grad, rtol=1e-4, atol=1e-4):
            logger.error("CPU和NPU输出不一致")
            raise AssertionError("CPU和NPU输出不一致")

        logger.info("✓ 参数化测试通过")
    except Exception as e:
        # 保存失败的测试数据
        raise e

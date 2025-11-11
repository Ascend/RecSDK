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
import random
import sysconfig
import torch
import pytest
import fbgemm_gpu

logging.getLogger().setLevel(logging.INFO)
DEVICE_ID = "npu:0"

# 加载NPU库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def lookup_cpu(weights, weights_offsets, indices, offsets, max_d):
    """
    CPU实现用于对比测试 - 与NPU版本保持参数一致

    Args:
        weights: dev_weights tensor [num_embeddings, embedding_dim]
        weights_offsets: weights_offsets tensor [num_tables] 或 None（单表）
        indices: indices tensor [total_indices]
        offsets: offsets tensor [num_tables + 1]
        max_d: maximum embedding dimension

    Returns:
        output tensor [total_indices, max_d]
    """
    # Validate inputs
    if not isinstance(weights, torch.Tensor) or weights.dim() != 2:
        raise ValueError("weights must be a 2D tensor")

    if not isinstance(indices, torch.Tensor) or indices.dim() != 1:
        raise ValueError("indices must be a 1D tensor")

    if not isinstance(offsets, torch.Tensor) or offsets.dim() != 1:
        raise ValueError("offsets must be a 1D tensor")

    if len(offsets) <= 1:
        raise ValueError("offsets must have at least 2 elements")

    # Check index bounds
    if indices.numel() > 0 and (indices.min() < 0 or indices.max() >= weights.size(0)):
        raise IndexError("indices out of bounds")

    # 计算表数量
    if weights_offsets is not None:
        num_tables = len(weights_offsets)
        # 确保weights_offsets是tensor且可索引
        if not isinstance(weights_offsets, torch.Tensor):
            raise ValueError("weights_offsets must be a tensor when provided")
    else:
        num_tables = 1

    # 验证offsets长度
    if len(offsets) != num_tables + 1:
        raise ValueError(f"offsets长度应为{num_tables + 1}，实际为{len(offsets)}")

    # 获取嵌入维度
    embedding_dim = weights.size(1) if weights.dim() > 1 else 1

    result = []

    # 统一处理单表和多表情况
    for i in range(num_tables):
        # 计算当前表的索引范围 - 直接使用offsets[i]到offsets[i+1]
        indices_start = int(offsets[i].item()) if hasattr(offsets[i], "item") else int(offsets[i])
        indices_end = int(offsets[i + 1].item()) if hasattr(offsets[i + 1], "item") else int(offsets[i + 1])

        # 获取当前表的权重偏移量，需要除以嵌入维度得到实际的索引偏移
        weight_offset = int(weights_offsets[i].item()) // embedding_dim if weights_offsets is not None else 0

        # 处理当前表的所有索引
        for j in range(indices_start, indices_end):
            if j < len(indices):
                idx = indices[j]
                actual_idx = int(idx.item()) + weight_offset if hasattr(idx, "item") else int(idx) + weight_offset

                # 确保索引在权重张量范围内
                if actual_idx < weights.size(0):
                    embedding = weights[actual_idx]
                    # 确保维度正确
                    if embedding.size(0) < max_d:
                        padded = torch.zeros(max_d, dtype=embedding.dtype)
                        padded[: embedding.size(0)] = embedding
                        result.append(padded)
                    elif embedding.size(0) > max_d:
                        result.append(embedding[:max_d])
                    else:
                        result.append(embedding)
                else:
                    logging.warning(f"索引{idx}超出权重范围")
                    result.append(torch.zeros(max_d, dtype=weights.dtype))

    return torch.stack(result, dim=0) if result else torch.empty(0, max_d, dtype=weights.dtype)


def lookup_npu(weights, weights_offsets, indices, offsets, max_d):
    """
    NPU实现使用自定义算子

    Args:
        weights: dev_weights tensor [num_embeddings, embedding_dim]
        weights_offsets: weights_offsets tensor [num_tables] 或 None（单表）
        indices: indices tensor [total_indices]
        offsets: offsets tensor [num_tables + 1]
        max_d: maximum embedding dimension

    Returns:
        output tensor [total_indices, max_d]
    """
    torch.npu.set_device(DEVICE_ID)

    # Move tensors to NPU
    weights_npu = weights.to(DEVICE_ID)
    indices_npu = indices.to(DEVICE_ID)
    offsets_npu = offsets.to(DEVICE_ID)

    # 处理权重偏移量
    if weights_offsets is not None:
        # 检查weights_offsets的数据类型是否为int64
        if weights_offsets.dtype != torch.int64:
            raise TypeError(f"weights_offsets的数据类型必须是torch.int64，但得到的是{weights_offsets.dtype}")
        weights_offsets_npu = weights_offsets.to(DEVICE_ID)
    else:
        # 单表情况，创建默认偏移量
        weights_offsets_npu = torch.tensor([0], dtype=torch.int64).to(DEVICE_ID)

    # Prepare parameters for the custom operator
    dev_weights = weights_npu.contiguous().view(-1)
    d_offsets = torch.tensor([0, max_d], dtype=torch.int32).to(DEVICE_ID)
    total_d = max_d
    max_d_param = max_d
    hash_size_cumsum = torch.tensor([0, weights.size(0)], dtype=indices.dtype).to(DEVICE_ID)
    total_hash_size_bits = 0
    pooling_mode = 2  # No pooling
    output_dtype_optional = 0

    # Call the custom operator with fbgemm namespace - only pass used parameters
    output = torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
        dev_weights,
        weights_offsets_npu,
        d_offsets,
        total_d,
        max_d_param,
        hash_size_cumsum,
        total_hash_size_bits,
        indices_npu,
        offsets_npu,
        pooling_mode,
        None,
        None,
        output_dtype_optional,
        None,
        None,
        None,
        0,
        0,
        0,
    )

    return output.detach().cpu()


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
@pytest.mark.parametrize("max_seq_len", [50, 1000])
@pytest.mark.parametrize("num_tables", [1, 5])
def test_dense_embedding_codegen_lookup_function_forward(
    num_embeddings, embedding_dim, batch_size, max_seq_len, num_tables
):
    """
    参数化测试正向传播函数
    """
    # 生成测试数据
    weights, weights_offsets, indices, offsets, max_d = generate_test_data(
        num_embeddings, embedding_dim, batch_size, max_seq_len, num_tables
    )

    logging.info(
        f"测试参数: num_embeddings={num_embeddings}, embedding_dim={embedding_dim}, "
        f"batch_size={batch_size}, max_seq_len={max_seq_len}, num_tables={num_tables}"
    )

    # 调用CPU版本
    cpu_output = lookup_cpu(weights, weights_offsets, indices, offsets, max_d)

    # 调用NPU版本
    npu_output = lookup_npu(weights, weights_offsets, indices, offsets, max_d)

    # 验证输出形状一致
    if cpu_output.shape != npu_output.shape:
        raise AssertionError(f"输出形状不匹配: CPU {cpu_output.shape} vs NPU {npu_output.shape}")

    # 验证输出结果一致
    if not torch.allclose(cpu_output, npu_output, rtol=1e-4, atol=1e-4):
        raise AssertionError("CPU和NPU输出不一致")

    logging.info("✓ 参数化测试通过")

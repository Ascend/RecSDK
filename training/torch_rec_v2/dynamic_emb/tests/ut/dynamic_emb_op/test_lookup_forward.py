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

import time
import pytest
import torch
import dynamic_emb_extensions
import torch.nn.functional as F


class EmbeddingTablePooler:
    """
    嵌入表池化器，用于对嵌入向量进行SUM/MEAN池化，并重新排列输出

    Attributes:
        combiner: 池化方式，0=SUM，1=MEAN
        total_dims: 输出张量的总维度
        accum_dims: 池化结果开始写入的列偏移
        ev_size: 每个嵌入向量的维度
        num_vec: 总向量数量
        batch_size: 批次大小
        K: 每个样本对应的向量数量 (num_vec // batch_size)
        output_dim: 每个样本的输出维度 (K * ev_size)
    """

    def __init__(
        self,
        combiner: int,
        total_dims: int,
        accum_dims: int,
        ev_size: int,
        num_vec: int,
        batch_size: int
    ):
        """
        初始化池化器

        Args:
            combiner: 池化方式，0=SUM，1=MEAN
            total_dims: 输出张量的总维度
            accum_dims: 池化结果开始写入的列偏移
            ev_size: 每个嵌入向量的维度
            num_vec: 总向量数量
            batch_size: 批次大小
        """
        # 参数校验
        assert combiner in (0, 1), "combiner 必须是 0 (SUM) 或 1 (MEAN)"
        assert num_vec % batch_size == 0, "num_vec 必须能被 batch_size 整除"

        # 初始化固定参数
        self.combiner = combiner
        self.total_dims = total_dims
        self.accum_dims = accum_dims
        self.ev_size = ev_size
        self.num_vec = num_vec
        self.batch_size = batch_size

        # 预计算固定值
        self.K = num_vec // batch_size
        self.output_dim = self.K * ev_size
        self.end_col = accum_dims + self.output_dim

        # 最终维度校验
        assert self.end_col <= total_dims, f"end_col ({self.end_col}) 不能超过 total_dims ({total_dims})"

        # 预计算新的索引排列（固定不变，只需计算一次）
        self.new_indices = []
        for b in range(self.batch_size):
            for k in range(self.K):
                self.new_indices.append(b + k * self.batch_size)
        # 转换为tensor以提高索引效率
        self.new_indices = torch.tensor(self.new_indices, dtype=torch.long)

    def pool(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        offset: torch.Tensor,
        inverse: torch.Tensor
    ) -> None:
        """
        执行池化操作，并将结果写入dst张量

        Args:
            src: 源嵌入张量，shape: [*, ev_size]
            dst: 目标输出张量，结果会写入该张量的指定位置
            offset: 偏移张量，用于划分每个分组的范围
            inverse: 索引张量，用于从src中收集嵌入向量
        """
        base = offset[0].item()
        pooled_list = []

        # 遍历每个分组进行池化
        for i in range(self.num_vec):
            start = offset[i].item() - base
            end = offset[i + 1].item() - base
            length = end - start

            if length <= 0:
                # 空分组，返回全零向量
                pooled_emb = torch.zeros(self.ev_size, dtype=src.dtype, device=src.device)
            else:
                # 获取ID列表并收集嵌入向量
                ids = inverse[start:end].to(torch.long)
                embs = src[ids]

                # 根据combiner类型进行池化
                if self.combiner == 0:
                    pooled_emb = embs.sum(dim=0)
                else:
                    pooled_emb = embs.mean(dim=0)

            pooled_list.append(pooled_emb)

        # 堆叠成 [num_vec, ev_size]
        pooled = torch.stack(pooled_list, dim=0)

        # 使用预计算的索引重新排列
        pooled = pooled[self.new_indices.to(pooled.device)]

        # 重塑并写入dst
        pooled_reshaped = pooled.view(self.batch_size, self.K, self.ev_size).flatten(1, 2)
        dst[:, self.accum_dims:self.end_col] = pooled_reshaped


# 用例1
@pytest.mark.parametrize("dtype_int", [torch.uint64, torch.int64])
@pytest.mark.parametrize("dtype_float", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("combiner", [0, 1])
def test_pooling_embeddings(dtype_int, dtype_float, device, combiner):
    torch.npu.set_device(device)
    src = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], dtype=dtype_float, device=f'npu:{device}')
    inverse = torch.tensor([0, 1, 2, 1, 3], dtype=dtype_int, device=f'npu:{device}')
    offset = torch.tensor([10, 12, 13, 14, 15], dtype=dtype_int, device=f'npu:{device}')

    batch_size = 2
    num_vec = 4
    ev_size = 4
    total_dims = 12
    accum_dims = 0

    '''
    预期结果：combiner = 0
    [[6,8,10,12,  5,6,7,8,       0,0,0,0],
     [9,10,11,12, 13,14,15,16    0,0,0,0]]
    '''

    pooler = EmbeddingTablePooler(
        combiner=combiner,
        total_dims=total_dims,
        accum_dims=accum_dims,
        ev_size=ev_size,
        num_vec=num_vec,
        batch_size=batch_size
    )

#   npu torch算子
    excepted = torch.zeros(batch_size, total_dims, dtype=dtype_float, device=f'npu:{device}')
    pooler.pool(src, excepted, offset, inverse)

#   自定义算子验证
    result = torch.zeros(batch_size, total_dims, dtype=dtype_float, device=f'npu:{device}')
    dynamic_emb_extensions.lookup_forward(src, result, offset,
        inverse, combiner, total_dims, accum_dims, ev_size, num_vec, batch_size)

    torch.testing.assert_close(excepted.cpu(), result.cpu())
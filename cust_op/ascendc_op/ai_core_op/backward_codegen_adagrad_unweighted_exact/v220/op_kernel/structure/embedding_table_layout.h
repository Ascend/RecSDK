/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef EMBEDDING_TABLE_LAYOUT_H
#define EMBEDDING_TABLE_LAYOUT_H

#include "kernel_operator.h"
#include "args_struct.h"
using namespace AscendC;

// 表布局元数据：EmbeddingTableLayout
struct EmbeddingTableLayout {
    GlobalTensor<int32_t> dOffsetsGT;       // dOffsets - 各表维度前缀和 [0, d0, d0+d1, ...]
    GlobalTensor<int64_t> weightsOffsetsGT; // weightsOffsets - 各表在 weight buffer 中的起始偏移
    GlobalTensor<int64_t> hashSizeCumsumGT; // hashSizeCumsum - 各表大小前缀和（用于全局索引）
    int64_t maxDim;                         // 最大embedding维度

    // 初始化方法
    __aicore__ inline void Init(Args& args, BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        dOffsetsGT.SetGlobalBuffer((__gm__ int32_t*) args.dOffsets, tilingData.dOffsetsDim0);
        weightsOffsetsGT.SetGlobalBuffer((__gm__ int64_t*) args.weightsOffsets, tilingData.weightsOffsetsDim0);
        hashSizeCumsumGT.SetGlobalBuffer((__gm__ int64_t*) args.hashSizeCumsum, tilingData.weightsOffsetsDim0 + 1);
        maxDim = tilingData.maxD;
    }

    __aicore__ inline int64_t GetMaxDim() const { return maxDim; }

    // 获取表维度
    __aicore__ inline int64_t GetEmbedDim(int64_t tableIdx) const
    {
        return dOffsetsGT.GetValue(tableIdx + 1) - dOffsetsGT.GetValue(tableIdx);
    }

    // 获取表在权重 buffer 中的基地址偏移
    __aicore__ inline int64_t GetWeightBaseOffset(int64_t tableIdx) const
    {
        return weightsOffsetsGT.GetValue(tableIdx);
    }

    // 根据全局 ID 计算表内局部 ID
    __aicore__ inline int64_t GetLocalId(int64_t globalId, int64_t tableIdx) const
    {
        return globalId - hashSizeCumsumGT.GetValue(tableIdx);
    }

    // 获取embedding维度的偏移量
    __aicore__ inline int64_t GetEmbeddingDimOffset(int64_t tableIdx) const { return dOffsetsGT.GetValue(tableIdx); }

    // 获取表的综合信息
    __aicore__ inline void GetTableInfo(int64_t tableIdx, int64_t& embedDim, int64_t& weightOffset) const
    {
        embedDim = GetEmbedDim(tableIdx);
        weightOffset = GetWeightBaseOffset(tableIdx);
    }

    // 获取表的维度和维度偏移
    __aicore__ inline void GetDimAndOffset(int64_t tableIdx, int64_t& embedDim, int64_t& embedOffset) const
    {
        embedDim = GetEmbedDim(tableIdx);
        embedOffset = GetEmbeddingDimOffset(tableIdx);
    }

    // 获取总哈希大小
    __aicore__ inline int64_t GetTotalHashSize() const { return hashSizeCumsumGT.GetValue(GetWeightsOffsetsDim0()); }

    __aicore__ inline uint64_t GetWeightsOffsetsDim0() const { return weightsOffsetsGT.GetSize(); }

    // 获取hashSizeCumsumGT的值
    __aicore__ inline int64_t GetTableSizePrefixSum(int64_t index) const { return hashSizeCumsumGT.GetValue(index); }

    // 获取weightsOffsetsGT的值
    __aicore__ inline int64_t GetWeightOffsetTableValue(int64_t index) const
    {
        return weightsOffsetsGT.GetValue(index);
    }

    __aicore__ inline int64_t LocateTableIndex(int64_t thisTableOffset) const
    {
        int64_t tableIndex = 0;
        for (int64_t i = GetWeightsOffsetsDim0(); i >= 0; i--) {
            if (thisTableOffset >= hashSizeCumsumGT.GetValue(i)) {
                tableIndex = i;
                break;
            }
        }
        return tableIndex;
    }
};

#endif // EMBEDDING_TABLE_LAYOUT_H
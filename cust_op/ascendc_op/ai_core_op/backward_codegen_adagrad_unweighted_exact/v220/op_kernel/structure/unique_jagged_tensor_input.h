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

#ifndef UNIQUE_JAGGED_TENSOR_INPUT_H
#define UNIQUE_JAGGED_TENSOR_INPUT_H

#include "kernel_operator.h"
using namespace AscendC;

// Unique Jagged Tensor Input 结构体
struct UniqueJaggedTensorInput {
    GlobalTensor<int64_t> uniqueIdsGT;                 // 去重后的 ID 列表
    GlobalTensor<int64_t> inverseIndicesGT;            // 原始 ID → uniqueId 映射
    GlobalTensor<int64_t> tableUniqueCountPrefixSumGT; // uniqueHashSize（每表唯一 ID 数前缀和）
    GlobalTensor<int64_t> tableRawCountPrefixSumGT;    // indiceSizeCumsum（每表原始 ID 总数前缀和）
    int64_t uniqueIdDim0;
    // 初始化方法
    __aicore__ inline void Init(Args& args, BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        this->uniqueIdsGT.SetGlobalBuffer((__gm__ int64_t*) args.uniqueId, tilingData.uniqueIdDim0);
        this->inverseIndicesGT.SetGlobalBuffer((__gm__ int64_t*) args.uniqueInverse, tilingData.indicesDim0);
        this->tableUniqueCountPrefixSumGT.SetGlobalBuffer((__gm__ int64_t*) args.uniqueHashSize,
                                                          tilingData.uniqueHashDim0);
        this->tableRawCountPrefixSumGT.SetGlobalBuffer((__gm__ int64_t*) args.indiceSizeCumsum,
                                                       tilingData.uniqueHashDim0);

        // len(uniqueId) = uniqueHash[-1]
        uniqueIdDim0 = tableUniqueCountPrefixSumGT.GetValue(tilingData.uniqueHashDim0 - 1);
    }

    __aicore__ inline int64_t GetUniqueIdDim0() const { return uniqueIdDim0; }

    __aicore__ inline GlobalTensor<int64_t>& GetInverseIndicesGT() { return inverseIndicesGT; }

    __aicore__ inline int64_t GetUniqueHashIdCount() const { return tableUniqueCountPrefixSumGT.GetSize(); }

    __aicore__ inline int64_t GetUniqueIdx(const int64_t offset) const { return uniqueIdsGT.GetValue(offset); }

    __aicore__ inline int64_t GetUniqueCountPrefixSum(const int64_t offset) const
    {
        return tableUniqueCountPrefixSumGT.GetValue(offset);
    }

    // 获取表的起始偏移
    __aicore__ inline int64_t GetRawCount(int64_t tableIdx) const
    {
        if (tableIdx == 0) {
            return 0;
        }
            
        return tableRawCountPrefixSumGT.GetValue(tableIdx);
    }
};

#endif
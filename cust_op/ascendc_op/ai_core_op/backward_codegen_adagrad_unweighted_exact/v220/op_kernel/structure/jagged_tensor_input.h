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

#ifndef JAGGED_TENSOR_INPUT_H
#define JAGGED_TENSOR_INPUT_H

#include "kernel_operator.h"
#include "../structure/args_struct.h"

using namespace AscendC;

// jagged tensor输入数据
struct JaggedTensorInput {
    GlobalTensor<int64_t> indicesGT; // indices张量（根据是否启用hash可能指向hashIndices）
    GlobalTensor<int64_t> offsetsGT; // offsets张量
    int64_t batchSize;

    // 初始化方法
    __aicore__ inline void Init(Args& args, BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        // 根据enableHash决定使用哪个地址作为ID张量
        __gm__ int64_t* idTensorAddr =
            tilingData.enableHash ? (__gm__ int64_t*) args.hashIndices : (__gm__ int64_t*) args.indices;
        indicesGT.SetGlobalBuffer(idTensorAddr, tilingData.indicesDim0);
        offsetsGT.SetGlobalBuffer((__gm__ int64_t*) args.offsets, tilingData.offsetsDim0);
        batchSize = (tilingData.offsetsDim0 - 1) / (tilingData.dOffsetsDim0 - 1);
    }

    // 获取ID值的方法
    __aicore__ inline int64_t GetId(int64_t index) const { return indicesGT.GetValue(index); }

    // 获取偏移值的方法
    __aicore__ inline int64_t GetOffset(int64_t index) const { return offsetsGT.GetValue(index); }

    // 获取ID张量的维度大小
    __aicore__ inline int64_t GetIndicesDimSize() const { return indicesGT.GetSize(); }

    // 获取偏移张量的维度大小
    __aicore__ inline int64_t GetOffsetsDimSize() const { return offsetsGT.GetSize(); }

    // 检查索引是否在指定offset范围内
    __aicore__ inline bool IsInOffsetRange(int64_t index, int64_t offsetIndex) const
    {
        return (index >= GetOffset(offsetIndex) && index < GetOffset(offsetIndex + 1));
    }

    // 获取两个offset之间的差值（即bag的长度）
    __aicore__ inline int64_t GetBagLength(int64_t offsetIndex) const
    {
        return GetOffset(offsetIndex + 1) - GetOffset(offsetIndex);
    }

    __aicore__ inline int64_t GetBatchSize() const { return batchSize; }

    __aicore__ inline int64_t GetOffsetsSize() const { return offsetsGT.GetSize(); }

    __aicore__ inline int64_t LocateOffsetIndex(const int64_t offset) const
    {
        int64_t thisOffsetIndex = 0;
        for (int64_t i = GetOffsetsSize() - 1; i >= 0; i--) {
            if (offset >= GetOffset(i)) {
                thisOffsetIndex = i;
                break;
            }
        }
        return thisOffsetIndex;
    }
};

#endif // JAGGED_TENSOR_INPUT_H
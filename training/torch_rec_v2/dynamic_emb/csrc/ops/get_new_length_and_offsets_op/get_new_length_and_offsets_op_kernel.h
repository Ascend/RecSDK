/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GET_NEW_LENGTH_AND_OFFSETS_OP_KERNEL_H
#define GET_NEW_LENGTH_AND_OFFSETS_OP_KERNEL_H

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

namespace DynamicEmbeddingGetNewLengthAndOffsetsOPSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 1;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t CACHE_ALIGN = 64;

template <typename T>
__aicore__ inline int64_t BinarySearchLastLeIdx(__gm__ const T* const arr, int64_t num, T target)
{
    int64_t start = 0;
    int64_t end = num;
    while (start < end) {
        int64_t middle = start + (end - start) / 2;
        T value = arr[middle];
        if (value <= target) {
            start = middle + 1;
        } else {
            end = middle;
        }
    }
    return (start == num && arr[start - 1] != target) ? num : start - 1;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void GetNewLengthAndOffsetsKernel(
    __gm__ uint64_t* dUniqueOffsets,         // 各table的唯一元素偏移
    __gm__ int64_t* dTableOffsetsInFeature,  // 各table包含的feature范围
    __gm__ T* newOffsets, __gm__ T* newLenghths, int tableNum, int64_t newLengthsSize, int localBatchSize)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t coreNum = AscendC::Simt::GetBlockNum();
    int32_t threadTotal = coreNum * threadNumPerCore;
    int32_t globalTid = coreId * threadNumPerCore + tid;

    // 数据划分
    int64_t baseChunkSize = newLengthsSize / threadTotal;
    int64_t remainder = newLengthsSize % threadTotal;
    int64_t startIdx = 0;
    int64_t endIdx = 0;

    if (globalTid < remainder) {
        startIdx = globalTid * (baseChunkSize + 1);
        endIdx = startIdx + baseChunkSize + 1;
    } else {
        startIdx = remainder * (baseChunkSize + 1) + (globalTid - remainder) * baseChunkSize;
        endIdx = startIdx + baseChunkSize;
    }

    endIdx = AscendC::Simt::Min(endIdx, newLengthsSize);

    for (int64_t i = startIdx; i < endIdx; ++i) {
        // 1) 计算对应的featureId和所属tableId
        int featureId = i / localBatchSize;
        int64_t tableId = BinarySearchLastLeIdx(dTableOffsetsInFeature, static_cast<int64_t>(tableNum + 1),
                                                static_cast<int64_t>(featureId));

        // 2) 计算当前table的总bucket数和当前i对应的bucketId
        int64_t table_feature_count = dTableOffsetsInFeature[tableId + 1] - dTableOffsetsInFeature[tableId];
        int64_t table_buckets = table_feature_count * localBatchSize;
        int64_t bucketId = i - (dTableOffsetsInFeature[tableId] * localBatchSize);

        // 3) 计算当前table的唯一元素总数
        uint64_t uniqueNum = dUniqueOffsets[tableId + 1] - dUniqueOffsets[tableId];

        // 4) 分配每个bucket均匀分配unique元素
        uint64_t bucketBase = uniqueNum / table_buckets;
        uint64_t bucketRemainder = uniqueNum % table_buckets;
        uint64_t tmpLength = bucketBase;
        uint64_t tmpOffset = dUniqueOffsets[tableId];

        // 每个多分配1个元素
        if (bucketId < bucketRemainder) {
            tmpLength += 1;
        }

        tmpOffset += (bucketId * bucketBase) + (bucketId < bucketRemainder ? bucketId : bucketRemainder);
        newLenghths[i] = static_cast<T>(tmpLength);
        newOffsets[i] = static_cast<T>(tmpOffset);

        if (i == newLengthsSize - 1) {
            newOffsets[newLengthsSize] = static_cast<T>(tmpOffset + tmpLength);
        }
    }

    AscendC::Simt::ThreadBarrier();
}

}  // namespace DynamicEmbeddingGetNewLengthAndOffsetsOPSimt

#endif  // GET_NEW_LENGTH_AND_OFFSETS_OP_KERNEL_H

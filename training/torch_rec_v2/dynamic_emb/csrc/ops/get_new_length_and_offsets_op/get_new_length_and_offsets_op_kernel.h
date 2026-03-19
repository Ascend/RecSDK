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
constexpr int32_t MAX_TABLE_NUM_USE_QUICK_DIVIDE = 200;

/* 本函数原公式为：bucket = idx % mySize
 * 现由快除法计算出div_res = idx / mySize，再由bucket = idx - div_res * mySize
 * uint64_t快除法：
 * r = x / y，如果y的值固定，则除法可以等效替换为如下公式
 * 预计算部分（host）：
 * 1. shift = ceil(log2(y))
 * 2. magic = ceil(2 ^ (64 + shift) / y)
 * 运行时计算部分（本函数内）：
 * 3. q = (x * magic) >> 64 由__umul64hi完成
 * 4. t = ((x - q) >> 1) + q
 * 5. r = t >> (shift - 1)
 */
__aicore__ inline uint64_t QuickRem(const uint64_t& x, const uint64_t& divisorMagic, const uint64_t& divisorShift,
                                    const uint64_t& y)
{
    uint64_t divTmp = __umul64hi(x, divisorMagic);
    divTmp = ((x - divTmp) >> 1) + divTmp;
    uint64_t divResult = divTmp >> divisorShift;
    return x - divResult * y;
}

__aicore__ inline uint64_t QuickDiv(const uint64_t& x, const uint64_t& divisorMagic, const uint64_t& divisorShift)
{
    uint64_t divTmp = __umul64hi(x, divisorMagic);
    divTmp = ((x - divTmp) >> 1) + divTmp;
    uint64_t divResult = divTmp >> divisorShift;
    return divResult;
}

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
__aicore__ inline int64_t BinarySearchLastLeIdxUB(__ubuf__ const T* const arr, int64_t num, T target)
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

struct TableCache {
    int64_t tableStartFeature;
    int64_t tableEndFeature;
    int64_t table_feature_count;
    int64_t table_buckets;
    int64_t bucketBaseOffset;
    int64_t nextTableBucketStart;
    uint64_t unique_num;
    uint64_t baseUniqueOffset;
    uint64_t bucketBase;
    uint64_t bucketRemainder;
};

__aicore__ inline void UpdateTableCache(int64_t tableId, int localBatchSize,
    __ubuf__ int64_t* dTableOffsetsInFeature, __ubuf__ uint64_t* dUniqueOffsets,
    __ubuf__ uint64_t* blkSizeMagicShifts, bool isQuickDivide, TableCache& cache)
{
    cache.tableStartFeature = dTableOffsetsInFeature[tableId];
    cache.tableEndFeature = dTableOffsetsInFeature[tableId + 1];
    cache.table_feature_count = cache.tableEndFeature - cache.tableStartFeature;
    cache.table_buckets = cache.table_feature_count * localBatchSize;
    cache.bucketBaseOffset = cache.tableStartFeature * localBatchSize;
    cache.nextTableBucketStart = cache.tableEndFeature * localBatchSize;

    cache.unique_num = dUniqueOffsets[tableId + 1] - dUniqueOffsets[tableId];
    cache.baseUniqueOffset = dUniqueOffsets[tableId];
    if (cache.table_buckets > 0) {
        if (isQuickDivide) {
            uint64_t blkSizeMagic = blkSizeMagicShifts[tableId * 2];
            uint64_t blkSizeShift = blkSizeMagicShifts[tableId * 2 + 1];
            cache.bucketBase = QuickDiv(cache.unique_num, blkSizeMagic, blkSizeShift);
            cache.bucketRemainder = QuickRem(cache.unique_num, blkSizeMagic, blkSizeShift, cache.table_buckets);
        } else {
            cache.bucketBase = cache.unique_num / cache.table_buckets;
            cache.bucketRemainder = cache.unique_num % cache.table_buckets;
        }
    } else {
        cache.bucketBase = 0;
        cache.bucketRemainder = 0;
    }
}

template <typename T, bool isQuickDivide>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void GetNewLengthAndOffsetsKernel(
    __ubuf__ uint64_t* dUniqueOffsets,         // 各table的唯一元素偏移
    __ubuf__ int64_t* dTableOffsetsInFeature,  // 各table包含的feature范围
    __ubuf__ uint64_t* blkSizeMagicShifts,     // 各table包含的快除参数
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

    // 由于 i 在循环中单调递增，可通过“边界推进”方式避免每次进行二分查找
    int64_t tableId = BinarySearchLastLeIdxUB(
        dTableOffsetsInFeature, static_cast<int64_t>(tableNum + 1), static_cast<int64_t>(startIdx / localBatchSize));

    // 预分配缓存，减少循环里重复计算
    TableCache cache;
    UpdateTableCache(tableId, localBatchSize, dTableOffsetsInFeature,
                     dUniqueOffsets, blkSizeMagicShifts, isQuickDivide, cache);

    for (int64_t i = startIdx; i < endIdx; ++i) {
        // 计算对应的tableId（i单调增，每次最多跨越一个table边界）
        if (tableId + 1 < tableNum && i >= cache.nextTableBucketStart) {
            tableId++;
            UpdateTableCache(tableId, localBatchSize, dTableOffsetsInFeature, dUniqueOffsets, blkSizeMagicShifts,
                             isQuickDivide, cache);
        }

        // 计算当前table的总bucket数和当前i对应的bucketId
        int64_t bucketId = i - cache.bucketBaseOffset;

        uint64_t tmpLength = cache.bucketBase;
        uint64_t tmpOffset = cache.baseUniqueOffset;

        // 每个多分配1个元素
        if (bucketId < cache.bucketRemainder) {
            tmpLength += 1;
        }

        tmpOffset += (bucketId * cache.bucketBase) +
                     (bucketId < cache.bucketRemainder ? bucketId : cache.bucketRemainder);
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

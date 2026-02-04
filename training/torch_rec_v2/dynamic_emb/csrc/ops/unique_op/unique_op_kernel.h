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

#ifndef UNIQUE_OP_KERNEL_H
#define UNIQUE_OP_KERNEL_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;
using KeyType = int64_t;

namespace DynamicEmbeddingHashUniqueOPSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t CACHE_ALIGN = 64;
constexpr int64_t EMPTY_COUNT = -1;
constexpr int64_t INT64_EMPTY_KEY = std::numeric_limits<int64_t>::min();

template <typename T>
__aicore__ inline T Min(const T& a, const T& b)
{
    return (a < b) ? a : b;
}

__aicore__ __inline__ uint64_t MurmurHash3_64(uint64_t const& key)
{
    uint64_t k = key;
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void ResetHashTable(
    __gm__ KeyType* hashTableKeyGm, __gm__ int64_t* hashTableCountGm, __gm__ int64_t* globalCounterGm,
    const int64_t hashTableCapacity, const KeyType emptyKey, int32_t coreNum)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t threadTotal = coreNum * threadNumPerCore;
    int32_t globalTid = coreId * threadNumPerCore + tid;

    // 重置哈希表
    int64_t baseChunkSize = hashTableCapacity / threadTotal;
    int64_t remainder = hashTableCapacity % threadTotal;
    int64_t startIdx = (globalTid < remainder)
                           ? (globalTid * (baseChunkSize + 1))
                           : (remainder * (baseChunkSize + 1) + (globalTid - remainder) * baseChunkSize);
    int64_t endIdx = Min(startIdx + baseChunkSize + (globalTid < remainder ? 1 : 0), hashTableCapacity);

    for (int64_t idx = startIdx; idx < endIdx; ++idx) {
        hashTableKeyGm[idx] = emptyKey;
        hashTableCountGm[idx] = EMPTY_COUNT;
    }

    if (tid == 0) {
        *globalCounterGm = 0;
    }

    AscendC::Simt::ThreadBarrier();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void HashUniqueInsertCompute(
    __gm__ const KeyType* keyGm, __gm__ KeyType* uniqueKeyGm, __gm__ int64_t* restoreIndexGm, __gm__ int64_t* countGm,
    __gm__ KeyType* hashTableKeyGm,    // 哈希表key槽位
    __gm__ int64_t* hashTableCountGm,  // 存储uniqueIdx
    __gm__ int64_t* globalCounterGm,   // 全局计数器
    const int32_t totalLength,         // 原始key长度
    const int64_t hashTableCapacity,   // 哈希表容量
    const KeyType emptyKey,            // 哈希表空key标记
    int32_t coreNum, __gm__ int32_t* uniqueSrcIndices)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t threadTotal = coreNum * threadNumPerCore;
    int32_t globalTid = coreId * threadNumPerCore + tid;

    // 1. 划分数据
    int64_t baseChunkSize = totalLength / threadTotal;
    int64_t remainder = totalLength % threadTotal;
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (globalTid < remainder) {
        // 前 remainder 个线程，每个多处理1个元素
        startIdx = globalTid * (baseChunkSize + 1);
        endIdx = startIdx + baseChunkSize + 1;
    } else {
        startIdx = remainder * (baseChunkSize + 1) + (globalTid - remainder) * baseChunkSize;
        endIdx = startIdx + baseChunkSize;
    }
    endIdx = Min(endIdx, static_cast<int64_t>(totalLength));

    for (int64_t idx = startIdx; idx < endIdx; ++idx) {
        KeyType targetKey = keyGm[idx];
        int64_t uniqueIdx = EMPTY_COUNT;
        uint64_t hashValue = MurmurHash3_64(targetKey);
        int64_t hashIndex = static_cast<int64_t>(hashValue % hashTableCapacity);
        int64_t counter = 0;

        // 开放寻址法 hash
        while (true) {
            if (counter >= hashTableCapacity) {
                break;
            }
            volatile KeyType existingKey = hashTableKeyGm[hashIndex];
            volatile __gm__ int64_t& slotUniqueIdx = hashTableCountGm[hashIndex];

            if (existingKey == emptyKey) {
                KeyType oldKey = AscendC::Simt::AtomicCas(&hashTableKeyGm[hashIndex], emptyKey, targetKey);

                if (oldKey == emptyKey) {
                    int64_t oldCount = AscendC::Simt::AtomicAdd<int64_t>(globalCounterGm, 1LL);
                    uniqueIdx = oldCount;
                    uniqueKeyGm[uniqueIdx] = targetKey;
                    uniqueSrcIndices[uniqueIdx] = idx;
                    countGm[uniqueIdx] = 1LL;
                    AscendC::Simt::ThreadFence();
                    slotUniqueIdx = uniqueIdx;

                    break;
                } else if (oldKey == targetKey) {
                    // CAS失败但槽位已存在targetKey：自旋等待uniqueIdx就绪
                    while (slotUniqueIdx == EMPTY_COUNT) {}
                    uniqueIdx = slotUniqueIdx;
                    AscendC::Simt::AtomicAdd<int64_t>(&countGm[uniqueIdx], 1LL);
                    break;
                }
            } else if (existingKey == targetKey) {
                // 槽位已存在targetKey：等待uniqueIdx就绪=
                while (slotUniqueIdx == EMPTY_COUNT) {}
                uniqueIdx = slotUniqueIdx;
                AscendC::Simt::AtomicAdd<int64_t>(&countGm[uniqueIdx], 1LL);
                break;
            }

            counter++;
            hashIndex = (hashIndex + 1) % hashTableCapacity;
        }

        AscendC::Simt::AtomicExch<int64_t>(restoreIndexGm + idx, uniqueIdx);
    }

    AscendC::Simt::ThreadBarrier();
}

}  // namespace DynamicEmbeddingHashUniqueOPSimt

#endif  // UNIQUE_OP_KERNEL_H
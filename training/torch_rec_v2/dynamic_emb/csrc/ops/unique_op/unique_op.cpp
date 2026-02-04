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

#include <type_traits>

#include "unique_op_kernel.h"

#define DTYPE_DISPATCH(isInt64, DTYPE, ...) \
    do {                                    \
        if ((isInt64)) {                    \
            using DTYPE = int64_t;          \
            __VA_ARGS__;                    \
        }                                   \
    } while (0)

extern "C" __global__ __aicore__ void unique_op(GM_ADDR key, GM_ADDR uniqueKey, GM_ADDR restoreIndex, GM_ADDR count,
                                                GM_ADDR workspace, int64_t hashTableCapacity, int32_t totalLength,
                                                GM_ADDR uniqueSrcIndices)
{
    int32_t coreId = AscendC::GetBlockIdx();
    int32_t coreNum = AscendC::GetBlockNum();
    bool isInt64 = true;

    DTYPE_DISPATCH(isInt64, KeyType, {
        using namespace AscendC;
        using KeyType = int64_t;
        __gm__ const KeyType* keyGm = reinterpret_cast<__gm__ const KeyType*>(key);
        __gm__ KeyType* uniqueKeyGm = reinterpret_cast<__gm__ KeyType*>(uniqueKey);
        __gm__ int32_t* uniqueSrcIndicesGm = reinterpret_cast<__gm__ int32_t*>(uniqueSrcIndices);
        __gm__ int64_t* restoreIndexGm = reinterpret_cast<__gm__ int64_t*>(restoreIndex);
        __gm__ int64_t* countGm = reinterpret_cast<__gm__ int64_t*>(count);

        __gm__ uint8_t* workspaceBuf = reinterpret_cast<__gm__ uint8_t*>(workspace);
        size_t offset = 0;
        // 哈希表key槽位
        size_t keySlotSize = hashTableCapacity * sizeof(KeyType);
        __gm__ KeyType* hashTableKeyGm = reinterpret_cast<__gm__ KeyType*>(workspaceBuf + offset);
        offset += keySlotSize;

        // 哈希表count槽位
        size_t countSlotSize = hashTableCapacity * sizeof(int64_t);
        __gm__ int64_t* hashTableCountGm = reinterpret_cast<__gm__ int64_t*>(workspaceBuf + offset);
        offset += countSlotSize;

        // 全局计数器
        __gm__ int64_t* globalCounterGm = reinterpret_cast<__gm__ int64_t*>(workspaceBuf + offset);
        offset += sizeof(int64_t);

        KeyType emptyKey = static_cast<KeyType>(DynamicEmbeddingHashUniqueOPSimt::INT64_EMPTY_KEY);

        Simt::VF_CALL<DynamicEmbeddingHashUniqueOPSimt::ResetHashTable>(
            Simt::Dim3{DynamicEmbeddingHashUniqueOPSimt::MAX_THREADS_PER_BLOCK, 1, 1}, hashTableKeyGm, hashTableCountGm,
            globalCounterGm, hashTableCapacity, emptyKey, coreNum);
        SyncAll();

        Simt::VF_CALL<DynamicEmbeddingHashUniqueOPSimt::HashUniqueInsertCompute>(
            Simt::Dim3{DynamicEmbeddingHashUniqueOPSimt::MAX_THREADS_PER_BLOCK, 1, 1}, keyGm, uniqueKeyGm,
            restoreIndexGm, countGm, hashTableKeyGm, hashTableCountGm, globalCounterGm, totalLength, hashTableCapacity,
            emptyKey, coreNum, uniqueSrcIndicesGm);

        SyncAll();
    });
}
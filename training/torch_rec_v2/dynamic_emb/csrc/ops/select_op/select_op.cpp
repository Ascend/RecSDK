/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "select_op_kernel.h"

#define KEY_TYPE_DISPATCH(isUInt64, KEY_TYPE, ...) \
    do {                                           \
        if (isUInt64) {                            \
            using KEY_TYPE = uint64_t;             \
            __VA_ARGS__;                           \
        } else {                                   \
            using KEY_TYPE = int64_t;              \
            __VA_ARGS__;                           \
        }                                          \
    } while (0)

namespace {

template <bool SelectIndex, typename KeyType>
__aicore__ inline void RunSelectOp(GM_ADDR flags, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR numSelected,
                                   GM_ADDR workspace, int64_t numTotal, int32_t isSmall, int32_t totalBlocks)
{
    using namespace AscendC;
    using namespace DynamicEmbeddingSelectOPSimt;

    int32_t coreId = AscendC::GetBlockIdx();
    int32_t coreNum = AscendC::GetBlockNum();

    __gm__ bool* flagsGm = reinterpret_cast<__gm__ bool*>(flags);
    __gm__ KeyType* inputsGm = reinterpret_cast<__gm__ KeyType*>(inputs);
    __gm__ KeyType* outputsGm = reinterpret_cast<__gm__ KeyType*>(outputs);
    __gm__ int64_t* numSelectedGm = reinterpret_cast<__gm__ int64_t*>(numSelected);

    __gm__ uint8_t* workspaceBuf = reinterpret_cast<__gm__ uint8_t*>(workspace);
    __gm__ int64_t* prefixGm = reinterpret_cast<__gm__ int64_t*>(workspaceBuf);
    int32_t stride = CACHE_ALIGN / static_cast<int32_t>(sizeof(int64_t));
    __gm__ int64_t* blockSumsGm = prefixGm + numTotal;

    TPipe pipe;
    TBuf<TPosition::VECCALC> sharedMem;
    pipe.InitBuffer(sharedMem, MAX_WARPS * static_cast<int32_t>(sizeof(int64_t)));
    LocalTensor<int64_t> sharedTensor = sharedMem.Get<int64_t>();
    __ubuf__ int64_t* sharedMemory = reinterpret_cast<__ubuf__ int64_t*>(sharedTensor.GetPhyAddr());

    int32_t blockStartIdx = 0;
    int32_t curBlocksCount = 0;

    if (isSmall > 0) {
        Simt::VF_CALL<FlagPrefixSumSmall<int64_t>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, flagsGm, prefixGm,
                                                   blockSumsGm, sharedMemory, static_cast<int32_t>(numTotal), coreNum,
                                                   stride);
        SyncAll();
        if (totalBlocks > 1) {
            Simt::VF_CALL<FlagPrefixSumSmallUpdate<int64_t>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, prefixGm,
                                                             blockSumsGm, static_cast<int32_t>(numTotal), coreNum,
                                                             stride);
            SyncAll();
        }
    } else {
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        blockStartIdx = 0;
        if (coreId < remainderBlocks) {
            blockStartIdx = coreId * (blocksPerCore + 1);
        } else {
            blockStartIdx = remainderBlocks * (blocksPerCore + 1) + (coreId - remainderBlocks) * blocksPerCore;
        }

        Simt::VF_CALL<FlagPrefixSumLarge<int64_t>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, flagsGm, prefixGm,
                                                   blockSumsGm, sharedMemory, static_cast<int32_t>(numTotal),
                                                   blockStartIdx, curBlocksCount, stride);
        SyncAll();
        Simt::VF_CALL<FlagPrefixSumLargeUpdate<int64_t>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, prefixGm, blockSumsGm,
                                                         static_cast<int32_t>(numTotal), blockStartIdx, curBlocksCount,
                                                         stride);
        SyncAll();
    }

    if (isSmall > 0) {
        Simt::VF_CALL<SelectScatterSmall<KeyType, int64_t, SelectIndex>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1},
                                                                         flagsGm, inputsGm, outputsGm, prefixGm,
                                                                         static_cast<int32_t>(numTotal));
    } else {
        Simt::VF_CALL<SelectScatterLarge<KeyType, int64_t, SelectIndex>>(
            Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, flagsGm, inputsGm, outputsGm, prefixGm,
            static_cast<int32_t>(numTotal), blockStartIdx, curBlocksCount);
    }
    SyncAll();

    if (coreId == 0) {
        Simt::VF_CALL<WriteNumSelectedVF<int64_t>>(Simt::Dim3{1, 1, 1}, prefixGm, flagsGm, numSelectedGm,
                                                   static_cast<int32_t>(numTotal));
    }
    SyncAll();
}

}  // namespace

extern "C" __global__ __aicore__ void select_op(GM_ADDR flags, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR numSelected,
                                                GM_ADDR workspace, int64_t numTotal, int32_t isUInt64, int32_t isSmall,
                                                int32_t totalBlocks)
{
    KEY_TYPE_DISPATCH(isUInt64 > 0, KeyType, {
        RunSelectOp<false, KeyType>(flags, inputs, outputs, numSelected, workspace, numTotal, isSmall, totalBlocks);
    });
}

extern "C" __global__ __aicore__ void select_index_op(GM_ADDR flags, GM_ADDR outputIndices, GM_ADDR numSelected,
                                                      GM_ADDR workspace, int64_t numTotal, int32_t isUInt64,
                                                      int32_t isSmall, int32_t totalBlocks)
{
    KEY_TYPE_DISPATCH(isUInt64 > 0, KeyType, {
        RunSelectOp<true, KeyType>(flags, nullptr, outputIndices, numSelected, workspace, numTotal, isSmall,
                                   totalBlocks);
    });
}

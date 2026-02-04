/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <type_traits>
#include "block_bucketsize_sparse_features_kernel.h"
#include "kernel_operator.h"

#define DTYPE_DISPATCH(isInt32, DTYPE, ...) \
    do {                                    \
        if (isInt32) {                      \
            using DTYPE = int32_t;          \
            __VA_ARGS__;                    \
        } else {                            \
            using DTYPE = int64_t;          \
            __VA_ARGS__;                    \
        }                                   \
    } while (0)

extern "C" __global__ __aicore__ void block_bucketsize_sparse_features(
    GM_ADDR lengths, GM_ADDR indices, GM_ADDR weights, GM_ADDR offsets, GM_ADDR dist_type_per_feature,
    GM_ADDR blockSizes, GM_ADDR newLengths, GM_ADDR newIndices, GM_ADDR newWeights, GM_ADDR newPos,
    GM_ADDR unbucketPermute, GM_ADDR currentOffsets, int32_t lengthsSize, int32_t indicesSize, int32_t npuSize,
    int32_t B, int32_t isInt32,
    int32_t hasWeights,        // 是否启用权重
    int32_t sequenceFlag,      // 是否启用反桶化
    int32_t bucketizePosFlag,  // 是否启用分桶位置
    int32_t isSmall, int32_t newLengthsTotalSize, int32_t totalBlocks)
{
    int32_t coreId = AscendC::GetBlockIdx();
    int32_t coreNum = AscendC::GetBlockNum();

    DTYPE_DISPATCH(isInt32 > 0, DTYPE_X, {
        using namespace AscendC;
        using namespace BlockBucketSizeSparseSimt;
        __gm__ DTYPE_X* lengthsGm = reinterpret_cast<__gm__ DTYPE_X*>(lengths);
        __gm__ DTYPE_X* indicesGm = reinterpret_cast<__gm__ DTYPE_X*>(indices);
        __gm__ int64_t* offsetsGm = reinterpret_cast<__gm__ int64_t*>(offsets);
        __gm__ DTYPE_X* distTypeGm = reinterpret_cast<__gm__ DTYPE_X*>(dist_type_per_feature);
        __gm__ DTYPE_X* blockSizesGm = reinterpret_cast<__gm__ DTYPE_X*>(blockSizes);
        __gm__ DTYPE_X* newLengthsGm = reinterpret_cast<__gm__ DTYPE_X*>(newLengths);
        __gm__ DTYPE_X* newIndicesGm = reinterpret_cast<__gm__ DTYPE_X*>(newIndices);
        __gm__ DTYPE_X* newPosGm = reinterpret_cast<__gm__ DTYPE_X*>(newPos);
        __gm__ DTYPE_X* unbucketizePermuteGm = reinterpret_cast<__gm__ DTYPE_X*>(unbucketPermute);
        __gm__ DTYPE_X* currOffsetsGm = reinterpret_cast<__gm__ DTYPE_X*>(currentOffsets);

        const int32_t lenSize = static_cast<int32_t>(lengthsSize);
        const int32_t mySize = static_cast<int32_t>(npuSize);
        const int32_t batchSizeB = static_cast<int32_t>(B);
        const bool hasWeight = (hasWeights > 0);
        const bool sequence = (sequenceFlag > 0);
        const bool bucketizePos = (bucketizePosFlag > 0);

        __gm__ float* weightsGm = hasWeight ? reinterpret_cast<__gm__ float*>(weights) : nullptr;
        __gm__ float* newWeightsGm = hasWeight ? reinterpret_cast<__gm__ float*>(newWeights) : nullptr;

        Simt::VF_CALL<SimtComputeNewLengths<DTYPE_X>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm,
                                                      blockSizesGm, distTypeGm, newLengthsGm, lenSize, batchSizeB,
                                                      mySize);
        SyncAll();

        // 排他累加和
        {
            TPipe pipe;
            TBuf<TPosition::VECCALC> sharedMem;
            pipe.InitBuffer(sharedMem, MAX_WARPS * sizeof(DTYPE_X));
            LocalTensor<DTYPE_X> sharedTensor = sharedMem.Get<DTYPE_X>();
            __ubuf__ DTYPE_X* sharedMemory = reinterpret_cast<__ubuf__ DTYPE_X*>(sharedTensor.GetPhyAddr());

            // blockSums
            int32_t stride = CACHE_ALIGN / sizeof(DTYPE_X);
            __gm__ DTYPE_X* blockSums = currOffsetsGm + newLengthsTotalSize;

            if (isSmall > 0) {
                Simt::VF_CALL<SimtSmallDataCompute<DTYPE_X>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, newLengthsGm,
                                                             currOffsetsGm, blockSums, sharedMemory,
                                                             newLengthsTotalSize, coreNum);
                SyncAll();
                if (totalBlocks > 1) {
                    Simt::VF_CALL<SimtSmallDataUpdate<DTYPE_X>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, currOffsetsGm,
                                                                blockSums, newLengthsTotalSize, coreNum);
                    SyncAll();
                }
            } else {
                int32_t blocksPerCore = totalBlocks / coreNum;
                int32_t remainderBlocks = totalBlocks % coreNum;
                int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
                int32_t blockStartIdx = 0;
                if (coreId < remainderBlocks) {
                    blockStartIdx = coreId * (blocksPerCore + 1);
                } else {
                    blockStartIdx = remainderBlocks * (blocksPerCore + 1) + (coreId - remainderBlocks) * blocksPerCore;
                }

                Simt::VF_CALL<SimtLargeDataCompute<DTYPE_X>>(
                    Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, newLengthsGm, currOffsetsGm, blockSums, sharedMemory,
                    newLengthsTotalSize, coreNum, blockStartIdx, curBlocksCount);
                SyncAll();
                Simt::VF_CALL<SimtLargeDataUpdate<DTYPE_X>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, currOffsetsGm,
                                                            blockSums, newLengthsTotalSize, coreNum, blockStartIdx,
                                                            curBlocksCount);
                SyncAll();
            }
        }
        SyncAll();

        if (sequence && hasWeight && bucketizePos) {
            Simt::VF_CALL<SimtRearrangeData<true, true, true, DTYPE_X>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1},
                                                                        offsetsGm,             // offsets
                                                                        indicesGm,             // indices
                                                                        weightsGm,             // weights
                                                                        blockSizesGm,          // block_sizes
                                                                        distTypeGm,            // dist_type_per_feature
                                                                        currOffsetsGm,         // current_offsets
                                                                        newIndicesGm,          // new_indices
                                                                        newWeightsGm,          // new_weights
                                                                        newPosGm,              // new_pos
                                                                        unbucketizePermuteGm,  // unbucketize_permute
                                                                        lenSize,               // lengths_size
                                                                        batchSizeB,            // B
                                                                        mySize                 // my_size
            );
        } else if (sequence && hasWeight && !bucketizePos) {
            Simt::VF_CALL<SimtRearrangeData<true, true, false, DTYPE_X>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm, weightsGm, blockSizesGm, distTypeGm,
                currOffsetsGm, newIndicesGm, newWeightsGm, nullptr, unbucketizePermuteGm, lenSize, batchSizeB, mySize);
        } else if (sequence && !hasWeight && bucketizePos) {
            Simt::VF_CALL<SimtRearrangeData<true, false, true, DTYPE_X>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm, nullptr, blockSizesGm, distTypeGm,
                currOffsetsGm, newIndicesGm, nullptr, newPosGm, unbucketizePermuteGm, lenSize, batchSizeB, mySize);
        } else if (!sequence && hasWeight && bucketizePos) {
            Simt::VF_CALL<SimtRearrangeData<false, true, true, DTYPE_X>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm, weightsGm, blockSizesGm, distTypeGm,
                currOffsetsGm, newIndicesGm, newWeightsGm, newPosGm, nullptr, lenSize, batchSizeB, mySize);
        } else if (sequence && !hasWeight && !bucketizePos) {
            Simt::VF_CALL<SimtRearrangeData<true, false, false, DTYPE_X>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm, nullptr, blockSizesGm, distTypeGm,
                currOffsetsGm, newIndicesGm, nullptr, nullptr, unbucketizePermuteGm, lenSize, batchSizeB, mySize);
        } else if (!sequence && hasWeight && !bucketizePos) {
            Simt::VF_CALL<SimtRearrangeData<false, true, false, DTYPE_X>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm, weightsGm, blockSizesGm, distTypeGm,
                currOffsetsGm, newIndicesGm, newWeightsGm, nullptr, nullptr, lenSize, batchSizeB, mySize);
        } else if (!sequence && !hasWeight && bucketizePos) {
            Simt::VF_CALL<SimtRearrangeData<false, false, true, DTYPE_X>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm, nullptr, blockSizesGm, distTypeGm,
                currOffsetsGm, newIndicesGm, nullptr, newPosGm, nullptr, lenSize, batchSizeB, mySize);
        } else {
            Simt::VF_CALL<SimtRearrangeData<false, false, false, DTYPE_X>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, indicesGm, nullptr, blockSizesGm, distTypeGm,
                currOffsetsGm, newIndicesGm, nullptr, nullptr, nullptr, lenSize, batchSizeB, mySize);
        }
        SyncAll();
    });
}

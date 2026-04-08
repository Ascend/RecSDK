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

#ifndef BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_SIMPLIFIED_H
#define BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_SIMPLIFIED_H

#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernel_operator.h"
#include "block_bucketize_sparse_features_common.h"

using namespace AscendC;

namespace BlockBucketizeSparseFeaturesKernelSimplified {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t SCATTER_THREADS_PER_BLOCK = 512; // 1024线程数将导致寄存器溢出到栈，性能反而降低
constexpr int32_t WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / warpSize;
constexpr int32_t SCATTER_WARPS_PER_BLOCK = SCATTER_THREADS_PER_BLOCK / warpSize;
constexpr int32_t MAX_FEATURE_NUM_USE_QUICK_DIVIDE = 500;

template <typename T>
__aicore__ inline constexpr T UintDivMaxDividend()
{
    return static_cast<T>(std::numeric_limits<typename std::make_signed<T>::type>::max());
}

template <typename IndexT, bool useQuickDiv>
__aicore__ inline IndexT ComputeBucket(
    typename std::make_unsigned<IndexT>::type idx,
    typename std::make_unsigned<IndexT>::type blkSize,
    typename std::make_unsigned<IndexT>::type blkSizeMulMySize,
    typename std::make_unsigned<IndexT>::type mySize,
    typename std::make_unsigned<IndexT>::type mySizeMagic,
    uint32_t mySizeShift,
    const __ubuf__ typename std::make_unsigned<IndexT>::type* blkSizeMagicShifts,
    int32_t featureIndex,
    bool isPowerOfTwo)
{
    using UIndexT = typename std::make_unsigned<IndexT>::type;
    if (idx < blkSizeMulMySize) {
        if constexpr (useQuickDiv) {
            return static_cast<IndexT>(AscendC::Simt::UintDiv<UIndexT>(
                idx,
                blkSizeMagicShifts[featureIndex * 2],
                blkSizeMagicShifts[featureIndex * 2 + 1]));
        }
        return static_cast<IndexT>(idx / blkSize);
    }

    if (isPowerOfTwo) {
        return static_cast<IndexT>(idx & (mySize - 1));
    }
    if (idx > UintDivMaxDividend<UIndexT>()) {
        return static_cast<IndexT>(idx % mySize);
    }
    const UIndexT q = AscendC::Simt::UintDiv<UIndexT>(idx, mySizeMagic, static_cast<UIndexT>(mySizeShift));
    return static_cast<IndexT>(idx - q * mySize);
}

template <typename IndexT, bool useQuickDiv>
__aicore__ inline IndexT ComputeNewIndex(
    typename std::make_unsigned<IndexT>::type idx,
    typename std::make_unsigned<IndexT>::type blkSize,
    typename std::make_unsigned<IndexT>::type blkSizeMulMySize,
    typename std::make_unsigned<IndexT>::type mySize,
    typename std::make_unsigned<IndexT>::type mySizeMagic,
    uint32_t mySizeShift,
    const __ubuf__ typename std::make_unsigned<IndexT>::type* blkSizeMagicShifts,
    int32_t featureIndex)
{
    using UIndexT = typename std::make_unsigned<IndexT>::type;
    if (idx < blkSizeMulMySize) {
        if constexpr (useQuickDiv) {
            const UIndexT q = AscendC::Simt::UintDiv<UIndexT>(
                idx,
                blkSizeMagicShifts[featureIndex * 2],
                blkSizeMagicShifts[featureIndex * 2 + 1]);
            return static_cast<IndexT>(idx - q * blkSize);
        }
        return static_cast<IndexT>(idx % blkSize);
    }
    if (idx > UintDivMaxDividend<UIndexT>()) {
        return static_cast<IndexT>(idx / mySize);
    }
    return static_cast<IndexT>(AscendC::Simt::UintDiv<UIndexT>(idx, mySizeMagic, static_cast<UIndexT>(mySizeShift)));
}

template <typename OffsetT, typename IndexT, bool useQuickDiv, bool batchSizeIsOne>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtComputeNewLengthsGmAtomic(
    const __gm__ OffsetT* offsets,
    const __gm__ IndexT* indices,
    const __gm__ IndexT* blockSizes,
    __gm__ OffsetT* newLengths,
    int32_t lengthsSize,
    int32_t mySize,
    int32_t rowBegin,
    int32_t rowEnd,
    uint64_t mySizeDivMagic,
    uint32_t mySizeDivShift,
    const __ubuf__ typename std::make_unsigned<IndexT>::type* blkSizeMagicShifts,
    uint64_t batchSizeDivMagic,
    uint32_t batchSizeDivShift,
    bool isPowerOfTwo)
{
    if (rowEnd <= rowBegin) {
        return;
    }

    using UIndexT = typename std::make_unsigned<IndexT>::type;
    const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    const int32_t warpId = threadIdx / warpSize;
    const int32_t laneId = threadIdx % warpSize;
    const UIndexT mySizeU = static_cast<UIndexT>(mySize);
    const UIndexT mySizeMagicU = static_cast<UIndexT>(mySizeDivMagic);

    const int32_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;

    for (int32_t rowIdx = rowBegin + warpId; rowIdx < clampedEnd; rowIdx += WARPS_PER_BLOCK) {
        const int32_t featureIndex = batchSizeIsOne ? rowIdx : static_cast<int32_t>(AscendC::Simt::UintDiv<uint64_t>(
                  static_cast<uint64_t>(static_cast<uint32_t>(rowIdx)), batchSizeDivMagic, static_cast<uint64_t>(batchSizeDivShift)));
        const UIndexT blkSize = static_cast<UIndexT>(blockSizes[featureIndex]);
        const UIndexT blkSizeMulMySize = blkSize * mySizeU;
        const OffsetT rowStart = (rowIdx == 0) ? static_cast<OffsetT>(0) : offsets[rowIdx - 1];
        const OffsetT rowEndIdx = offsets[rowIdx];

        for (OffsetT idxPos = rowStart + laneId; idxPos < rowEndIdx; idxPos += warpSize) {
            const UIndexT idx = static_cast<UIndexT>(indices[idxPos]);
            const IndexT bucket = ComputeBucket<IndexT, useQuickDiv>(
                idx, blkSize, blkSizeMulMySize, mySizeU,
                mySizeMagicU, mySizeDivShift, blkSizeMagicShifts, featureIndex, isPowerOfTwo);
            asc_atomic_add(&newLengths[static_cast<int32_t>(bucket) * lengthsSize + rowIdx],
                           static_cast<OffsetT>(1));
        }
    }
}

template <bool hasWeight, bool bucketizePos, typename OffsetT, typename IndexT, bool useQuickDiv, bool batchSizeIsOne>
__simt_vf__ __aicore__ LAUNCH_BOUND(SCATTER_THREADS_PER_BLOCK) inline void SimtScatterNewIndicesPooledUbAtomic(
    const __gm__ OffsetT* offsets,
    const __gm__ IndexT* indices,
    const __gm__ float* weights,
    const __gm__ IndexT* blockSizes,
    const __gm__ OffsetT* currOffsets,
    __gm__ IndexT* newIndices,
    __gm__ float* newWeights,
    __gm__ IndexT* newPos,
    __ubuf__ int32_t* ubCountersBase,
    int32_t lengthsSize,
    int32_t mySize,
    int32_t indicesSize,
    int32_t rowBegin,
    int32_t rowEnd,
    uint64_t mySizeDivMagic,
    uint32_t mySizeDivShift,
    const __ubuf__ typename std::make_unsigned<IndexT>::type* blkSizeMagicShifts,
    uint64_t batchSizeDivMagic,
    uint32_t batchSizeDivShift,
    bool isPowerOfTwo)
{
    if (rowEnd <= rowBegin) {
        return;
    }

    using UIndexT = typename std::make_unsigned<IndexT>::type;
    const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    const int32_t warpId = threadIdx / warpSize;
    const int32_t laneId = threadIdx % warpSize;
    const UIndexT mySizeU = static_cast<UIndexT>(mySize);
    const UIndexT mySizeMagicU = static_cast<UIndexT>(mySizeDivMagic);

    __ubuf__ int32_t* ubCounters = ubCountersBase + (warpId * mySize);
    const int32_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;

    for (int32_t rowIdx = rowBegin + warpId; rowIdx < clampedEnd; rowIdx += SCATTER_WARPS_PER_BLOCK) {
        for (int32_t bkt = laneId; bkt < mySize; bkt += warpSize) {
            ubCounters[bkt] = 0;
        }

        const int32_t featureIndex = batchSizeIsOne ? rowIdx : static_cast<int32_t>(AscendC::Simt::UintDiv<uint64_t>(
                  static_cast<uint64_t>(static_cast<uint32_t>(rowIdx)), batchSizeDivMagic, static_cast<uint64_t>(batchSizeDivShift)));
        const UIndexT blkSize = static_cast<UIndexT>(blockSizes[featureIndex]);
        const UIndexT blkSizeMulMySize = blkSize * mySizeU;
        const OffsetT rowStart = (rowIdx == 0) ? static_cast<OffsetT>(0) : offsets[rowIdx - 1];
        const OffsetT rowEndIdx = offsets[rowIdx];

        for (OffsetT idxPos = rowStart + laneId; idxPos < rowEndIdx; idxPos += warpSize) {
            const UIndexT idx = static_cast<UIndexT>(indices[idxPos]);
            const IndexT bucket = ComputeBucket<IndexT, useQuickDiv>(
                idx, blkSize, blkSizeMulMySize, mySizeU,
                mySizeMagicU, mySizeDivShift, blkSizeMagicShifts, featureIndex, isPowerOfTwo);
            const IndexT finalIndex = ComputeNewIndex<IndexT, useQuickDiv>(
                idx, blkSize, blkSizeMulMySize, mySizeU,
                mySizeMagicU, mySizeDivShift, blkSizeMagicShifts, featureIndex);
            const int32_t relativePos = asc_atomic_add(&ubCounters[static_cast<int32_t>(bucket)], 1);
            const OffsetT baseOffset = currOffsets[static_cast<int32_t>(bucket) * lengthsSize + rowIdx];
            const OffsetT writePos = baseOffset + static_cast<OffsetT>(relativePos);
            if (writePos < static_cast<OffsetT>(indicesSize)) {
                newIndices[writePos] = finalIndex;
                if constexpr (hasWeight) {
                    newWeights[writePos] = weights[idxPos];
                }
                if constexpr (bucketizePos) {
                    newPos[writePos] = static_cast<IndexT>(idxPos - rowStart);
                }
            }
        }
    }
}

template <bool hasWeight, bool bucketizePos, typename OffsetT, typename IndexT, bool useQuickDiv, bool batchSizeIsOne>
__simt_vf__ __aicore__ LAUNCH_BOUND(SCATTER_THREADS_PER_BLOCK) inline void SimtScatterNewIndicesRowsAtomic(
    const __gm__ OffsetT* offsets,
    const __gm__ IndexT* indices,
    const __gm__ float* weights,
    const __gm__ IndexT* blockSizes,
    __gm__ OffsetT* currOffsets,
    __gm__ IndexT* newIndices,
    __gm__ float* newWeights,
    __gm__ IndexT* newPos,
    __gm__ IndexT* unbucketizePermute,
    int32_t lengthsSize,
    int32_t mySize,
    int32_t indicesSize,
    int32_t rowBegin,
    int32_t rowEnd,
    uint64_t mySizeDivMagic,
    uint32_t mySizeDivShift,
    const __ubuf__ typename std::make_unsigned<IndexT>::type* blkSizeMagicShifts,
    uint64_t batchSizeDivMagic,
    uint32_t batchSizeDivShift,
    bool isPowerOfTwo)
{
    if (rowEnd <= rowBegin) {
        return;
    }

    using UIndexT = typename std::make_unsigned<IndexT>::type;
    const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    const int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    const UIndexT mySizeU = static_cast<UIndexT>(mySize);
    const UIndexT mySizeMagicU = static_cast<UIndexT>(mySizeDivMagic);
    const int32_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;

    for (int32_t rowIdx = rowBegin + threadIdx; rowIdx < clampedEnd; rowIdx += blockDim) {
        const int32_t featureIndex = batchSizeIsOne ? rowIdx : static_cast<int32_t>(AscendC::Simt::UintDiv<uint64_t>(
                  static_cast<uint64_t>(static_cast<uint32_t>(rowIdx)), batchSizeDivMagic, static_cast<uint64_t>(batchSizeDivShift)));
        const UIndexT blkSize = static_cast<UIndexT>(blockSizes[featureIndex]);
        const UIndexT blkSizeMulMySize = blkSize * mySizeU;
        const OffsetT rowStart = (rowIdx == 0) ? static_cast<OffsetT>(0) : offsets[rowIdx - 1];
        const OffsetT rowEndIdx = offsets[rowIdx];

        for (OffsetT idxPos = rowStart; idxPos < rowEndIdx; ++idxPos) {
            const UIndexT idx = static_cast<UIndexT>(indices[idxPos]);
            const IndexT bucket = ComputeBucket<IndexT, useQuickDiv>(
                idx, blkSize, blkSizeMulMySize, mySizeU,
                mySizeMagicU, mySizeDivShift, blkSizeMagicShifts, featureIndex, isPowerOfTwo);
            const IndexT finalIndex = ComputeNewIndex<IndexT, useQuickDiv>(
                idx, blkSize, blkSizeMulMySize, mySizeU,
                mySizeMagicU, mySizeDivShift, blkSizeMagicShifts, featureIndex);
            const int32_t slot = static_cast<int32_t>(bucket) * lengthsSize + rowIdx;
            const OffsetT writeCursor = currOffsets[slot];
            currOffsets[slot] = writeCursor + 1;
            if (writeCursor < static_cast<OffsetT>(indicesSize)) {
                newIndices[writeCursor] = finalIndex;
                unbucketizePermute[idxPos] = static_cast<IndexT>(writeCursor);
                if constexpr (hasWeight) {
                    newWeights[writeCursor] = weights[idxPos];
                }
                if constexpr (bucketizePos) {
                    newPos[writeCursor] = static_cast<IndexT>(idxPos - rowStart);
                }
            }
        }
    }
}

} // namespace BlockBucketizeSparseFeaturesKernelSimplified

#endif // BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_SIMPLIFIED_H

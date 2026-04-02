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

#ifndef BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_FULL_H
#define BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_FULL_H

#include <cstdint>
#include <type_traits>
#include <limits>

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "block_bucketize_sparse_features_common.h"
#include "fast_divmod.h"

using namespace AscendC;
using namespace BlockBucketizeSparseFeaturesCommon;

template <typename OffsetT, typename IndexT>
class BlockBucketizeSparseFeaturesKernelFull {
public:
    struct KernelArgs {
        __gm__ IndexT* indices;
        __gm__ IndexT* blockSizes;
        __gm__ IndexT* totalNumBlocks;
        __gm__ void* blockBucketizePosList;
        __gm__ uint64_t* blockBucketizePosPtrs;
        __gm__ int64_t* blockBucketizePosLens;
        __gm__ OffsetT* newLengths;
        __gm__ IndexT* newIndices;
        __gm__ IndexT* unbucketizePermute;
        __gm__ float* weights;
        __gm__ float* newWeights;
        __gm__ IndexT* newPos;
        __gm__ OffsetT* offsets;
        __gm__ OffsetT* newOffsets;
        __gm__ OffsetT* batchSizeOffsets;
        int32_t lengthsSize;
        int32_t indicesSize;
        int32_t numFeatures;
        int32_t batchSize;
        int32_t mySize;
        bool sequenceEnabled;
        bool weightsEnabled;
        bool bucketizePosEnabled;
        bool keepOrigIdxEnabled;
        bool totalNumBlocksEnabled;
        bool batchSizePerFeatureEnabled;
        bool blockBucketizePosEnabled;
        uint64_t mySizeDivMagic;
        uint32_t mySizeDivShift;
    };

    __aicore__ inline BlockBucketizeSparseFeaturesKernelFull(KernelArgs& args)
        :indices(args.indices),
         blockSizes(args.blockSizes),
         totalNumBlocks(args.totalNumBlocks),
         blockBucketizePosList(args.blockBucketizePosList),
         blockBucketizePosPtrs(args.blockBucketizePosPtrs),
         blockBucketizePosLens(args.blockBucketizePosLens),
         newLengths(args.newLengths),
         newIndices(args.newIndices),
         unbucketizePermute(args.unbucketizePermute),
         weights(args.weights),
         newWeights(args.newWeights),
         newPos(args.newPos),
         offsets(args.offsets),
         newOffsets(args.newOffsets),
         batchSizeOffsets(args.batchSizeOffsets),
         lengthsSize(args.lengthsSize),
         indicesSize(args.indicesSize),
         numFeatures(args.numFeatures),
         batchSize(args.batchSize),
         mySize(args.mySize),
         sequenceEnabled(args.sequenceEnabled),
         weightsEnabled(args.weightsEnabled),
         bucketizePosEnabled(args.bucketizePosEnabled),
         keepOrigIdxEnabled(args.keepOrigIdxEnabled),
         totalNumBlocksEnabled(args.totalNumBlocksEnabled),
         batchSizePerFeatureEnabled(args.batchSizePerFeatureEnabled),
         blockBucketizePosEnabled(args.blockBucketizePosEnabled),
         mySizeDivMagic(args.mySizeDivMagic),
         mySizeDivShift(args.mySizeDivShift)
    {
        const int32_t pooledUbBytes = WARPS_PER_BLOCK * args.mySize *
                                      static_cast<int32_t>(sizeof(int32_t));
        pipe.InitBuffer(ubBuf, pooledUbBytes);
        ubTensor = ubBuf.Get<OffsetT>();
        ubPtr = reinterpret_cast<__ubuf__ OffsetT*>(ubTensor.GetPhyAddr());
    }

    __aicore__ inline void ProcessComputeNewLengths()
    {
        const int32_t coreNum = static_cast<int32_t>(AscendC::GetBlockNum());
        const int32_t coreId = static_cast<int32_t>(AscendC::GetBlockIdx());

        WorkRange rowRange = {0};
        ComputeWorkRange(lengthsSize, coreId, coreNum, rowRange);

        if (blockBucketizePosEnabled) {
            BuildBucketizePosTables();
        }

        AscendC::SyncAll();

        AscendC::Simt::VF_CALL<SimtComputeNewLengthsGmAtomic>(
            AscendC::Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1},
            blockSizes,
            totalNumBlocks,
            offsets,
            indices,
            newLengths,
            lengthsSize,
            batchSize,
            mySize,
            rowRange.begin,
            rowRange.end,
            totalNumBlocksEnabled,
            batchSizeOffsets,
            numFeatures,
            batchSizePerFeatureEnabled,
            blockBucketizePosPtrs,
            blockBucketizePosLens,
            bucketizePosEnabled,
            mySizeDivMagic,
            mySizeDivShift);
    }

    __aicore__ inline void ProcessScatterNewIndices()
    {
        const int32_t coreNum = static_cast<int32_t>(AscendC::GetBlockNum());
        const int32_t coreId = static_cast<int32_t>(AscendC::GetBlockIdx());

        WorkRange rowRange = {0};
        ComputeWorkRange(lengthsSize, coreId, coreNum, rowRange);

        if (blockBucketizePosEnabled) {
            BuildBucketizePosTables();
        }

        AscendC::SyncAll();

        if (!sequenceEnabled) {
            AscendC::Simt::VF_CALL<SimtScatterNewIndicesPooledUbAtomic>(
                AscendC::Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1},
                blockSizes,
                totalNumBlocks,
                offsets,
                indices,
                weights,
                newWeights,
                newPos,
                newOffsets,
                newIndices,
                ubPtr,
                lengthsSize,
                batchSize,
                mySize,
                indicesSize,
                rowRange.begin,
                rowRange.end,
                weightsEnabled,
                bucketizePosEnabled,
                keepOrigIdxEnabled,
                totalNumBlocksEnabled,
                batchSizeOffsets,
                numFeatures,
                batchSizePerFeatureEnabled,
                blockBucketizePosPtrs,
                blockBucketizePosLens,
                mySizeDivMagic,
                mySizeDivShift);
        } else {
            AscendC::Simt::VF_CALL<SimtScatterNewIndicesRowsAtomic>(
                AscendC::Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1},
                blockSizes,
                totalNumBlocks,
                offsets,
                indices,
                weights,
                newWeights,
                newPos,
                newOffsets,
                newIndices,
                unbucketizePermute,
                lengthsSize,
                batchSize,
                mySize,
                indicesSize,
                rowRange.begin,
                rowRange.end,
                sequenceEnabled,
                weightsEnabled,
                bucketizePosEnabled,
                keepOrigIdxEnabled,
                totalNumBlocksEnabled,
                batchSizeOffsets,
                numFeatures,
                batchSizePerFeatureEnabled,
                blockBucketizePosPtrs,
                blockBucketizePosLens,
                mySizeDivMagic,
                mySizeDivShift);
        }
    }
private:
    __aicore__ inline void BuildBucketizePosTables()
    {
        if (blockBucketizePosList == nullptr || blockBucketizePosPtrs == nullptr ||
            blockBucketizePosLens == nullptr) {
            return;
        }

        AscendC::ListTensorDesc listDesc;
        listDesc.Init(blockBucketizePosList);

        const int32_t coreNum = static_cast<int32_t>(AscendC::GetBlockNum());
        const int32_t coreId = static_cast<int32_t>(AscendC::GetBlockIdx());

        const int32_t totalBlocks = (numFeatures + CACHE_LINE_ELEMS_64BIT - 1) / CACHE_LINE_ELEMS_64BIT;

        int32_t startBlock = 0;
        int32_t myBlocks = 0;

        if (totalBlocks > 0) {
            const int32_t blocksPerCore = totalBlocks / coreNum;
            const int32_t remainder = totalBlocks % coreNum;
            startBlock = coreId * blocksPerCore + ((coreId < remainder) ? coreId : remainder);
            myBlocks = blocksPerCore + ((coreId < remainder) ? 1 : 0);
        }

        if (myBlocks <= 0) {
            return;
        }

        const int32_t startFeature = startBlock * CACHE_LINE_ELEMS_64BIT;
        const int32_t endFeature = (startFeature + myBlocks * CACHE_LINE_ELEMS_64BIT) > numFeatures ?
                                   numFeatures : (startFeature + myBlocks * CACHE_LINE_ELEMS_64BIT);

        for (int32_t featureIndex = startFeature; featureIndex < endFeature; ++featureIndex) {
            uint64_t ptrVal = 0;
            int64_t lenVal = 0;
            uint64_t shapeBuf = 0;
            AscendC::TensorDesc<IndexT> desc;
            desc.SetShapeAddr(&shapeBuf);
            listDesc.GetDesc(desc, static_cast<uint32_t>(featureIndex));
            const uint32_t dim = desc.GetDim();
            if (dim == 1) {
                lenVal = static_cast<int64_t>(desc.GetShape(0));
                ptrVal = reinterpret_cast<uint64_t>(listDesc.GetDataPtr<IndexT>(static_cast<uint32_t>(featureIndex)));
            }
            blockBucketizePosPtrs[featureIndex] = ptrVal;
            blockBucketizePosLens[featureIndex] = lenVal;
            if ((featureIndex & 7) == 7) {
                dcci(blockBucketizePosPtrs + (featureIndex - 7),
                    cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
                dcci(blockBucketizePosLens + (featureIndex - 7),
                    cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
            }
        }

        if ((endFeature & 7) != 0) {
            const int32_t lastBlockStart = (endFeature & ~7);
            dcci(blockBucketizePosPtrs + lastBlockStart, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
            dcci(blockBucketizePosLens + lastBlockStart, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        }
    }

    static constexpr int32_t MAX_THREADS_PER_BLOCK = 512;
    static constexpr int32_t WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / warpSize;
    static constexpr int32_t CACHE_LINE_ELEMS_64BIT = 8;

    __gm__ IndexT* indices;
    __gm__ IndexT* blockSizes;
    __gm__ IndexT* totalNumBlocks;
    __gm__ void* blockBucketizePosList;
    __gm__ uint64_t* blockBucketizePosPtrs;
    __gm__ int64_t* blockBucketizePosLens;
    __gm__ OffsetT* newLengths;
    __gm__ IndexT* newIndices;
    __gm__ IndexT* unbucketizePermute;
    __gm__ float* weights;
    __gm__ float* newWeights;
    __gm__ IndexT* newPos;
    __gm__ OffsetT* offsets;
    __gm__ OffsetT* newOffsets;
    __gm__ OffsetT* batchSizeOffsets;
    int32_t lengthsSize;
    int32_t indicesSize;
    int32_t numFeatures;
    int32_t batchSize;
    int32_t mySize;
    bool sequenceEnabled;
    bool weightsEnabled;
    bool bucketizePosEnabled;
    bool keepOrigIdxEnabled;
    bool totalNumBlocksEnabled;
    bool batchSizePerFeatureEnabled;
    bool blockBucketizePosEnabled;
    uint64_t mySizeDivMagic;
    uint32_t mySizeDivShift;

    TPipe pipe;
    TBuf<TPosition::VECCALC> ubBuf;
    LocalTensor<OffsetT> ubTensor;
    __ubuf__ OffsetT* ubPtr;

    __aicore__ static inline int32_t ResolveFeatureIndexForRow(
        int32_t row,
        int32_t batchSize,
        const __gm__ OffsetT* batchSizeOffsets,
        int32_t numFeatures,
        bool useBatchSizePerFeature)
    {
        if (!useBatchSizePerFeature) {
            const int32_t safeBatch = (batchSize <= 0) ? 1 : batchSize;
            return row / safeBatch;
        }
        if (batchSizeOffsets == nullptr || numFeatures <= 0) {
            return 0;
        }
        int32_t left = 0;
        int32_t right = numFeatures;
        while (left < right) {
            const int32_t mid = (left + right + 1) >> 1;
            if (static_cast<int32_t>(batchSizeOffsets[mid]) <= row) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    // 每个 Warp 处理一个 Row
    __simt_vf__ __aicore__ static LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtComputeNewLengthsGmAtomic(
        const __gm__ IndexT* blockSizes,
        const __gm__ IndexT* totalNumBlocks,
        const __gm__ OffsetT* offsets,
        const __gm__ IndexT* indices,
        __gm__ OffsetT* newLengths,
        int32_t lengthsSize,
        int32_t batchSize,
        int32_t mySize,
        int32_t rowBegin,
        int32_t rowEnd,
        bool totalNumBlocksEnabled,
        const __gm__ OffsetT* batchSizeOffsets,
        int32_t numFeatures,
        bool batchSizePerFeatureEnabled,
        const __gm__ uint64_t* blockBucketizePosPtrs,
        const __gm__ int64_t* blockBucketizePosLens,
        bool bucketizePosEnabled,
        uint64_t mySizeDivMagic,
        uint32_t mySizeDivShift)
    {
        if (rowEnd <= rowBegin) {
            return;
        }
        const int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
        const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
        const int32_t warpsPerBlock = blockDim / warpSize;
        const int32_t warpId = threadIdx / warpSize;
        const int32_t laneId = threadIdx % warpSize;
        using UIndexT = std::make_unsigned_t<IndexT>;
        const IndexT mySizeIdx = static_cast<IndexT>(mySize);
        const int32_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;
        const bool hasTotalBlocks = totalNumBlocksEnabled && (totalNumBlocks != nullptr);
        const bool hasPosList = (blockBucketizePosPtrs != nullptr) && (blockBucketizePosLens != nullptr);
        const FastDivmod<UIndexT> fdMySize(
            static_cast<UIndexT>(mySizeDivMagic), mySizeDivShift, static_cast<UIndexT>(mySizeIdx));

        for (int32_t rowIdx = rowBegin + warpId; rowIdx < clampedEnd; rowIdx += warpsPerBlock) {
            const int32_t featureIndex = ResolveFeatureIndexForRow(
                rowIdx, batchSize, batchSizeOffsets, numFeatures, batchSizePerFeatureEnabled);
            if (featureIndex < 0 || featureIndex >= numFeatures) {
                continue;
            }
            __gm__ IndexT* posPtr = nullptr;
            int64_t posLen = 0;
            if (hasPosList) {
                posPtr = reinterpret_cast<__gm__ IndexT*>(blockBucketizePosPtrs[featureIndex]);
                posLen = static_cast<int64_t>(blockBucketizePosLens[featureIndex]);
            }
            const bool usePos = hasPosList && (posPtr != nullptr) && (posLen > 0);
            const IndexT blkSizeVal = blockSizes[featureIndex];
            if (blkSizeVal < 0) {
                continue;
            }
            IndexT globalBlocks = hasTotalBlocks ? totalNumBlocks[featureIndex] : mySizeIdx;
            if (globalBlocks <= 0) {
                globalBlocks = mySizeIdx;
            }
            IndexT localBlocks = hasTotalBlocks ? static_cast<IndexT>(globalBlocks / mySizeIdx)
                                                : static_cast<IndexT>(1);
            localBlocks = (localBlocks <= 0) ? static_cast<IndexT>(1) : localBlocks;
            const IndexT globalIdxSize = blkSizeVal * globalBlocks;
            const IndexT localIdxSize = blkSizeVal * localBlocks;
            const OffsetT rowStart = (rowIdx == 0) ? static_cast<OffsetT>(0) : offsets[rowIdx - 1];
            const OffsetT rowEndIdx = offsets[rowIdx];

            const UIndexT uLocalIdxSize = static_cast<UIndexT>(localIdxSize);
            const UIndexT uGlobalBlocks = static_cast<UIndexT>(globalBlocks);
            const UIndexT uLocalBlocks = static_cast<UIndexT>(localBlocks);

            const IndexT blkScalar = (usePos && blkSizeVal == 0 && globalBlocks > 0)
                ? static_cast<IndexT>(static_cast<UIndexT>(posPtr[posLen - 1]) / uGlobalBlocks)
                : static_cast<IndexT>(1);
            for (OffsetT i = rowStart + laneId; i < rowEndIdx; i += warpSize) {
                const UIndexT idxUnsigned = static_cast<UIndexT>(indices[i]);
                IndexT bucket;
                if (usePos) {
                    UIndexT idxAdj = idxUnsigned;
                    if (blkSizeVal == 0 && globalBlocks > 0) {
                        idxAdj = (idxUnsigned % uGlobalBlocks) * static_cast<UIndexT>(blkScalar);
                    }
                    int64_t first = 0;
                    int64_t last = posLen;
                    while (first < last) {
                        const int64_t mid = first + ((last - first) >> 1);
                        if (static_cast<UIndexT>(posPtr[mid]) <= idxAdj) {
                            first = mid + 1;
                        } else {
                            last = mid;
                        }
                    }
                    const int64_t lb = first - 1;
                    if (lb >= 0) {
                        bucket = static_cast<IndexT>((lb < static_cast<int64_t>(mySizeIdx)) ?
                            lb : static_cast<int64_t>(fdMySize.Mod(idxUnsigned)));
                    } else {
                        bucket = static_cast<IndexT>(fdMySize.Mod(idxUnsigned));
                    }
                } else {
                    if (blkSizeVal != 0 && idxUnsigned < static_cast<UIndexT>(globalIdxSize)) {
                        bucket = static_cast<IndexT>(idxUnsigned / uLocalIdxSize);
                    } else {
                        bucket = static_cast<IndexT>((idxUnsigned % uGlobalBlocks) / uLocalBlocks);
                    }
                }
                asc_atomic_add(&newLengths[static_cast<int32_t>(bucket) * lengthsSize + rowIdx],
                               static_cast<OffsetT>(1));
            }
        }
    }

    __simt_vf__ __aicore__ static LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtScatterNewIndicesPooledUbAtomic(
        const __gm__ IndexT* blockSizes,
        const __gm__ IndexT* totalNumBlocks,
        const __gm__ OffsetT* offsets,
        const __gm__ IndexT* indices,
        const __gm__ float* weights,
        __gm__ float* newWeights,
        __gm__ IndexT* newPos,
        const __gm__ OffsetT* newOffsets,
        __gm__ IndexT* newIndices,
        __ubuf__ OffsetT* ubPtr,
        int32_t lengthsSize,
        int32_t batchSize,
        int32_t mySize,
        int32_t indicesSize,
        int32_t rowBegin,
        int32_t rowEnd,
        bool weightsEnabled,
        bool bucketizePosEnabled,
        bool keepOrigIdxEnabled,
        bool totalNumBlocksEnabled,
        const __gm__ OffsetT* batchSizeOffsets,
        int32_t numFeatures,
        bool batchSizePerFeatureEnabled,
        const __gm__ uint64_t* blockBucketizePosPtrs,
        const __gm__ int64_t* blockBucketizePosLens,
        uint64_t mySizeDivMagic,
        uint32_t mySizeDivShift)
    {
        if (rowEnd <= rowBegin) {
            return;
        }
        const int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
        const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
        const int32_t warpsPerBlock = blockDim / warpSize;
        const int32_t warpId = threadIdx / warpSize;
        const int32_t laneId = threadIdx % warpSize;
        using UIndexT = std::make_unsigned_t<IndexT>;
        const IndexT mySizeIdx = static_cast<IndexT>(mySize);
        const bool writeWeights = weightsEnabled && (weights != nullptr) && (newWeights != nullptr);
        const bool writePositions = bucketizePosEnabled && (newPos != nullptr);
        const int32_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;
        const bool hasTotalBlocks = totalNumBlocksEnabled && (totalNumBlocks != nullptr);
        const bool hasPosList = (blockBucketizePosPtrs != nullptr) && (blockBucketizePosLens != nullptr);
        const FastDivmod<UIndexT> fdMySize(
            static_cast<UIndexT>(mySizeDivMagic), mySizeDivShift, static_cast<UIndexT>(mySizeIdx));

        __ubuf__ int32_t* ubCounters = reinterpret_cast<__ubuf__ int32_t*>(ubPtr)
                                       + (warpId * mySize);

        for (int32_t rowIdx = rowBegin + warpId; rowIdx < clampedEnd; rowIdx += warpsPerBlock) {
            for (int32_t b = laneId; b < mySize; b += warpSize) {
                ubCounters[b] = 0;
            }

            const int32_t featureIndex = ResolveFeatureIndexForRow(
                rowIdx, batchSize, batchSizeOffsets, numFeatures, batchSizePerFeatureEnabled);
            if (featureIndex < 0 || featureIndex >= numFeatures) {
                continue;
            }
            __gm__ IndexT* posPtr = nullptr;
            int64_t posLen = 0;
            if (hasPosList) {
                posPtr = reinterpret_cast<__gm__ IndexT*>(blockBucketizePosPtrs[featureIndex]);
                posLen = blockBucketizePosLens[featureIndex];
            }
            const bool usePos = hasPosList && (posPtr != nullptr) && (posLen > 0);
            const IndexT blkSizeVal = blockSizes[featureIndex];
            if (blkSizeVal < 0) {
                continue;
            }
            IndexT globalBlocks = hasTotalBlocks ? totalNumBlocks[featureIndex] : mySizeIdx;
            if (globalBlocks <= 0) {
                globalBlocks = mySizeIdx;
            }
            IndexT localBlocks = hasTotalBlocks ? static_cast<IndexT>(globalBlocks / mySizeIdx)
                                                : static_cast<IndexT>(1);
            localBlocks = (localBlocks <= 0) ? static_cast<IndexT>(1) : localBlocks;
            const IndexT globalIdxSize = blkSizeVal * globalBlocks;
            const IndexT localIdxSize = blkSizeVal * localBlocks;
            const OffsetT rowStart = (rowIdx == 0) ? static_cast<OffsetT>(0) : offsets[rowIdx - 1];
            const OffsetT rowEndIdx = offsets[rowIdx];

            const UIndexT uLocalIdxSize = static_cast<UIndexT>(localIdxSize);
            const UIndexT uGlobalBlocks = static_cast<UIndexT>(globalBlocks);
            const UIndexT uLocalBlocks = static_cast<UIndexT>(localBlocks);

            const IndexT blkScalar = (usePos && blkSizeVal == 0 && globalBlocks > 0)
                ? static_cast<IndexT>(static_cast<UIndexT>(posPtr[posLen - 1]) / uGlobalBlocks)
                : static_cast<IndexT>(1);

            for (OffsetT i = rowStart + laneId; i < rowEndIdx; i += warpSize) {
                const UIndexT idxUnsigned = static_cast<UIndexT>(indices[i]);
                IndexT bucket;
                IndexT finalIndex;
                if (usePos) {
                    UIndexT idxAdj = idxUnsigned;
                    if (blkSizeVal == 0 && globalBlocks > 0) {
                        idxAdj = (idxUnsigned % uGlobalBlocks) * static_cast<UIndexT>(blkScalar);
                    }
                    int64_t first = 0;
                    int64_t last = posLen;
                    while (first < last) {
                        const int64_t mid = first + ((last - first) >> 1);
                        if (static_cast<UIndexT>(posPtr[mid]) <= idxAdj) {
                            first = mid + 1;
                        } else {
                            last = mid;
                        }
                    }
                    const int64_t lb = first - 1;
                    if (lb >= 0) {
                        bucket = static_cast<IndexT>((lb < static_cast<int64_t>(mySizeIdx)) ?
                            lb : static_cast<int64_t>(fdMySize.Mod(idxUnsigned)));
                    } else {
                        bucket = static_cast<IndexT>(fdMySize.Mod(idxUnsigned));
                    }
                    if (keepOrigIdxEnabled) {
                        finalIndex = static_cast<IndexT>(idxUnsigned);
                    } else if (blkSizeVal == 0 && globalBlocks > 0) {
                        finalIndex = static_cast<IndexT>(idxUnsigned / uGlobalBlocks);
                    } else if (lb >= 0 && lb < static_cast<int64_t>(mySizeIdx)) {
                        finalIndex = static_cast<IndexT>(idxUnsigned - static_cast<UIndexT>(posPtr[lb]));
                    } else {
                        finalIndex = static_cast<IndexT>(fdMySize.Div(idxUnsigned));
                    }
                } else {
                    if (blkSizeVal != 0 && idxUnsigned < static_cast<UIndexT>(globalIdxSize)) {
                        bucket = static_cast<IndexT>(idxUnsigned / uLocalIdxSize);
                        finalIndex = keepOrigIdxEnabled ?
                            static_cast<IndexT>(idxUnsigned) : static_cast<IndexT>(idxUnsigned % uLocalIdxSize);
                    } else {
                        bucket = static_cast<IndexT>((idxUnsigned % uGlobalBlocks) / uLocalBlocks);
                        finalIndex = keepOrigIdxEnabled ?
                            static_cast<IndexT>(idxUnsigned) :
                            static_cast<IndexT>(idxUnsigned / uGlobalBlocks);
                    }
                }
                const int32_t relativePos = asc_atomic_add(&ubCounters[bucket], 1);
                const OffsetT baseOffset = newOffsets[static_cast<int32_t>(bucket) * lengthsSize + rowIdx];
                const OffsetT writePos = baseOffset + static_cast<OffsetT>(relativePos);
                if (writePos < static_cast<OffsetT>(indicesSize)) {
                    newIndices[writePos] = finalIndex;
                    if (writeWeights) {
                        newWeights[writePos] = weights[i];
                    }
                    if (writePositions) {
                        newPos[writePos] = static_cast<IndexT>(i - rowStart);
                    }
                }
            }
        }
    }

    __simt_vf__ __aicore__ static LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtScatterNewIndicesRowsAtomic(
        const __gm__ IndexT* blockSizes,
        const __gm__ IndexT* totalNumBlocks,
        const __gm__ OffsetT* offsets,
        const __gm__ IndexT* indices,
        const __gm__ float* weights,
        __gm__ float* newWeights,
        __gm__ IndexT* newPos,
        __gm__ OffsetT* newOffsets,
        __gm__ IndexT* newIndices,
        __gm__ IndexT* unbucketizePermute,
        int32_t lengthsSize,
        int32_t batchSize,
        int32_t mySize,
        int32_t indicesSize,
        int32_t rowBegin,
        int32_t rowEnd,
        bool sequenceEnabled,
        bool weightsEnabled,
        bool bucketizePosEnabled,
        bool keepOrigIdxEnabled,
        bool totalNumBlocksEnabled,
        const __gm__ OffsetT* batchSizeOffsets,
        int32_t numFeatures,
        bool batchSizePerFeatureEnabled,
        const __gm__ uint64_t* blockBucketizePosPtrs,
        const __gm__ int64_t* blockBucketizePosLens,
        uint64_t mySizeDivMagic,
        uint32_t mySizeDivShift)
    {
        if (rowEnd <= rowBegin) {
            return;
        }
        using UIndexT = std::make_unsigned_t<IndexT>;
        const int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
        const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
        const IndexT mySizeIdx = static_cast<IndexT>(mySize);
        const bool writeWeights = weightsEnabled && (weights != nullptr) && (newWeights != nullptr);
        const bool writePositions = bucketizePosEnabled && (newPos != nullptr);
        const bool writeUnbucketizePermute = sequenceEnabled && (unbucketizePermute != nullptr);
        const int32_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;
        const bool hasTotalBlocks = totalNumBlocksEnabled && (totalNumBlocks != nullptr);
        const bool hasPosList = (blockBucketizePosPtrs != nullptr) && (blockBucketizePosLens != nullptr);
        const FastDivmod<UIndexT> fdMySize(
            static_cast<UIndexT>(mySizeDivMagic), mySizeDivShift, static_cast<UIndexT>(mySizeIdx));
        for (int32_t rowIdx = rowBegin + threadIdx; rowIdx < clampedEnd; rowIdx += blockDim) {
            const int32_t featureIndex = ResolveFeatureIndexForRow(
                rowIdx, batchSize, batchSizeOffsets, numFeatures, batchSizePerFeatureEnabled);
            if (featureIndex < 0 || featureIndex >= numFeatures) {
                continue;
            }
            __gm__ IndexT* posPtr = nullptr;
            int64_t posLen = 0;
            if (hasPosList) {
                posPtr = reinterpret_cast<__gm__ IndexT*>(blockBucketizePosPtrs[featureIndex]);
                posLen = blockBucketizePosLens[featureIndex];
            }
            const bool usePos = hasPosList && (posPtr != nullptr) && (posLen > 0);
            const IndexT blkSizeVal = blockSizes[featureIndex];
            if (blkSizeVal < 0) {
                continue;
            }
            IndexT globalBlocks = hasTotalBlocks ? totalNumBlocks[featureIndex] : mySizeIdx;
            if (globalBlocks <= 0) {
                globalBlocks = mySizeIdx;
            }
            IndexT localBlocks = hasTotalBlocks ? static_cast<IndexT>(globalBlocks / mySizeIdx)
                                                : static_cast<IndexT>(1);
            localBlocks = (localBlocks <= 0) ? static_cast<IndexT>(1) : localBlocks;
            const IndexT globalIdxSize = blkSizeVal * globalBlocks;
            const IndexT localIdxSize = blkSizeVal * localBlocks;
            const OffsetT rowStart = (rowIdx == 0) ? static_cast<OffsetT>(0) : offsets[rowIdx - 1];
            const OffsetT rowEndIdx = offsets[rowIdx];

            const UIndexT uLocalIdxSize = static_cast<UIndexT>(localIdxSize);
            const UIndexT uGlobalBlocks = static_cast<UIndexT>(globalBlocks);
            const UIndexT uLocalBlocks = static_cast<UIndexT>(localBlocks);

            const IndexT blkScalar = (usePos && blkSizeVal == 0 && globalBlocks > 0)
                ? static_cast<IndexT>(static_cast<UIndexT>(posPtr[posLen - 1]) / uGlobalBlocks)
                : static_cast<IndexT>(1);

            for (OffsetT i = rowStart; i < rowEndIdx; ++i) {
                const UIndexT idxUnsigned = static_cast<UIndexT>(indices[i]);
                IndexT bucket;
                IndexT finalIndex;
                if (usePos) {
                    UIndexT idxAdj = idxUnsigned;
                    if (blkSizeVal == 0 && globalBlocks > 0) {
                        idxAdj = (idxUnsigned % uGlobalBlocks) * static_cast<UIndexT>(blkScalar);
                    }
                    int64_t first = 0;
                    int64_t last = posLen;
                    while (first < last) {
                        const int64_t mid = first + ((last - first) >> 1);
                        if (static_cast<UIndexT>(posPtr[mid]) <= idxAdj) {
                            first = mid + 1;
                        } else {
                            last = mid;
                        }
                    }
                    const int64_t lb = first - 1;
                    if (lb >= 0) {
                        bucket = static_cast<IndexT>((lb < static_cast<int64_t>(mySizeIdx)) ?
                            lb : static_cast<int64_t>(fdMySize.Mod(idxUnsigned)));
                    } else {
                        bucket = static_cast<IndexT>(fdMySize.Mod(idxUnsigned));
                    }
                    if (keepOrigIdxEnabled) {
                        finalIndex = static_cast<IndexT>(idxUnsigned);
                    } else if (blkSizeVal == 0 && globalBlocks > 0) {
                        finalIndex = static_cast<IndexT>(idxUnsigned / uGlobalBlocks);
                    } else if (lb >= 0 && lb < static_cast<int64_t>(mySizeIdx)) {
                        finalIndex = static_cast<IndexT>(idxUnsigned - static_cast<UIndexT>(posPtr[lb]));
                    } else {
                        finalIndex = static_cast<IndexT>(fdMySize.Div(idxUnsigned));
                    }
                } else {
                    if (blkSizeVal != 0 && idxUnsigned < static_cast<UIndexT>(globalIdxSize)) {
                        bucket = static_cast<IndexT>(idxUnsigned / uLocalIdxSize);
                        finalIndex = keepOrigIdxEnabled ?
                            static_cast<IndexT>(idxUnsigned) : static_cast<IndexT>(idxUnsigned % uLocalIdxSize);
                    } else {
                        bucket = static_cast<IndexT>((idxUnsigned % uGlobalBlocks) / uLocalBlocks);
                        finalIndex = keepOrigIdxEnabled ?
                            static_cast<IndexT>(idxUnsigned) :
                            static_cast<IndexT>(idxUnsigned / uGlobalBlocks);
                    }
                }
                const int32_t slot = static_cast<int32_t>(bucket) * lengthsSize + rowIdx;
                const OffsetT writeCursor = newOffsets[slot];
                newOffsets[slot] = writeCursor + 1;
                if (writeCursor < static_cast<OffsetT>(indicesSize)) {
                    newIndices[writeCursor] = finalIndex;
                    if (writeUnbucketizePermute) {
                        unbucketizePermute[i] = static_cast<IndexT>(writeCursor);
                    }
                    if (writeWeights) {
                        newWeights[writeCursor] = weights[i];
                    }
                    if (writePositions) {
                        newPos[writeCursor] = static_cast<IndexT>(i - rowStart);
                    }
                }
            }
        }
    }
};

#endif // BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_FULL_H

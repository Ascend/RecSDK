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

#ifndef BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_H
#define BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_H

#include <cstdint>
#include <type_traits>
#include <limits>

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "asynchronous_complete_cumsum_kernel.h"
#include "fast_divmod.h"

using namespace AscendC;

template <typename OffsetT, typename IndexT>
class BlockBucketizeSparseFeaturesKernel {
public:
    struct KernelArgs {
        __gm__ OffsetT* lengths;
        __gm__ IndexT* indices;
        __gm__ IndexT* blockSizes;
        __gm__ OffsetT* batchSizePerFeature;
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
        __gm__ OffsetT* writeOffsets;
        __gm__ OffsetT* batchSizeOffsets;
        __gm__ OffsetT* blockSums;
        int64_t lengthsSize;
        int64_t indicesSize;
        int64_t numFeatures;
        int64_t batchSize;
        int64_t mySize;
        int64_t newLengthsSize;
        bool sequenceEnabled;
        bool weightsEnabled;
        bool bucketizePosEnabled;
        bool keepOrigIdxEnabled;
        bool totalNumBlocksEnabled;
        int64_t maxBatchSize;
        bool batchSizePerFeatureEnabled;
        bool blockBucketizePosEnabled;
        uint64_t mySizeDivMagic;
        uint32_t mySizeDivShift;
    };

    __aicore__ inline BlockBucketizeSparseFeaturesKernel(KernelArgs& args)
        :lengths(args.lengths),
         indices(args.indices),
         blockSizes(args.blockSizes),
         batchSizePerFeature(args.batchSizePerFeature),
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
         writeOffsets(args.writeOffsets),
         batchSizeOffsets(args.batchSizeOffsets),
         blockSums(args.blockSums),
         lengthsSize(args.lengthsSize),
         indicesSize(args.indicesSize),
         numFeatures(args.numFeatures),
         batchSize(args.batchSize),
         mySize(args.mySize),
         newLengthsSize(args.newLengthsSize),
         sequenceEnabled(args.sequenceEnabled),
         weightsEnabled(args.weightsEnabled),
         bucketizePosEnabled(args.bucketizePosEnabled),
         keepOrigIdxEnabled(args.keepOrigIdxEnabled),
         totalNumBlocksEnabled(args.totalNumBlocksEnabled),
         maxBatchSize(args.maxBatchSize),
         batchSizePerFeatureEnabled(args.batchSizePerFeatureEnabled),
         blockBucketizePosEnabled(args.blockBucketizePosEnabled),
         mySizeDivMagic(args.mySizeDivMagic),
         mySizeDivShift(args.mySizeDivShift)
    {
        // UB空间分配：取 cumsum 和 pooled scatter 所需字节数的较大值
        // cumsum 阶段使用 OffsetT 类型元素
        const int64_t cumsumUbBytes = (static_cast<int64_t>(MAX_THREADS_PER_BLOCK) + 1) *
                                      static_cast<int64_t>(sizeof(OffsetT));
        // pooled scatter 阶段使用 int32_t 计数器（UB asc_atomic_add 仅支持 int32_t）
        const int64_t pooledWarps = static_cast<int64_t>(WARPS_PER_BLOCK);
        const int64_t pooledUbBytes = pooledWarps * static_cast<int64_t>(args.mySize) *
                                      static_cast<int64_t>(sizeof(int32_t));
        const int64_t ubBytes = (cumsumUbBytes > pooledUbBytes) ? cumsumUbBytes : pooledUbBytes;
        pipe.InitBuffer(ubBuf, static_cast<int32_t>(ubBytes));
        ubTensor = ubBuf.Get<OffsetT>();
        ubPtr = reinterpret_cast<__ubuf__ OffsetT*>(ubTensor.GetPhyAddr());
    }

    __aicore__ inline void Process()
    {
        const int32_t coreNum = static_cast<int32_t>(AscendC::GetBlockNum());
        const int32_t coreId = static_cast<int32_t>(AscendC::GetBlockIdx());

        WorkRange rowRange = {0};
        ComputeWorkRange(lengthsSize, coreId, coreNum, rowRange);

        if (blockBucketizePosEnabled) {
            BuildBucketizePosTables();
        }

        // Stage 1: 对lengths做前缀和得到offsets
        RunCumsum(lengths, offsets, blockSums, ubPtr, lengthsSize);
        AscendC::SyncAll();

        // Stage 1.5: 构建batch_size_per_feature的前缀和
        if (batchSizePerFeatureEnabled) {
            RunCumsum(batchSizePerFeature, batchSizeOffsets, blockSums, ubPtr, numFeatures);
            AscendC::SyncAll();
        }

        // Stage 2 合并：UB asc_atomic_add 一次遍历统计 new_lengths
        AscendC::Simt::VF_CALL<SimtComputeNewLengthsUbAtomic>(
            AscendC::Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1},
            blockSizes,
            totalNumBlocks,
            offsets,
            indices,
            newLengths,
            ubPtr,
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
        AscendC::SyncAll();

        // Stage 3: 对new_lengths做前缀和，得到bucket化后的offsets
        RunCumsum(newLengths, newOffsets, blockSums, ubPtr, newLengthsSize);
        AscendC::SyncAll();

        // Stage 4: 再次遍历indices，按bucket顺序写出new_indices
        if (!sequenceEnabled) {
            // Pooled 路径
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
            // Sequence模式：（单线程-per-Row，GM读写）
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
    // 统一 cumsum 入口：对 input[0..totalLength-1] 做前缀和写入 output，blockSums 复用共享 buffer
    __aicore__ inline void RunCumsum(__gm__ OffsetT* input,
                                     __gm__ OffsetT* output,
                                     __gm__ OffsetT* blockSumsPtr,
                                     __ubuf__ OffsetT* ubMem,
                                     int64_t totalLength)
    {
        if (totalLength <= 0) {
            return;
        }
        const int32_t coreNum = static_cast<int32_t>(AscendC::GetBlockNum());
        const int32_t coreId = static_cast<int32_t>(AscendC::GetBlockIdx());
        const bool isInt32 = (sizeof(OffsetT) == 4);
        const int64_t smallThreshold = isInt32 ? (24 * CUMSUM_THREADS_PER_BLOCK) : (44 * CUMSUM_THREADS_PER_BLOCK);
        const bool isSmall = (totalLength <= smallThreshold);
        const int64_t perBlockCapacity = isSmall ?
            static_cast<int64_t>(CUMSUM_THREADS_PER_BLOCK) :
            static_cast<int64_t>(CUMSUM_THREADS_PER_BLOCK) * AsynchronousCompleteCumsumSimt::MAX_ELEMENTS_PER_THREAD;
        const int64_t totalBlocksVal = (totalLength + perBlockCapacity - 1) / perBlockCapacity;
        const int32_t totalBlocks = static_cast<int32_t>(totalBlocksVal > 0 ? totalBlocksVal : 1);

        if (totalBlocks <= 0) {
            return;
        }

        const int32_t totalLength32 = static_cast<int32_t>(totalLength > 0 ? totalLength : 0);
        if (isSmall && totalBlocks <= coreNum) {
            AscendC::Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtSmallDataCompute<OffsetT>>(
                AscendC::Simt::Dim3{CUMSUM_THREADS_PER_BLOCK, 1, 1},
                input, output, blockSumsPtr, ubMem, totalLength32, totalBlocks);
            AscendC::SyncAll();
            if (totalBlocks > 1) {
                AscendC::Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtSmallDataUpdate<OffsetT>>(
                    AscendC::Simt::Dim3{CUMSUM_THREADS_PER_BLOCK, 1, 1},
                    output, blockSumsPtr, totalLength32, totalBlocks);
            }
        } else {
            const int32_t blocksPerCore = totalBlocks / coreNum;
            const int32_t remainderBlocks = totalBlocks % coreNum;
            const int32_t blockStartIdx =
                coreId * blocksPerCore + (coreId < remainderBlocks ? coreId : remainderBlocks);
            const int32_t curBlocksCount = blocksPerCore + (coreId < remainderBlocks ? 1 : 0);

            AscendC::Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtLargeDataCompute<OffsetT>>(
                AscendC::Simt::Dim3{CUMSUM_THREADS_PER_BLOCK, 1, 1},
                input, output, blockSumsPtr, ubMem, totalLength32, totalBlocks, blockStartIdx, curBlocksCount);
            AscendC::SyncAll();
            AscendC::Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtLargeDataUpdate<OffsetT>>(
                AscendC::Simt::Dim3{CUMSUM_THREADS_PER_BLOCK, 1, 1},
                output, blockSumsPtr, totalLength32, totalBlocks, blockStartIdx, curBlocksCount);
        }
    }

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

        const int64_t totalBlocks = (numFeatures + CACHE_LINE_ELEMS_64BIT - 1) / CACHE_LINE_ELEMS_64BIT;

        int64_t startBlock = 0;
        int64_t myBlocks = 0;

        if (totalBlocks > 0) {
            const int64_t blocksPerCore = totalBlocks / coreNum;
            const int64_t remainder = totalBlocks % coreNum;
            startBlock = coreId * blocksPerCore + ((coreId < remainder) ? coreId : remainder);
            myBlocks = blocksPerCore + ((coreId < remainder) ? 1 : 0);
        }

        if (myBlocks <= 0) {
            return;
        }

        const int64_t startFeature = startBlock * CACHE_LINE_ELEMS_64BIT;
        const int64_t endFeature = (startFeature + myBlocks * CACHE_LINE_ELEMS_64BIT) > numFeatures ?
                                   numFeatures : (startFeature + myBlocks * CACHE_LINE_ELEMS_64BIT);

        for (int64_t featureIndex = startFeature; featureIndex < endFeature; ++featureIndex) {
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
            int64_t lastBlockStart = (endFeature & ~7);
            dcci(blockBucketizePosPtrs + lastBlockStart, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
            dcci(blockBucketizePosLens + lastBlockStart, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        }
    }

    struct WorkRange {
        int64_t begin;
        int64_t end;
    };

    static constexpr int32_t MAX_THREADS_PER_BLOCK = 512;
    static constexpr int32_t WARP_SIZE = 32;
    static constexpr int32_t WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / WARP_SIZE;
    static constexpr int64_t CACHE_LINE_ELEMS_64BIT = 8;  // 64 bytes / 8 bytes per element
    static constexpr int32_t CUMSUM_THREADS_PER_BLOCK = AsynchronousCompleteCumsumSimt::MAX_THREADS_PER_BLOCK;

    __gm__ OffsetT* lengths;
    __gm__ IndexT* indices;
    __gm__ IndexT* blockSizes;
    __gm__ OffsetT* batchSizePerFeature;
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
    __gm__ OffsetT* writeOffsets;
    __gm__ OffsetT* batchSizeOffsets;
    __gm__ OffsetT* blockSums;
    int64_t lengthsSize;
    int64_t indicesSize;
    int64_t numFeatures;
    int64_t batchSize;
    int64_t mySize;
    int64_t newLengthsSize;
    bool sequenceEnabled;
    bool weightsEnabled;
    bool bucketizePosEnabled;
    bool keepOrigIdxEnabled;
    bool totalNumBlocksEnabled;
    int64_t maxBatchSize;
    bool batchSizePerFeatureEnabled;
    bool blockBucketizePosEnabled;
    uint64_t mySizeDivMagic;
    uint32_t mySizeDivShift;

    TPipe pipe;
    TBuf<TPosition::VECCALC> ubBuf;
    LocalTensor<OffsetT> ubTensor;
    __ubuf__ OffsetT* ubPtr;

    __aicore__ static inline int64_t ResolveFeatureIndexForRow(
        int64_t row,
        int64_t batchSize,
        const __gm__ OffsetT* batchSizeOffsets,
        int64_t numFeatures,
        bool useBatchSizePerFeature)
    {
        if (!useBatchSizePerFeature) {
            const int64_t safeBatch = (batchSize <= 0) ? 1 : batchSize;
            return row / safeBatch;
        }
        if (batchSizeOffsets == nullptr || numFeatures <= 0) {
            return 0;
        }
        int64_t left = 0;
        int64_t right = numFeatures;
        while (left < right) {
            const int64_t mid = (left + right + 1) >> 1;
            const int64_t offset = static_cast<int64_t>(batchSizeOffsets[mid]);
            if (offset <= row) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    __aicore__ static inline void ComputeWorkRange(int64_t totalSize, int32_t coreId, int32_t coreNum, WorkRange &range)
    {
        if (totalSize <= 0 || coreNum <= 0 || coreId < 0 || coreId >= coreNum) {
            range = {0, 0};
            return;
        }
        const int64_t base = totalSize / coreNum;
        const int64_t remainder = totalSize % coreNum;
        const int64_t begin = coreId * base + ((coreId < remainder) ? coreId : remainder);
        range = {begin, begin + base + ((coreId < remainder) ? 1 : 0)};
    }

    // 每个 Warp 处理一个 Row
    __simt_vf__ __aicore__ static LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtComputeNewLengthsUbAtomic(
        const __gm__ IndexT* blockSizes,
        const __gm__ IndexT* totalNumBlocks,
        const __gm__ OffsetT* offsets,
        const __gm__ IndexT* indices,
        __gm__ OffsetT* newLengths,
        __ubuf__ OffsetT* ubPtr,
        int64_t lengthsSize,
        int64_t batchSize,
        int64_t mySize,
        int64_t rowBegin,
        int64_t rowEnd,
        bool totalNumBlocksEnabled,
        const __gm__ OffsetT* batchSizeOffsets,
        int64_t numFeatures,
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
        const int32_t warpsPerBlock = blockDim / WARP_SIZE;
        const int32_t warpId = threadIdx / WARP_SIZE;
        const int32_t laneId = threadIdx % WARP_SIZE;
        using UIndexT = std::make_unsigned_t<IndexT>;
        const IndexT mySizeUnsigned = static_cast<IndexT>(static_cast<int64_t>(mySize));
        const int64_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;
        const bool hasTotalBlocks = totalNumBlocksEnabled && (totalNumBlocks != nullptr);
        const bool hasPosList = (blockBucketizePosPtrs != nullptr) && (blockBucketizePosLens != nullptr);
        const FastDivmod<UIndexT> fdMySize(
            static_cast<UIndexT>(mySizeDivMagic), mySizeDivShift, static_cast<UIndexT>(mySizeUnsigned));

        __ubuf__ int32_t* ubCounters = reinterpret_cast<__ubuf__ int32_t*>(ubPtr)
                                       + (warpId * mySize);

        for (int64_t rowIdx = rowBegin + warpId; rowIdx < clampedEnd; rowIdx += warpsPerBlock) {
            for (int64_t b = laneId; b < mySize; b += WARP_SIZE) {
                ubCounters[b] = 0;
            }

            const int64_t featureIndex = ResolveFeatureIndexForRow(
                rowIdx, batchSize, batchSizeOffsets, numFeatures, batchSizePerFeatureEnabled);
            if (featureIndex < 0 || featureIndex >= numFeatures) {
                for (int64_t b = laneId; b < mySize; b += WARP_SIZE) {
                    newLengths[b * lengthsSize + rowIdx] = static_cast<OffsetT>(0);
                }
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
                for (int64_t b = laneId; b < mySize; b += WARP_SIZE) {
                    newLengths[b * lengthsSize + rowIdx] = static_cast<OffsetT>(0);
                }
                continue;
            }
            IndexT globalBlocks = hasTotalBlocks ? totalNumBlocks[featureIndex] : mySizeUnsigned;
            if (globalBlocks <= 0) {
                globalBlocks = mySizeUnsigned;
            }
            IndexT localBlocks = hasTotalBlocks ? static_cast<IndexT>(globalBlocks / mySizeUnsigned)
                                                : static_cast<IndexT>(1);
            localBlocks = (localBlocks <= 0) ? static_cast<IndexT>(1) : localBlocks;
            const IndexT globalIdxSize = blkSizeVal * globalBlocks;
            const IndexT localIdxSize = blkSizeVal * localBlocks;
            const int64_t rowStart = static_cast<int64_t>(offsets[rowIdx]);
            const int64_t rowIndicesEnd = static_cast<int64_t>(offsets[rowIdx + 1]);

            const UIndexT uLocalIdxSize = static_cast<UIndexT>(localIdxSize);
            const UIndexT uGlobalBlocks = static_cast<UIndexT>(globalBlocks);
            const UIndexT uLocalBlocks = static_cast<UIndexT>(localBlocks);

            const IndexT blkScalar = (usePos && blkSizeVal == 0 && globalBlocks > 0)
                ? static_cast<IndexT>(static_cast<UIndexT>(posPtr[posLen - 1]) / uGlobalBlocks)
                : static_cast<IndexT>(1);

            // 一次遍历 indices，UB asc_atomic_add 统计
            for (int64_t i = rowStart + laneId; i < rowIndicesEnd; i += WARP_SIZE) {
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
                        bucket = static_cast<IndexT>((lb < static_cast<int64_t>(mySizeUnsigned)) ?
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
                asc_atomic_add(&ubCounters[bucket], 1);
            }

            // 将 UB 计数器写回 newLengths
            for (int64_t b = laneId; b < mySize; b += WARP_SIZE) {
                newLengths[b * lengthsSize + rowIdx] = static_cast<OffsetT>(ubCounters[b]);
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
        int64_t lengthsSize,
        int64_t batchSize,
        int64_t mySize,
        int64_t indicesSize,
        int64_t rowBegin,
        int64_t rowEnd,
        bool weightsEnabled,
        bool bucketizePosEnabled,
        bool keepOrigIdxEnabled,
        bool totalNumBlocksEnabled,
        const __gm__ OffsetT* batchSizeOffsets,
        int64_t numFeatures,
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
        const int32_t warpsPerBlock = blockDim / WARP_SIZE;
        const int32_t warpId = threadIdx / WARP_SIZE;
        const int32_t laneId = threadIdx % WARP_SIZE;
        using UIndexT = std::make_unsigned_t<IndexT>;
        const IndexT mySizeUnsigned = static_cast<IndexT>(static_cast<int64_t>(mySize));
        const bool writeWeights = weightsEnabled && (weights != nullptr) && (newWeights != nullptr);
        const bool writePositions = bucketizePosEnabled && (newPos != nullptr);
        const int64_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;
        const bool hasTotalBlocks = totalNumBlocksEnabled && (totalNumBlocks != nullptr);
        const bool hasPosList = (blockBucketizePosPtrs != nullptr) && (blockBucketizePosLens != nullptr);
        const FastDivmod<UIndexT> fdMySize(
            static_cast<UIndexT>(mySizeDivMagic), mySizeDivShift, static_cast<UIndexT>(mySizeUnsigned));

        __ubuf__ int32_t* ubCounters = reinterpret_cast<__ubuf__ int32_t*>(ubPtr)
                                       + (warpId * mySize);

        for (int64_t rowIdx = rowBegin + warpId; rowIdx < clampedEnd; rowIdx += warpsPerBlock) {
            for (int64_t b = laneId; b < mySize; b += WARP_SIZE) {
                ubCounters[b] = 0;
            }

            const int64_t featureIndex = ResolveFeatureIndexForRow(
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
            IndexT globalBlocks = hasTotalBlocks ? totalNumBlocks[featureIndex] : mySizeUnsigned;
            if (globalBlocks <= 0) {
                globalBlocks = mySizeUnsigned;
            }
            IndexT localBlocks = hasTotalBlocks ? static_cast<IndexT>(globalBlocks / mySizeUnsigned)
                                                : static_cast<IndexT>(1);
            localBlocks = (localBlocks <= 0) ? static_cast<IndexT>(1) : localBlocks;
            const IndexT globalIdxSize = blkSizeVal * globalBlocks;
            const IndexT localIdxSize = blkSizeVal * localBlocks;
            const int64_t rowStart = static_cast<int64_t>(offsets[rowIdx]);
            const int64_t rowIndicesEnd = static_cast<int64_t>(offsets[rowIdx + 1]);

            const UIndexT uLocalIdxSize = static_cast<UIndexT>(localIdxSize);
            const UIndexT uGlobalBlocks = static_cast<UIndexT>(globalBlocks);
            const UIndexT uLocalBlocks = static_cast<UIndexT>(localBlocks);

            const IndexT blkScalar = (usePos && blkSizeVal == 0 && globalBlocks > 0)
                ? static_cast<IndexT>(static_cast<UIndexT>(posPtr[posLen - 1]) / uGlobalBlocks)
                : static_cast<IndexT>(1);

            for (int64_t i = rowStart + laneId; i < rowIndicesEnd; i += WARP_SIZE) {
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
                        bucket = static_cast<IndexT>((lb < static_cast<int64_t>(mySizeUnsigned)) ?
                            lb : static_cast<int64_t>(fdMySize.Mod(idxUnsigned)));
                    } else {
                        bucket = static_cast<IndexT>(fdMySize.Mod(idxUnsigned));
                    }
                    if (keepOrigIdxEnabled) {
                        finalIndex = static_cast<IndexT>(idxUnsigned);
                    } else if (blkSizeVal == 0 && globalBlocks > 0) {
                        finalIndex = static_cast<IndexT>(idxUnsigned / uGlobalBlocks);
                    } else if (lb >= 0 && lb < static_cast<int64_t>(mySizeUnsigned)) {
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
                const int64_t baseOffset =
                    static_cast<int64_t>(newOffsets[static_cast<int64_t>(bucket) * lengthsSize + rowIdx]);
                const int64_t writePos = baseOffset + static_cast<int64_t>(relativePos);
                if (writePos < indicesSize) {
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
        __gm__ OffsetT* writeOffsets,
        __gm__ IndexT* newIndices,
        __gm__ IndexT* unbucketizePermute,
        int64_t lengthsSize,
        int64_t batchSize,
        int64_t mySize,
        int64_t indicesSize,
        int64_t rowBegin,
        int64_t rowEnd,
        bool sequenceEnabled,
        bool weightsEnabled,
        bool bucketizePosEnabled,
        bool keepOrigIdxEnabled,
        bool totalNumBlocksEnabled,
        const __gm__ OffsetT* batchSizeOffsets,
        int64_t numFeatures,
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
        OffsetT writeCursor;
        const int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
        const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
        const IndexT mySizeUnsigned = static_cast<IndexT>(static_cast<int64_t>(mySize));
        const bool writeWeights = weightsEnabled && (weights != nullptr) && (newWeights != nullptr);
        const bool writePositions = bucketizePosEnabled && (newPos != nullptr);
        const bool writeUnbucketizePermute = sequenceEnabled && (unbucketizePermute != nullptr);
        const int64_t clampedEnd = (rowEnd > lengthsSize) ? lengthsSize : rowEnd;
        const bool hasTotalBlocks = totalNumBlocksEnabled && (totalNumBlocks != nullptr);
        const bool hasPosList = (blockBucketizePosPtrs != nullptr) && (blockBucketizePosLens != nullptr);
        const FastDivmod<UIndexT> fdMySize(
            static_cast<UIndexT>(mySizeDivMagic), mySizeDivShift, static_cast<UIndexT>(mySizeUnsigned));
        for (int64_t rowIdx = rowBegin + threadIdx; rowIdx < clampedEnd; rowIdx += blockDim) {
            const int64_t featureIndex = ResolveFeatureIndexForRow(
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
            IndexT globalBlocks = hasTotalBlocks ? totalNumBlocks[featureIndex] : mySizeUnsigned;
            if (globalBlocks <= 0) {
                globalBlocks = mySizeUnsigned;
            }
            IndexT localBlocks = hasTotalBlocks ? static_cast<IndexT>(globalBlocks / mySizeUnsigned)
                                                : static_cast<IndexT>(1);
            localBlocks = (localBlocks <= 0) ? static_cast<IndexT>(1) : localBlocks;
            const IndexT globalIdxSize = blkSizeVal * globalBlocks;
            const IndexT localIdxSize = blkSizeVal * localBlocks;
            const int64_t rowStart = static_cast<int64_t>(offsets[rowIdx]);
            const int64_t rowIndicesEnd = static_cast<int64_t>(offsets[rowIdx + 1]);

            const UIndexT uLocalIdxSize = static_cast<UIndexT>(localIdxSize);
            const UIndexT uGlobalBlocks = static_cast<UIndexT>(globalBlocks);
            const UIndexT uLocalBlocks = static_cast<UIndexT>(localBlocks);

            const IndexT blkScalar = (usePos && blkSizeVal == 0 && globalBlocks > 0)
                ? static_cast<IndexT>(static_cast<UIndexT>(posPtr[posLen - 1]) / uGlobalBlocks)
                : static_cast<IndexT>(1);

            for (int64_t i = rowStart; i < rowIndicesEnd; ++i) {
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
                        bucket = static_cast<IndexT>((lb < static_cast<int64_t>(mySizeUnsigned)) ?
                            lb : static_cast<int64_t>(fdMySize.Mod(idxUnsigned)));
                    } else {
                        bucket = static_cast<IndexT>(fdMySize.Mod(idxUnsigned));
                    }
                    if (keepOrigIdxEnabled) {
                        finalIndex = static_cast<IndexT>(idxUnsigned);
                    } else if (blkSizeVal == 0 && globalBlocks > 0) {
                        finalIndex = static_cast<IndexT>(idxUnsigned / uGlobalBlocks);
                    } else if (lb >= 0 && lb < static_cast<int64_t>(mySizeUnsigned)) {
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
                const int64_t slot = static_cast<int64_t>(bucket) * lengthsSize + rowIdx;
                writeCursor = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                    L1CacheType::NON_CACHEABLE>(writeOffsets + slot);
                __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                    L1CacheType::NON_CACHEABLE>(writeOffsets + slot, writeCursor + 1);
                const int64_t writeIdx = static_cast<int64_t>(writeCursor);
                if (writeIdx < indicesSize) {
                    newIndices[writeIdx] = finalIndex;
                    if (writeUnbucketizePermute) {
                        unbucketizePermute[i] = static_cast<IndexT>(writeIdx);
                    }
                    if (writeWeights) {
                        newWeights[writeIdx] = weights[i];
                    }
                    if (writePositions) {
                        newPos[writeIdx] = static_cast<IndexT>(i - rowStart);
                    }
                }
            }
        }
    }
};

#endif // BLOCK_BUCKETIZE_SPARSE_FEATURES_KERNEL_H

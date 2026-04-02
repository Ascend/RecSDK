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

#include "block_bucketize_sparse_features_kernel_full.h"
#include "block_bucketize_sparse_features_kernel_simplified.h"

struct ComputeNewLengthsParams {
    GM_ADDR indices;
    GM_ADDR blockSizes;
    GM_ADDR offsets;
    GM_ADDR totalNumBlocks;
    GM_ADDR batchSizeOffsets;
    GM_ADDR blockBucketizePos;
    GM_ADDR newLengths;
    GM_ADDR workspace;
};

template <typename OffsetT, typename IndexT>
__aicore__ inline void RunComputeNewLengthsFullKernel(
    const ComputeNewLengthsParams& p,
    const BlockBucketizeSparseFeaturesTilingData& td)
{
    __gm__ uint8_t* wsBase = reinterpret_cast<__gm__ uint8_t*>(GetUserWorkspace(p.workspace));
    __gm__ uint64_t* posPtrs = td.enableBlockBucketizePos ?
        reinterpret_cast<__gm__ uint64_t*>(wsBase + td.posPtrsOffset) : nullptr;
    __gm__ int64_t* posLens = td.enableBlockBucketizePos ?
        reinterpret_cast<__gm__ int64_t*>(wsBase + td.posLensOffset) : nullptr;

    typename BlockBucketizeSparseFeaturesKernelFull<OffsetT, IndexT>::KernelArgs args{
        reinterpret_cast<__gm__ IndexT*>(p.indices),
        reinterpret_cast<__gm__ IndexT*>(p.blockSizes),
        reinterpret_cast<__gm__ IndexT*>(p.totalNumBlocks),
        reinterpret_cast<__gm__ void*>(p.blockBucketizePos),
        posPtrs,
        posLens,
        reinterpret_cast<__gm__ OffsetT*>(p.newLengths),
        nullptr,    // newIndices
        nullptr,    // unbucketizePermute
        nullptr,    // weights
        nullptr,    // newWeights
        nullptr,    // newPos
        reinterpret_cast<__gm__ OffsetT*>(p.offsets),
        nullptr,    // newOffsets
        reinterpret_cast<__gm__ OffsetT*>(p.batchSizeOffsets),
        static_cast<int32_t>(td.lengthsSize),
        static_cast<int32_t>(td.indicesSize),
        static_cast<int32_t>(td.numFeatures),
        static_cast<int32_t>(td.batchSize),
        static_cast<int32_t>(td.mySize),
        false,      // sequenceEnabled
        false,      // weightsEnabled
        td.enableBucketizePos,
        false,      // keepOrigIdxEnabled
        td.enableTotalNumBlocks,
        td.enableBatchSizePerFeature,
        td.enableBlockBucketizePos,
        td.mySizeDivMagic,
        td.mySizeDivShift};

    BlockBucketizeSparseFeaturesKernelFull<OffsetT, IndexT> kernel(args);
    kernel.ProcessComputeNewLengths();
}

#define SIMPLIFIED_WARP_CALL_COMPUTE_NEWLENGTHS(QD, B1)                                      \
    do {                                                                                     \
        Simt::VF_CALL<BlockBucketizeSparseFeaturesKernelSimplified::                                         \
            SimtComputeNewLengthsGmAtomic<OffsetT, IndexT, QD, B1>>(                         \
            Simt::Dim3{BlockBucketizeSparseFeaturesKernelSimplified::MAX_THREADS_PER_BLOCK, 1, 1},         \
            offsetsGm, indicesGm, blockSizesGm, newLengthsGm,                               \
            lengthsSize, mySize,                                                             \
            rowRange.begin, rowRange.end,                                                    \
            td.mySizeDivMagic, td.mySizeDivShift, blkSizeMagicShifts,                        \
            td.batchSizeDivMagic, td.batchSizeDivShift, isPowerOfTwo);                       \
    } while (0)

template <typename OffsetT, typename IndexT>
__aicore__ inline void RunComputeNewLengthsSimplifiedKernel(
    const ComputeNewLengthsParams& p,
    const BlockBucketizeSparseFeaturesTilingData& td)
{
    using namespace AscendC;
    using namespace BlockBucketizeSparseFeaturesKernelSimplified;
    using namespace BlockBucketizeSparseFeaturesCommon;
    using UIndexT = typename std::make_unsigned<IndexT>::type;

    const int32_t coreId = GetBlockIdx();
    const int32_t coreNum = GetBlockNum();

    __gm__ OffsetT* offsetsGm = reinterpret_cast<__gm__ OffsetT*>(p.offsets);
    __gm__ IndexT* indicesGm = reinterpret_cast<__gm__ IndexT*>(p.indices);
    __gm__ IndexT* blockSizesGm = reinterpret_cast<__gm__ IndexT*>(p.blockSizes);
    __gm__ OffsetT* newLengthsGm = reinterpret_cast<__gm__ OffsetT*>(p.newLengths);

    const int32_t lengthsSize = static_cast<int32_t>(td.lengthsSize);
    const int32_t mySize = static_cast<int32_t>(td.mySize);
    const int32_t featureNum = static_cast<int32_t>(td.numFeatures);
    const bool batchSizeIsOne = (td.batchSize == 1);
    const bool isPowerOfTwo = ((mySize & (mySize - 1)) == 0);

    WorkRange rowRange = {0};
    ComputeWorkRange(lengthsSize, coreId, coreNum, rowRange);

    bool useQuickDivide = (featureNum <= MAX_FEATURE_NUM_USE_QUICK_DIVIDE) && (mySize > 1);
    const int32_t quickDivBytes = useQuickDivide
        ? static_cast<int32_t>(featureNum * 2 * sizeof(UIndexT))
        : 0;
    const int32_t totalUbBytes = quickDivBytes;

    TPipe ubPipe;
    TBuf<TPosition::VECCALC> ubBuf;
    ubPipe.InitBuffer(ubBuf, (totalUbBytes > 0) ? totalUbBytes : static_cast<int32_t>(sizeof(int32_t)));
    LocalTensor<uint8_t> ubRaw = ubBuf.Get<uint8_t>();
    __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(ubRaw.GetPhyAddr());

    __ubuf__ UIndexT* blkSizeMagicShifts = reinterpret_cast<__ubuf__ UIndexT*>(ubBase);

    if (useQuickDivide) {
        for (int32_t i = 0; i < featureNum; ++i) {
            const UIndexT blkSize = static_cast<UIndexT>(blockSizesGm[i]);
            if (blkSize <= 1) {
                useQuickDivide = false;
                break;
            }
            UIndexT magic = 0;
            UIndexT shift = 0;
            GetUintDivMagicAndShift(magic, shift, blkSize);
            blkSizeMagicShifts[i * 2] = magic;
            blkSizeMagicShifts[i * 2 + 1] = shift;
        }
    }
    SyncAll();

    if (useQuickDivide && batchSizeIsOne) {
        SIMPLIFIED_WARP_CALL_COMPUTE_NEWLENGTHS(true, true);
    } else if (useQuickDivide) {
        SIMPLIFIED_WARP_CALL_COMPUTE_NEWLENGTHS(true, false);
    } else if (batchSizeIsOne) {
        SIMPLIFIED_WARP_CALL_COMPUTE_NEWLENGTHS(false, true);
    } else {
        SIMPLIFIED_WARP_CALL_COMPUTE_NEWLENGTHS(false, false);
    }
}

#undef SIMPLIFIED_WARP_CALL_COMPUTE_NEWLENGTHS

extern "C" __global__ __aicore__ void block_bucketize_sparse_features_compute_new_lengths(
    /* input - required */
    GM_ADDR indices,
    GM_ADDR block_sizes,
    GM_ADDR offsets,
    /* input - optional */
    GM_ADDR total_num_blocks,
    GM_ADDR batch_size_offsets,
    /* input - dynamic */
    GM_ADDR block_bucketize_pos,
    /* output - required */
    GM_ADDR new_lengths,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    ComputeNewLengthsParams params{
        indices, block_sizes, offsets,
        total_num_blocks, batch_size_offsets, block_bucketize_pos,
        new_lengths, workspace};

    if (TILING_KEY_IS(0)) {
        RunComputeNewLengthsFullKernel<DTYPE_OFFSETS, DTYPE_INDICES>(params, tilingData);
    } else if (TILING_KEY_IS(1)) {
        RunComputeNewLengthsSimplifiedKernel<DTYPE_OFFSETS, DTYPE_INDICES>(params, tilingData);
    }
}

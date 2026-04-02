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

struct ScatterNewIndicesParams {
    GM_ADDR indices;
    GM_ADDR blockSizes;
    GM_ADDR offsets;
    GM_ADDR newOffsets;
    GM_ADDR weights;
    GM_ADDR totalNumBlocks;
    GM_ADDR batchSizeOffsets;
    GM_ADDR blockBucketizePos;
    GM_ADDR newIndices;
    GM_ADDR newWeights;
    GM_ADDR newPos;
    GM_ADDR unbucketizePermute;
    GM_ADDR workspace;
};

template <typename OffsetT, typename IndexT>
__aicore__ inline void RunScatterNewIndicesFullKernel(
    const ScatterNewIndicesParams& p,
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
        nullptr,    // newLengths
        reinterpret_cast<__gm__ IndexT*>(p.newIndices),
        reinterpret_cast<__gm__ IndexT*>(p.unbucketizePermute),
        reinterpret_cast<__gm__ float*>(p.weights),
        reinterpret_cast<__gm__ float*>(p.newWeights),
        reinterpret_cast<__gm__ IndexT*>(p.newPos),
        reinterpret_cast<__gm__ OffsetT*>(p.offsets),
        reinterpret_cast<__gm__ OffsetT*>(p.newOffsets),
        reinterpret_cast<__gm__ OffsetT*>(p.batchSizeOffsets),
        static_cast<int32_t>(td.lengthsSize),
        static_cast<int32_t>(td.indicesSize),
        static_cast<int32_t>(td.numFeatures),
        static_cast<int32_t>(td.batchSize),
        static_cast<int32_t>(td.mySize),
        td.enableSequence,
        td.enableWeights,
        td.enableBucketizePos,
        td.enableKeepOrigIdx,
        td.enableTotalNumBlocks,
        td.enableBatchSizePerFeature,
        td.enableBlockBucketizePos,
        td.mySizeDivMagic,
        td.mySizeDivShift};

    BlockBucketizeSparseFeaturesKernelFull<OffsetT, IndexT> kernel(args);
    kernel.ProcessScatterNewIndices();
}

#define SIMPLIFIED_WARP_CALL_REARRANGE_POOLED(WT, POS, QD, B1)                               \
    do {                                                                                     \
        Simt::VF_CALL<BlockBucketizeSparseFeaturesKernelSimplified::                                         \
            SimtScatterNewIndicesPooledUbAtomic<WT, POS, OffsetT, IndexT, QD, B1>>(         \
            Simt::Dim3{BlockBucketizeSparseFeaturesKernelSimplified::SCATTER_THREADS_PER_BLOCK, 1, 1},     \
            offsetsGm, indicesGm, (WT) ? weightsGm : nullptr, blockSizesGm,                 \
            currOffsetsGm, newIndicesGm, (WT) ? newWeightsGm : nullptr,                     \
            (POS) ? newPosGm : nullptr, ubCounters,                                          \
            lengthsSize, mySize, indicesSize,                                                \
            rowRange.begin, rowRange.end,                                                    \
            td.mySizeDivMagic, td.mySizeDivShift, blkSizeMagicShifts,                        \
            td.batchSizeDivMagic, td.batchSizeDivShift, isPowerOfTwo);                       \
    } while (0)

#define SIMPLIFIED_WARP_CALL_REARRANGE_SEQUENCE(WT, POS, QD, B1)                             \
    do {                                                                                     \
        Simt::VF_CALL<BlockBucketizeSparseFeaturesKernelSimplified::                                         \
            SimtScatterNewIndicesRowsAtomic<WT, POS, OffsetT, IndexT, QD, B1>>(             \
            Simt::Dim3{BlockBucketizeSparseFeaturesKernelSimplified::SCATTER_THREADS_PER_BLOCK, 1, 1},     \
            offsetsGm, indicesGm, (WT) ? weightsGm : nullptr, blockSizesGm,                 \
            currOffsetsGm, newIndicesGm, (WT) ? newWeightsGm : nullptr,                     \
            (POS) ? newPosGm : nullptr, unbucketizePermuteGm,                                \
            lengthsSize, mySize, indicesSize,                                                \
            rowRange.begin, rowRange.end,                                                    \
            td.mySizeDivMagic, td.mySizeDivShift, blkSizeMagicShifts,                        \
            td.batchSizeDivMagic, td.batchSizeDivShift, isPowerOfTwo);                       \
    } while (0)

template <typename OffsetT, typename IndexT>
__aicore__ inline void RunScatterNewIndicesSimplifiedKernel(
    const ScatterNewIndicesParams& p,
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
    __gm__ IndexT* newIndicesGm = reinterpret_cast<__gm__ IndexT*>(p.newIndices);
    __gm__ IndexT* newPosGm = reinterpret_cast<__gm__ IndexT*>(p.newPos);
    __gm__ IndexT* unbucketizePermuteGm = reinterpret_cast<__gm__ IndexT*>(p.unbucketizePermute);
    __gm__ OffsetT* currOffsetsGm = reinterpret_cast<__gm__ OffsetT*>(p.newOffsets);

    __gm__ float* weightsGm = td.enableWeights ? reinterpret_cast<__gm__ float*>(p.weights) : nullptr;
    __gm__ float* newWeightsGm = td.enableWeights ? reinterpret_cast<__gm__ float*>(p.newWeights) : nullptr;

    const int32_t lengthsSize = static_cast<int32_t>(td.lengthsSize);
    const int32_t mySize = static_cast<int32_t>(td.mySize);
    const int32_t indicesSize = static_cast<int32_t>(td.indicesSize);
    const int32_t featureNum = static_cast<int32_t>(td.numFeatures);
    const bool hasWeight = td.enableWeights;
    const bool bucketizePos = td.enableBucketizePos;
    const bool sequence = td.enableSequence;
    const bool batchSizeIsOne = (td.batchSize == 1);
    const bool isPowerOfTwo = ((mySize & (mySize - 1)) == 0);

    WorkRange rowRange = {0};
    ComputeWorkRange(lengthsSize, coreId, coreNum, rowRange);

    bool useQuickDivide = (featureNum <= MAX_FEATURE_NUM_USE_QUICK_DIVIDE) && (mySize > 1);

    const int32_t quickDivBytes = useQuickDivide
        ? static_cast<int32_t>(featureNum * 2 * sizeof(UIndexT))
        : 0;
    const int32_t pooledUbBytes =
        SCATTER_WARPS_PER_BLOCK * mySize * static_cast<int32_t>(sizeof(int32_t));
    const int32_t totalUbBytes = quickDivBytes + pooledUbBytes;

    TPipe ubPipe;
    TBuf<TPosition::VECCALC> ubBuf;
    ubPipe.InitBuffer(ubBuf, totalUbBytes);
    LocalTensor<uint8_t> ubRaw = ubBuf.Get<uint8_t>();
    __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(ubRaw.GetPhyAddr());

    __ubuf__ UIndexT* blkSizeMagicShifts = reinterpret_cast<__ubuf__ UIndexT*>(ubBase);
    __ubuf__ int32_t* ubCounters = reinterpret_cast<__ubuf__ int32_t*>(ubBase + quickDivBytes);

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

#define SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(MACRO, HW, BP, QD, B1)                           \
    do {                                                                                     \
        if ((HW) && (BP)) {                                                                  \
            MACRO(true, true, QD, B1);                                                       \
        } else if (HW) {                                                                     \
            MACRO(true, false, QD, B1);                                                      \
        } else if (BP) {                                                                     \
            MACRO(false, true, QD, B1);                                                      \
        } else {                                                                             \
            MACRO(false, false, QD, B1);                                                     \
        }                                                                                    \
    } while (0)

    if (sequence) {
        if (useQuickDivide && batchSizeIsOne) {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_SEQUENCE, hasWeight, bucketizePos, true, true);
        } else if (useQuickDivide) {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_SEQUENCE, hasWeight, bucketizePos, true, false);
        } else if (batchSizeIsOne) {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_SEQUENCE, hasWeight, bucketizePos, false, true);
        } else {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_SEQUENCE, hasWeight, bucketizePos, false, false);
        }
    } else {
        if (useQuickDivide && batchSizeIsOne) {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_POOLED, hasWeight, bucketizePos, true, true);
        } else if (useQuickDivide) {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_POOLED, hasWeight, bucketizePos, true, false);
        } else if (batchSizeIsOne) {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_POOLED, hasWeight, bucketizePos, false, true);
        } else {
            SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS(SIMPLIFIED_WARP_CALL_REARRANGE_POOLED, hasWeight, bucketizePos, false, false);
        }
    }

#undef SIMPLIFIED_WARP_DISPATCH_WEIGHTS_POS
}

#undef SIMPLIFIED_WARP_CALL_REARRANGE_POOLED
#undef SIMPLIFIED_WARP_CALL_REARRANGE_SEQUENCE

extern "C" __global__ __aicore__ void block_bucketize_sparse_features_scatter_new_indices(
    /* input - required */
    GM_ADDR indices,
    GM_ADDR block_sizes,
    GM_ADDR offsets,
    GM_ADDR new_offsets,
    /* input - optional */
    GM_ADDR weights,
    GM_ADDR total_num_blocks,
    GM_ADDR batch_size_offsets,
    /* input - dynamic */
    GM_ADDR block_bucketize_pos,
    /* output - required */
    GM_ADDR new_indices,
    /* output - optional */
    GM_ADDR new_weights,
    GM_ADDR new_pos,
    GM_ADDR unbucketize_permute,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    ScatterNewIndicesParams params{
        indices, block_sizes, offsets, new_offsets,
        weights, total_num_blocks, batch_size_offsets, block_bucketize_pos,
        new_indices, new_weights, new_pos, unbucketize_permute, workspace};

    if (TILING_KEY_IS(0)) {
        RunScatterNewIndicesFullKernel<DTYPE_OFFSETS, DTYPE_INDICES>(params, tilingData);
    } else if (TILING_KEY_IS(1)) {
        RunScatterNewIndicesSimplifiedKernel<DTYPE_OFFSETS, DTYPE_INDICES>(params, tilingData);
    }
}

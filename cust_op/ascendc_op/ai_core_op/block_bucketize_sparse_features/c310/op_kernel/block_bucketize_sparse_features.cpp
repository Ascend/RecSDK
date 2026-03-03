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

#include "block_bucketize_sparse_features_kernel.h"

struct KernelEntryParams {
    GM_ADDR lengths;
    GM_ADDR indices;
    GM_ADDR blockSizes;
    GM_ADDR weights;
    GM_ADDR batchSizePerFeature;
    GM_ADDR totalNumBlocks;
    GM_ADDR blockBucketizePos;
    GM_ADDR newLengths;
    GM_ADDR newIndices;
    GM_ADDR newWeights;
    GM_ADDR newPos;
    GM_ADDR unbucketizePermute;
    GM_ADDR workspace;
};

template <typename OffsetT, typename IndexT>
__aicore__ inline void RunSimtKernel(
    const KernelEntryParams& p,
    const BlockBucketizeSparseFeaturesTilingData& td)
{
    __gm__ uint8_t* wsBase = reinterpret_cast<__gm__ uint8_t*>(GetUserWorkspace(p.workspace));
    __gm__ OffsetT* offsets = reinterpret_cast<__gm__ OffsetT*>(wsBase + td.offsetsOffset);
    __gm__ OffsetT* newOffsets = reinterpret_cast<__gm__ OffsetT*>(wsBase + td.newOffsetsOffset);
    __gm__ OffsetT* writeOffsets = reinterpret_cast<__gm__ OffsetT*>(wsBase + td.writeOffsetsOffset);
    __gm__ OffsetT* batchSizeOffsets = (td.batchSizeOffsetsOffset != 0) ?
        reinterpret_cast<__gm__ OffsetT*>(wsBase + td.batchSizeOffsetsOffset) : nullptr;
    __gm__ uint64_t* posPtrs = (td.posPtrsOffset != 0) ?
        reinterpret_cast<__gm__ uint64_t*>(wsBase + td.posPtrsOffset) : nullptr;
    __gm__ int64_t* posLens = (td.posLensOffset != 0) ?
        reinterpret_cast<__gm__ int64_t*>(wsBase + td.posLensOffset) : nullptr;
    __gm__ OffsetT* blockSums = reinterpret_cast<__gm__ OffsetT*>(wsBase + td.blockSumsOffset);

    typename BlockBucketizeSparseFeaturesKernel<OffsetT, IndexT>::KernelArgs args{
        reinterpret_cast<__gm__ OffsetT*>(p.lengths),
        reinterpret_cast<__gm__ IndexT*>(p.indices),
        reinterpret_cast<__gm__ IndexT*>(p.blockSizes),
        reinterpret_cast<__gm__ OffsetT*>(p.batchSizePerFeature),
        reinterpret_cast<__gm__ IndexT*>(p.totalNumBlocks),
        reinterpret_cast<__gm__ void*>(p.blockBucketizePos),
        posPtrs,
        posLens,
        reinterpret_cast<__gm__ OffsetT*>(p.newLengths),
        reinterpret_cast<__gm__ IndexT*>(p.newIndices),
        reinterpret_cast<__gm__ IndexT*>(p.unbucketizePermute),
        reinterpret_cast<__gm__ float*>(p.weights),
        reinterpret_cast<__gm__ float*>(p.newWeights),
        reinterpret_cast<__gm__ IndexT*>(p.newPos),
        offsets,
        newOffsets,
        writeOffsets,
        batchSizeOffsets,
        blockSums,
        td.lengthsSize,
        td.indicesSize,
        td.numFeatures,
        td.batchSize,
        td.mySize,
        td.newLengthsSize,
        td.enableSequence,
        td.enableWeights,
        td.enableBucketizePos,
        td.enableKeepOrigIdx,
        td.enableTotalNumBlocks,
        td.maxBatchSize,
        td.enableBatchSizePerFeature,
        td.enableBlockBucketizePos,
        td.mySizeDivMagic,
        td.mySizeDivShift};

    BlockBucketizeSparseFeaturesKernel<OffsetT, IndexT> kernel(args);
    kernel.Process();
}

extern "C" __global__ __aicore__ void block_bucketize_sparse_features(
    /* input */
    GM_ADDR lengths,
    GM_ADDR indices,
    GM_ADDR block_sizes,
    GM_ADDR weights,
    GM_ADDR batch_size_per_feature,
    GM_ADDR total_num_blocks,
    GM_ADDR block_bucketize_pos,
    /* output */
    GM_ADDR new_lengths,
    GM_ADDR new_indices,
    GM_ADDR new_weights,
    GM_ADDR new_pos,
    GM_ADDR unbucketize_permute,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    KernelEntryParams params{
        lengths, indices, block_sizes, weights,
        batch_size_per_feature, total_num_blocks, block_bucketize_pos,
        new_lengths, new_indices, new_weights, new_pos,
        unbucketize_permute, workspace};

    RunSimtKernel<DTYPE_LENGTHS, DTYPE_INDICES>(params, tilingData);
}

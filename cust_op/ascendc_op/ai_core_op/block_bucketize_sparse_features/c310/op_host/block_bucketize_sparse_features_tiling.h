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

#ifndef BLOCK_BUCKETIZE_SPARSE_FEATURES_TILING_H
#define BLOCK_BUCKETIZE_SPARSE_FEATURES_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(BlockBucketizeSparseFeaturesTilingData)
    TILING_DATA_FIELD_DEF(int64_t, lengthsSize);
    TILING_DATA_FIELD_DEF(int64_t, indicesSize);
    TILING_DATA_FIELD_DEF(int64_t, numFeatures);
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(int64_t, mySize);
    TILING_DATA_FIELD_DEF(int64_t, newLengthsSize);
    TILING_DATA_FIELD_DEF(int64_t, maxBatchSize);
    TILING_DATA_FIELD_DEF(bool, enableSequence);
    TILING_DATA_FIELD_DEF(bool, enableWeights);
    TILING_DATA_FIELD_DEF(bool, enableBucketizePos);
    TILING_DATA_FIELD_DEF(bool, enableKeepOrigIdx);
    TILING_DATA_FIELD_DEF(bool, enableTotalNumBlocks);
    TILING_DATA_FIELD_DEF(bool, enableBatchSizePerFeature);
    TILING_DATA_FIELD_DEF(bool, enableBlockBucketizePos);
    TILING_DATA_FIELD_DEF(uint64_t, mySizeDivMagic);
    TILING_DATA_FIELD_DEF(uint32_t, mySizeDivShift);
    TILING_DATA_FIELD_DEF(uint64_t, posPtrsOffset);
    TILING_DATA_FIELD_DEF(uint64_t, posLensOffset);
    TILING_DATA_FIELD_DEF(uint64_t, batchSizeDivMagic);
    TILING_DATA_FIELD_DEF(uint32_t, batchSizeDivShift);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockBucketizeSparseFeaturesComputeNewLengths, BlockBucketizeSparseFeaturesTilingData)
REGISTER_TILING_DATA_CLASS(BlockBucketizeSparseFeaturesScatterNewIndices, BlockBucketizeSparseFeaturesTilingData)

} // namespace optiling

#endif // BLOCK_BUCKETIZE_SPARSE_FEATURES_TILING_H

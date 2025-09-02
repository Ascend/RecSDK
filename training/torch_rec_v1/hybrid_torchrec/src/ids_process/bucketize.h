/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef HYBRID_BUCKETIZE_H
#define HYBRID_BUCKETIZE_H
#include "torch/torch.h"

namespace hybrid {
using BucketResult = std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>,
                     std::optional<at::Tensor>,
                     std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor>;
BucketResult BlockBucketizeSparseFeaturesCpu(
    const at::Tensor& lengths, const at::Tensor& indices,
    const bool bucketizePos, const bool sequence,
    const at::Tensor& blockSizes, const int64_t bucketSize,
    const std::optional<at::Tensor>& totalNumBlocks,
    const std::optional<at::Tensor>& weights,
    const std::optional<at::Tensor>& batchSizePerFeature, const int64_t maxBatchSize,
    const std::optional<std::vector<at::Tensor>>& blockBucketizePos,
    const bool returnBucketMapping, const bool keepOrigIdx, const bool doUnique,
    const bool returnCount);
}
#endif
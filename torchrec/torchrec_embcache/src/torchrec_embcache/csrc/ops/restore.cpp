/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "restore.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

namespace Embcache {

void Restore(const at::Tensor& uniqueIndices, const at::Tensor& uniqueInverse, const at::Tensor& uniqueOffset,
             const std::vector<int64_t>& offsetsPerTable, at::Tensor& hashIndices)
{
    TORCH_CHECK(uniqueIndices.dim() == 1 && uniqueInverse.dim() == 1 && uniqueOffset.dim() == 1,
                "unique tensors must be 1-D");

    TORCH_CHECK(uniqueIndices.scalar_type() == at::kLong && uniqueInverse.scalar_type() == at::kLong &&
                    uniqueOffset.scalar_type() == at::kLong && hashIndices.scalar_type() == at::kLong,
                "dtype tensors must be int64");

    TORCH_CHECK(uniqueIndices.device() == uniqueInverse.device() && uniqueIndices.device() == uniqueOffset.device() &&
                    uniqueIndices.device() == hashIndices.device(),
                "tensor device mismatch");

    TORCH_CHECK(hashIndices.numel() == uniqueInverse.numel(), "hashIndices length must equal uniqueInverse length");

    const int64_t nTables = static_cast<int64_t>(offsetsPerTable.size()) - 1;

    const int64_t* uniqueOffsetPtr = uniqueOffset.data_ptr<int64_t>();
    const int64_t* offsetsPtr = offsetsPerTable.data();
    const int64_t* uniqueIndicesPtr = uniqueIndices.data_ptr<int64_t>();
    const int64_t* uniqueInversePtr = uniqueInverse.data_ptr<int64_t>();
    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();

    at::parallel_for(0, nTables, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t shift = uniqueOffsetPtr[i];
            const int64_t rStart = offsetsPtr[i];
            const int64_t rEnd = offsetsPtr[i + 1];

            for (int64_t r = rStart; r < rEnd; ++r) {
                const int64_t globalIdx = uniqueInversePtr[r] + shift;
                hashIndicesPtr[r] = uniqueIndicesPtr[globalIdx];
            }
        }
    });
}

AsyncTask<void> RestoreAsync(const torch::Tensor& uniqueIndices, const torch::Tensor& uniqueInverse,
                             const torch::Tensor& uniqueOffset, const std::vector<int64_t>& offsetsPerTable,
                             torch::Tensor& hashIndices)
{
    return AsyncTask<void>(
        [=]() mutable { Restore(uniqueIndices, uniqueInverse, uniqueOffset, offsetsPerTable, hashIndices); });
}

}  // namespace Embcache
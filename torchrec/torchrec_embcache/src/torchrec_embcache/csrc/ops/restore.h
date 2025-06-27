/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef TORCHREC_RESTORE_H
#define TORCHREC_RESTORE_H

#include <torch/extension.h>
#include <utils/async_task.h>

namespace Embcache {
void Restore(const torch::Tensor& uniqueIndices, const torch::Tensor& uniqueInverse, const torch::Tensor& uniqueOffset,
             const std::vector<int64_t>& offsetsPerTable, torch::Tensor& hashIndices);

AsyncTask<void> RestoreAsync(const torch::Tensor& uniqueIndices, const torch::Tensor& uniqueInverse,
                             const torch::Tensor& uniqueOffset, const std::vector<int64_t>& offsetsPerTable,
                             torch::Tensor& hashIndices);
}  // namespace Embcache

#endif  // TORCHREC_RESTORE_H

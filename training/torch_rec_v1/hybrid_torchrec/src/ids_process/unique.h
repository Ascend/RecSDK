/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef HYBRID_UNIQUE_H
#define HYBRID_UNIQUE_H
#include <omp.h>
#include <torch/torch.h>

namespace hybrid {
std::tuple<at::Tensor, at::Tensor> UniqueParallel(const at::Tensor& ids);
}  // namespace hybrid
#endif
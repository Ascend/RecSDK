/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================== */

/*!
 * \file l0_hstu_attn_metadata_backward.h
 * \brief HstuAttnMetadataBackward 的 l0 接口（把算子挂到 launcher 队列）。
 */

#ifndef L0_HSTU_ATTN_METADATA_BACKWARD_H
#define L0_HSTU_ATTN_METADATA_BACKWARD_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* HstuAttnMetadataBackward(const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv,
                                          const aclTensor* numContexts, const aclTensor* numTargets, int64_t numHeads,
                                          int64_t headDim, int64_t maskMode, int64_t targetGroupSize,
                                          const aclTensor* fullCnt, const aclTensor* maskCnt, const aclTensor* metadata,
                                          aclOpExecutor* executor);
}  // namespace l0op

#endif  // L0_HSTU_ATTN_METADATA_BACKWARD_H

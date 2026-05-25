/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CUSTOM_KERNEL_OPS_H
#define CUSTOM_KERNEL_OPS_H

#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace dyn_emb {
void load_from_pointer_hybrid_ops(void* pointers, void* dst, uint32_t dim, uint32_t num, aclrtStream stream,
                                  uint32_t coreNum, uint32_t oType, uint64_t totalUbSize);
}  // namespace dyn_emb

#endif  // CUSTOM_KERNEL_OPS_H

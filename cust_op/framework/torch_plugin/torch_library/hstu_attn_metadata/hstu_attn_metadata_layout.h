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
 * \file hstu_attn_metadata_layout.h
 * \brief hstu_attn_metadata 前向 / 反向两个 torch 绑定共享的 metadata 容量口径。
 *
 * 常量镜像算子侧布局头（唯一事实源在算子仓，不在这里）：
 *   - 前向 cust_op/ascendc_op/ai_core_op/hstu_attn_metadata/op_kernel_aicpu/hstu_attn_metadata.h
 *   - 反向 cust_op/ascendc_op/ai_core_op/hstu_attn_metadata_backward/op_kernel_aicpu/hstu_attn_metadata_backward.h
 * 反向布局与前向同构（其布局头注释明确这一点）：每个 section 内 AIC/AIV core 各占
 * METADATA_STRIDE 个 int32，HEAD 段额外 METADATA_STRIDE 个 int32，故两侧共用同一容量公式：
 *
 *   elems(sections) = (sections * (AIC_CORE_NUM + AIV_CORE_NUM) + 1) * METADATA_STRIDE
 *
 * 前向 sections = batch * num_heads_kv；反向 sections 取 sectionNum 上界 = batch * num_heads。
 * 绑定侧容量只需 >= kernel 实际用量（多出来的尾部算子不触碰）；若算子侧布局变更，
 * 必须同步修改本文件，否则 kernel 容量校验会在运行期拒绝。
 */

#ifndef HSTU_ATTN_METADATA_LAYOUT_H
#define HSTU_ATTN_METADATA_LAYOUT_H

#include <cstdint>

namespace hstu_meta {

// metadata 内存布局常量：每个 section 内 AIC/AIV core 各占 16 个 int32；HEAD 段额外 16 个 int32。
constexpr int64_t AIC_CORE_NUM = 36;
constexpr int64_t AIV_CORE_NUM = 72;
constexpr int64_t METADATA_STRIDE = 16;
// metadata 张量按 4096 个 int32 对齐分配。
constexpr int64_t METADATA_ALIGN = 4096;

/*! \brief sections 个 section 的 metadata 所需 int32 元素数（容量上界口径）。 */
inline int64_t HstuMetadataCapacityElems(int64_t sections)
{
    return (sections * (AIC_CORE_NUM + AIV_CORE_NUM) + 1) * METADATA_STRIDE;
}

}  // namespace hstu_meta

#endif  // HSTU_ATTN_METADATA_LAYOUT_H

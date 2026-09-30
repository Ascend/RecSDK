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
 * \file aclnn_hstu_attn_metadata_backward.h
 * \brief HstuAttnMetadataBackward 的 aclnn 两段式对外接口。
 */

#ifndef ACLNN_HSTU_ATTN_METADATA_BACKWARD_H
#define ACLNN_HSTU_ATTN_METADATA_BACKWARD_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief 第一段：入参校验 + 建图，回填 workspace 大小与 executor。
 *
 * \param cuSeqlensQ   [batch+1] int32/int64，q 序列 offsets（必选）
 * \param cuSeqlensKv  [batch+1] int32/int64，k 序列 offsets（必选）
 * \param numContexts  [batch] int32/int64，每 batch 的 context 行数
 *                     （causal 专属，可选：nullptr / 空 tensor = history-only；
 *                      非 causal 模式必须为空）
 * \param numTargets   [batch] int32/int64，每 batch 的 target 行数
 *                     （causal 专属，可选：nullptr / 空 tensor = 无 target 封顶；
 *                      非 causal 模式必须为空）
 * \param numHeads     注意力头数（必选属性）
 * \param headDim      注意力头维（必选属性）
 * \param maskMode     mask 模式（必选属性）：0=无 mask，1=causal（随 numContexts/numTargets/
 *                     targetGroupSize 生效），2=arbitrary（随 fullCnt/maskCnt 生效）
 * \param targetGroupSize causal 的 target 项组大小（标量属性）：<= 0 = 无 target；
 *                         其余 mask 模式忽略该值（建议传 0）
 * \param fullCnt      [batch*maxBlkCntK] int32，每个 K 行块的 full 类 Q 列块数
 *                     （arbitrary 专属，必选；其余 mask 模式必须传 nullptr）
 * \param maskCnt      [batch*maxBlkCntK] int32，每个 K 行块的 mask 类 Q 列块数
 *                     （arbitrary 专属，必选；其余 mask 模式必须传 nullptr）
 * \param metadata     [N] int32 输出，HEAD+FA+FD 布局（必选）
 */
__attribute__((visibility("default"))) aclnnStatus aclnnHstuAttnMetadataBackwardGetWorkspaceSize(
    const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv, const aclTensor* numContexts,
    const aclTensor* numTargets, int64_t numHeads, int64_t headDim, int64_t maskMode, int64_t targetGroupSize,
    const aclTensor* fullCnt, const aclTensor* maskCnt, const aclTensor* metadata, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/*! \brief 第二段：把 executor 下发到 stream。 */
__attribute__((visibility("default"))) aclnnStatus aclnnHstuAttnMetadataBackward(void* workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor* executor,
                                                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_HSTU_ATTN_METADATA_BACKWARD_H

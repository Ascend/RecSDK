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

#include "aclnn_hstu_attn_metadata_backward.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "../hstu_attn_metadata_backward_check.h"
#include "l0_hstu_attn_metadata_backward.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnHstuAttnMetadataBackwardGetWorkspaceSize(const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv,
                                                          const aclTensor* numContexts, const aclTensor* numTargets,
                                                          int64_t numHeads, int64_t headDim, int64_t maskMode,
                                                          int64_t targetGroupSize, const aclTensor* fullCnt,
                                                          const aclTensor* maskCnt, const aclTensor* metadata,
                                                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnHstuAttnMetadataBackward,
                   DFX_IN(cuSeqlensQ, cuSeqlensKv, numContexts, numTargets, numHeads, headDim, maskMode,
                          targetGroupSize, fullCnt, maskCnt),
                   DFX_OUT(metadata));

    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret =
        HstuAttnMetadataBackwardCheck::ParamsCheck(cuSeqlensQ, cuSeqlensKv, numContexts, numTargets, fullCnt, maskCnt,
                                                   numHeads, headDim, maskMode, targetGroupSize, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto output =
        l0op::HstuAttnMetadataBackward(cuSeqlensQ, cuSeqlensKv, numContexts, numTargets, numHeads, headDim, maskMode,
                                       targetGroupSize, fullCnt, maskCnt, metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 纯 AICPU 算子，无 workspace 需求
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHstuAttnMetadataBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHstuAttnMetadataBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

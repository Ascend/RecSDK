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

#include "l0_hstu_attn_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(HstuAttnMetadata);

const aclTensor* HstuAttnMetadata(const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKvOptional,
                                  const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
                                  int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                  int64_t numHeadsKv, int64_t headDim, int64_t maskMode, int64_t winLeft,
                                  int64_t winRight, const char* layoutQ, const char* layoutKv, const char* layoutOut,
                                  const char* socVersion, int64_t aicCoreNum, int64_t aivCoreNum,
                                  const aclTensor* metaData, aclOpExecutor* executor)
{
    L0_DFX(HstuAttnMetadata, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize,
           maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, winLeft, winRight, layoutQ, layoutKv,
           layoutOut, socVersion, aicCoreNum, aivCoreNum, metaData);

    static internal::AicpuTaskSpace space("HstuAttnMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        HstuAttnMetadata,
        OP_ATTR_NAMES({"batch_size", "max_seqlen_q", "max_seqlen_kv", "num_heads_q", "num_heads_kv", "head_dim",
                       "mask_mode", "win_left", "win_right", "layout_q", "layout_kv", "layout_out", "soc_version",
                       "aic_core_num", "aiv_core_num"}),
        OP_INPUT(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional), OP_OUTPUT(metaData),
        OP_ATTR(batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, winLeft, winRight,
                layoutQ, layoutKv, layoutOut, socVersion, aicCoreNum, aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HstuAttnMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return metaData;
}

}  // namespace l0op

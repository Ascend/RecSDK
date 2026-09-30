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

#include "l0_hstu_attn_metadata_backward.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(HstuAttnMetadataBackward);

const aclTensor* HstuAttnMetadataBackward(const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv,
                                          const aclTensor* numContexts, const aclTensor* numTargets, int64_t numHeads,
                                          int64_t headDim, int64_t maskMode, int64_t targetGroupSize,
                                          const aclTensor* fullCnt, const aclTensor* maskCnt, const aclTensor* metadata,
                                          aclOpExecutor* executor)
{
    L0_DFX(HstuAttnMetadataBackward, cuSeqlensQ, cuSeqlensKv, numContexts, numTargets, numHeads, headDim, maskMode,
           targetGroupSize, fullCnt, maskCnt, metadata);

    static internal::AicpuTaskSpace space("HstuAttnMetadataBackward");

    // 入参顺序必须与 kernel 侧 InputIdx 严格一致：cu_seqlens_q, cu_seqlens_kv, num_contexts,
    // num_targets, full_cnt, mask_cnt（num_contexts / num_targets 是 causal 专属槽位、
    // full_cnt / mask_cnt 是 arbitrary 专属槽位，非当前模式的段传空占位）
    // 属性顺序必须与 kernel 侧 kAttr* 读取的名字严格一致：num_heads, head_dim, mask_mode,
    // target_group_size（causal 的 target 项组大小）
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(HstuAttnMetadataBackward,
                                          OP_ATTR_NAMES({"num_heads", "head_dim", "mask_mode", "target_group_size"}),
                                          OP_INPUT(cuSeqlensQ, cuSeqlensKv, numContexts, numTargets, fullCnt, maskCnt),
                                          OP_OUTPUT(metadata), OP_ATTR(numHeads, headDim, maskMode, targetGroupSize));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HstuAttnMetadataBackward ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metadata;
}

}  // namespace l0op

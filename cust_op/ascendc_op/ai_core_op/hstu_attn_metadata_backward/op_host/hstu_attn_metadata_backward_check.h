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
 * \file hstu_attn_metadata_backward_check.h
 * \brief HstuAttnMetadataBackward 的 host 侧入参校验。
 *
 * 入参为 q/k 的 offsets + 可选 num_contexts/num_targets（causal 专属段）+ 可选 full_cnt/mask_cnt
 * （arbitrary 专属段）+ 四个属性（num_heads / head_dim / mask_mode / target_group_size），
 * 外加一个 metadata 输出。
 * 这里只校验「可以不看数据就判定的性质」：存在性、维度、dtype、取值域、两侧一致性；
 * offsets 的单调性与首元素为 0 由 kernel 侧读数据时校验；num_contexts/num_targets 的
 * clamp 语义（[0, seqLenQ] / [0, min(seqLenQ, seqLenK)]）与 cnt 的 numel 精确校验
 * 同样只能由 kernel 读到 offsets 内容后进行。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_CHECK_H
#define HSTU_ATTN_METADATA_BACKWARD_CHECK_H

#include <unordered_set>

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

class HstuAttnMetadataBackwardCheck {
public:
    // mask_mode 取值域（wire 值与 kernel 侧 MaskMode 枚举一致）：0=nomask，1=causal，2=arbitrary。
    static constexpr int64_t MASK_MODE_NO_MASK = 0;
    static constexpr int64_t MASK_MODE_CAUSAL = 1;
    static constexpr int64_t MASK_MODE_ARBITRARY = 2;

    static inline aclnnStatus ParamsCheck(const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv,
                                          const aclTensor* numContexts, const aclTensor* numTargets,
                                          const aclTensor* fullCnt, const aclTensor* maskCnt, int64_t numHeads,
                                          int64_t headDim, int64_t maskMode, int64_t targetGroupSize,
                                          const aclTensor* metadata);

private:
    static inline bool IsTensorExist(const aclTensor* tensor);

    static inline bool IsOffsetDtype(const aclTensor* tensor);

    static inline aclnnStatus CheckAttr(int64_t numHeads, int64_t headDim, int64_t maskMode, int64_t targetGroupSize);

    static inline aclnnStatus CheckCntTensors(const aclTensor* fullCnt, const aclTensor* maskCnt, int64_t maskMode);

    static inline aclnnStatus CheckCausalTensors(const aclTensor* numContexts, const aclTensor* numTargets,
                                                 const aclTensor* cuSeqlensQ, int64_t maskMode);

    static inline aclnnStatus CheckOffsetTensor(const aclTensor* tensor, const char* name);

    static inline aclnnStatus CheckMetadataTensor(const aclTensor* metadata);

    static inline aclnnStatus CheckConsistency(const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv);
};

inline aclnnStatus HstuAttnMetadataBackwardCheck::ParamsCheck(const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv,
                                                              const aclTensor* numContexts, const aclTensor* numTargets,
                                                              const aclTensor* fullCnt, const aclTensor* maskCnt,
                                                              int64_t numHeads, int64_t headDim, int64_t maskMode,
                                                              int64_t targetGroupSize, const aclTensor* metadata)
{
    auto ret = CheckAttr(numHeads, headDim, maskMode, targetGroupSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckCntTensors(fullCnt, maskCnt, maskMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckCausalTensors(numContexts, numTargets, cuSeqlensQ, maskMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckOffsetTensor(cuSeqlensQ, "cuSeqlensQ");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckOffsetTensor(cuSeqlensKv, "cuSeqlensKv");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckConsistency(cuSeqlensQ, cuSeqlensKv);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckMetadataTensor(metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

inline bool HstuAttnMetadataBackwardCheck::IsTensorExist(const aclTensor* tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0) &&
           (tensor->GetData() != nullptr);
}

inline bool HstuAttnMetadataBackwardCheck::IsOffsetDtype(const aclTensor* tensor)
{
    const auto dtype = tensor->GetDataType();
    return (dtype == op::DataType::DT_INT32) || (dtype == op::DataType::DT_INT64);
}

inline aclnnStatus HstuAttnMetadataBackwardCheck::CheckAttr(int64_t numHeads, int64_t headDim, int64_t maskMode,
                                                            int64_t targetGroupSize)
{
    CHECK_COND(numHeads > 0, ACLNN_ERR_PARAM_INVALID, "numHeads must be greater than 0, but got %ld", numHeads);

    // mask_mode 取值域为 [NO_MASK, ARBITRARY]：三类均已实现（no-mask / causal / arbitrary），
    // 各自专属段的入参校验见 CheckCausalTensors / CheckCntTensors。
    CHECK_COND(maskMode >= MASK_MODE_NO_MASK && maskMode <= MASK_MODE_ARBITRARY, ACLNN_ERR_PARAM_INVALID,
               "maskMode must be in [%ld, %ld], but got %ld", MASK_MODE_NO_MASK, MASK_MODE_ARBITRARY, maskMode);

    // target_group_size 是 causal 的 target 项组大小（标量属性，其余模式忽略该值）；
    // 负值拒绝（<= 0 在 kernel 侧语义为「无 target」，合法）
    CHECK_COND(targetGroupSize >= 0, ACLNN_ERR_PARAM_INVALID, "targetGroupSize must be non-negative, but got %ld",
               targetGroupSize);

    constexpr int64_t HEAD_DIM_32 = 32;
    constexpr int64_t HEAD_DIM_64 = 64;
    constexpr int64_t HEAD_DIM_128 = 128;
    constexpr int64_t HEAD_DIM_256 = 256;
    static const std::unordered_set<int64_t> headDimSet = {HEAD_DIM_32, HEAD_DIM_64, HEAD_DIM_128, HEAD_DIM_256};
    CHECK_COND(headDimSet.count(headDim) > 0, ACLNN_ERR_PARAM_INVALID,
               "headDim only supports %ld, %ld, %ld, %ld, but got %ld", HEAD_DIM_32, HEAD_DIM_64, HEAD_DIM_128,
               HEAD_DIM_256, headDim);

    return ACLNN_SUCCESS;
}

inline aclnnStatus HstuAttnMetadataBackwardCheck::CheckCntTensors(const aclTensor* fullCnt, const aclTensor* maskCnt,
                                                                  int64_t maskMode)
{
    // full_cnt / mask_cnt 是 arbitrary 专属输入：arbitrary 必须成对提供（int32、非空），
    // 其余 mask 模式必须缺席（槽位分段固定，非当前模式的段拒收非空值，见 README 扩展约定）。
    // 逻辑布局 [batchSize, maxBlkCntK] 的 numel 精确校验只能由 kernel 做
    // （host 拿不到 offsets 内容与块档位推导结果），host 层不判 rank。
    if (maskMode == MASK_MODE_ARBITRARY) {
        CHECK_COND(IsTensorExist(fullCnt) && IsTensorExist(maskCnt), ACLNN_ERR_PARAM_NULLPTR,
                   "maskMode %ld (arbitrary) requires full_cnt and mask_cnt, but got null", maskMode);
        CHECK_COND(fullCnt->GetDataType() == op::DataType::DT_INT32 && maskCnt->GetDataType() == op::DataType::DT_INT32,
                   ACLNN_ERR_PARAM_INVALID, "full_cnt / mask_cnt must be int32, but got dtype %d / %d",
                   static_cast<int32_t>(fullCnt->GetDataType()), static_cast<int32_t>(maskCnt->GetDataType()));
    } else {
        CHECK_COND(!IsTensorExist(fullCnt) && !IsTensorExist(maskCnt), ACLNN_ERR_PARAM_INVALID,
                   "full_cnt / mask_cnt are arbitrary-mode-only inputs and must be null for maskMode %ld", maskMode);
    }

    return ACLNN_SUCCESS;
}

inline aclnnStatus HstuAttnMetadataBackwardCheck::CheckCausalTensors(const aclTensor* numContexts,
                                                                     const aclTensor* numTargets,
                                                                     const aclTensor* cuSeqlensQ, int64_t maskMode)
{
    // num_contexts / num_targets 是 causal 专属输入（槽位分段固定，非当前模式的段拒收非空值，
    // 见 README 扩展约定）。「未提供」= null 或空 tensor（两种绑定链路都合法）；
    // causal 下二者独立可选：未提供 = history-only / 无 target 封顶；
    // 提供时必须 1D、长度 == batch、int32/int64。per-batch 值的 clamp
    // （[0, seqLenQ] / [0, min(seqLenQ, seqLenK)]）由 kernel 侧 FromCtx 读 offsets 后执行。
    const aclTensor* tensors[2] = {numContexts, numTargets};
    const char* names[2] = {"numContexts", "numTargets"};
    for (size_t i = 0; i < 2; ++i) {
        if (!IsTensorExist(tensors[i])) {
            continue;  // null / 空 tensor = 未提供，任何 mask 模式都接受
        }
        CHECK_COND(maskMode == MASK_MODE_CAUSAL, ACLNN_ERR_PARAM_INVALID,
                   "%s is causal-mode-only input and must be empty for maskMode %ld", names[i], maskMode);
        CHECK_COND(tensors[i]->GetViewShape().GetDimNum() == 1, ACLNN_ERR_PARAM_INVALID,
                   "%s must be 1D, but got %ld dims", names[i], tensors[i]->GetViewShape().GetDimNum());
        // 非空时长度必须等于 batch（= cuSeqlensQ 长度 - 1）
        CHECK_COND(tensors[i]->GetViewShape().GetDim(0) == cuSeqlensQ->GetViewShape().GetDim(0) - 1,
                   ACLNN_ERR_PARAM_INVALID, "%s length must equal batch size %ld, but got %ld", names[i],
                   cuSeqlensQ->GetViewShape().GetDim(0) - 1, tensors[i]->GetViewShape().GetDim(0));
        CHECK_COND(IsOffsetDtype(tensors[i]), ACLNN_ERR_PARAM_INVALID, "%s must be int32 or int64, but got dtype %d",
                   names[i], static_cast<int32_t>(tensors[i]->GetDataType()));
    }

    return ACLNN_SUCCESS;
}

inline aclnnStatus HstuAttnMetadataBackwardCheck::CheckOffsetTensor(const aclTensor* tensor, const char* name)
{
    CHECK_COND(IsTensorExist(tensor), ACLNN_ERR_PARAM_NULLPTR, "%s should be provided, but got null", name);
    CHECK_COND(tensor->GetViewShape().GetDimNum() == 1, ACLNN_ERR_PARAM_INVALID, "%s must be 1D, but got %ld dims",
               name, tensor->GetViewShape().GetDimNum());
    // offsets 长度 = batch + 1，故至少 2 才能表达一个非空 batch
    CHECK_COND(tensor->GetViewShape().GetDim(0) >= 2, ACLNN_ERR_PARAM_INVALID,
               "%s shape must be at least (2,), but got (%ld,)", name, tensor->GetViewShape().GetDim(0));
    CHECK_COND(IsOffsetDtype(tensor), ACLNN_ERR_PARAM_INVALID, "%s must be int32 or int64, but got dtype %d", name,
               static_cast<int32_t>(tensor->GetDataType()));

    return ACLNN_SUCCESS;
}

inline aclnnStatus HstuAttnMetadataBackwardCheck::CheckMetadataTensor(const aclTensor* metadata)
{
    CHECK_COND(IsTensorExist(metadata), ACLNN_ERR_PARAM_NULLPTR, "metadata should be provided, but got null");
    CHECK_COND(metadata->GetViewShape().GetDimNum() == 1, ACLNN_ERR_PARAM_INVALID,
               "metadata must be 1D, but got %ld dims", metadata->GetViewShape().GetDimNum());
    CHECK_COND(metadata->GetDataType() == op::DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID,
               "metadata must be int32, but got dtype %d", static_cast<int32_t>(metadata->GetDataType()));

    return ACLNN_SUCCESS;
}

inline aclnnStatus HstuAttnMetadataBackwardCheck::CheckConsistency(const aclTensor* cuSeqlensQ,
                                                                   const aclTensor* cuSeqlensKv)
{
    // q/k 两条 offsets 描述同一个 batch 维度，长度必须一致（batchSize = len - 1）
    CHECK_COND(cuSeqlensQ->GetViewShape().GetDim(0) == cuSeqlensKv->GetViewShape().GetDim(0), ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensQ and cuSeqlensKv must have the same length, but got %ld and %ld",
               cuSeqlensQ->GetViewShape().GetDim(0), cuSeqlensKv->GetViewShape().GetDim(0));

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif  // HSTU_ATTN_METADATA_BACKWARD_CHECK_H

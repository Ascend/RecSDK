/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_paged.h"

constexpr uint32_t DIM_2 = 2;

namespace HstuDenseForward {

REGISTER_POLICY(LAYOUT_TYPE::PAGED, std::make_shared<TilingPolicyPaged>());

bool TilingPolicyPaged::TilingWorkSpace(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_CHECK_PTR_NULL(currentWorkspace, return false);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * tiling.get_dim() * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;
    // 3: midk midv attnscore
    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem * 3) * sizeof(float);
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize;
    return true;
}

bool TilingPolicyPaged::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    int64_t batchSize;
    int64_t seqLensQ;
    int64_t headNumQ;
    int64_t headDimQ;
    int64_t maxSeqLensQ;
    int64_t seqLensK;
    int64_t headNumK;
    int64_t headDimK;
    int64_t maxSeqLensK;

    auto seqOffsetQShape = context->GetOptionalInputShape(INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX)->GetStorageShape();
    batchSize = seqOffsetQShape.GetDim(0) - 1;

    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
              OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return false);

    auto qShape = context->GetInputShape(INPUT_INDEX_T::Q_INDEX)->GetStorageShape();
    auto kShape = context->GetInputShape(INPUT_INDEX_T::K_INDEX)->GetStorageShape();
    auto vShape = context->GetInputShape(INPUT_INDEX_T::V_INDEX)->GetStorageShape();

    OPS_CHECK(qShape.GetDimNum() != JAGGED_DIM_NUM,
              OPS_LOG_E("", "Jagged QKV should have 3 dimensions, but get %d", qShape.GetDimNum()), return false);

    // Q: [bs, n, d]
    seqLensQ = qShape.GetDim(0);
    headNumQ = qShape.GetDim(1);
    headDimQ = qShape.GetDim(2);
    maxSeqLensQ = tiling.get_maxSeqLenq();
    // K: [bs, n, d]
    seqLensK = kShape.GetDim(0);
    headNumK = kShape.GetDim(1);
    headDimK = kShape.GetDim(2);
    maxSeqLensK = tiling.get_maxSeqLenk();

    tiling.set_batchSize(batchSize);
    tiling.set_headNum(headNumQ);
    tiling.set_dim(headDimQ);
    tiling.set_seqLen(maxSeqLensQ);

    OPS_CHECK(kShape != vShape, OPS_LOG_E("", "K, V shape mismatch"), return false);
    OPS_CHECK(seqLensQ > seqLensK, OPS_LOG_E("", "Q, K, V seqlen mismatch"), return false);
    OPS_CHECK(headDimQ != headDimK, OPS_LOG_E("", "Q, K, V headDIM Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensQ, headNumQ, headDimQ),
              OPS_LOG_E("", "Q Jagged Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensK, headNumK, headDimK),
              OPS_LOG_E("", "K Jagged Shape Check failed"), return false);
    if (headNumQ != headNumK) {
        OPS_CHECK((headNumQ % headNumK != 0), OPS_LOG_E("", "For GQA, headNumQ must be divisible by headNumK"),
                  return false);
    }

    tiling.set_headRatio(headNumQ / headNumK);
    tiling.set_headNumK(headNumK);

    uint32_t masktype = tiling.get_maskType();
    auto *isDeltaQK = context->GetAttrs()->GetAttrPointer<uint32_t>(ATTR_INDEX_T::IS_DELTA_QK_INDEX);
    if (!*isDeltaQK) {
        OPS_CHECK(seqLensQ != seqLensK, OPS_LOG_E("", "Q, K seqLens mismatch"), return false);
    }

    auto numContext = context->GetOptionalInputShape(INPUT_INDEX_T::NUM_CONTEXT_INDEX);
    if (numContext != nullptr) {
        auto numCtxShape = numContext->GetStorageShape();
        int64_t numCtxDim = numCtxShape.GetDimNum();
        OPS_CHECK(numCtxDim != CONTEXT_DIM_NUM,
                  OPS_LOG_E("", "num_context should have %d dimension, but get %d", CONTEXT_DIM_NUM, numCtxDim),
                  return false);
        int64_t batchSizeCtx = numCtxShape.GetDim(0);
        OPS_CHECK(batchSizeCtx != batchSize,
                  OPS_LOG_E("", "The length of num_context expect %lld, but get %lld", batchSize, batchSizeCtx),
                  return false);
    }

    auto numTarget = context->GetOptionalInputShape(INPUT_INDEX_T::NUM_TARGET_INDEX);
    if (numTarget != nullptr) {
        auto numTarShape = numTarget->GetStorageShape();
        int64_t numTarDim = numTarShape.GetDimNum();
        OPS_CHECK(numTarDim != CONTEXT_DIM_NUM,
                  OPS_LOG_E("", "num_target should have %d dimension, but get %d", CONTEXT_DIM_NUM, numTarDim),
                  return false);
        int64_t batchSizeTar = numTarShape.GetDim(0);
        OPS_CHECK(batchSizeTar != batchSize,
                  OPS_LOG_E("", "The length of num_target expect %lld, but get %lld", batchSize, batchSizeTar),
                  return false);
    }
    return TilingShapePaged(context, tiling);
}

bool TilingPolicyPaged::TilingShapePaged(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    // paged_ids、page_offsets、为空
    auto pagedIds = context->GetOptionalInputTensor(INPUT_INDEX_T::PAGE_IDS_INDEX);
    auto pagedOffset = context->GetOptionalInputTensor(INPUT_INDEX_T::PAGE_OFFSETS_INDEX);
    auto lastPageLen = context->GetOptionalInputTensor(INPUT_INDEX_T::LAST_PAGE_LEN_INDEX);
    auto kvCache = context->GetOptionalInputTensor(INPUT_INDEX_T::KV_CACHE_INDEX);
    OPS_CHECK(pagedIds == nullptr || pagedOffset == nullptr || lastPageLen == nullptr || kvCache == nullptr,
              OPS_LOG_E("Tiling Debug", "pagedIds, pagedOffset, lastPageLen and kvCache should not be nullptr."),
              return false);

    OPS_LOG_E_IF_NULL("kvCache shape", context->GetInputShape(INPUT_INDEX_T::KV_CACHE_INDEX), return false);
    OPS_LOG_E_IF_NULL("q_offsets shape", context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX), return false);
    OPS_LOG_E_IF_NULL("k_offsets shape", context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_K_INDEX), return false);
    OPS_LOG_E_IF_NULL("t_offsets shape", context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_T_INDEX), return false);

    // [page_num, 2, paged_size, num_head, head_dim]
    auto pageSize = context->GetInputShape(INPUT_INDEX_T::KV_CACHE_INDEX)->GetStorageShape().GetDim(DIM_2);
    auto offsetsLenQ = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX)->GetStorageShape().GetDim(0);
    auto offsetsLenK = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_K_INDEX)->GetStorageShape().GetDim(0);
    auto offsetsLenT = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_T_INDEX)->GetStorageShape().GetDim(0);

    OPS_CHECK(offsetsLenQ != offsetsLenK || offsetsLenQ != offsetsLenT,
        OPS_LOG_E("Tiling Debug", "offsetsLenQ, offsetsLenK and offsetsLenT should have the same shape."),
        return false);
    // pageSize [32, 64, 128, 256]
    OPS_CHECK(BLOCK_HEIGHT % pageSize != 0 || pageSize < MIN_PAGE_SIZE,
        OPS_LOG_E("Tiling Debug", "PageSize should be 32, 64, 128 or 256."),
        return false);
    tiling.set_pageSize(pageSize);
    return true;
}

bool TilingPolicyPaged::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    context->SetTilingKey(PAGED_TILING_KEY);
    return true;
}
}
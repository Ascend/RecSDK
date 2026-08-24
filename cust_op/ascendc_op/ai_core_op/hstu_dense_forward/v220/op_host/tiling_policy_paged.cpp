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
#include "tiling_policy_paged.h"

constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;

using namespace HstuForward;

namespace HstuPagedForward {

ge::graphStatus TilingPolicyPaged::TilingProcess(gert::TilingContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);

    optiling::HstuPagedForwardTilingData tiling;

    // step1: get attribute
    OPS_CHECK(!TilingAttribute(context, tiling), OPS_LOG_E("", "TilingAttribute is failed.\n"),
              return ge::GRAPH_FAILED);

    // step2: get key shape form input
    OPS_CHECK(!TilingShape(context, tiling), OPS_LOG_E("", "TilingShape is failed.\n"), return ge::GRAPH_FAILED);

    // step3: tiling core
    OPS_CHECK(!TilingCore(context), OPS_LOG_E("", "TilingCore is failed.\n"), return ge::GRAPH_FAILED);

    // step5: set tiling key
    OPS_CHECK(!TilingKeySet(context, tiling), OPS_LOG_E("", "TilingKeySet is failed.\n"), return ge::GRAPH_FAILED);

    // step6: tiling save to buffer
    OPS_CHECK(!TilingSaveToBuffer(context, tiling), OPS_LOG_E("", "TilingSaveToBuffer is failed.\n"),
              return ge::GRAPH_FAILED);

    // step7: set workspace
    OPS_CHECK(!TilingWorkSpace(context), OPS_LOG_E("", "Set workspace size is failed.\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool TilingPolicyPaged::TilingAttribute(gert::TilingContext* context, optiling::HstuPagedForwardTilingData& tiling)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const uint32_t* maskType = attrs->GetAttrPointer<uint32_t>(PAGED_ATTR_INDEX_T::MASKTYPE_INDEX);
    OPS_CHECK_PTR_NULL(maskType, return false);

    const uint32_t* maxSeqLen = attrs->GetAttrPointer<uint32_t>(PAGED_ATTR_INDEX_T::MAX_SEQ_Q_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLen, return false);

    const uint32_t* maxSeqLenk = attrs->GetAttrPointer<uint32_t>(PAGED_ATTR_INDEX_T::MAX_SEQ_K_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLenk, return false);

    const float* siluScale = attrs->GetAttrPointer<float>(PAGED_ATTR_INDEX_T::SILU_SCALE_INDEX);
    OPS_CHECK_PTR_NULL(siluScale, return false);

    const float* alpha = attrs->GetAttrPointer<float>(PAGED_ATTR_INDEX_T::ALPHA_INDEX);
    OPS_CHECK_PTR_NULL(alpha, return false);

    const bool* deterministic = attrs->GetAttrPointer<bool>(PAGED_ATTR_INDEX_T::DETERMINISTIC_INDEX);
    OPS_CHECK_PTR_NULL(deterministic, return false);

    const uint32_t* targetGroupSize = attrs->GetAttrPointer<uint32_t>(PAGED_ATTR_INDEX_T::TARGET_GROUP_SIZE_INDEX);
    OPS_CHECK_PTR_NULL(targetGroupSize, return false);

    auto biasTensor = context->GetOptionalInputTensor(PAGED_INPUT_INDEX_T::ATTN_BIAS_INDEX);
    bool enableBias = (biasTensor != nullptr);
    tiling.set_enableBias(enableBias);

    tiling.set_maskType(*maskType);
    tiling.set_siluScale(*siluScale);
    tiling.set_alpha(*alpha);
    tiling.set_maxSeqLen(*maxSeqLen);
    tiling.set_maxSeqLenq(*maxSeqLen);
    tiling.set_maxSeqLenk(*maxSeqLenk);
    tiling.set_targetGroupSize(*targetGroupSize);
    tiling.set_deterministic(*deterministic);
    tiling.set_blockHeight(BLOCK_HEIGHT);
    return true;
}

bool TilingPolicyPaged::TilingShape(gert::TilingContext* context, optiling::HstuPagedForwardTilingData& tiling)
{
    int64_t batchSize;
    int64_t seqLensQ;
    int64_t headNumQ;
    int64_t headDimQ;
    int64_t maxSeqLensQ;
    int64_t seqLensK;
    int64_t headNumK;
    int64_t headDimK;
    int64_t headDimV;
    int64_t maxSeqLensK;

    auto seqOffsetQShape = context->GetInputShape(PAGED_INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX)->GetStorageShape();
    batchSize = seqOffsetQShape.GetDim(0) - 1;

    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
              OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return false);

    auto qShape = context->GetInputShape(PAGED_INPUT_INDEX_T::Q_INDEX)->GetStorageShape();
    auto kShape = context->GetInputShape(PAGED_INPUT_INDEX_T::K_INDEX)->GetStorageShape();
    auto vShape = context->GetInputShape(PAGED_INPUT_INDEX_T::V_INDEX)->GetStorageShape();

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

    headDimV = vShape.GetDim(2);
    tiling.set_vDim(headDimV);

    OPS_CHECK(kShape != vShape, OPS_LOG_E("", "K, V shape mismatch"), return false);
    OPS_CHECK(seqLensQ > seqLensK, OPS_LOG_E("", "Q, K, V seqlen mismatch"), return false);
    OPS_CHECK(headDimQ != headDimK, OPS_LOG_E("", "Q, K, V headDIM Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensQ, headNumQ, headDimQ),
              OPS_LOG_E("", "Q Jagged Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensK, headNumK, headDimK),
              OPS_LOG_E("", "K Jagged Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensK, headNumK, headDimV, true),
              OPS_LOG_E("", "V Jagged Shape Check failed"), return false);
    if (headNumQ != headNumK) {
        OPS_CHECK((headNumQ % headNumK != 0), OPS_LOG_E("", "For GQA, headNumQ must be divisible by headNumK"),
                  return false);
    }

    tiling.set_headRatio(headNumQ / headNumK);
    tiling.set_headNumK(headNumK);

    uint32_t masktype = tiling.get_maskType();

    const bool* deterministic = context->GetAttrs()->GetAttrPointer<bool>(PAGED_ATTR_INDEX_T::DETERMINISTIC_INDEX);
    OPS_CHECK_PTR_NULL(deterministic, return false);
    tiling.set_deterministic(*deterministic);

    auto numContext = context->GetOptionalInputShape(PAGED_INPUT_INDEX_T::NUM_CONTEXT_INDEX);
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

    auto numTarget = context->GetOptionalInputShape(PAGED_INPUT_INDEX_T::NUM_TARGET_INDEX);
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

bool TilingPolicyPaged::TilingShapePaged(gert::TilingContext* context, optiling::HstuPagedForwardTilingData& tiling)
{
    // paged_ids、page_offsets、为空
    auto pagedIds = context->GetOptionalInputTensor(PAGED_INPUT_INDEX_T::PAGE_IDS_INDEX);
    auto pagedOffset = context->GetOptionalInputTensor(PAGED_INPUT_INDEX_T::PAGE_OFFSETS_INDEX);
    auto lastPageLen = context->GetOptionalInputTensor(PAGED_INPUT_INDEX_T::LAST_PAGE_LEN_INDEX);
    auto kvCache = context->GetOptionalInputTensor(PAGED_INPUT_INDEX_T::KV_CACHE_INDEX);
    auto kCache = context->GetOptionalInputTensor(PAGED_INPUT_INDEX_T::K_CACHE_INDEX);
    auto vCache = context->GetOptionalInputTensor(PAGED_INPUT_INDEX_T::V_CACHE_INDEX);

    bool hasSplitCache = (kCache != nullptr && vCache != nullptr);
    bool hasCombinedCache = (kvCache != nullptr);
    OPS_CHECK(hasSplitCache && hasCombinedCache,
              OPS_LOG_E("Tiling Debug", "kv_cache and k_cache/v_cache cannot be provided at the same time."),
              return false);
    OPS_CHECK(!hasSplitCache && !hasCombinedCache,
              OPS_LOG_E("Tiling Debug", "must provide kv_cache or k_cache/v_cache."), return false);
    OPS_CHECK(pagedIds == nullptr || pagedOffset == nullptr || lastPageLen == nullptr,
              OPS_LOG_E("Tiling Debug", "pagedIds, pagedOffset and lastPageLen should not be nullptr."), return false);

    int64_t pageSize;
    if (hasSplitCache) {
        OPS_LOG_E_IF_NULL("k_cache shape", context->GetInputShape(PAGED_INPUT_INDEX_T::K_CACHE_INDEX), return false);
        OPS_LOG_E_IF_NULL("v_cache shape", context->GetInputShape(PAGED_INPUT_INDEX_T::V_CACHE_INDEX), return false);
        // [page_num, paged_size, num_head, head_dim]
        pageSize = context->GetInputShape(PAGED_INPUT_INDEX_T::K_CACHE_INDEX)->GetStorageShape().GetDim(DIM_1);
    } else {
        OPS_LOG_E_IF_NULL("kvCache shape", context->GetInputShape(PAGED_INPUT_INDEX_T::KV_CACHE_INDEX), return false);
        // [page_num, 2, paged_size, num_head, head_dim]
        pageSize = context->GetInputShape(PAGED_INPUT_INDEX_T::KV_CACHE_INDEX)->GetStorageShape().GetDim(DIM_2);
    }
    tiling.set_enableSplitCache(hasSplitCache);

    OPS_LOG_E_IF_NULL("q_offsets shape", context->GetInputShape(PAGED_INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX), return false);
    OPS_LOG_E_IF_NULL("k_offsets shape", context->GetInputShape(PAGED_INPUT_INDEX_T::SEQ_OFFSET_K_INDEX), return false);
    OPS_LOG_E_IF_NULL("t_offsets shape", context->GetInputShape(PAGED_INPUT_INDEX_T::SEQ_OFFSET_T_INDEX), return false);

    auto offsetsLenQ = context->GetInputShape(PAGED_INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX)->GetStorageShape().GetDim(0);
    auto offsetsLenK = context->GetInputShape(PAGED_INPUT_INDEX_T::SEQ_OFFSET_K_INDEX)->GetStorageShape().GetDim(0);
    auto offsetsLenT = context->GetInputShape(PAGED_INPUT_INDEX_T::SEQ_OFFSET_T_INDEX)->GetStorageShape().GetDim(0);

    OPS_CHECK(offsetsLenQ != offsetsLenK || offsetsLenQ != offsetsLenT,
              OPS_LOG_E("Tiling Debug", "offsetsLenQ, offsetsLenK and offsetsLenT should have the same shape."),
              return false);
    // pageSize [32, 64, 128, 256]
    OPS_CHECK(BLOCK_HEIGHT % pageSize != 0 || pageSize < MIN_PAGE_SIZE,
              OPS_LOG_E("Tiling Debug", "PageSize should be 32, 64, 128 or 256."), return false);
    tiling.set_pageSize(pageSize);
    return true;
}

bool TilingPolicyPaged::TilingKeySet(gert::TilingContext* context, optiling::HstuPagedForwardTilingData& tiling)
{
    uint32_t typeTilingKey = PAGED_TILING_KEY & 0x3;
    TilingKeyParam param{.enableBias = tiling.get_enableBias(),
                         .deterministic = tiling.get_deterministic(),
                         .maskType = tiling.get_maskType(),
                         .dimQ = tiling.get_dim(),
                         .dimV = tiling.get_vDim(),
                         .maxSeqLenQ = tiling.get_maxSeqLenq(),
                         .maxSeqLenK = tiling.get_maxSeqLenk()};
    return TilingKeySetImpl(context, param, typeTilingKey);
}

bool TilingPolicyPaged::TilingSaveToBuffer(gert::TilingContext* context, optiling::HstuPagedForwardTilingData& tiling)
{
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return false);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return true;
}

bool TilingPolicyPaged::TilingWorkSpace(gert::TilingContext* context)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_CHECK_PTR_NULL(currentWorkspace, return false);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * MAX_DIM * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;
    // 3: midk midv attnscore
    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem * 3) * sizeof(float);
    int64_t syncSize = coreNum * VCORE_NUM_IN_ONE_AIC * DATA_ALIGN_BYTES;
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize + syncSize;
    return true;
}
}  // namespace HstuPagedForward

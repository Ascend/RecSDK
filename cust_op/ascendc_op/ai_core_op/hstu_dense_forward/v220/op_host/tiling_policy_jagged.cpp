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


#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>

#include "common_host.h"
#include "register/op_def_registry.h"
#include "tiling_policy_jagged.h"

constexpr bool JAGGED_TASK_ASSIGN_DEBUG = false;

#if JAGGED_TASK_ASSIGN_DEBUG
#include <chrono>
#endif

using namespace HstuForward;

namespace HstuJaggedForward {

ge::graphStatus TilingPolicyJagged::TilingProcess(gert::TilingContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);

    optiling::HstuJaggedForwardTilingData tiling;

    // step1: get attribute
    OPS_CHECK(!TilingAttribute(context, tiling), OPS_LOG_E("", "TilingAttribute is failed.\n"),
              return ge::GRAPH_FAILED);

    // step2: get key shape form input
    OPS_CHECK(!TilingShape(context, tiling), OPS_LOG_E("", "TilingShape is failed.\n"), return ge::GRAPH_FAILED);

    // step3: tiling core
    OPS_CHECK(!TilingCore(context), OPS_LOG_E("", "TilingCore is failed.\n"), return ge::GRAPH_FAILED);

    // step4: set tiling key
    OPS_CHECK(!TilingKeySet(context, tiling), OPS_LOG_E("", "TilingKeySet is failed.\n"), return ge::GRAPH_FAILED);

    // step5: tiling save to buffer
    OPS_CHECK(!TilingSaveToBuffer(context, tiling), OPS_LOG_E("", "TilingSaveToBuffer is failed.\n"),
              return ge::GRAPH_FAILED);

    // step6: set workspace
    OPS_CHECK(!TilingWorkSpace(context), OPS_LOG_E("", "Set workspace size is failed.\n"),
    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool TilingPolicyJagged::TilingAttribute(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const uint32_t *maskType = attrs->GetAttrPointer<uint32_t>(JAGGED_ATTR_INDEX_T::MASKTYPE_INDEX);
    OPS_CHECK_PTR_NULL(maskType, return false);

    const uint32_t *maxSeqLen = attrs->GetAttrPointer<uint32_t>(JAGGED_ATTR_INDEX_T::MAX_SEQ_Q_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLen, return false);

    const uint32_t *maxSeqLenk = attrs->GetAttrPointer<uint32_t>(JAGGED_ATTR_INDEX_T::MAX_SEQ_K_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLenk, return false);

    const float *siluScale = attrs->GetAttrPointer<float>(JAGGED_ATTR_INDEX_T::SILU_SCALE_INDEX);
    OPS_CHECK_PTR_NULL(siluScale, return false);

    const uint32_t *targetGroupSize = attrs->GetAttrPointer<uint32_t>(JAGGED_ATTR_INDEX_T::TARGET_GROUP_SIZE_INDEX);
    OPS_CHECK_PTR_NULL(targetGroupSize, return false);

    const float *alpha = attrs->GetAttrPointer<float>(JAGGED_ATTR_INDEX_T::ALPHA_INDEX);
    OPS_CHECK_PTR_NULL(alpha, return false);

    const bool *deterministic = attrs->GetAttrPointer<bool>(JAGGED_ATTR_INDEX_T::DETERMINISTIC_INDEX);
    OPS_CHECK_PTR_NULL(deterministic, return false);

    auto biasTensor = context->GetOptionalInputTensor(JAGGED_INPUT_INDEX_T::ATTN_BIAS_INDEX);
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

bool TilingPolicyJagged::TilingShape(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling)
{
    int64_t batchSize;
    int64_t seqlenBatchSumQ;
    int64_t headNumQ;
    int64_t headDimQ;
    int64_t maxSeqLensQ;
    int64_t seqlenBatchSumK;
    int64_t headNumK;
    int64_t headDimK;
    int64_t maxSeqLensK;
    int64_t seqlenBatchSumV;
    int64_t headNumV;
    int64_t headDimV;

    auto seqOffsetQShape = context->GetInputShape(JAGGED_INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX)->GetStorageShape();
    batchSize = seqOffsetQShape.GetDim(0) - 1;

    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
              OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return false);

    OPS_CHECK_PTR_NULL(context->GetInputShape(JAGGED_INPUT_INDEX_T::Q_INDEX), return false);

    auto qShape = context->GetInputShape(JAGGED_INPUT_INDEX_T::Q_INDEX)->GetStorageShape();
    auto kShape = context->GetInputShape(JAGGED_INPUT_INDEX_T::K_INDEX)->GetStorageShape();
    auto vShape = context->GetInputShape(JAGGED_INPUT_INDEX_T::V_INDEX)->GetStorageShape();

    OPS_CHECK(qShape.GetDimNum() != JAGGED_DIM_NUM,
              OPS_LOG_E("", "Jagged QKV should have 3 dimensions, but get %d", qShape.GetDimNum()), return false);

    // Q: [bs, n, d]
    seqlenBatchSumQ = qShape.GetDim(0);
    headNumQ = qShape.GetDim(1);
    headDimQ = qShape.GetDim(2);
    maxSeqLensQ = tiling.get_maxSeqLenq();
    // K: [bs, n, d]
    seqlenBatchSumK = kShape.GetDim(0);
    headNumK = kShape.GetDim(1);
    headDimK = kShape.GetDim(2);
    maxSeqLensK = tiling.get_maxSeqLenk();
    // V: [bs, n, d]
    seqlenBatchSumV = vShape.GetDim(0);
    headNumV = vShape.GetDim(1);
    headDimV = vShape.GetDim(2);

    tiling.set_batchSize(batchSize);
    tiling.set_headNum(headNumQ);
    tiling.set_dim(headDimQ);
    tiling.set_vDim(headDimV);
    tiling.set_seqLen(maxSeqLensQ);

    OPS_CHECK(seqlenBatchSumK != seqlenBatchSumV, OPS_LOG_E("", "K, V seqLens mismatch"), return false);
    OPS_CHECK(headNumK != headNumV, OPS_LOG_E("", "K, V headNum mismatch"), return false);
    OPS_CHECK(headDimQ != headDimK, OPS_LOG_E("", "Q, K, V headDIM Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensQ, headNumQ, headDimQ),
              OPS_LOG_E("", "Q Jagged Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensK, headNumK, headDimK),
              OPS_LOG_E("", "K Jagged Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensK, headNumK, headDimV, true),
              OPS_LOG_E("", "V Jagged Shape Check failed"), return false);
    if (headNumQ != headNumK) {
        OPS_CHECK((headNumQ % headNumK) != 0, OPS_LOG_E("", "For GQA, headNumQ must be divisible by headNumK"),
                  return false);
    }

    tiling.set_headRatio(headNumQ / headNumK);
    tiling.set_headNumK(headNumK);

    uint32_t masktype = tiling.get_maskType();

    auto numContext = context->GetOptionalInputShape(JAGGED_INPUT_INDEX_T::NUM_CONTEXT_INDEX);
    bool enableNumContext = (numContext != nullptr);
    tiling.set_enableNumContext(enableNumContext);
    if (enableNumContext) {
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

    auto numTarget = context->GetOptionalInputShape(JAGGED_INPUT_INDEX_T::NUM_TARGET_INDEX);
    bool enableNumTarget = (numTarget != nullptr);
    tiling.set_enableNumTarget(enableNumTarget);
    if (enableNumTarget) {
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
    return true;
}

bool TilingPolicyJagged::TilingKeySet(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling)
{
    uint32_t typeTilingKey = JAGGED_TILING_KEY & 0x3;
    TilingKeyParam param {
        .enableBias = tiling.get_enableBias(),
        .deterministic = tiling.get_deterministic(),
        .maskType = tiling.get_maskType(),
        .dimQ = tiling.get_dim(),
        .dimV = tiling.get_vDim(),
        .maxSeqLenQ = tiling.get_maxSeqLenq(),
        .maxSeqLenK = tiling.get_maxSeqLenk()
    };
    return TilingKeySetImpl(context, param, typeTilingKey);
}

bool TilingPolicyJagged::TilingSaveToBuffer(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling)
{
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return false);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return true;
}
}  // namespace HstuDenseForward

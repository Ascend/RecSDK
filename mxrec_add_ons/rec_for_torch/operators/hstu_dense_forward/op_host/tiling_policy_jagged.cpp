/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "../../../common/common_host.h"
#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_jagged.h"

constexpr bool JAGGED_TASK_ASSIGN_DEBUG = false;

#if JAGGED_TASK_ASSIGN_DEBUG
#include <chrono>
#endif

namespace HstuDenseForward {

REGISTER_POLICY(LAYOUT_TYPE::JAGGED, std::make_shared<TilingPolicyJagged>());

bool TilingPolicyJagged::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
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
    OPS_CHECK(headNumQ != headNumK, OPS_LOG_E("", "Q, K, V headNum Shape Check failed"), return false);
    OPS_CHECK(headDimQ != headDimK, OPS_LOG_E("", "Q, K, V headDIM Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensQ, headNumQ, headDimQ),
              OPS_LOG_E("", "Q Jagged Shape Check failed"), return false);
    OPS_CHECK(!GeneralShapeCheck(batchSize, maxSeqLensK, headNumK, headDimK),
              OPS_LOG_E("", "K Jagged Shape Check failed"), return false);

    uint32_t masktype = tiling.get_maskType();
    auto *isDeltaQK = context->GetAttrs()->GetAttrPointer<uint32_t>(ATTR_INDEX_T::IS_DELTA_QK_INDEX);
    if (!*isDeltaQK) {
        OPS_CHECK(kShape != qShape, OPS_LOG_E("", "Q, K shape mismatch"), return false);
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
    return true;
}

bool TilingPolicyJagged::TilingCore(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();

    size_t aicCoreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    return true;
}

bool TilingPolicyJagged::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    context->SetTilingKey(JAGGED_TILING_KEY);
    return true;
}
}  // namespace HstuDenseForward

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

bool TilingPolicyJagged::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    int64_t batchSize;
    int64_t headNum;
    int64_t headDIM;
    int64_t seqLens;

    auto seqOffsetQShape = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX)->GetStorageShape();
    batchSize = seqOffsetQShape.GetDim(0) - 1;

    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
              OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return false);

    auto qShape = context->GetInputShape(INPUT_INDEX_T::Q_INDEX)->GetStorageShape();
    auto kShape = context->GetInputShape(INPUT_INDEX_T::K_INDEX)->GetStorageShape();
    auto vShape = context->GetInputShape(INPUT_INDEX_T::V_INDEX)->GetStorageShape();

    OPS_CHECK(qShape.GetDimNum() != JAGGED_DIM_NUM,
              OPS_LOG_E("", "Jagged QKV should have 3 dimensions, but get %d", qShape.GetDimNum()), return false);
    OPS_CHECK(!(qShape == kShape && qShape == vShape), OPS_LOG_E("", "Q, K, V shape mismatch"), return false);

    // Q: [bs, n, d]
    headNum = qShape.GetDim(1);
    headDIM = qShape.GetDim(2);
    seqLens = tiling.get_maxSeqLen();

    tiling.set_batchSize(batchSize);
    tiling.set_headNum(headNum);
    tiling.set_dim(headDIM);
    tiling.set_seqLen(seqLens);

    OPS_CHECK(!GeneralShapeCheck(batchSize, seqLens, headNum, headDIM),
              OPS_LOG_E("", "Jagged Shape Check failed"), return false);

    uint32_t masktype = tiling.get_maskType();
    if (masktype == 0) {
        auto numCtxShape = context->GetInputShape(INPUT_INDEX_T::NUM_CONTEXT_INDEX)->GetStorageShape();
        auto numTargetShape = context->GetInputShape(INPUT_INDEX_T::NUM_TARGET_INDEX)->GetStorageShape();
        int64_t batchSizeCtx = numCtxShape.GetDim(0);
        OPS_CHECK(batchSizeCtx != batchSize,
                  OPS_LOG_E("", "The length of num_context expect %lld, but get %lld", batchSize, batchSizeCtx),
                  return false);
        OPS_CHECK(numCtxShape != numTargetShape, OPS_LOG_E("", "num_context, num_target shape mismatch"), return false);
    }

    return true;
}

bool TilingPolicyJagged::TilingCore(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();

    size_t aicCoreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    
    return true;
}

bool TilingPolicyJagged::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    context->SetTilingKey(JAGGED_TILING_KEY);
    return true;
}

void TilingPolicyJagged::DumpTiling(optiling::HstuDenseForwardTilingData &tiling)
{
    this->TilingPolicy::DumpTiling(tiling);
}
}  // namespace HstuDenseForward
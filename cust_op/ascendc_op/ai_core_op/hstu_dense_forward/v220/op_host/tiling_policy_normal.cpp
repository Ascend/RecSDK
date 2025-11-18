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

#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_normal.h"

namespace HstuDenseForward {

REGISTER_POLICY(LAYOUT_TYPE::NORMAL, std::make_shared<TilingPolicyNormal>());

bool TilingPolicyNormal::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    auto qShape = context->GetInputShape(INPUT_INDEX_T::Q_INDEX)->GetStorageShape();
    auto kShape = context->GetInputShape(INPUT_INDEX_T::K_INDEX)->GetStorageShape();
    auto vShape = context->GetInputShape(INPUT_INDEX_T::V_INDEX)->GetStorageShape();

    OPS_CHECK(qShape.GetDimNum() != NORMAL_DIM_NUM,
              OPS_LOG_E("", "Normal QKV should have 4 dimensions, but get %d", qShape.GetDimNum()), return false);
    OPS_CHECK(!(qShape == kShape && kShape == vShape), OPS_LOG_E("", "Q, K, V shape mismatch"), return false);
    
    // Q: [b, s, n, d]
    int64_t batchSize = qShape.GetDim(0);
    tiling.set_batchSize(batchSize);
    int64_t seqLen = qShape.GetDim(1);
    tiling.set_maxSeqLenk(seqLen);
    tiling.set_seqLen(seqLen);
    int64_t headNum = qShape.GetDim(2);
    tiling.set_headNum(headNum);
    int64_t dim = qShape.GetDim(3);
    tiling.set_dim(dim);

    OPS_CHECK(!GeneralShapeCheck(batchSize, seqLen, headNum, dim), OPS_LOG_E("", "Shape Check failed"), return false);
    return true;
}

bool TilingPolicyNormal::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    context->SetTilingKey(NORMAL_TILING_KEY);
    return true;
}
}  // namespace HstuDenseForward

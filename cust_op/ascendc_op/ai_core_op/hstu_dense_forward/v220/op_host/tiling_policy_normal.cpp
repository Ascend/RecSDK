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

constexpr int QKV_DIM = 4;

namespace HstuDenseForward {

REGISTER_POLICY(LAYOUT_TYPE::NORMAL, std::make_shared<TilingPolicyNormal>());

bool TilingPolicyNormal::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    auto qShape = context->GetInputShape(0)->GetStorageShape();

    int64_t batchSize = qShape.GetDim(0);
    tiling.set_batchSize(batchSize);
    int64_t seqLen = qShape.GetDim(1);
    tiling.set_seqLen(seqLen);
    int64_t headNum = qShape.GetDim(2);
    tiling.set_headNum(headNum);
    int64_t dim = qShape.GetDim(3);
    tiling.set_dim(dim);

    if (!QKVShapeCheck(context, QKV_DIM)) {
        return false;
    }
    OPS_CHECK(!GeneralShapeCheck(batchSize, seqLen, headNum, dim), OPS_LOG_E("", "Shape Check failed"), return false);
    return true;
}

bool TilingPolicyNormal::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    OPS_LOG_E_IF_NULL("query", context->GetInputTensor(0), return false);
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        context->SetTilingKey(FLOAT_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        context->SetTilingKey(FLOAT16_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_BF16) {
        context->SetTilingKey(BF16_TILING_KEY);
    } else {
        OPS_LOG_E("", "invalid datatype, only support fp32, fp16, bf16.\n");
        return false;
    }

    return true;
}

}
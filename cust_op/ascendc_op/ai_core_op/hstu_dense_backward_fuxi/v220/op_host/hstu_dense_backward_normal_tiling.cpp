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

#include "hstu_dense_backward_normal_tiling.h"

namespace optiling {
ge::graphStatus CheckMaskTypeAndBias(gert::TilingContext *context,
                                     HstuDenseBackwardFuxiTilingData &tiling)
{
    auto batchSize = tiling.get_batchSize();
    auto headNum = tiling.get_headNum();
    auto maxSeqLen = tiling.get_maxSeqLen();
    auto maskType = tiling.get_maskType();

    auto posBias = context->GetOptionalInputTensor(INDEX_T::INDEX_BIAS_POSITION);
    auto tsBias = context->GetOptionalInputTensor(INDEX_T::INDEX_BIAS_TIMESTAMP);
    if (posBias == nullptr || tsBias == nullptr) {
        tiling.set_enableBias(0);
    } else {
        tiling.set_enableBias(1);
    }

    if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
        auto mask = context->GetOptionalInputTensor(INDEX_T::INDEX_4);
        OPS_LOG_E_IF_NULL("mask can't be none when maskType is MASK_CUSTOM, mask", mask, return ge::GRAPH_FAILED);

        auto maskShape = context->GetInputShape(INDEX_T::INDEX_4)->GetStorageShape();
        OPS_LOG_E_IF(maskShape.GetDimNum() != MASK_DIM_NUM,
                     context,
                     return ge::GRAPH_FAILED,
                     "mask dim num is not %d", MASK_DIM_NUM);

        OPS_LOG_E_IF(maskShape.GetDim(INDEX_T::INDEX_0) != batchSize ||
                     maskShape.GetDim(INDEX_T::INDEX_1) != headNum ||
                     maskShape.GetDim(INDEX_T::INDEX_2) != maxSeqLen ||
                     maskShape.GetDim(INDEX_T::INDEX_3) != maxSeqLen,
                     context,
                     return ge::GRAPH_FAILED,
                     "mask shape must be {batchSize, headNum, seqLen, seqLen}");
    } else if (IfMask(maskType, MaskType::MASK_TRIL) ||
               IfMask(maskType, MaskType::MASK_NONE)) {
        // do nothing
    } else if (IfMask(maskType, MaskType::MASK_TRIU)) {
        OPS_LOG_E("Check", "maskType:MASK_TRIU is not support yet");
        return ge::GRAPH_FAILED;
    } else {
        OPS_LOG_E("Check", "supported maskType list is [MASK_TRIL, MASK_NONE, MASK_CUSTOM]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
}
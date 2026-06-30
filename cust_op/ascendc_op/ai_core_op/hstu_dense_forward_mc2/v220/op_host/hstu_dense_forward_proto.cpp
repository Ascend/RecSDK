/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "hcom_topo_info.h"

using namespace ge;
namespace ops {
constexpr size_t DIM_FOUR = 4UL;

constexpr size_t COMBINE_INPUT_Q_INDEX = 0;
constexpr size_t COMBINE_INPUT_K_INDEX = 1;
constexpr size_t COMBINE_INPUT_V_INDEX = 2;
constexpr size_t COMBINE_OUTPUT_INDEX = 0;

static ge::graphStatus InferShapeHstuDenseForward(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(COMBINE_INPUT_Q_INDEX);
    if (qShape == nullptr)
        return GRAPH_FAILED;
    const gert::Shape* kShape = context->GetInputShape(COMBINE_INPUT_K_INDEX);
    if (kShape == nullptr)
        return GRAPH_FAILED;
    const gert::Shape* vShape = context->GetInputShape(COMBINE_INPUT_V_INDEX);
    if (vShape == nullptr)
        return GRAPH_FAILED;

    gert::Shape* outputShape = context->GetOutputShape(COMBINE_OUTPUT_INDEX);
    if (outputShape == nullptr)
        return GRAPH_FAILED;

    bool sameShape = (qShape->GetDimNum() == kShape->GetDimNum()) && (qShape->GetDimNum() == vShape->GetDimNum());
    if (!sameShape) {
        return ge::FAILED;
    }

    outputShape->SetDimNum(DIM_FOUR);
    outputShape->SetDim(0U, qShape->GetDim(0));
    outputShape->SetDim(1U, qShape->GetDim(1));
    outputShape->SetDim(2U, qShape->GetDim(2));
    outputShape->SetDim(3U, qShape->GetDim(3));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHstuDenseForward(gert::InferDataTypeContext* context)
{
    auto xDtype = context->GetInputDataType(COMBINE_INPUT_Q_INDEX);
    context->SetOutputDataType(COMBINE_OUTPUT_INDEX, xDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HstuDenseForward)
    .InferShape(InferShapeHstuDenseForward)
    .InferDataType(InferDataTypeHstuDenseForward);
}  // namespace ops

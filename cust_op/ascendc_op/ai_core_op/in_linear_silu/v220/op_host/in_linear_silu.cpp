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

#include "in_linear_silu_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
constexpr size_t ATTR_INDEX_SPLIT_LIST = 0;
constexpr size_t ATTR_INDEX_REQUIRES_GRAD = 1;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    InLinearSiluTilingData tiling;
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(INPUT_INDEX_X), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("wShape", context->GetInputShape(INPUT_INDEX_WEIGHT), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xInputDesc", context->GetInputDesc(0), return ge::GRAPH_FAILED);
    const ge::DataType selfDataType = context->GetInputDesc(0)->GetDataType();
    const gert::Shape xShape = context->GetInputShape(INPUT_INDEX_X)->GetStorageShape();
    const gert::Shape weightShape = context->GetInputShape(INPUT_INDEX_WEIGHT)->GetStorageShape();
    tiling.set_m(xShape.GetDim(0));
    tiling.set_k(xShape.GetDim(1));
    tiling.set_n(weightShape.GetDim(0));
    const auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    auto splitArgList = attrs->GetListInt(ATTR_INDEX_SPLIT_LIST)->GetData();
    OPS_LOG_E_IF_NULL("splitArgList", splitArgList, return ge::GRAPH_FAILED);
    tiling.set_splitList((int64_t*)splitArgList);

    tiling.set_transB(true);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t totalCoreNum = ascendcPlatform.GetCoreNumAic();

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    context->SetBlockDim(totalCoreNum);
    context->SetTilingKey(static_cast<uint64_t>(selfDataType));
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* x_shape = context->GetInputShape(INPUT_INDEX_X);
    OPS_LOG_E_IF_NULL("x_shape", x_shape, return ge::GRAPH_FAILED);
    const gert::Shape* weight_shape = context->GetInputShape(INPUT_INDEX_WEIGHT);
    OPS_LOG_E_IF_NULL("weight_shape", weight_shape, return ge::GRAPH_FAILED);
    const auto attrs = context->GetAttrs();
    auto split_arg_list = attrs->GetListInt(ATTR_INDEX_SPLIT_LIST)->GetData();

    for (int i = 0; i < 4; ++i) {
        const auto output_shape = context->GetOutputShape(i);
        string ops_desc = "output_shape" + std::to_string(i);
        OPS_LOG_E_IF_NULL(ops_desc.c_str(), output_shape, return ge::GRAPH_FAILED);
        output_shape->SetDimNum(2);
        output_shape->SetDim(0, x_shape->GetDim(0));
        output_shape->SetDim(1, split_arg_list[i]);
    }

    const auto linear_output_shape = context->GetOutputShape(4);
    OPS_LOG_E_IF_NULL("linear_output_shape", linear_output_shape, return ge::GRAPH_FAILED);
    linear_output_shape->SetDimNum(2);
    linear_output_shape->SetDim(0, x_shape->GetDim(0));
    linear_output_shape->SetDim(1, weight_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, inputDataType);
    context->SetOutputDataType(2, inputDataType);
    context->SetOutputDataType(3, inputDataType);
    context->SetOutputDataType(4, inputDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class InLinearSilu : public OpDef {
public:
    explicit InLinearSilu(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("user")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("linear_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("split_arg_list").ListInt();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(InLinearSilu);
}  // namespace ops
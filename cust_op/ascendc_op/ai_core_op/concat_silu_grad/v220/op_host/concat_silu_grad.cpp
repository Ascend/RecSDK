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
#include "concat_silu_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
constexpr size_t INPUT_INDEX_SILU_INPUT = 4;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConcatSiluGradTilingData tiling;

    const ge::DataType selfDataType = context->GetInputDesc(0)->GetDataType();
    if (selfDataType != ge::DataType::DT_FLOAT && selfDataType != ge::DataType::DT_FLOAT16 &&
        selfDataType != ge::DataType::DT_BF16) {
        OPS_LOG_E("", "invalid data type, only support float/fp16/bf16\n");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape siluInputShape = context->GetInputShape(INPUT_INDEX_SILU_INPUT)->GetStorageShape();
    OPS_CHECK(
        siluInputShape.GetDimNum() < 2,
        OPS_LOG_E("[ERROR]", "Invalid input0 shape, dim num must be 2, but got: %d.", siluInputShape.GetDimNum()),
        return ge::GRAPH_FAILED)
    auto row = siluInputShape.GetDim(0);
    auto col = siluInputShape.GetDim(1);
    tiling.set_m(row);
    tiling.set_n(col);

    int64_t splitArgList[MAX_SPLIT_NUM];
    for (int i = 0; i < MAX_SPLIT_NUM; i++) {
        const gert::Shape gradShape = context->GetInputShape(i)->GetStorageShape();
        splitArgList[i] = gradShape.GetDim(1);
    }
    tiling.set_splitList((int64_t*)splitArgList);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t totalCoreNum = ascendcPlatform.GetCoreNumAic();

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    context->SetBlockDim(totalCoreNum);
    context->SetTilingKey(static_cast<uint64_t>(selfDataType));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class ConcatSiluGrad : public OpDef {
public:
    explicit ConcatSiluGrad(const char* name) : OpDef(name)
    {
        this->Input("grad1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("grad2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("grad3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("grad4")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("silu_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_silu_input").ParamType(REQUIRED).Follow("silu_input");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(ConcatSiluGrad);
}  // namespace ops

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

#include "gather_for_rank1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace optiling {
constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int AT_LEAST_INDEX_UB_SIZE = 32;
constexpr int MAX_DIM = 20480;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GatherForRank1TilingData tiling;

    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexShape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("x", context->GetInputTensor(0), return ge::GRAPH_FAILED);

    int64_t xDim0 = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    int64_t indexDim0 = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    ge::DataType xType = context->GetInputTensor(0)->GetDataType();
    int dataTypeTilingKey = xType;

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;

    if (xDim0 > MAX_DIM) {
        OPS_LOG_E("[ERROR]", "Invalid shape of xDim0, it should be less than %d, but it's %lld!", MAX_DIM, xDim0);
        return ge::GRAPH_FAILED;
    }

    tiling.set_xDim0(xDim0);
    tiling.set_indexDim0(indexDim0);
    tiling.set_ubCanUsed(ubCanUsed);

    context->SetTilingKey(dataTypeTilingKey);

    context->SetBlockDim(coreNum);

    if (context->GetRawTilingData() == nullptr) {
        OPS_LOG_E("[ERROR]", "GetRawTilingData Failed!");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* x1_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);

    OPS_LOG_E_IF_NULL("x1_shape", x1_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("y_shape", y_shape, return ge::GRAPH_FAILED);

    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetDynamicInputDataType(0, 0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class GatherForRank1 : public OpDef {
public:
    explicit GatherForRank1(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
#ifdef SUPPORT_V200
            .DataType({ge::DT_INT32, ge::DT_INT32})
#else
            .DataType({ge::DT_INT64, ge::DT_INT64})
#endif
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910", aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend310p", aicore_config);
    }
};

OP_ADD(GatherForRank1);
}  // namespace ops
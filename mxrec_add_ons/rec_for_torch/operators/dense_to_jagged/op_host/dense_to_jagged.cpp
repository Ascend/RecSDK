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
#include "dense_to_jagged_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../../../common/ops_log.h"

namespace optiling {
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_512 = 512;
constexpr int32_t RESERVER_UB_SIZE = (20 * 1024);
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t TYPE_FLOAT = 0;
constexpr int32_t TYPE_INT64 = 9;
constexpr int32_t SIZEOF_FLOAT = 4;
constexpr int32_t SIZEOF_INT64 = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("denseShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetShape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("dense", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offset", context->GetInputTensor(1), return ge::GRAPH_FAILED);
    auto denseShape = context->GetInputShape(0)->GetStorageShape();
    auto offsetShape = context->GetInputShape(1)->GetStorageShape();
    auto denseType = context->GetInputTensor(0)->GetDataType();
    auto offsetType = context->GetInputTensor(1)->GetDataType();

    OPS_CHECK(denseShape.GetDim(DIM0) != offsetShape.GetDim(DIM0) - 1,
        OPS_LOG_E("[ERROR]", "dense shape[0] != offset shape[0] - 1"), return ge::GRAPH_FAILED);
   
    DenseToJaggedTilling tilingData;
    // Platform configuration
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = 0 + systemWorkspacesSize;
#ifndef SUPPORT_V200
    size_t coreNum = ascnedPlatform.GetCoreNumAiv();
#else
    size_t coreNum = 1;
#endif
    uint64_t ub;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;

    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    const int32_t* outDim0 = context->GetAttrs()->GetAttrPointer<int32_t>(0);
    int outDim1 = denseShape.GetDim(DIM2);
    int64_t jaggedTotal = *outDim0 * outDim1;
    int64_t denseTotal = denseShape.GetDim(DIM0) * denseShape.GetDim(DIM1) * denseShape.GetDim(DIM2);
    
    if (coreNum == 0) {
        printf("[ERROR] aiv core num == 0!");
        return ge::GRAPH_FAILED;
    }
    int singleCoreBatch = (offsetShape.GetDim(0) - 1) / coreNum;
    int left = (offsetShape.GetDim(0) - 1) % coreNum;
    int singleLoopSize = ub / 2 / ALIGN_512 * ALIGN_512;

    tilingData.set_denseDim1(denseShape.GetDim(DIM1));
    tilingData.set_denseDim2(denseShape.GetDim(DIM2));
    tilingData.set_left(left);
    tilingData.set_singleCoreBatch(singleCoreBatch);
    tilingData.set_singleLoopSize(singleLoopSize);
    tilingData.set_denseType(denseType);
    tilingData.set_offsetType(offsetType);
    tilingData.set_denseTotal(denseTotal);
    tilingData.set_jaggedTotal(jaggedTotal);

    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
constexpr int32_t DIM2 = 2;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* denseShape = context->GetInputShape(0);
    gert::Shape* jaggedShape = context->GetOutputShape(0);

    OPS_LOG_E_IF_NULL("denseShape", denseShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("jaggedShape", jaggedShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    const int32_t* jaggedDim0 = context->GetAttrs()->GetAttrPointer<int32_t>(0);

    jaggedShape->SetDimNum(DIM2);
    jaggedShape->SetDim(0, *jaggedDim0);
    jaggedShape->SetDim(1, denseShape->GetDim(DIM2));

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class DenseToJagged : public OpDef {
public:
    explicit DenseToJagged(const char* name) : OpDef(name)
    {
        this->Input("dense")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("jagged_dense")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("jagged_dim0").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(DenseToJagged);
}

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
#include "dense_to_jagged_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace optiling {
// 常量定义
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_512 = 512;
constexpr int32_t RESERVER_UB_SIZE = (20 * 1024); // 20KB
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;

constexpr int32_t JAGGED_DIM0_INDEX = 0;
constexpr int32_t INPUT_DENSE_INDEX = 0;
constexpr int32_t INPUT_OFFSET_INDEX = 1;
constexpr int32_t OUTPUT_JAGGED_INDEX = 0;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("denseShape", context->GetInputShape(INPUT_DENSE_INDEX),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetShape", context->GetInputShape(INPUT_OFFSET_INDEX),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("dense", context->GetInputTensor(INPUT_DENSE_INDEX),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offset", context->GetInputTensor(INPUT_OFFSET_INDEX),
                      return ge::GRAPH_FAILED);

    // 获取输入形状和类型
    auto denseShape = context->GetInputShape(INPUT_DENSE_INDEX)->GetStorageShape();
    auto offsetShape = context->GetInputShape(INPUT_OFFSET_INDEX)->GetStorageShape();
    auto denseType = context->GetInputTensor(INPUT_DENSE_INDEX)->GetDataType();
    auto offsetType = context->GetInputTensor(INPUT_OFFSET_INDEX)->GetDataType();

    OPS_CHECK(denseShape.GetDim(DIM0) != offsetShape.GetDim(DIM0) - 1,
        OPS_LOG_E("[ERROR]", "dense shape[0] != offset shape[0] - 1"), return ge::GRAPH_FAILED);

    OPS_CHECK(offsetShape.GetDim(DIM0) > std::numeric_limits<int>::max(),
        OPS_LOG_E("[ERROR]", "offset shape[0] > %d\n", std::numeric_limits<int>::max()), return ge::GRAPH_FAILED);

    // Platform configuration
    size_t usrSize = 0;
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    currentWorkspace[0] = usrSize + systemWorkspacesSize;
#ifndef SUPPORT_V200
    size_t coreNum = ascnedPlatform.GetCoreNumAiv();
#else
    size_t coreNum = 1;
#endif
    uint64_t ubSize = 0;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    const int32_t* outDim0 = context->GetAttrs()->GetAttrPointer<int32_t>(JAGGED_DIM0_INDEX);
    OPS_LOG_E_IF_NULL("outDim0", outDim0, return ge::GRAPH_FAILED);
    int outDim1 = denseShape.GetDim(DIM2);
    int64_t jaggedTotal = *outDim0 * outDim1;
    int64_t denseTotal = denseShape.GetDim(DIM0) * denseShape.GetDim(DIM1) * denseShape.GetDim(DIM2);

    OPS_CHECK(coreNum == 0, OPS_LOG_E("[ERROR]", "aiv core num == 0"), return ge::GRAPH_FAILED);
    int singleCoreBatch = (offsetShape.GetDim(DIM0) - 1) / coreNum;
    int left = (offsetShape.GetDim(DIM0) - 1) % coreNum;
    int singleLoopSize = (ubSize - RESERVER_UB_SIZE) / 2 / ALIGN_512 * ALIGN_512;

    // 设置分片数据
    DenseToJaggedTilling tilingData;
    tilingData.set_denseDim1(denseShape.GetDim(DIM1));
    tilingData.set_denseDim2(denseShape.GetDim(DIM2));
    tilingData.set_left(left);
    tilingData.set_singleCoreBatch(singleCoreBatch);
    tilingData.set_singleLoopSize(singleLoopSize);
    tilingData.set_denseType(denseType);
    tilingData.set_offsetType(offsetType);
    tilingData.set_denseTotal(denseTotal);
    tilingData.set_jaggedTotal(jaggedTotal);

    // 保存分片数据
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
using optiling::INPUT_DENSE_INDEX;
using optiling::OUTPUT_JAGGED_INDEX;
using optiling::JAGGED_DIM0_INDEX;
using optiling::DIM0;
using optiling::DIM1;
using optiling::DIM2;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* denseShape = context->GetInputShape(INPUT_DENSE_INDEX);
    gert::Shape* jaggedShape = context->GetOutputShape(OUTPUT_JAGGED_INDEX);

    OPS_LOG_E_IF_NULL("denseShape", denseShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("jaggedShape", jaggedShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    const int32_t* jaggedDim0 = context->GetAttrs()
                                       ->GetAttrPointer<int32_t>(JAGGED_DIM0_INDEX);
    OPS_LOG_E_IF_NULL("jaggedDim0", jaggedDim0, return ge::GRAPH_FAILED);
    jaggedShape->SetDimNum(DIM2);
    jaggedShape->SetDim(DIM0, *jaggedDim0);
    jaggedShape->SetDim(DIM1, denseShape->GetDim(DIM2));

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_JAGGED_INDEX,
                               context->GetInputDataType(INPUT_DENSE_INDEX));
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
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("jagged_dense")
            .ParamType(REQUIRED)
            .Follow("dense", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Attr("jagged_dim0").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(DenseToJagged);
}

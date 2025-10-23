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

#include "jagged_to_padded_dense_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"
namespace optiling {

constexpr int GM_ALIGN = 64;
constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int DATA_TYPE_INT64 = 8;
constexpr int DATA_TYPE_INT32 = 4;
constexpr int DATA_TYPE_FLOAT32 = 4;
constexpr int NUM_QUEUE = 4;
constexpr int UB_ALIGN = 32;
constexpr int SUPORT_EMBEDDING_DIM_NUM = 2;

static void SetTypeTiling(gert::TilingContext* context, JaggedToPaddedDenseTilingData& tiling)
{
    int64_t bytesOfDataType = 0;
    ge::DataType dataType = context->GetInputTensor(0)->GetDataType();
    if (dataType == ge::DataType::DT_FLOAT) {
        bytesOfDataType = DATA_TYPE_FLOAT32;
    } else {
        bytesOfDataType = DATA_TYPE_INT64;
    }

    int64_t offsetDataType = 0;
    ge::DataType offsetDataTypeGe = context->GetInputTensor(1)->GetDataType();
    if (offsetDataTypeGe == ge::DataType::DT_INT64) {
        offsetDataType = DATA_TYPE_INT64;
    } else {
        offsetDataType = DATA_TYPE_INT32;
    }
    tiling.set_bytesOfDataType(bytesOfDataType);
    tiling.set_offsetDataType(offsetDataType);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    JaggedToPaddedDenseTilingData tiling;
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    OPS_LOG_E_IF_NULL("valuesShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesTensor", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsTensor", context->GetInputTensor(1), return ge::GRAPH_FAILED);

    auto valuesShape = context->GetInputShape(0)->GetStorageShape();
    auto offsetsShape = context->GetInputShape(1)->GetStorageShape();

    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    ubCanUsed = ubCanUsed / UB_ALIGN / NUM_QUEUE * UB_ALIGN * NUM_QUEUE;
    tiling.set_ubCanUsed(ubCanUsed);

    if (valuesShape.GetDimNum() != SUPORT_EMBEDDING_DIM_NUM or offsetsShape.GetDimNum() != 1) {
        printf("[ERROR]jagged_to_padded_dense_tiling is only used for values with rank-3 and offset rank-1");
        return ge::GRAPH_FAILED;
    }

    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    // tiling core
    
    int64_t totalBatch = offsetsShape.GetDim(0) - 1;
    tiling.set_totalBatch(totalBatch);
    int64_t baseBatchLen = (offsetsShape.GetDim(0) - 1) / coreNum;
    tiling.set_baseBatchLen(baseBatchLen);
    int64_t tailSplitIndex = (offsetsShape.GetDim(0) - 1) % coreNum;
    tiling.set_tailSplitIndex(tailSplitIndex);
    int64_t valuesDim0 = valuesShape.GetDim(0);
    tiling.set_valuesDim0(valuesDim0);
    int64_t valuesDim1 = valuesShape.GetDim(1);
    tiling.set_valuesDim1(valuesDim1);
    int64_t offsetDim0 = offsetsShape.GetDim(0);
    tiling.set_offsetDim0(offsetDim0);
    int64_t outDim1 = *context->GetAttrs()->GetInt(0);
    tiling.set_outDim1(outDim1);
    SetTypeTiling(context, tiling);

    context->SetBlockDim(coreNum);

    OPS_LOG_E_IF_NULL("context->GetRawTilingData(0)", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    int64_t maxLen = *context->GetAttrs()->GetInt(0);
    const gert::Shape* valuesShape = context->GetInputShape(0);
    const gert::Shape* offsetsShape = context->GetInputShape(1);

    gert::Shape* outShape = context->GetOutputShape(0);

    OPS_LOG_E_IF_NULL("valuesShape", valuesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);

    int dimSize = 3;
    int dimIndex2 = 2;
    outShape->SetDimNum(dimSize);
    outShape->SetDim(0, offsetsShape->GetDim(0) - 1);
    outShape->SetDim(1, maxLen);
    outShape->SetDim(dimIndex2, valuesShape->GetDim(1));

    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class JaggedToPaddedDense : public OpDef {
public:
    explicit JaggedToPaddedDense(const char* name) : OpDef(name)
    {
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Attr("max_length").Int();
        this->Attr("padding_value").Float();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(JaggedToPaddedDense);
}  // namespace ops

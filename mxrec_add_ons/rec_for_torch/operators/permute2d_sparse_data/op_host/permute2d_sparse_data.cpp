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

#include "permute2d_sparse_data_tilling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../../../common/ops_log.h"
namespace optiling {

constexpr int GM_ALIGN = 64;
constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int DATA_TYPE_INT64 = 8;
constexpr int DATA_TYPE_INT32 = 4;
constexpr int DATA_TYPE_FLOAT32 = 4;
constexpr int NUM_QUEUE = 4;
constexpr int UB_ALIGN = 32;
constexpr int SUPPORT_EMBEDDING_DIM_NUM = 2;
constexpr int PERMUTE_INDEX = 0;
constexpr int LENGTH_INDEX = 1;
constexpr int VALUES_INDEX = 2;

static ge::graphStatus SetTypeTiling(gert::TilingContext* context, Permute2dSparseDataTilingData& tiling)
{
    // check tensor is nullptr
    OPS_LOG_E_IF_NULL("permute", context->GetInputTensor(PERMUTE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("length", context->GetInputTensor(LENGTH_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("value", context->GetInputTensor(VALUES_INDEX), return ge::GRAPH_FAILED);
    // permute: InputTensor(0), support int32
    int64_t permuteDataType = 0;
    ge::DataType permuteDataTypeGe = context->GetInputTensor(0)->GetDataType();
    if (permuteDataTypeGe == ge::DataType::DT_INT32) {
        permuteDataType = DATA_TYPE_INT32;
    }

    // lengths: InputTensor(1), support int64、int32
    int64_t lengthsDataType = 0;
    ge::DataType lengthsDataTypeGe = context->GetInputTensor(1)->GetDataType();
    if (lengthsDataTypeGe == ge::DataType::DT_INT64) {
        lengthsDataType = DATA_TYPE_INT64;
    } else {
        lengthsDataType = DATA_TYPE_INT32;
    }

    // value: InputTensor(2), support int64、int32、fp32
    int64_t valueDataType = 0;
    ge::DataType dataType = context->GetInputTensor(2)->GetDataType();
    if (dataType == ge::DataType::DT_INT32) {
        valueDataType = DATA_TYPE_INT32;
    } else if (dataType == ge::DataType::DT_INT64) {
        valueDataType = DATA_TYPE_INT64;
    } else {
        valueDataType = DATA_TYPE_FLOAT32;
    }

    tiling.set_valueDataType(valueDataType);
    tiling.set_permuteDataType(permuteDataType);
    tiling.set_lengthsDataType(lengthsDataType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("permuteShape", context->GetInputShape(PERMUTE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lengthsShape", context->GetInputShape(LENGTH_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesShape", context->GetInputShape(VALUES_INDEX), return ge::GRAPH_FAILED);

    Permute2dSparseDataTilingData tiling;
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto permuteShape = context->GetInputShape(0)->GetStorageShape();
    auto lengthsShape = context->GetInputShape(1)->GetStorageShape();
    auto valuesShape = context->GetInputShape(2)->GetStorageShape();

    // set ub
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = (ubCanUsed - RESERVER_UB_SIZE) / UB_ALIGN / NUM_QUEUE * UB_ALIGN * NUM_QUEUE;
    tiling.set_ubCanUsed(ubCanUsed);

    // datatype check
    if ((permuteShape.GetDimNum() != 1) || (lengthsShape.GetDimNum() != SUPPORT_EMBEDDING_DIM_NUM) ||
        (permuteShape.GetDim(0) != lengthsShape.GetDim(0)))  {
        printf("[ERROR]permute shape or lengths shape is error.");
        return ge::GRAPH_FAILED;
    }

    // set coreNUm
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_coreNum(coreNum);

    // tiling core
    int64_t totalBatch = permuteShape.GetDim(0);
    tiling.set_totalBatch(totalBatch);
    int64_t baseBatchLen = (permuteShape.GetDim(0)) / coreNum;
    tiling.set_baseBatchLen(baseBatchLen);
    int64_t tailSplitIndex = (permuteShape.GetDim(0)) % coreNum;
    tiling.set_tailSplitIndex(tailSplitIndex);

    // set data dim
    int64_t permuteDim0 = permuteShape.GetDim(0);
    tiling.set_permuteDim0(permuteDim0);
    int64_t lengthsT = lengthsShape.GetDim(0);
    tiling.set_lengthsT(lengthsT);
    int64_t lengthsB = lengthsShape.GetDim(1);
    tiling.set_lengthsB(lengthsB);
    int64_t valuesDim = valuesShape.GetDim(0);
    tiling.set_valuesDim(valuesDim);

    // apply workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize + (lengthsT + 1) * GM_ALIGN + (lengthsT + 1) * GM_ALIGN * coreNum;

    OPS_LOG_E_IF(SetTypeTiling(context, tiling) == ge::GRAPH_FAILED, context, return ge::GRAPH_FAILED,
                "SetTypeTiling Failed.");

    context->SetBlockDim(coreNum);

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
  
    const gert::Shape* permuteShape = context->GetInputShape(optiling::PERMUTE_INDEX);
    const gert::Shape* lengthsShape = context->GetInputShape(optiling::LENGTH_INDEX);
    const gert::Shape* valuesShape = context->GetInputShape(optiling::VALUES_INDEX);

    gert::Shape* outPermutedLengths = context->GetOutputShape(0);
    gert::Shape* outPermutedValues = context->GetOutputShape(1);

    OPS_LOG_E_IF_NULL("permuteShape", permuteShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lengthsShape", lengthsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesShape", valuesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outPermutedLengths", outPermutedLengths, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outPermutedValues", outPermutedValues, return ge::GRAPH_FAILED);

    int dimSize = 2;
    outPermutedLengths->SetDimNum(dimSize);
    outPermutedLengths->SetDim(0, lengthsShape->GetDim(0));
    outPermutedLengths->SetDim(1, lengthsShape->GetDim(1));

    outPermutedValues->SetDimNum(1);
    outPermutedValues->SetDim(0, valuesShape->GetDim(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Permute2dSparseData : public OpDef {
public:
    explicit Permute2dSparseData(const char* name) : OpDef(name)
    {
        this->Input("permute")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("permuted_lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("permuted_values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("permuted_weights")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("permuted_sum").Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Permute2dSparseData);
}  // namespace ops

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
#include <algorithm>
#include <cmath>
#include <limits>
#include "sgd_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t MAX_DIM_SIZE = 1024;
constexpr int RESERVE_UB_SIZE = 20 * 1024;

enum class WeightDecay : uint64_t {
    DISABLE = 0,
    ENABLE = 1
};

template <typename T>
static ge::graphStatus CheckNullPointer(T* pointer, const char* errorMessage)
{
    if (pointer == nullptr) {
        printf("%s nullptr\n", errorMessage);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingShapeInfo(const gert::TilingContext* context, SgdTilingData &tilingData)
{
    const gert::StorageShape* gradShape = context->GetInputShape(0);
    if (CheckNullPointer(gradShape, "gradShape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* inputVarShape = context->GetInputShape(2);
    if (CheckNullPointer(inputVarShape, "inputVarShape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingData.set_batchSize(gradShape->GetStorageShape().GetDim(0));
    tilingData.set_dimSize(gradShape->GetStorageShape().GetDim(1));
    tilingData.set_tableSize(inputVarShape->GetStorageShape().GetDim(0));

    if (tilingData.get_dimSize() > MAX_DIM_SIZE) {
        printf("dimSize %d must meet range[1, 1024]\n", tilingData.get_dimSize());
        return ge::GRAPH_FAILED;
    }

    auto tableDimSize = inputVarShape->GetStorageShape().GetDim(1);
    if (tilingData.get_dimSize() != tableDimSize) {
        printf("grad dim must equal table dim\n");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingAttr(gert::TilingContext* context, SgdTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    if (CheckNullPointer(attrs, "GetAttrs attrs") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto isClose = [](float value, float other) -> bool {
        return std::fabs(value - other) <= std::numeric_limits<float>::epsilon();
    };

    float weightDecay = *attrs->GetAttrPointer<float>(0);
    if (isClose(weightDecay, float(0.0f))) {
        context->SetTilingKey(static_cast<uint64_t>(WeightDecay::DISABLE));
    } else {
        context->SetTilingKey(static_cast<uint64_t>(WeightDecay::ENABLE));
    }
    tilingData.set_weightDecay(weightDecay);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCore(gert::TilingContext* context, SgdTilingData &tilingData)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivCoreNum = platformInfo.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        printf("coreNum is zero\n");
        return ge::GRAPH_FAILED;
    }

    uint64_t ub;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (ub <= RESERVE_UB_SIZE) {
        printf("ubSize is not enough\n");
        return ge::GRAPH_FAILED;
    }
    tilingData.set_ubFreeSize(ub - RESERVE_UB_SIZE);

    auto bs = tilingData.get_batchSize();
    uint32_t actualCoreNum = (bs >= aivCoreNum) ? aivCoreNum : bs;
    uint32_t splitNextCoreProcBs = bs / actualCoreNum;
    uint32_t splitPrevCoreProcBs = splitNextCoreProcBs + 1;
    uint32_t splitCoreIndex = bs % actualCoreNum;

    tilingData.set_actualCoreNum(actualCoreNum);
    tilingData.set_splitNextCoreProcBs(splitNextCoreProcBs);
    tilingData.set_splitPrevCoreProcBs(splitPrevCoreProcBs);
    tilingData.set_splitCoreIndex(splitCoreIndex);

    context->SetBlockDim(actualCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (CheckNullPointer(context, "context") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SgdTilingData tilingData;
    if (TilingShapeInfo(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingAttr(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingCore(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (optiling::CheckNullPointer(context, "context") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputVarShape = context->GetOutputShape(0);
    if (optiling::CheckNullPointer(outputVarShape, "outputVarShape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputVarShape = context->GetInputShape(2);
    if (optiling::CheckNullPointer(inputVarShape, "inputVarShape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    *outputVarShape = *inputVarShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType([[maybe_unused]] gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

}


namespace ops {
class Sgd : public OpDef {
public:
    explicit Sgd(const char* name) : OpDef(name)
    {
        this->Input("gradient")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("inputVar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("learningRate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("inputVar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("weightDecay").AttrType(OPTIONAL).Float(0.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Sgd);
}
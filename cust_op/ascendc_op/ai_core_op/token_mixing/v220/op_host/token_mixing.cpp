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

#include "token_mixing_tiling.h"

#include <map>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {
constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_GAMMA_INDEX = 2;
constexpr int32_t INPUT_BETA_INDEX = 3;
constexpr uint32_t DIM0 = 0;
constexpr uint32_t DIM1 = 1;
constexpr uint32_t DIM2 = 2;
constexpr uint32_t DIM3 = 3;
constexpr int32_t EPSILON_INDEX = 0;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int32_t RESERVERD_UB_SIZE = 20 * 1024;  // UB保留空间

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(INPUT_X_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gammaShape", context->GetInputShape(INPUT_GAMMA_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("betaShape", context->GetInputShape(INPUT_BETA_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("x", context->GetInputTensor(INPUT_X_INDEX), return ge::GRAPH_FAILED);

    auto inputTensor = context->GetInputTensor(0);
    OPS_LOG_E_IF_NULL("inputTensor", inputTensor, return ge::GRAPH_FAILED);
    ge::DataType inputDataType = inputTensor->GetDataType();
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    OPS_LOG_E_IF(xShape.GetDimNum() != 3, context, return ge::GRAPH_FAILED,
                 "[ERROR]TokenMixing required the dim of input_0 is 3");
    auto gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX)->GetStorageShape();
    OPS_LOG_E_IF(gammaShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]TokenMixing required the dim of input_1 is 1");
    auto betaShape = context->GetInputShape(INPUT_BETA_INDEX)->GetStorageShape();
    OPS_LOG_E_IF(betaShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]TokenMixing required the dim of input_2 is 1");
    const std::map<ge::DataType, int32_t> dtype2TilingKey = {{ge::DT_FLOAT, 0}};
    if (dtype2TilingKey.find(inputDataType) == dtype2TilingKey.end()) {
        OPS_LOG_E("[ERROR]Invalid data type. TokenMixing only support float", NULL);
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(dtype2TilingKey.at(inputDataType));

    uint32_t alignment = BLOCK_SIZE / sizeof(float);
    uint32_t xDim2WithPadding = (xShape[DIM2] + alignment - 1) / alignment * alignment;

    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    const float* epsilon = context->GetAttrs()->GetAttrPointer<float>(EPSILON_INDEX);
    OPS_LOG_E_IF_NULL("epsilon", epsilon, return ge::GRAPH_FAILED);

    // Platform configuration
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_LOG_E_IF(coreNum == 0, context, return ge::GRAPH_FAILED, "[ERROR] TokenMixing coreNum is invaild");
    // ub
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed -= RESERVERD_UB_SIZE;

    // 计算每行结果需要的字节数, x(H) + transpose(H) + add(H) + gamma(H) + beta(H) + tmp(H) + mean(1) + rstd(1) + one(1) + output(H)
    auto perRowMemory = (7 * xDim2WithPadding + 3) * sizeof(float);
    // 每个core每次可以处理的Row
    auto perCoreComputeRows = ubCanUsed / perRowMemory;
    // ub剩余的空间不足以执行一次循环
    OPS_LOG_E_IF(perCoreComputeRows == 0, context, return ge::GRAPH_FAILED,
                 "[ERROR]TokenMixing insufficient space in ub.");

    // 前n-1个core分配的行数
    uint64_t formerCoreRows = (xShape[DIM0] * xShape[DIM1] + coreNum - 1) / coreNum;
    uint64_t formerLoopCount = formerCoreRows / perCoreComputeRows;
    uint64_t formerRemainRows = formerCoreRows - perCoreComputeRows * formerLoopCount;
    // 重新调整最大核数
    if (xShape[DIM0] * xShape[DIM1] < formerCoreRows * (coreNum - 1)) {
        coreNum = (xShape[DIM0] * xShape[DIM1] + formerCoreRows - 1) / formerCoreRows;
    }
    // 最后一个核分配的数据
    uint64_t tailCoreRows = xShape[DIM0] * xShape[DIM1] - formerCoreRows * (coreNum - 1);
    uint64_t tailLoopCount = tailCoreRows / perCoreComputeRows;
    uint64_t tailRemainRows = tailCoreRows - perCoreComputeRows * tailLoopCount;

    // 设置分片数据
    TokenMixingTilingData tilingData;
    tilingData.set_xDim0(xShape[DIM0]);
    tilingData.set_xDim1(xShape[DIM1]);
    tilingData.set_xDim2(xShape[DIM2]);
    tilingData.set_xDim2WithPadding(xDim2WithPadding);
    tilingData.set_epsilon(*epsilon);
    tilingData.set_coreNum(coreNum);
    tilingData.set_perCoreComputeRows(perCoreComputeRows);
    tilingData.set_formerCoreRows(formerCoreRows);
    tilingData.set_formerLoopCount(formerLoopCount);
    tilingData.set_formerRemainRows(formerRemainRows);
    tilingData.set_tailCoreRows(tailCoreRows);
    tilingData.set_tailLoopCount(tailLoopCount);
    tilingData.set_tailRemainRows(tailRemainRows);
    context->SetBlockDim(coreNum);
    // 保存分片数据
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
using optiling::DIM0;
using optiling::DIM1;
using optiling::DIM2;
using optiling::DIM3;
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* meanShape = context->GetOutputShape(1);
    gert::Shape* rstdShape = context->GetOutputShape(2);

    OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("yShape", yShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("meanShape", meanShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("rstdShape", rstdShape, return ge::GRAPH_FAILED);

    yShape->SetDimNum(DIM3);
    yShape->SetDim(DIM0, xShape->GetDim(DIM0));
    yShape->SetDim(DIM1, xShape->GetDim(DIM2));
    yShape->SetDim(DIM2, xShape->GetDim(DIM1));
    meanShape->SetDimNum(DIM2);
    meanShape->SetDim(DIM0, xShape->GetDim(DIM0));
    meanShape->SetDim(DIM1, xShape->GetDim(DIM1));
    rstdShape->SetDimNum(DIM2);
    rstdShape->SetDim(DIM0, xShape->GetDim(DIM0));
    rstdShape->SetDim(DIM1, xShape->GetDim(DIM1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    context->SetOutputDataType(2, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class TokenMixing : public OpDef {
public:
    explicit TokenMixing(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x_t")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").Float();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(TokenMixing);
}  // namespace ops

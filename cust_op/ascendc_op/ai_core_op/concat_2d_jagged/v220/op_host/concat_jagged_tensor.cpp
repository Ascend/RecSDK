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

#include "concat_jagged_tensor_tiling.h"
#include "register/op_def_registry.h"
#include "register/register.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

constexpr uint32_t RESERVED_UB_SIZE = 1024;
constexpr uint32_t WORKSPACE_SIZE = 4 * 1024 * 1024;
constexpr uint32_t ALIGN = 32;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取硬件信息
    uint64_t ubSize = 0;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t maxUbSize = ubSize - RESERVED_UB_SIZE;
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[ERROR] need more than 0 ai core");
        return ge::GRAPH_FAILED;
    }
    // 计算偏移地址和大小
    auto offsets = context->GetAttrs()->GetListInt(0);
    int32_t offsetLen = *context->GetAttrs()->GetInt(1);
    int32_t jtNum =  *context->GetAttrs()->GetInt(2);

    uint32_t sliceSize[MAX_SLICE_SIZE] = {0};
    uint32_t inputOffsetBegin[MAX_SLICE_SIZE] = {0};
    uint32_t outputOffsetBegin[MAX_SLICE_SIZE] = {0};

    int64_t sliceIndex = 0;

    for (int i = 0; i < offsetLen - 1; i++) {
        for (int j = 0; j < jtNum; j++) {
            int64_t index = j * offsetLen + i;
            int64_t offset = offsets->GetData()[index];
            inputOffsetBegin[sliceIndex] = offset;
            sliceSize[sliceIndex] = offsets->GetData()[index + 1] - offsets->GetData()[index];
            sliceIndex++;
        }
    }

    int64_t outputOffset = 0;
    for (int i = 0; i < jtNum * (offsetLen - 1); i++) {
        outputOffsetBegin[i] = outputOffset;
        outputOffset += sliceSize[i];
    }
    // 计算切分逻辑
    uint64_t batchSize = jtNum * (offsetLen - 1);
    uint64_t formerCore = 0;
    uint64_t tailCore = 0;
    uint64_t batchNumInTail = 0;
    uint64_t batchNumInFormer = 0;
    if (batchSize < coreNum) {
        formerCore = batchSize;
        batchNumInFormer = 1;
    } else {
        formerCore = batchSize % coreNum;
        tailCore = coreNum - formerCore;
        batchNumInTail = batchSize / coreNum;
        batchNumInFormer = batchNumInTail + 1;
    }

    // 计算copy最大行数
    OPS_CHECK_PTR_NULL(context->GetRequiredInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputXDesc", context->GetInputDesc(0), return ge::GRAPH_FAILED);
    uint32_t inputColSize = context->GetRequiredInputShape(0)->GetOriginShape().GetDim(1);
    auto inputXDesc = context->GetInputDesc(0);
    ge::DataType xDtype{ge::DT_FLOAT};
    xDtype = inputXDesc->GetDataType();
    uint32_t ubMaxLength = maxUbSize / ge::GetSizeByDataType(xDtype);
    ubMaxLength = ubMaxLength & ~(ALIGN - 1);

    // 设置tiling
    ConcatJaggedTensorTilingData tiling;
    tiling.set_jtNum(jtNum);
    tiling.set_inputColSize(inputColSize);
    tiling.set_ubMaxLength(ubMaxLength);
    tiling.set_formerCore(formerCore);
    tiling.set_tailCore(tailCore);
    tiling.set_batchNumInTail(batchNumInTail);
    tiling.set_batchNumInFormer(batchNumInFormer);

    tiling.set_inputOffsetBegin(inputOffsetBegin);
    tiling.set_sliceSize(sliceSize);
    tiling.set_outputOffsetBegin(outputOffsetBegin);

    context->SetBlockDim(coreNum);
    OPS_CHECK_PTR_NULL(context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto offsets = context->GetAttrs()->GetListInt(0);
    int32_t offsetLen = *context->GetAttrs()->GetInt(1);
    int32_t jtNum = *context->GetAttrs()->GetInt(2);
    int64_t outputRow = 0;
    for (int i = 0; i < jtNum; i++) {
        outputRow += offsets->GetData()[i * offsetLen];
    }
    const gert::Shape *x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        OPS_LOG_E("[ERROR]", "InputShape should not be nullptr.");
        return ge::GRAPH_FAILED;
        }
    uint32_t dimNum = x_shape->GetDimNum();
    gert::Shape* y_shape  = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        OPS_LOG_E("[ERROR]", "OutputShape should not be nullptr.");
        return ge::GRAPH_FAILED;
    }
    y_shape->SetDimNum(dimNum);
    y_shape->SetDim(0, outputRow);
    y_shape->SetDim(1, x_shape->GetDim(1));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ConcatJaggedTensor : public OpDef {
public:
    explicit ConcatJaggedTensor(const char* name) : OpDef(name)
    {
        this->Input("values")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("offsets").ListInt();
        this->Attr("offsetLen").Int(0);
        this->Attr("jtNum").Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
    }
};
OP_ADD(ConcatJaggedTensor);
}
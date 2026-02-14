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
#include "multislice_concat.h"

#include <limits>
#include "register/op_def_registry.h"
#include "register/register.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "ops_log.h"

constexpr uint32_t PING_PANG_NUM = 2;
constexpr uint32_t ALIGN_BYTE = 32;
constexpr uint32_t MAX_PROC_COLUMN_BYTE = 1024;
constexpr uint64_t MIN_NEED_CORE_NUM = 6;
constexpr uint32_t ALL_CORE_BATCH_SIZE = 512;

/* 除输入Tensor外，共有5个输入
 * concatNum 输出tensor数量
 * concatSize 输出tensor的concat的子tensor数，int[concatNum]
 * sliceBegin 输出tensor的每个子tensor在输入tensor的偏移量，int[concatSize[0]..., concatSize[1]..., ...]
 * sliceLength 输出tensor的每个子tensor的长度，int[concatSize[0]..., concatSize[1]..., ...]
 * maxConcatSize 预留参数
 */
constexpr size_t CONCAT_NUM_INDEX = 0;
constexpr size_t CONCAT_SIZE_INDEX = 1;
constexpr size_t SLICE_BEGIN_INDEX = 2;
constexpr size_t SLICE_LENGTH_INDEX = 3;

namespace optiling {
static ge::graphStatus CalcTilingParams(MultisliceConcatTilingData& tiling, uint64_t maxColumnSizeActual,
                                        uint64_t ubSize, uint64_t batchSize, int32_t dTypeSize, uint64_t& coreNum)
{
    uint16_t maxProcColumnNum = 0;
    coreNum = std::min(coreNum, batchSize);
    OPS_LOG_E_IF((coreNum == 0), "Tiling error", return ge::GRAPH_FAILED, "coreNum[%lu] must be != 0", coreNum);

    uint64_t alignColumnNum = ALIGN_BYTE / dTypeSize;
    if ((maxColumnSizeActual * dTypeSize) > MAX_PROC_COLUMN_BYTE) {
        maxProcColumnNum = MAX_PROC_COLUMN_BYTE / dTypeSize;
    } else {
        maxProcColumnNum = (maxColumnSizeActual + alignColumnNum) / alignColumnNum * alignColumnNum;
    }

    uint64_t maxProcBatchSizePerCore = ubSize / (PING_PANG_NUM * maxProcColumnNum * dTypeSize);
    uint64_t needCoreNum = batchSize / maxProcBatchSizePerCore + 1;
    needCoreNum = std::max(MIN_NEED_CORE_NUM, needCoreNum);
    needCoreNum = std::max(batchSize, needCoreNum);
    if ((batchSize > ALL_CORE_BATCH_SIZE) && (needCoreNum < coreNum)) {
        needCoreNum = coreNum;
    }

    while (coreNum < needCoreNum) {
        maxProcColumnNum -= alignColumnNum;
        OPS_LOG_E_IF((maxProcColumnNum <= 0), "Tiling error", return ge::GRAPH_FAILED,
                     "batchSize[%lu] is too large to compute", batchSize);
        maxProcBatchSizePerCore = ubSize / (PING_PANG_NUM * maxProcColumnNum * dTypeSize);
        needCoreNum = batchSize / maxProcBatchSizePerCore + 1;
    }
    coreNum = needCoreNum;
    uint64_t formerCore = batchSize % coreNum;
    uint64_t tailCore = coreNum - formerCore;
    uint16_t batchNumInTail = batchSize / coreNum;
    uint16_t batchNumInFormer = batchNumInTail + 1;
    tiling.set_formerCore(formerCore);
    tiling.set_tailCore(tailCore);
    tiling.set_batchNumInTail(batchNumInTail);
    tiling.set_batchNumInFormer(batchNumInFormer);
    tiling.set_maxProColumnNum(maxProcColumnNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto inputShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("inputDesc", context->GetInputDesc(0), return ge::GRAPH_FAILED);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t batchSize = inputShape->GetStorageShape().GetDim(0);
    uint64_t inputCol = inputShape->GetStorageShape().GetDim(1);

    auto inputAttrs = context->GetAttrs();
    auto concatNumPtr = inputAttrs->GetInt(CONCAT_NUM_INDEX);
    OPS_LOG_E_IF_NULL("concatNum", concatNumPtr, return ge::GRAPH_FAILED);
    auto concatNum = *concatNumPtr;
    OPS_LOG_E_IF_NULL("inputAttrs", inputAttrs, return ge::GRAPH_FAILED);
    auto concatSizeAttr = inputAttrs->GetListInt(CONCAT_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("concatSize", concatSizeAttr, return ge::GRAPH_FAILED);
    auto sliceBeginAttr = inputAttrs->GetListInt(SLICE_BEGIN_INDEX);
    OPS_LOG_E_IF_NULL("sliceBegin", sliceBeginAttr, return ge::GRAPH_FAILED);
    auto sliceLengthAttr = inputAttrs->GetListInt(SLICE_LENGTH_INDEX);
    OPS_LOG_E_IF_NULL("sliceLength", sliceLengthAttr, return ge::GRAPH_FAILED);

    std::array<uint16_t, MAX_CONCAT_TENSOR_NUM> concatSize{};
    std::array<uint16_t, MAX_SLICE_NUM> sliceBegin{};
    std::array<uint16_t, MAX_SLICE_NUM> sliceLength{};
    int32_t sliceOffset = 0;
    uint64_t maxColumnSize = 0;
    for (int32_t i = 0; i < concatNum; i++) {
        concatSize[i] = concatSizeAttr->GetData()[i];
        for (int32_t j = 0; j < concatSize[i]; j++) {
            uint64_t offset = sliceOffset + j;
            sliceBegin[offset] = sliceBeginAttr->GetData()[offset];
            sliceLength[offset] = sliceLengthAttr->GetData()[offset];
            if (sliceLength[offset] == -1) {
                sliceLength[offset] = inputCol - sliceBegin[offset];
            }

            if (sliceLength[offset] > maxColumnSize) {
                maxColumnSize = sliceLength[offset];
            }
        }
        sliceOffset += concatSize[i];
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    MultisliceConcatTilingData tiling;
    auto dType = context->GetInputDesc(0)->GetDataType();
    int32_t dTypeSize = ge::GetSizeByDataType(dType);
    auto ret = CalcTilingParams(tiling, maxColumnSize, ubSize, batchSize, dTypeSize, coreNum);
    OPS_LOG_E_IF((ret != ge::GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "calc tiling params failed");

    tiling.set_colSize(inputCol);
    tiling.set_concatNum(concatNum);
    tiling.set_concatSize(concatSize.data());
    tiling.set_sliceBegin(sliceBegin.data());
    tiling.set_sliceLength(sliceLength.data());
    context->SetBlockDim(coreNum);

    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    workspaceSize[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto inputShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("inputShape", inputShape, return ge::GRAPH_FAILED);
    uint64_t inputRow = inputShape->GetDim(0);
    uint64_t inputCol = inputShape->GetDim(1);

    auto inputAttrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("inputAttrs", inputAttrs, return ge::GRAPH_FAILED);
    auto concatSize = inputAttrs->GetListInt(CONCAT_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("concatSize", concatSize, return ge::GRAPH_FAILED);
    auto sliceBegin = inputAttrs->GetListInt(SLICE_BEGIN_INDEX);
    OPS_LOG_E_IF_NULL("sliceBegin", sliceBegin, return ge::GRAPH_FAILED);
    auto sliceLength = inputAttrs->GetListInt(SLICE_LENGTH_INDEX);
    OPS_LOG_E_IF_NULL("sliceLength", sliceLength, return ge::GRAPH_FAILED);
    auto concatNumPtr = inputAttrs->GetInt(CONCAT_NUM_INDEX);
    OPS_LOG_E_IF_NULL("concatNum", concatNumPtr, return ge::GRAPH_FAILED);
    auto concatNum = *concatNumPtr;

    int32_t sliceOffset = 0;
    for (int32_t i = 0; i < concatNum; i++) {
        int32_t outConcatSize = concatSize->GetData()[i];
        int32_t outColSize = 0;
        for (int32_t concatIndex = 0; concatIndex < outConcatSize; concatIndex++) {
            int32_t offset = sliceOffset + concatIndex;
            int32_t curSliceBegin = sliceBegin->GetData()[offset];
            int32_t curSliceLength = sliceLength->GetData()[offset];
            if (curSliceLength == -1) {
                curSliceLength = inputCol - curSliceBegin;
            }
            outColSize += curSliceLength;
        }
        sliceOffset += outConcatSize;
        gert::Shape* outShape = context->GetOutputShape(i);
        OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);
        *outShape = *inputShape;
        outShape->SetDimNum(2);
        outShape->SetDim(0, inputRow);
        outShape->SetDim(1, outColSize);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputAttrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDataType(0);
    auto concatNum = *context->GetAttrs()->GetInt(CONCAT_NUM_INDEX);
    for (int32_t i = 0; i < concatNum; i++) {
        context->SetOutputDataType(i, dtype);
    }
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MultisliceConcat : public OpDef {
public:
    explicit MultisliceConcat(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("concat_num").Int();
        this->Attr("concat_size").ListInt();
        this->Attr("slice_begin").ListInt();
        this->Attr("slice_length").ListInt();

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MultisliceConcat);
}  // namespace ops
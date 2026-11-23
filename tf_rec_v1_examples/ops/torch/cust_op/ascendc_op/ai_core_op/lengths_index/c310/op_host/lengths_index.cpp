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

#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "lengths_index_tiling.h"

namespace {
    constexpr int NUM_QUEUE = 4;
    constexpr int UB_ALIGN = 32;
    constexpr int32_t MAX_THREADS_PER_BLOCK = 512;
    constexpr int OFFSETS_INDEX = 0;
    constexpr int OUTPUTSIZE_INDEX = 0;
    constexpr int NUMSEQ_INDEX = 1;
    constexpr int OUTPUT_INDEX = 0;
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // check input
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsTensor", context->GetInputTensor(OFFSETS_INDEX), return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // check attrs
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* outputSizePtr = attrs->GetAttrPointer<int64_t>(OUTPUTSIZE_INDEX);
    OPS_LOG_E_IF_NULL("output_size attr", outputSizePtr, return ge::GRAPH_FAILED);
    int64_t outputSize = *outputSizePtr;
    const int64_t* numSeqPtr = attrs->GetAttrPointer<int64_t>(NUMSEQ_INDEX);
    OPS_LOG_E_IF_NULL("num_seq attr", numSeqPtr, return ge::GRAPH_FAILED);
    int64_t numSeq = *numSeqPtr;

    // check dataType
    auto offsetsTensor = context->GetInputTensor(OFFSETS_INDEX);
    ge::DataType offsetsDataType = offsetsTensor->GetDataType();
    OPS_CHECK(offsetsDataType != ge::DT_INT32 && offsetsDataType != ge::DT_INT64,
              OPS_LOG_E("[ERROR]Invalid data type",
                        "LengthsIndex only support int64 and int32."),
              return ge::GRAPH_FAILED);

    // check dimension: offsets must be 1D
    uint32_t dimNum = context->GetInputShape(OFFSETS_INDEX)->GetOriginShape().GetDimNum();
    OPS_LOG_E_IF(dimNum != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]LengthsIndex requires the dim of input-0 is 1");

    // set coreNum
    size_t coreNum = ascendPlatform.GetCoreNumAiv();

    // set ub
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed / UB_ALIGN / NUM_QUEUE * UB_ALIGN * NUM_QUEUE;

    // apply workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspacesSize;

    int64_t offsetsSize = context->GetInputShape(OFFSETS_INDEX)->GetOriginShape().GetShapeSize();
    OPS_LOG_E_IF(numSeq < 0, context, return ge::GRAPH_FAILED,
                 "[ERROR] num_seq must be >= 0");
    OPS_LOG_E_IF(outputSize < 0, context, return ge::GRAPH_FAILED,
                 "[ERROR] output_size must be >= 0");
    OPS_LOG_E_IF(offsetsSize != numSeq && offsetsSize != numSeq + 1, context, return ge::GRAPH_FAILED,
                 "[ERROR] offsets size should equal num_seq(exclusive) or num_seq + 1(complete)");

    // Calculate vector size based on output_size / num_seq ratio
    int64_t avgLen = (numSeq > 0) ? (outputSize / numSeq) : 0;
    uint32_t vectorSize;

    if (avgLen < 2) {
        vectorSize = 2;
    } else if (avgLen < 4) {
        vectorSize = 4;
    } else if (avgLen < 64) {
        vectorSize = 8;
    } else if (avgLen < 128) {
        vectorSize = 16;
    } else {
        vectorSize = 32;
    }

    uint32_t rowsPerBlock = MAX_THREADS_PER_BLOCK / vectorSize;
    int64_t totalBlocks = (numSeq + rowsPerBlock - 1) / rowsPerBlock;
    size_t actualCoreNum = totalBlocks > coreNum ? coreNum : totalBlocks;
    int64_t blocksPerCore = 0;
    int32_t remainderBlocks = 0;
    if (actualCoreNum > 0) {
        blocksPerCore = totalBlocks / actualCoreNum;
        remainderBlocks = static_cast<int32_t>(totalBlocks % actualCoreNum);
    }

    LengthsIndexTilingData tiling;
    tiling.set_numSeq(numSeq);
    tiling.set_outputSize(outputSize);
    tiling.set_totalBlocks(totalBlocks);
    tiling.set_blocksPerCore(blocksPerCore);
    tiling.set_remainderBlocks(remainderBlocks);
    tiling.set_vectorSize(vectorSize);
    tiling.set_rowsPerBlock(rowsPerBlock);
    tiling.set_ubCanUsed(ubCanUsed);

    context->SetBlockDim(actualCoreNum > 0 ? actualCoreNum : 1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* offsetsShape = context->GetInputShape(OFFSETS_INDEX);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    // Get output_size from attributes
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* outputSizePtr = attrs->GetAttrPointer<int64_t>(OUTPUTSIZE_INDEX);
    OPS_LOG_E_IF_NULL("output_size attr", outputSizePtr, return ge::GRAPH_FAILED);
    int64_t outputSize = *outputSizePtr;

    outputShape->SetDimNum(1);
    outputShape->SetDim(0, outputSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(OFFSETS_INDEX);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(OUTPUT_INDEX, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LengthsIndex : public OpDef {
public:
    explicit LengthsIndex(const char* name) : OpDef(name)
    {
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("output_size").Int();
        this->Attr("num_seq").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(LengthsIndex);
}  // namespace ops
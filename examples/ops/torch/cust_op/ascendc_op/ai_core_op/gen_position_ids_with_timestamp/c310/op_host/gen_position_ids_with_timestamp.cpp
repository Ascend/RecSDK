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
#include "gen_position_ids_with_timestamp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
constexpr int32_t SEQLEN_INDEX = 0;
constexpr int32_t SEQLEN_OFFSETS_INDEX = 1;
constexpr int32_t TIMESTAMPS_INDEX = 2;
constexpr int32_t OUTPUT_INDEX = 0;
constexpr int32_t ATTR_TIME_SCALE = 0;
constexpr int32_t DIM0 = 0;
}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("seqlenShape", context->GetInputShape(SEQLEN_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("seqlenTensor", context->GetInputTensor(SEQLEN_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("seqlenOffsetsShape", context->GetInputShape(SEQLEN_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("seqlenOffsetsTensor", context->GetInputTensor(SEQLEN_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("timestampsShape", context->GetInputShape(TIMESTAMPS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("timestampsTensor", context->GetInputTensor(TIMESTAMPS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& seqlenShape = context->GetInputShape(SEQLEN_INDEX)->GetOriginShape();
    const gert::Shape& seqlenOffsetsShape = context->GetInputShape(SEQLEN_OFFSETS_INDEX)->GetOriginShape();
    const gert::Shape& timestampsShape = context->GetInputShape(TIMESTAMPS_INDEX)->GetOriginShape();

    const ge::DataType seqlenDtype = context->GetInputTensor(SEQLEN_INDEX)->GetDataType();
    const ge::DataType seqlenOffDtype = context->GetInputTensor(SEQLEN_OFFSETS_INDEX)->GetDataType();
    const ge::DataType timestampsDtype = context->GetInputTensor(TIMESTAMPS_INDEX)->GetDataType();
    if (seqlenDtype != ge::DT_INT32 || seqlenOffDtype != ge::DT_INT32 || timestampsDtype != ge::DT_INT32) {
        OPS_LOG_E("Tiling", "seqlen, seqlen_offsets, timestamps must be int32.");
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_E_IF(seqlenShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED, "[ERROR] seqlen must be 1D.");
    OPS_LOG_E_IF(seqlenOffsetsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR] seqlen_offsets must be 1D.");
    OPS_LOG_E_IF(timestampsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED, "[ERROR] timestamps must be 1D.");

    int64_t batchSize = seqlenShape.GetDim(DIM0);
    int64_t offsetsLen = seqlenOffsetsShape.GetDim(DIM0);
    int64_t totalSeqLen = timestampsShape.GetDim(DIM0);

    OPS_LOG_E_IF(batchSize < 1, context, return ge::GRAPH_FAILED, "[ERROR] batchSize must be >= 1, got %lld.",
                 (long long)batchSize);
    OPS_LOG_E_IF(offsetsLen != batchSize + 1, context, return ge::GRAPH_FAILED,
                 "[ERROR] seqlen_offsets length must be batchSize+1, expected %lld, got %lld.",
                 (long long)(batchSize + 1), (long long)offsetsLen);
    OPS_LOG_E_IF(totalSeqLen < 1, context, return ge::GRAPH_FAILED, "[ERROR] totalSeqLen must be >= 1, got %lld.",
                 (long long)totalSeqLen);

    const auto* attrs = context->GetAttrs();
    const float* timeScalePtr = attrs->GetAttrPointer<float>(ATTR_TIME_SCALE);
    OPS_LOG_E_IF_NULL("timeScale attr", timeScalePtr, return ge::GRAPH_FAILED);
    float timeScale = *timeScalePtr;
    OPS_LOG_E_IF(timeScale <= 0.0f, context, return ge::GRAPH_FAILED,
                 "[ERROR] timeScale must be positive, got %f. Default when omitted in graph is 300.0.", timeScale);
    const float invTimeScale = 1.0f / timeScale;

    auto ascendCPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const size_t maxCores = ascendCPlatform.GetCoreNumAiv();
    OPS_LOG_E_IF(maxCores == 0, context, return ge::GRAPH_FAILED, "[ERROR] aiv core num is zero.");

    // 与 CUDA grid.x 对齐：每个 sample 一个 logical block；核数取 min(batch, maxCores)，负载由 blocksPerCore/remainder
    // 均分
    const int64_t totalBlocks = batchSize;
    size_t actualCoreNum = totalBlocks > static_cast<int64_t>(maxCores) ? maxCores : static_cast<size_t>(totalBlocks);
    int64_t blocksPerCore = 0;
    int32_t remainderBlocks = 0;
    if (actualCoreNum > 0) {
        blocksPerCore = totalBlocks / static_cast<int64_t>(actualCoreNum);
        remainderBlocks = static_cast<int32_t>(totalBlocks % static_cast<int64_t>(actualCoreNum));
    }

    // 1U：无符号 1，与 GetWorkspaceSizes 期望的「workspace 个数」类型一致（避免有符号/无符号混用告警）。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1U);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    const size_t systemWorkspacesSize = ascendCPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    GenPositionIdsWithTimestampTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_invTimeScale(invTimeScale);
    tiling.set_blocksPerCore(blocksPerCore);
    tiling.set_remainderBlocks(remainderBlocks);

    context->SetBlockDim(actualCoreNum > 0 ? static_cast<uint32_t>(actualCoreNum) : 1U);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
// 输出 position_ids 与 timestamps 同秩、逐维相同（一维时长度均为 total_seq_len）。
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* timestampsShape = context->GetInputShape(TIMESTAMPS_INDEX);
    OPS_LOG_E_IF_NULL("timestampsShape", timestampsShape, return ge::GRAPH_FAILED);

    gert::Shape* positionIdsShape = context->GetOutputShape(OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL("positionIdsShape", positionIdsShape, return ge::GRAPH_FAILED);

    const int32_t dimNum = timestampsShape->GetDimNum();
    if (dimNum < 0) {
        return ge::GRAPH_FAILED;
    }
    positionIdsShape->SetDimNum(dimNum);
    for (int32_t dimIdx = 0; dimIdx < dimNum; ++dimIdx) {
        positionIdsShape->SetDim(dimIdx, timestampsShape->GetDim(dimIdx));
    }
    return ge::GRAPH_SUCCESS;
}

// 算子语义固定：position_ids 为 int32（与 OpDef / 适配层 / CUDA 参考一致），不随其它输入 dtype 推导。
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(OUTPUT_INDEX, ge::DT_INT32)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class GenPositionIdsWithTimestamp : public OpDef {
public:
    explicit GenPositionIdsWithTimestamp(const char* name) : OpDef(name)
    {
        this->Input("seqlen")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("seqlen_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("timestamps")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("position_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("timeScale").AttrType(OPTIONAL).Float(300.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(GenPositionIdsWithTimestamp);
}  // namespace ops
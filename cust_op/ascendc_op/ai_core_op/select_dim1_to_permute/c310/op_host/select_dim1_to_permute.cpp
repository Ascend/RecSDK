/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "select_dim1_to_permute_tiling.h"

#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"

namespace {
constexpr int32_t INDICES_INDEX = 0;
constexpr int32_t BATCHSIZE_INDEX = 0;
constexpr int32_t LENGTHSSIZE_INDEX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int64_t USE_BUFFER_NUM = 1;
constexpr int64_t USE_QUEUE_NUM = 1;
constexpr int DCACHE_SIZE = 128 * 1024;
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SelectDim1ToPermuteTilingData tiling;
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("context->GetAttrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lengthsSize", context->GetAttrs()->GetInt(LENGTHSSIZE_INDEX), return ge::GRAPH_FAILED);

    int64_t batchSize = *context->GetAttrs()->GetInt(BATCHSIZE_INDEX);  // 稀疏矩阵一行元素个数
    OPS_LOG_E_IF(batchSize <= 0, context, return ge::GRAPH_FAILED,
                 "[ERROR]SelectDim1ToPermuteTilingData required batchSize must be a positive number");
    int64_t lengthsSize = *context->GetAttrs()->GetInt(LENGTHSSIZE_INDEX);  // 稀疏矩阵一行元素个数
    OPS_LOG_E_IF(lengthsSize <= 0, context, return ge::GRAPH_FAILED,
                 "[ERROR]SelectDim1ToPermuteTilingData required lengthsSize must be a positive number");
    int32_t batchNum = lengthsSize / batchSize;  // 稀疏矩阵一列元素个数
    tiling.set_batchSize(batchSize);
    tiling.set_lengthsSize(lengthsSize);
    int64_t indicesLength = context->GetInputShape(INDICES_INDEX)->GetOriginShape().GetShapeSize();

    auto indicesTensor = context->GetInputTensor(0);
    auto lengthsTensor = context->GetInputTensor(1);
    OPS_LOG_E_IF_NULL("indicesTensor", indicesTensor, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lengthsTensor", lengthsTensor, return ge::GRAPH_FAILED);
    int indicesBytes = indicesTensor->GetDataType() == ge::DT_INT64 ? sizeof(int64_t) : sizeof(int32_t);
    uint64_t indicesAlignment = BLOCK_SIZE / indicesBytes;
    int64_t indicesLengthWithPadding = (indicesLength + indicesAlignment - 1) / indicesAlignment * indicesAlignment;
    uint64_t permuteLengthWithPadding = static_cast<uint64_t>(batchNum) * indicesLengthWithPadding;
    OPS_LOG_E_IF(
        permuteLengthWithPadding > std::numeric_limits<int64_t>::max(), context, return ge::GRAPH_FAILED,
        "[ERROR]SelectDim1ToPermuteTilingData required permuteLengthWithPadding little than %lld, but get %llu",
        std::numeric_limits<int64_t>::max(), permuteLengthWithPadding);
    // 获取平台信息
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    workspaceSize[0] = systemWorkspacesSize;

    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[ERROR] need more than 0 ai core");
        return ge::GRAPH_FAILED;
    }
    int64_t totalTasks = batchNum;
    int64_t actualCoreNum = totalTasks < coreNum ? totalTasks : coreNum;
    int32_t splitBaseLen = totalTasks / actualCoreNum;    // 每核基础块数k,
    int32_t tailSplitIndex = totalTasks % actualCoreNum;  // 余数块数l
    // ub
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    int lengthsBytes = lengthsTensor->GetDataType() == ge::DT_INT64 ? sizeof(int64_t) : sizeof(int32_t);
    uint64_t lengthsAlignment = BLOCK_SIZE / lengthsBytes;
    // 执行时, lengths以batchSizeWithPadding为单位处理
    int64_t batchSizeWithPadding = (batchSize + lengthsAlignment - 1) / lengthsAlignment * lengthsAlignment;
    int64_t lengthsUbSize = batchSizeWithPadding * USE_QUEUE_NUM * lengthsBytes;
    ubCanUsed -= lengthsUbSize;
    int64_t blockLen = ubCanUsed / USE_QUEUE_NUM / USE_BUFFER_NUM / indicesBytes;
    blockLen = blockLen / indicesAlignment * indicesAlignment;
    blockLen = std::min(blockLen, indicesLengthWithPadding);
    OPS_LOG_E_IF(blockLen <= 0, context, return ge::GRAPH_FAILED,
                 "[ERROR]SelectDim1ToPermuteTilingData required blockLen must be a positive number");
    tiling.set_indicesLength(indicesLength);
    tiling.set_batchNum(batchNum);
    tiling.set_splitBaseLen(splitBaseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);
    tiling.set_blockLen(blockLen);
    tiling.set_batchSizeWithPadding(batchSizeWithPadding);

    context->SetBlockDim(actualCoreNum);
    context->SetLocalMemorySize(DCACHE_SIZE);

    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());

    tilingData->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto indicesDataType = context->GetInputDataType(INDICES_INDEX);
    context->SetOutputDataType(0, indicesDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SelectDim1ToPermute : public OpDef {
public:
    explicit SelectDim1ToPermute(const char* name) : OpDef(name)
    {
        this->Input("indices").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("lengths").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Output("permute").ParamType(REQUIRED).Follow("indices", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Output("outputLengths")
            .ParamType(REQUIRED)
            .Follow("lengths", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Attr("batchSize").Int(0);
        this->Attr("lengthsSize").Int(0);

        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SelectDim1ToPermute);
}  // namespace ops

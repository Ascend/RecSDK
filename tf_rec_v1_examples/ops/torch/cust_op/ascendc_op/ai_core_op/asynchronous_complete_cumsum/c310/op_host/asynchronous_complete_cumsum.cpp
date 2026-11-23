/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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
#include "asynchronous_complete_cumsum_tiling.h"

namespace {
    constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
    constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
    constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / 32;
    constexpr int DCACHE_SIZE = 128 * 1024;
    constexpr int32_t MULTIPLIER = 2;
    constexpr int32_t DIVISOR = 4;
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor", context->GetInputTensor(0), return ge::GRAPH_FAILED);

    int64_t inputLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto inputTensor = context->GetInputTensor(0);
    ge::DataType inputDataType = inputTensor->GetDataType();

    uint32_t dimNum = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    OPS_LOG_E_IF(dimNum != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]AsynchronousCompleteCumsum required the dim of input-0 is 1");

    OPS_CHECK(inputDataType != ge::DT_INT32 && inputDataType != ge::DT_INT64,
              OPS_LOG_E("[ERROR]Invalid data type",
                        "AsynchronousCompleteCumsum only support int64 and int32."),
              return ge::GRAPH_FAILED);

    // 获取可用核心数，但只使用实际需要的核心数
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t maxCores = ascendPlatform.GetCoreNumAiv();
    int32_t SMALL_DATA_THRESHOLD_32 = maxCores * MAX_THREADS_PER_BLOCK / DIVISOR;
    int32_t SMALL_DATA_THRESHOLD_64 = maxCores * MAX_THREADS_PER_BLOCK * MULTIPLIER + SMALL_DATA_THRESHOLD_32;

    int32_t elementsPerBlock = MAX_THREADS_PER_BLOCK * MAX_ELEMENTS_PER_THREAD;
    bool isSmall = false;
    if (inputLength <= (inputDataType == ge::DT_INT32 ? SMALL_DATA_THRESHOLD_32 : SMALL_DATA_THRESHOLD_64)) {
        isSmall = true;
        elementsPerBlock = MAX_THREADS_PER_BLOCK;
    }
    int64_t totalBlocks = (inputLength + elementsPerBlock - 1) / elementsPerBlock;

    bool isFullCore = (totalBlocks > maxCores);
    size_t coreNum = isFullCore ? maxCores : totalBlocks;
    if (coreNum == 0) {
        OPS_LOG_E(context, "[ERROR] need more than 0 ai core");
        return ge::GRAPH_FAILED;
    }
    int64_t blocksPerCore = totalBlocks / coreNum;                      // 每核基础块数k
    int32_t remainderBlocks = totalBlocks % coreNum;                    // 余数块数l

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();

    size_t blockSumsSize = totalBlocks * sizeof(int64_t);
    size_t userWorkspaceSize = blockSumsSize;
    workspaceSize[0] = systemWorkspacesSize + userWorkspaceSize;

    AsynchronousCompleteCumsumTilingData tiling;
    tiling.set_totalLength(inputLength);
    tiling.set_totalBlocks(totalBlocks);
    tiling.set_blocksPerCore(blocksPerCore);
    tiling.set_remainderBlocks(remainderBlocks);
    tiling.set_elementsPerBlock(elementsPerBlock);
    tiling.set_isSmall(isSmall);
    tiling.set_isFullCore(isFullCore);

    context->SetBlockDim(coreNum);
    context->SetLocalMemorySize(DCACHE_SIZE);
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("yShape", yShape, return ge::GRAPH_FAILED);

    int64_t inputLength = xShape->GetDim(0);
    int64_t outputDim = 0;
    if (inputLength == -1) {
        outputDim = -1;
    } else {
        OPS_CHECK(inputLength <= 0 || inputLength >= std::numeric_limits<int64_t>::max(),
                  OPS_LOG_E("[ERROR]", "inputLength limit (0, %lld), but get %lld\n",
                            std::numeric_limits<int64_t>::max(), inputLength),
                  return ge::GRAPH_FAILED);
        outputDim = inputLength + 1;
    }
    yShape->SetDimNum(1);
    yShape->SetDim(0, outputDim);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(0);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AsynchronousCompleteCumsum : public OpDef {
public:
    explicit AsynchronousCompleteCumsum(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(AsynchronousCompleteCumsum);
}

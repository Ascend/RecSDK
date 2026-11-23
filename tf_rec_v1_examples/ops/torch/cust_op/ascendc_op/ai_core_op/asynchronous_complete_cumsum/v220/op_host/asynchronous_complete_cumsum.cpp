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

#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "asynchronous_complete_cumsum_tiling.h"

namespace {
    constexpr int32_t BLOCK_SIZE = 256;                    // 每块元素数量
    constexpr int32_t CACHE_LINE_SIZE = 64;               // Cache Line大小
    constexpr int32_t RESERVERD_UB_SIZE = 20 * 1024;      // UB保留空间
    constexpr int NUM_QUEUE = 2;
    constexpr int UB_ALIGN = 32;
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 参数验证
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor", context->GetInputTensor(0), return ge::GRAPH_FAILED);

    // 获取输入信息
    int64_t inputLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto inputTensor = context->GetInputTensor(0);
    ge::DataType inputDataType = inputTensor->GetDataType();

    auto dimNum = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    OPS_LOG_E_IF(dimNum != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]AsynchronousCompleteCumsum required the dim of input-0 is 1");

    OPS_CHECK(inputDataType != ge::DT_INT32 && inputDataType != ge::DT_INT64,
              OPS_LOG_E("[ERROR]Invalid data type",
                        "AsynchronousCompleteCumsum only support int64 and int32."),
              return ge::GRAPH_FAILED);
    
    // 获取平台信息
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    
    // 获取UB大小
    uint64_t ubTotal;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotal);
    int64_t ubAvailable = (ubTotal - RESERVERD_UB_SIZE) / UB_ALIGN / NUM_QUEUE * UB_ALIGN * NUM_QUEUE;

    // 计算分块策略
    int64_t totalBlocks = (inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 向上取整
    if (totalBlocks < coreNum) {
        coreNum = totalBlocks;
    }
    
    if (coreNum == 0) {
        OPS_LOG_E(context, "[ERROR] need more than 0 ai core");
        return ge::GRAPH_FAILED;
    }
    int64_t blocksPerCore = totalBlocks / coreNum;                      // 每核基础块数k
    int64_t remainderBlocks = totalBlocks % coreNum;                    // 余数块数l

    // 配置Workspace
    // 每个块需要一个cache line（64字节）来避免false sharing
    // 实际只用每个cache line的前sizeof(T)字节存储部分和
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    int64_t userWorkspaceSize = totalBlocks * CACHE_LINE_SIZE;  // 每个块一个完整cache line
    workspaceSize[0] = systemWorkspacesSize + userWorkspaceSize;

    AsynchronousCompleteCumsumTilingData tiling;
    tiling.set_totalLength(inputLength);
    tiling.set_totalBlocks(totalBlocks);
    tiling.set_blocksPerCore(blocksPerCore);
    tiling.set_remainderBlocks(remainderBlocks);

    context->SetBlockDim(coreNum);
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
        // 动态shape下，输入shape为-1，输出shape也为-1
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
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    // 输出数据类型与输入相同
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

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(AsynchronousCompleteCumsum);
}

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
#include "invert_permute_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace {
    constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
    constexpr int32_t MIN_THREADS_PER_BLOCK = 512;
    constexpr int32_t SMALL_DATA_LENGTH = 3072;
    constexpr int32_t ELEMS_ADD_THREAD = 2; // 每个线程处理元素个数大于2时，单核实际线程数为1024
    constexpr int32_t PARAS_CONTROL = 4;    // 控制参数，当总元素数超过某值时，核数增加
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    InvertPermuteTilingData tiling;

    int32_t xDim0 = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    ge::DataType xType = context->GetInputTensor(0)->GetDataType();

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = ascendPlatform.GetCoreNumAiv();

    uint32_t actualThreadsPerBlock;
    int32_t blockDim;
    if (xDim0 <= SMALL_DATA_LENGTH) {
        blockDim = 1;
    } else {
        blockDim = std::min((xDim0 + MAX_THREADS_PER_BLOCK - MAX_THREADS_PER_BLOCK / PARAS_CONTROL) \
                            / MAX_THREADS_PER_BLOCK, coreNum);
    }
    actualThreadsPerBlock = xDim0 < blockDim * MIN_THREADS_PER_BLOCK * ELEMS_ADD_THREAD ? \
                                    MIN_THREADS_PER_BLOCK: MAX_THREADS_PER_BLOCK;

    context->SetBlockDim(blockDim);

    if (xType == ge::DT_INT64) {
        context->SetTilingKey(0);
    } else if (xType == ge::DT_INT32) {
        context->SetTilingKey(1);
    } else {
        OPS_LOG_E("[ERROR]Invalid data type. InvertPermute only support int64 and int32.", NULL);
        return ge::GRAPH_FAILED;
    }

    tiling.set_xDim0(xDim0);
    tiling.set_actualThreadsPerBlock(actualThreadsPerBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) // 从输入shape推导出输出的shape
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED); // 检查context是否是空指针
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    OPS_LOG_E_IF_NULL("x_shape", x_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("y_shape", y_shape, return ge::GRAPH_FAILED);

    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class InvertPermute : public OpDef {
public:
    explicit InvertPermute(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(InvertPermute);
}
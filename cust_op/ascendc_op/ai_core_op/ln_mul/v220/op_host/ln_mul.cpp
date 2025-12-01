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

#include "ln_mul_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {
constexpr int RESERVER_UB_SIZE = 40 * 1024;
constexpr int BLOCK_SIZE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // platform info
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    size_t coreNum = ascendPlatform.GetCoreNumAiv();

    // ub
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;

    // 输入shape
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::Shape shape = x1_shape->GetStorageShape();
    auto aLength = shape[0];  // A
    auto rLength = shape[1];  // R
    auto rLengthWithPadding = ((rLength * sizeof(float) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE /
                              sizeof(float);  // rLengthWithPadding 32位对齐

    // 计算一行结果需要的字节数: x(R)+u(R)+gamma(R)+beta(R)+tmp(R)+output(R)+mean(1)+rstd(1)+oneLocal(1)+reducesum(1)
    auto perRowMemory = (rLengthWithPadding * 6 + 4) * sizeof(float);
    // 每个核每次能处理的行数
    auto perCoreComputeRows = ubCanUsed / perRowMemory;

    // 前n个核分配的数据
    if (coreNum == 0) {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }
    uint32_t baseCoreRows = aLength / coreNum;    // 均分到每个核的数据行数
    uint32_t formerCoreNums = aLength % coreNum;  // 剩余的数据行数，并将剩余的行数均分到前面的核
    uint32_t formerCoreRows = baseCoreRows + 1;
    uint64_t loopCountFormer = formerCoreRows / perCoreComputeRows;
    uint64_t formerRowLeft = formerCoreRows - loopCountFormer * perCoreComputeRows;

    // 后面的核分配的数据
    uint64_t loopCountTail = baseCoreRows / perCoreComputeRows;
    uint64_t tailRowLeft = baseCoreRows - loopCountTail * perCoreComputeRows;

    // 设置tiling参数
    LnMulTilingData tiling;
    tiling.set_aLength(aLength);
    tiling.set_rLength(rLength);
    tiling.set_rLengthWithPadding(rLengthWithPadding);
    tiling.set_epsilon(0.00001);
    tiling.set_coreNum(coreNum);
    tiling.set_perCoreComputeRows(perCoreComputeRows);
    tiling.set_formerCoreRows(formerCoreRows);
    tiling.set_loopCountFormer(loopCountFormer);
    tiling.set_formerRowLeft(formerRowLeft);
    tiling.set_loopCountTail(loopCountTail);
    tiling.set_tailRowLeft(tailRowLeft);
    tiling.set_baseCoreRows(baseCoreRows);
    tiling.set_formerCoreNums(formerCoreNums);

    context->SetBlockDim(coreNum);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LnMul : public OpDef {
public:
    explicit LnMul(const char* name) : OpDef(name)
    {
        this->Input("inputXGm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("inputUGm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gammaGm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("betaGm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("outputGm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend310p", aicore_config);
        this->AICore().AddConfig("ascend910_95", aicore_config);
    }
};

OP_ADD(LnMul);
}  // namespace ops
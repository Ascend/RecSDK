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
#include "norm_multiply_dropout_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"
constexpr uint64_t UB_RESERVED_LENGTH = 2048;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    NormMultiplyDropoutGradTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("x input shape", xShape, return ge::GRAPH_FAILED);
    auto xStorageShape = xShape->GetStorageShape();
    OPS_CHECK(xStorageShape.GetDimNum() != 2,
              OPS_LOG_E("", "Input x dim num must be 2, but get %d", xStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    int32_t xRowCount = xStorageShape.GetDim(0);
    int32_t xColCount = xStorageShape.GetDim(1);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();

    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize -= UB_RESERVED_LENGTH;

    int32_t useCoreNum = coreNum > xRowCount ? xRowCount : coreNum;

    // 每行内存系数（不含常量）
    int32_t coeffPerRow = xColCount * sizeof(float) * 12 +           // x, u, output, norm, ln_out (5个)
                            xColCount / 8 +                            // mask (uint8)
                            sizeof(float) * 2 +                        // mean, rstd (2个)
                            (xColCount / 64 + 1) * sizeof(float) * 2;  // meanBuf, squareBuf

    // 常量内存
    int32_t constMemory = xColCount * sizeof(float) * 2 +  // weight, bias
                           20 * 1024;                       // dropBuf (20KB)

    // 计算最大行数
    int32_t singleBlockRows = (ubSize - constMemory) / coeffPerRow;

    // 向下对齐到2的幂次或保持整数
    singleBlockRows = std::max(1, singleBlockRows);  // 至少1行

    int32_t bigCoreNum = xRowCount % useCoreNum;
    int32_t littleCoreNum = useCoreNum - bigCoreNum;
    int32_t avgCoreCalcRows = xRowCount / useCoreNum;

    context->SetBlockDim(useCoreNum);

    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_littleCoreNum(littleCoreNum);
    tiling.set_avgCoreCalcRows(avgCoreCalcRows);
    tiling.set_xRowCount(xRowCount);
    tiling.set_xColCount(xColCount);
    tiling.set_singleBlockRows(singleBlockRows);
    tiling.set_useCoreNum(useCoreNum);

    float eps = *(context->GetAttrs()->GetAttrPointer<float>(0));
    float dropoutRatio = *(context->GetAttrs()->GetAttrPointer<float>(1));
    tiling.set_eps(eps);
    tiling.set_dropoutRatio(dropoutRatio);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);
    *outputShape = *xShape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class NormMultiplyDropoutGrad : public OpDef {
public:
    explicit NormMultiplyDropoutGrad(const char* name) : OpDef(name)
    {
        this->Input("d_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("u")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("d_u")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("d_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("d_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("d_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("eps").Float();
        this->Attr("dropout_ratio").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(NormMultiplyDropoutGrad);
}  // namespace ops

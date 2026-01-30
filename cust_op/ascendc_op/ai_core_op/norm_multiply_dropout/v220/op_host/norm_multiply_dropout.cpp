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
#include "norm_multiply_dropout_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

constexpr uint64_t UB_RESERVED_LENGTH = 2048;
constexpr int DROP_TEMP_BYTE = 1024;
constexpr int DROP_BUF = 20 * 1024;
constexpr int X_INPUT_DIM_NUM = 2;
constexpr int TENSOR_NUM_PER_ROW = 7;
constexpr int MASK_BYTE = 8;
constexpr int MEAN_AND_RSTD_NUM = 3;
constexpr int MEAN_AND_SQUARE_COUNT = 64;
constexpr int MEAN_AND_SQUARE_BUF_NUM = 2;
constexpr int EPS_INDEX = 0;
constexpr int DROPOUT_RATIO_INDEX = 1;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    NormMultiplyDropoutTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    auto storageShape = xShape->GetStorageShape();
    OPS_CHECK(storageShape.GetDimNum() != X_INPUT_DIM_NUM,
              OPS_LOG_E("", "Input x dim num must be 2, but get %d", storageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    int32_t xRowCount = xShape->GetStorageShape().GetDim(0);
    int32_t xColCount = xShape->GetStorageShape().GetDim(1);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();

    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize -= UB_RESERVED_LENGTH;

    int32_t useCoreNum = coreNum > xRowCount ? xRowCount : coreNum;

    // 每行内存系数（不含常量）
    int32_t coeffPerRow =
        xColCount * sizeof(float) * TENSOR_NUM_PER_ROW +  // x, u, output, norm, ln_out (5个) + cast
        xColCount / MASK_BYTE +                           // mask (uint8)
        sizeof(float) * MEAN_AND_RSTD_NUM +               // mean, rstd (2个)
        (xColCount / MEAN_AND_SQUARE_COUNT + 1) * sizeof(float) * MEAN_AND_SQUARE_BUF_NUM;  // meanBuf, squareBuf

    // 常量内存
    int32_t constMemory = xColCount * sizeof(float) +  // weight, bias
                          DROP_BUF;         // dropBuf (20KB)

    // 计算最大行数
    OPS_CHECK(coeffPerRow == 0, OPS_LOG_E("", "coeffPerRow must be not 0, but get 0."), return ge::GRAPH_FAILED);
    int32_t singleBlockRows = (ubSize - constMemory) / coeffPerRow;

    int32_t bigCoreNum = xRowCount % useCoreNum;
    int32_t littleCoreNum = useCoreNum - bigCoreNum;
    OPS_CHECK(useCoreNum == 0, OPS_LOG_E("", "useCoreNum must be not 0, but get 0."), return ge::GRAPH_FAILED);
    int32_t avgCoreCalcRows = xRowCount / useCoreNum;

    context->SetBlockDim(useCoreNum);

    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_littleCoreNum(littleCoreNum);
    tiling.set_avgCoreCalcRows(avgCoreCalcRows);
    tiling.set_xRowCount(xRowCount);
    tiling.set_xColCount(xColCount);
    tiling.set_singleBlockRows(singleBlockRows);
    tiling.set_useCoreNum(useCoreNum);
    tiling.set_dropTmpMem(DROP_TEMP_BYTE);

    OPS_LOG_E_IF_NULL("GetAttrs ret", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("AttrPointer index 0", context->GetAttrs()->GetAttrPointer<float>(EPS_INDEX),
                      return ge::GRAPH_FAILED);
    float eps = *(context->GetAttrs()->GetAttrPointer<float>(EPS_INDEX));
    float dropoutRatio = *(context->GetAttrs()->GetAttrPointer<float>(DROPOUT_RATIO_INDEX));
    tiling.set_eps(eps);
    tiling.set_dropoutRatio(dropoutRatio);
    OPS_LOG_E_IF_NULL("GetRawTilingData ret", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);
    *outputShape = *xShape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class NormMultiplyDropout : public OpDef {
public:
    explicit NormMultiplyDropout(const char* name) : OpDef(name)
    {
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
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
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

OP_ADD(NormMultiplyDropout);
}  // namespace ops

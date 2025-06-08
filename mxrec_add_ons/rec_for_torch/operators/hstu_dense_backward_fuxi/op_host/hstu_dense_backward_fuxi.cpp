/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "register/op_def_registry.h"

#include "hstu_dense_backward_normal_tiling.h"
#include "hstu_dense_backward_jagged_tiling.h"

namespace optiling {
static ge::graphStatus TilingCommonFunc(gert::TilingContext *context, HstuDenseBackwardFuxiTilingData &tiling)
{
    int64_t headDim = tiling.get_headDim();
    int64_t blockHeight = tiling.get_blockHeight();
    int64_t dataTypeLength = tiling.get_dataTypeLength();
    int64_t headNum = tiling.get_headNum();
    int64_t maxSeqLen = tiling.get_maxSeqLen();
    int64_t batchSize = tiling.get_batchSize();

    matmul_tiling::DataType dataType;
    ge::DataType gradType = context->GetInputTensor(INDEX_T::INDEX_0)->GetDataType();
    if (gradType == ge::DataType::DT_FLOAT) {
        dataType = matmul_tiling::DataType::DT_FLOAT;
    } else if (gradType == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    } else if (gradType == ge::DataType::DT_BF16) {
        dataType = matmul_tiling::DataType::DT_BFLOAT16;
    } else {
        OPS_LOG_E("TilingCommonFunc", "invalid datatype, only support float/fp16/bf16");
        return ge::GRAPH_FAILED;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    size_t vecCoreNum = ascendPlatform.GetCoreNumAiv();

    int64_t qkMatmulTempSpace = blockHeight * blockHeight;
    int64_t gvMatmulTempSpace = blockHeight * blockHeight;
    int64_t gpvMatmulTempSpace = blockHeight * blockHeight;
    int64_t gtvMatmulTempSpace = blockHeight * blockHeight;

    int64_t scoreTempSpace = blockHeight * blockHeight;
    int64_t biasTimestampTempSpace = blockHeight * blockHeight;
    int64_t biasPositionTempSpace = blockHeight * blockHeight;

    int64_t vGradAccumTempSpace = blockHeight * headDim;
    int64_t kGradAccumTempSpace = blockHeight * headDim;
    int64_t biasTimestampAccumTempSpace = blockHeight * headDim;
    int64_t biasPositionAccumTempSpace = blockHeight * headDim;

    int64_t maskTempSpace = blockHeight * blockHeight;
    int64_t attnBiasGradTempSpace = batchSize * headNum * maxSeqLen * maxSeqLen * dataTypeLength;

    int64_t totalTempSpaceForOneVec =
        MID_USE_TIMES *
            ((vGradAccumTempSpace + kGradAccumTempSpace + biasTimestampAccumTempSpace +
              biasPositionAccumTempSpace) * sizeof(float) +
             (qkMatmulTempSpace + gvMatmulTempSpace + scoreTempSpace + biasTimestampTempSpace + biasPositionTempSpace +
              gpvMatmulTempSpace + gtvMatmulTempSpace) * dataTypeLength) +
        maskTempSpace * dataTypeLength;

    int64_t workspaceSize = vecCoreNum * totalTempSpaceForOneVec + attnBiasGradTempSpace;

    size_t *currentWorkspace = context->GetWorkspaceSizes(INDEX_T::INDEX_1);
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = workspaceSize + systemWorkspaceSize;

    matmul_tiling::MatmulApiTiling qkMatmul(ascendPlatform);
    qkMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    qkMatmul.SetOrgShape(blockHeight, blockHeight, headDim);
    qkMatmul.SetShape(blockHeight, blockHeight, headDim);
    qkMatmul.SetBias(false);
    qkMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling qGradMatmul(ascendPlatform);
    qGradMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qGradMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qGradMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
    qGradMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    qGradMatmul.SetOrgShape(blockHeight, headDim, blockHeight);
    qGradMatmul.SetShape(blockHeight, headDim, blockHeight);
    qGradMatmul.SetBias(false);
    qGradMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling kGradMatmul(ascendPlatform);
    kGradMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    kGradMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    kGradMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
    kGradMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    kGradMatmul.SetOrgShape(blockHeight, headDim, blockHeight);
    kGradMatmul.SetShape(blockHeight, headDim, blockHeight);
    kGradMatmul.SetBias(false);
    kGradMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling vGradMatmul(ascendPlatform);
    vGradMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    vGradMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    vGradMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
    vGradMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    vGradMatmul.SetOrgShape(blockHeight, headDim, blockHeight);
    vGradMatmul.SetShape(blockHeight, headDim, blockHeight);
    vGradMatmul.SetBias(false);
    vGradMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling biasMaskMatmul(ascendPlatform);
    biasMaskMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    biasMaskMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    biasMaskMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);
    biasMaskMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    biasMaskMatmul.SetOrgShape(blockHeight, headDim, blockHeight);
    biasMaskMatmul.SetShape(blockHeight, headDim, blockHeight);
    biasMaskMatmul.SetBias(false);
    biasMaskMatmul.SetBufferSpace(-1, -1, -1);

    if (qkMatmul.GetTiling(tiling.qkMatmul) == -1 ||
        qGradMatmul.GetTiling(tiling.qGradMatmul) == -1 ||
        kGradMatmul.GetTiling(tiling.kGradMatmul) == -1 ||
        vGradMatmul.GetTiling(tiling.vGradMatmul) == -1 ||
        biasMaskMatmul.GetTiling(tiling.biasMaskMatmul) == -1) {
        OPS_LOG_E("TilingCommonFunc", "Get Matmul Tiling failed");
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace optiling {
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    InputLayout layout;
    OPS_LOG_E_IF(GetInputLayout(attrs, layout) == ge::GRAPH_FAILED,
                 context, return ge::GRAPH_FAILED, "GetInputLayout failed");

    HstuDenseBackwardFuxiTilingData tiling;

    if (layout == InputLayout::JAGGED) {
        TilingJaggedFunc(context, attrs, tiling);
    } else {
        OPS_LOG_E("TilingFunc", "invalid layout, only support jagged");
        return ge::GRAPH_FAILED;
    }

    return TilingCommonFunc(context, tiling);
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    InputLayout layout;
    OPS_LOG_E_IF(GetInputLayout(attrs, layout) == ge::GRAPH_FAILED,
                 context, return ge::GRAPH_FAILED, "GetInputLayout failed");
    ge::graphStatus result = ge::GRAPH_SUCCESS;
    if (layout == InputLayout::JAGGED) {
        result = optiling::JaggedInferShape(context);
    } else {
        OPS_LOG_E("InferShape", "invalid layout, only support jagged");
        return ge::GRAPH_FAILED;
    }
    return result;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext *context)
{
    // q dataType
    auto dataType = context->GetInputDataType(INDEX_T::INDEX_1);

    context->SetOutputDataType(INDEX_T::INDEX_0, dataType);
    context->SetOutputDataType(INDEX_T::INDEX_1, dataType);
    context->SetOutputDataType(INDEX_T::INDEX_2, dataType);
    context->SetOutputDataType(INDEX_T::INDEX_3, dataType);
    context->SetOutputDataType(INDEX_T::INDEX_4, dataType);
    context->SetOutputDataType(INDEX_T::INDEX_5, dataType);
    context->SetOutputDataType(INDEX_T::INDEX_6, dataType);

    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class HstuDenseBackwardFuxi : public OpDef {
public:
    explicit HstuDenseBackwardFuxi(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias_position")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias_timestamp")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // 可以和grad合并
        this->Input("grad_bias_position")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("grad_bias_timestamp")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // ./可以和grad合并

        this->Output("q_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("k_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("v_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("position_bias_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("timestamp_bias_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("vbpos_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("vbts_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("layout").String("jagged");
        this->Attr("mask_type").Int();
        this->Attr("max_seq_len").Int();
        this->Attr("silu_scale").Float();
        this->Attr("seq_offsets").AttrType(OPTIONAL).ListInt();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910", aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(HstuDenseBackwardFuxi);
} // namespace ops

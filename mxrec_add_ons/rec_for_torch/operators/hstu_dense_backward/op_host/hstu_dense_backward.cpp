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

static void SetQKVGrad(matmul_tiling::MatmulApiTiling &matmul)
{
    matmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
    matmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    matmul.SetOrgShape(blockHeight, headDim, blockHeight);
    matmul.SetShape(blockHeight, headDim, blockHeight);
    matmul.SetBias(false);
    matmul.SetBufferSpace(-1, -1, -1);
}

static void SetQKMatmul(matmul_tiling::MatmulApiTiling &matmul)
{
    matmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    matmul.SetOrgShape(blockHeight, blockHeight, headDim);
    matmul.SetShape(blockHeight, blockHeight, headDim);
    matmul.SetBias(false);
    matmul.SetBufferSpace(-1, -1, -1);}
}

static ge::graphStatus GetDataType(gert::TilingContext *context, matmul_tiling::DataType &dataType)
{
    ge::DataType gradType = context->GetInputTensor(INDEX_T::INDEX_0)->GetDataType();
    if (gradType == ge::DataType::DT_FLOAT) {
        dataType = matmul_tiling::DataType::DT_FLOAT;
    } else if (gradType == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    } else if (gradType == ge::DataType::DT_BF16) {
        dataType = matmul_tiling::DataType::DT_BFLOAT16;
    } else {
        OPS_LOG_E("", "invalid datatype, only support float/fp16/bf16\n");
        return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus TilingCommonFunc(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t headDim = tiling.get_headDim();
    int64_t blockHeight = tiling.get_blockHeight();
    int64_t dataTypeLength = tiling.get_dataTypeLength();

    matmul_tiling::DataType dataType;
    OPS_CHECK(GetDataType(context, dataType) == ge::GRAPH_FAILED, OPS_LOG_E("", "GetDataType failed\n"),
        return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    size_t vecCoreNum = ascendPlatform.GetCoreNumAiv();

    int64_t qkMatmulTempSpace = blockHeight * blockHeight;
    int64_t gvMatmulTempSpace = blockHeight * blockHeight;

    int64_t scoreTempSpace = blockHeight * blockHeight;

    int64_t vGradAccumTempSpace = blockHeight * headDim;
    int64_t kGradAccumTempSpace = blockHeight * headDim;

    int64_t maskTempSpace = blockHeight * blockHeight;

    int64_t totalTempSpaceForOneVec =  MID_USE_TIMES *
            ((vGradAccumTempSpace + kGradAccumTempSpace) * sizeof(float) +
             (qkMatmulTempSpace + gvMatmulTempSpace + scoreTempSpace) * dataTypeLength) +
        maskTempSpace * dataTypeLength;

    int64_t workspaceSize = vecCoreNum * totalTempSpaceForOneVec;

    size_t *currentWorkspace = context->GetWorkspaceSizes(INDEX_T::INDEX_1);
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = workspaceSize + systemWorkspaceSize;

    matmul_tiling::MatmulApiTiling qkMatmul(ascendPlatform);
    SetQKMatmul(qkMatmul);

    matmul_tiling::MatmulApiTiling qGradMatmul(ascendPlatform);
    SetQKVGrad(qGradMatmul);

    matmul_tiling::MatmulApiTiling kGradMatmul(ascendPlatform);
    SetQKVGrad(kGradMatmul);

    matmul_tiling::MatmulApiTiling vGradMatmul(ascendPlatform);
    SetQKVGrad(vGradMatmul);

    if (qkMatmul.GetTiling(tiling.qkMatmul) == -1 || qGradMatmul.GetTiling(tiling.qGradMatmul) == -1 ||
        kGradMatmul.GetTiling(tiling.kGradMatmul) == -1 || vGradMatmul.GetTiling(tiling.vGradMatmul) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace optiling {
ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return ge::GRAPH_FAILED);

    InputLayout layout;
    OPS_CHECK(GetInputLayout(attrs, layout) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "GetInputLayout failed\n"),
                return ge::GRAPH_FAILED);

    HstuDenseBackwardTilingData tiling;

    if (layout == InputLayout::JAGGED) {
        TilingJaggedFunc(context, attrs, tiling);
    } else {
        TilingNormalFunc(context, attrs, tiling);
    }

    return TilingCommonFunc(context, tiling);
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return ge::GRAPH_FAILED);

    InputLayout layout;
    OPS_CHECK(GetInputLayout(attrs, layout) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "GetInputLayout failed\n"),
                return ge::GRAPH_FAILED);
    ge::graphStatus result = ge::GRAPH_SUCCESS;
    if (layout == InputLayout::JAGGED) {
        result = optiling::JaggedInferShape(context);
    } else {
        result = optiling::NormalInferShape(context);
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

    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class HstuDenseBackward : public OpDef {
public:
    explicit HstuDenseBackward(const char *name) : OpDef(name)
    {
        this->Input("grad").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("attn_bias").ParamType(OPTIONAL).DataType
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("q_grad").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("k_grad").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("v_grad").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attn_bias_grad").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("layout").String("normal");
        this->Attr("mask_type").Int();
        this->Attr("max_seq_len").Int();
        this->Attr("silu_scale").Float();
        this->Attr("seq_offsets").AttrType(OPTIONAL).ListInt();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true).ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
        .ExtendCfgInfo("coreType.value", "AiCore").ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910", aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(HstuDenseBackward);
} // namespace ops

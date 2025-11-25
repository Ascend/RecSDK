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

#include "register/op_def_registry.h"

#include "hstu_dense_backward_normal_tiling.h"
#include "hstu_dense_backward_jagged_tiling.h"

#include "matmul_check.h"
using namespace MatmulTilingCheck;
namespace optiling {
static ge::graphStatus TilingCommonFunc(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t batchSize = tiling.get_batchSize();
    int64_t headNum = tiling.get_headNum();
    int64_t headDim = tiling.get_headDim();
    int64_t blockHeight = tiling.get_blockHeight();
    int64_t dataTypeLength = tiling.get_dataTypeLength();
    int64_t maxSeqLen = tiling.get_maxSeqLen();
    int32_t enableBias = tiling.get_enableBias();
    int32_t isNormal = tiling.get_isNormal();

    matmul_tiling::DataType dataType;
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

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    size_t vecCoreNum = ascendPlatform.GetCoreNumAiv();

    int64_t qkMatmulTempSpace = blockHeight * blockHeight;
    int64_t gvMatmulTempSpace = blockHeight * blockHeight;

    int64_t scoreTempSpace = blockHeight * blockHeight;

    int64_t vGradAccumTempSpace = blockHeight * headDim;
    int64_t kGradAccumTempSpace = blockHeight * headDim;

    int64_t maskTempSpace = blockHeight * blockHeight;

    int64_t totalTempSpaceForOneVec =
        MID_USE_TIMES *
            ((vGradAccumTempSpace + kGradAccumTempSpace) * sizeof(float) +
             (qkMatmulTempSpace + gvMatmulTempSpace + scoreTempSpace) * dataTypeLength) +
        maskTempSpace * dataTypeLength;

    int64_t workspaceSize = vecCoreNum * totalTempSpaceForOneVec;

    if (!isNormal && !enableBias) {
        int64_t biasGradTempSpace = blockHeight * blockHeight;
        int64_t qGradAccumTempSpace = batchSize * headNum * maxSeqLen * headDim;
        workspaceSize += biasGradTempSpace * dataTypeLength * MID_USE_TIMES * vecCoreNum +
            qGradAccumTempSpace * sizeof(float);
    }

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

    if (qkMatmul.GetTiling(tiling.qkMatmul) == -1 ||
        qGradMatmul.GetTiling(tiling.qGradMatmul) == -1 ||
        kGradMatmul.GetTiling(tiling.kGradMatmul) == -1 ||
        vGradMatmul.GetTiling(tiling.vGradMatmul) == -1) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckBaseMNK(tiling.qkMatmul, dataTypeLength, dataTypeLength) ||
        !CheckBaseMNK(tiling.qGradMatmul, dataTypeLength, sizeof(float)) ||
        !CheckBaseMNK(tiling.kGradMatmul, dataTypeLength, sizeof(float)) ||
        !CheckBaseMNK(tiling.vGradMatmul, dataTypeLength, sizeof(float))) {
        return ge::GRAPH_FAILED;
    }

    if (gradType == ge::DataType::DT_BF16) {
        // matmul计算时，左矩阵一次拷入L1中的大小为depthA1*baseM*baseK，右矩阵一次拷入L1中的大小为depthB1*baseN*baseK
        // 理论上在调用matmul.GetTiling时，matmul内部会自动算出最优depth值，但不能使所有场景性能最优
        // 所以这里通过设置depth大小，调整一次拷入L1的数据多少，达到优化目的
        // 经测试，depth=4时已有shape性能最佳，并不适用所有shape场景。后续有其他shape可通过该值调整获得最优性能。
        int64_t depth = 4;

        OPS_CHECK(depth * (qkMatmul.GetBaseM() * qkMatmul.GetBaseK() + qkMatmul.GetBaseK() * qkMatmul.GetBaseN()) >
            L1_BUFFER_SIZE, OPS_LOG_E("", "The qkMatmul depth is set too high\n"), return ge::GRAPH_FAILED);
        tiling.qkMatmul.set_depthA1(depth);
        tiling.qkMatmul.set_depthB1(depth);

        OPS_CHECK(depth * (qGradMatmul.GetBaseM() * qGradMatmul.GetBaseK() +
            qGradMatmul.GetBaseK() * qGradMatmul.GetBaseN()) > L1_BUFFER_SIZE,
            OPS_LOG_E("", "The qGradMatmul depth is set too high\n"), return ge::GRAPH_FAILED);
        tiling.qGradMatmul.set_depthA1(depth);
        tiling.qGradMatmul.set_depthB1(depth);

        OPS_CHECK(depth * (kGradMatmul.GetBaseM() * kGradMatmul.GetBaseK() +
            kGradMatmul.GetBaseK() * kGradMatmul.GetBaseN()) > L1_BUFFER_SIZE,
            OPS_LOG_E("", "The kGradMatmul depth is set too high\n"), return ge::GRAPH_FAILED);
        tiling.kGradMatmul.set_depthA1(depth);
        tiling.kGradMatmul.set_depthB1(depth);

        OPS_CHECK(depth * (vGradMatmul.GetBaseM() * vGradMatmul.GetBaseK() +
            vGradMatmul.GetBaseK() * vGradMatmul.GetBaseN()) > L1_BUFFER_SIZE,
            OPS_LOG_E("", "The vGradMatmul depth is set too high\n"), return ge::GRAPH_FAILED);
        tiling.vGradMatmul.set_depthA1(depth);
        tiling.vGradMatmul.set_depthB1(depth);
    }

    context->SetBlockDim(coreNum);
    tiling.set_aivNum(vecCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace optiling {
ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
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
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
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
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("q")
            .ParamType(REQUIRED)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("attn_bias")
            .ParamType(OPTIONAL)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("num_context")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("num_target")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});

        this->Output("q_grad")
            .ParamType(OPTIONAL)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("k_grad")
            .ParamType(OPTIONAL)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("v_grad")
            .ParamType(OPTIONAL)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("attn_bias_grad")
            .ParamType(OPTIONAL)
            .Follow("grad", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Attr("layout").String("normal");
        this->Attr("mask_type").Int();
        this->Attr("max_seq_len").Int();
        this->Attr("silu_scale").Float();
        this->Attr("target_group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend910_95", aicore_config);
    }
};

OP_ADD(HstuDenseBackward);
} // namespace ops

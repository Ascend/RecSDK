/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "attention_fusion_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr int TRANSPOSE_AGLING_SPECAIL_CASE[] = {1000, 50};
constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int MAX_BATCH_SIZE = 2000;
constexpr int MAX_DIM = 1000;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int TRANSPOSE_ALIGNMENT = 16;
constexpr int TRANSPOSE_TYPE = 7;
constexpr int KEY_DIM1_COPY_ALIGN_MODE = 1;
constexpr int KEY_DIM1_COPY_TRANSPOSE_ALIGN_MODE = 2;
constexpr int KEY_DIM1_COPY_PAD_MODE = 3;
constexpr int L0C_MAX_SIZE = 32 * 1024;

int CeilDiv(int a, int b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

int GCD(int a, int b)
{
    while (a != b) {
        if (a > b) {
            a -= b;
        } else {
            b -= a;
        }
    }
    return a;
}

int LCM(int x, int y)
{
    int gcdResult = GCD(x, y);
    if (gcdResult == 0) {
        return 0;
    }
    return x * y / gcdResult;
}

static int32_t GradMatmulTiling(gert::TilingContext* context, AttentionFusionGradTilingData& tilingData)
{
    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(2)->GetStorageShape();
    auto kShape = context->GetInputShape(3)->GetStorageShape();
    auto vShape = context->GetInputShape(4)->GetStorageShape();

    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int DimIndex2 = 2;
    // qkMatmul configuration
    matmul_tiling::MatmulApiTiling gardV(ascnedPlatform);
    gardV.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetShape(kShape.GetDim(1), vShape.GetDim(DimIndex2), qShape.GetDim(1));

    gardV.SetBias(false);
    gardV.SetBufferSpace(-1, L0C_MAX_SIZE, -1);

    matmul_tiling::MatmulApiTiling gardS(ascnedPlatform);
    gardS.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetShape(qShape.GetDim(1), vShape.GetDim(1), vShape.GetDim(DimIndex2));

    gardS.SetBias(false);
    gardS.SetBufferSpace(-1, L0C_MAX_SIZE, -1);

    matmul_tiling::MatmulApiTiling gardQ(ascnedPlatform);
    gardQ.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetShape(qShape.GetDim(1), qShape.GetDim(DimIndex2), kShape.GetDim(1));

    gardQ.SetBias(false);
    gardQ.SetBufferSpace(-1, L0C_MAX_SIZE, -1);

    matmul_tiling::MatmulApiTiling gardK(ascnedPlatform);
    gardK.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetShape(kShape.GetDim(1), qShape.GetDim(DimIndex2), qShape.GetDim(1));

    gardK.SetBias(false);
    gardK.SetBufferSpace(-1, L0C_MAX_SIZE, -1);

    if (gardV.GetTiling(tilingData.gardVMatmulTiling) == -1 || gardS.GetTiling(tilingData.gardSMatmulTiling) == -1 ||
        gardK.GetTiling(tilingData.gardKMatmulTiling) == -1 || gardQ.GetTiling(tilingData.gardQMatmulTiling) == -1) {
        return 1;
    }
    return 0;
}

static int32_t GradSoftmaxTiling(gert::TilingContext* context, AttentionFusionGradTilingData& tilingData, uint64_t ub)
{
    int dimIndex2 = 2;
    int dimIndex3 = 3;
    int dimIndex4 = 4;
    auto qShape = context->GetInputShape(dimIndex2)->GetStorageShape();
    auto kShape = context->GetInputShape(dimIndex3)->GetStorageShape();
    auto vShape = context->GetInputShape(dimIndex4)->GetStorageShape();
    if (qShape.GetDim(0) > MAX_BATCH_SIZE || qShape.GetDim(1) > MAX_DIM || qShape.GetDim(dimIndex2) > MAX_DIM ||
        kShape.GetDim(1) > MAX_DIM || kShape.GetDim(dimIndex2) > MAX_DIM || vShape.GetDim(1) > MAX_DIM ||
        vShape.GetDim(dimIndex2) > MAX_DIM) {
        printf("This shape is out of range(0, 1000)");
    }

    // Platform configuration
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t softmaxOutSize = qShape.GetDim(0) * qShape.GetDim(1) * kShape.GetDim(1) * sizeof(float);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = softmaxOutSize + systemWorkspacesSize;

    int paddingKeyDim1 = CeilDiv(kShape.GetDim(1), FLOAT_ALIGNMENT) * FLOAT_ALIGNMENT;
    int numRowOfNormalizeOne = ub / 4 / sizeof(float) / paddingKeyDim1;
    int keyDim1Align;
    int transposeAlignDim = LCM(kShape.GetDim(1), TRANSPOSE_ALIGNMENT) / kShape.GetDim(1);

    if (kShape.GetDim(1) % FLOAT_ALIGNMENT == 0) {
        keyDim1Align = KEY_DIM1_COPY_ALIGN_MODE;
    } else if (qShape.GetDim(1) == TRANSPOSE_AGLING_SPECAIL_CASE[0] &&
               kShape.GetDim(1) == TRANSPOSE_AGLING_SPECAIL_CASE[1]) {
        numRowOfNormalizeOne = TRANSPOSE_ALIGNMENT * transposeAlignDim;
        keyDim1Align = KEY_DIM1_COPY_TRANSPOSE_ALIGN_MODE;
    } else {
        keyDim1Align = KEY_DIM1_COPY_PAD_MODE;
    }

    const ge::Shape softmaxShape({numRowOfNormalizeOne, paddingKeyDim1});
    uint64_t toTalUb;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    uint64_t localWorkSpaceSize = toTalUb - 4 * numRowOfNormalizeOne * paddingKeyDim1 * sizeof(float);
    float attenDimSqrt = 1 / std::sqrt(qShape.GetDim(2));

    // set attr
    tilingData.set_attnDim(qShape.GetDim(dimIndex2));
    tilingData.set_queryDim1(qShape.GetDim(1));
    tilingData.set_queryDim2(qShape.GetDim(dimIndex2));
    tilingData.set_keyDim1(kShape.GetDim(1));
    tilingData.set_keyDim2(kShape.GetDim(dimIndex2));
    tilingData.set_valueDim1(vShape.GetDim(1));
    tilingData.set_valueDim2(vShape.GetDim(dimIndex2));
    tilingData.set_batchNum(qShape.GetDim(0));
    tilingData.set_numRowOfNormalizeOne(numRowOfNormalizeOne);
    tilingData.set_paddingKeyDim1(paddingKeyDim1);
    tilingData.set_attenDimSqrt(attenDimSqrt);
    tilingData.set_keyDim1Align(keyDim1Align);

    AscendC::SoftMaxGradTilingFunc(softmaxShape, sizeof(float), localWorkSpaceSize, tilingData.softMaxGradTiling);
    return 0;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AttentionFusionGradTilingData tilingData;

    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascnedPlatform.GetCoreNumAic();

    uint64_t ub;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;

    if (GradMatmulTiling(context, tilingData) != 0 || GradSoftmaxTiling(context, tilingData, ub) != 0) {
        return ge::GRAPH_FAILED;
    }

    auto kShape = context->GetInputShape(3)->GetStorageShape();
    int keyDim1 = kShape.GetDim(1);
    if (keyDim1 == 0) {
        return ge::GRAPH_FAILED;
    }

    int paddingKeyDim1 = CeilDiv(keyDim1, FLOAT_ALIGNMENT) * FLOAT_ALIGNMENT;
    int transposeAlignDim = LCM(keyDim1, TRANSPOSE_ALIGNMENT) / keyDim1;

    std::vector<int64_t> shapeVec = {TRANSPOSE_ALIGNMENT, keyDim1 * transposeAlignDim};
    ge::Shape srcShape(shapeVec);
    AscendC::GetConfusionTransposeTilingInfo(srcShape, 0, sizeof(float), TRANSPOSE_TYPE,
                                             tilingData.unAlign2AlignStep1Tiling);

    std::vector<int64_t> shapeVec1 = {paddingKeyDim1 * transposeAlignDim, TRANSPOSE_ALIGNMENT};
    ge::Shape srcShape1(shapeVec1);
    AscendC::GetConfusionTransposeTilingInfo(srcShape1, 0, sizeof(float), TRANSPOSE_TYPE,
                                             tilingData.unAlign2AlignStep2Tiling);

    std::vector<int64_t> shapeVec2 = {TRANSPOSE_ALIGNMENT, paddingKeyDim1 * transposeAlignDim};
    ge::Shape srcShape2(shapeVec2);
    AscendC::GetConfusionTransposeTilingInfo(srcShape2, 0, sizeof(float), TRANSPOSE_TYPE,
                                             tilingData.Align2UnAlignStep1Tiling);

    std::vector<int64_t> shapeVec3 = {keyDim1 * transposeAlignDim, TRANSPOSE_ALIGNMENT};
    ge::Shape srcShape3(shapeVec3);
    AscendC::GetConfusionTransposeTilingInfo(srcShape3, 0, sizeof(float), TRANSPOSE_TYPE,
                                             tilingData.Align2UnAlignStep2Tiling);

    tilingData.set_transposeAlignDim(transposeAlignDim);

    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    int index2 = 2;
    int index3 = 3;
    int index4 = 4;
    const gert::Shape* qShape = context->GetInputShape(index2);
    const gert::Shape* kShape = context->GetInputShape(index3);
    const gert::Shape* vShape = context->GetInputShape(index4);
    gert::Shape* gradQShape = context->GetOutputShape(0);
    gert::Shape* gradKShape = context->GetOutputShape(1);
    gert::Shape* gradVShape = context->GetOutputShape(index2);
    *gradQShape = *qShape;
    *gradKShape = *kShape;
    *gradVShape = *vShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    int index2 = 2;
    int index3 = 3;
    int index4 = 4;
    context->SetOutputDataType(0, context->GetInputDataType(index2));
    context->SetOutputDataType(1, context->GetInputDataType(index3));
    context->SetOutputDataType(index2, context->GetInputDataType(index4));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AttentionFusionGrad : public OpDef {
public:
    explicit AttentionFusionGrad(const char* name) : OpDef(name)
    {
        this->Input("dout")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("softmax_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");
    }
};

OP_ADD(AttentionFusionGrad);
}  // namespace ops

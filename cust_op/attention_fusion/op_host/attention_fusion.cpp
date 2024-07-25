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

#include <cstdint>
#include <cmath>
#include "attention_fusion_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr int32_t ALIGN_32 = (32 / sizeof(float));
constexpr int32_t RESERVER_UB_SIZE = (20 * 1024);
constexpr int32_t ALREADY_ALIGNED = 1;
constexpr int32_t SPECIAL_CASE = 2;
constexpr int32_t SPECIAL_ROW_SIZE = (16 * 16);
constexpr int32_t SPECIAL_Q_DIM1 = 500;
constexpr int32_t SPECIAL_K_DIM1 = 50;
constexpr int32_t ORIG_UNPAD_DIM0 = 16;
constexpr int32_t ORIG_UNPAD_DIM1 = (16 * 50);
constexpr int32_t TRANSPOSED_PADDED_DIM0 = (56 * 16);
constexpr int32_t TRANSPOSED_PADDED_DIM1 = 16;
constexpr int32_t ORIG_PADDED_DIM0 = 16;
constexpr int32_t ORIG_PADDED_DIM1 = (56 * 16);
constexpr int32_t TRANSPOSED_UNPAD_DIM0 = (16 * 50);
constexpr int32_t TRANSPOSED_UNPAD_DIM1 = 16;
constexpr int32_t TRANSPOSE_CONST = 7;
constexpr int32_t SHAPE_DIMS = 3;
constexpr int32_t UB_TILES = 3;
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;

static int32_t MatmulTiling(gert::TilingContext* context, AttentionFusionTilingData &tilingData)
{
    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascnedPlatform.GetCoreNumAic();
    // qkMatmul configuration
    matmul_tiling::MultiCoreMatmulTiling qkMm(ascnedPlatform);
    qkMm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetShape(qShape.GetDim(DIM1), kShape.GetDim(DIM1), kShape.GetDim(DIM2));

    qkMm.SetSingleShape(qShape.GetDim(DIM1), kShape.GetDim(DIM1), kShape.GetDim(DIM2));

    qkMm.SetBias(false);
    qkMm.SetBufferSpace(-1, -1, -1);
    qkMm.SetDim(coreNum);

    // kvBmm Matmul Tilling
    matmul_tiling::MultiCoreMatmulTiling kvMm(ascnedPlatform);
    kvMm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetShape(qShape.GetDim(DIM1), vShape.GetDim(DIM2), kShape.GetDim(DIM1));
    kvMm.SetSingleShape(qShape.GetDim(DIM1), vShape.GetDim(DIM2), kShape.GetDim(DIM1));

    kvMm.SetBias(false);
    kvMm.SetBufferSpace(-1, -1, -1);
    kvMm.SetDim(coreNum);

    // set tiling data
    tilingData.set_attnDim(qShape.GetDim(DIM2));
    tilingData.set_queryDim1(qShape.GetDim(DIM1));
    tilingData.set_queryDim2(qShape.GetDim(DIM2));
    tilingData.set_keyDim1(kShape.GetDim(DIM1));
    tilingData.set_valueDim2(vShape.GetDim(DIM2));
    tilingData.set_batchNum(qShape.GetDim(DIM0));

    // Get tilingData using on the kernel side
    if (qkMm.GetTiling(tilingData.qkMatmulTiling) == -1 ||
        kvMm.GetTiling(tilingData.kvMatmulTiling) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static int32_t SoftmaxTiling(gert::TilingContext* context, AttentionFusionTilingData &tilingData, uint64_t ub)
{
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    const int32_t* maskIsOn = context->GetAttrs()->GetAttrPointer<int32_t>(0);

    int numOfelement = kShape.GetDim(DIM1) / ALIGN_32;
    uint8_t attr = 0;
    int normalizeColumn = 0;

    // Get column with 32bytes alignment
    if ((kShape.GetDim(DIM1) % ALIGN_32) == 0) {
        normalizeColumn = numOfelement * ALIGN_32;
        attr = ALREADY_ALIGNED;
    } else if (kShape.GetDim(DIM1) == SPECIAL_K_DIM1 && qShape.GetDim(DIM1) > SPECIAL_Q_DIM1) {
        normalizeColumn = numOfelement * ALIGN_32 + ALIGN_32;
        attr = SPECIAL_CASE;
    } else {
        normalizeColumn = numOfelement * ALIGN_32 + ALIGN_32;
    }

    if ((sizeof(float) * normalizeColumn) > (ub / UB_TILES)) {
        printf("[ERROR] Key dim1 too large, please check key shape!");
        return ge::GRAPH_FAILED;
    }

    // Get how many rows that half of ub contains
    int normalizeRow = ub / UB_TILES / (sizeof(float) * normalizeColumn);
    if (normalizeRow > qShape.GetDim(DIM1)) {
        normalizeRow = qShape.GetDim(DIM1);
    }

    if (attr == SPECIAL_CASE) {
        normalizeRow = SPECIAL_ROW_SIZE;
    }

    // Get max Ub left for softmax shared tmp buffer
    uint64_t maxLocalWorkSize = ub - (normalizeRow * normalizeColumn * sizeof(float) * 2);

    const ge::Shape softmaxShape({normalizeRow, normalizeColumn});
    const uint32_t minLocalWorkSize = AscendC::GetSoftMaxMinTmpSize(softmaxShape, sizeof(float), false);
    if (minLocalWorkSize > maxLocalWorkSize) {
        printf("[ERROR] Softmax minimun workspace larger than max local workspace, please check input shape.");
        return ge::GRAPH_FAILED;
    }

    // divisor should not be 0
    if (normalizeRow == 0) {
        printf("[ERROR] divisor normalizeRow == 0.")
        return ge::GRAPH_FAILED;
    }
    int normalizeLoop = qShape.GetDim(DIM1) / normalizeRow;
    normalizeLoop = ((qShape.GetDim(DIM1) % normalizeRow) == 0) ? normalizeLoop : normalizeLoop + 1;
    float dimSqrt = (sqrt(qShape.GetDim(DIM2)) != 0) ? (1 / sqrt(qShape.GetDim(DIM2))) : 0;

    // set tiling data
    tilingData.set_normalizeAttr(attr);
    tilingData.set_normalizeLoop(normalizeLoop);
    tilingData.set_normalizeRow(normalizeRow);
    tilingData.set_normalizeColumn(normalizeColumn);
    tilingData.set_maskIsOn(*maskIsOn);
    tilingData.set_normalizeSqrt(dimSqrt);
    tilingData.set_maxSharedTmpBuf(maxLocalWorkSize);

    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), maxLocalWorkSize, tilingData.softMaxTilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AttentionFusionTilingData tilingData;

    // Platform configuration
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = 0 + systemWorkspacesSize;
    size_t coreNum = ascnedPlatform.GetCoreNumAic();

    uint64_t ub;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;

    if (MatmulTiling(context, tilingData) != ge::GRAPH_SUCCESS ||
        SoftmaxTiling(context, tilingData, ub) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Get tiling data for transposing the origin tensor, then the transposed tensor is going to be padded.
    std::vector<int64_t> shapeVec = {ORIG_UNPAD_DIM0, ORIG_UNPAD_DIM1};
    ge::Shape srcShape(shapeVec);
    AscendC::GetConfusionTransposeTilingInfo(srcShape, 0, sizeof(float), TRANSPOSE_CONST,
        tilingData.confusionTransposeTilingData);

    // Get tiling data for transposing the padded tensor.
    std::vector<int64_t> shapeVec1 = {TRANSPOSED_PADDED_DIM0, TRANSPOSED_PADDED_DIM1};
    ge::Shape srcShape1(shapeVec1);
    AscendC::GetConfusionTransposeTilingInfo(srcShape1, 0, sizeof(float), TRANSPOSE_CONST,
        tilingData.confusionTransposeTilingData1);

    // Get tiling data for transposing the padded tensor, then the transposed tensor is going to be unpadded.
    std::vector<int64_t> shapeVec2 = {ORIG_PADDED_DIM0, ORIG_PADDED_DIM1};
    ge::Shape srcShape2(shapeVec2);
    AscendC::GetConfusionTransposeTilingInfo(srcShape2, 0, sizeof(float), TRANSPOSE_CONST,
        tilingData.confusionTransposeTilingData2);

    // Get tiling data for transposing the unpadded tensor back to original shape.
    std::vector<int64_t> shapeVec3 = {TRANSPOSED_UNPAD_DIM0, TRANSPOSED_UNPAD_DIM1};
    ge::Shape srcShape3(shapeVec3);
    AscendC::GetConfusionTransposeTilingInfo(srcShape3, 0, sizeof(float), TRANSPOSE_CONST,
        tilingData.confusionTransposeTilingData3);

    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(0);
    const gert::Shape* kShape = context->GetInputShape(1);
    const gert::Shape* vShape = context->GetInputShape(2);

    gert::Shape* attnScoreShape = context->GetOutputShape(0);
    gert::Shape* softmaxOutShape = context->GetOutputShape(1);

    attnScoreShape->SetDimNum(SHAPE_DIMS);
    attnScoreShape->SetDim(0, qShape->GetDim(DIM0));
    attnScoreShape->SetDim(1, qShape->GetDim(DIM1));
    attnScoreShape->SetDim(2, vShape->GetDim(DIM2));

    softmaxOutShape->SetDimNum(SHAPE_DIMS);
    softmaxOutShape->SetDim(0, qShape->GetDim(DIM0));
    softmaxOutShape->SetDim(1, qShape->GetDim(DIM1));
    softmaxOutShape->SetDim(2, kShape->GetDim(DIM1));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return GRAPH_SUCCESS;
}
}


namespace ops {
class AttentionFusion : public OpDef {
public:
    explicit AttentionFusion(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("attn_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("atten_score")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mask_on").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AttentionFusion);
}

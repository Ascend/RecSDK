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

#include <cstdint>
#include <cstdio>
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "hstu_dense_forward_tiling.h"

using namespace ge;

namespace {
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t K_INDEX = 1;
constexpr uint32_t V_INDEX = 2;

constexpr uint32_t FOUR_DIMS = 4U;

constexpr int FLOAT_TILING_KEY = 2;
constexpr int BF16_TILING_KEY = 1;
constexpr int FLOAT16_TILING_KEY = 0;

constexpr int BLOCK_HEIGHT = 256;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int TRANS_PIPE_NUM = 4;

const char* K_INNER_DEBUG = "HstuDenseForward Tiling Debug";
}  // namespace

namespace optiling {
static ge::graphStatus HstuDenseForwardTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    HstuDenseForwardTilingData* tilingData = context->GetTilingData<HstuDenseForwardTilingData>();
    if (tilingData == nullptr) {
        fprintf(stderr, "[TILING] ERROR: GetTilingData returned nullptr\n");
        return ge::GRAPH_FAILED;
    }

    // get input shape
    const gert::StorageShape* qStorageShape = context->GetInputShape(Q_INDEX);
    const gert::StorageShape* kStorageShape = context->GetInputShape(K_INDEX);
    const gert::StorageShape* vStorageShape = context->GetInputShape(V_INDEX);
    if (qStorageShape == nullptr || kStorageShape == nullptr || vStorageShape == nullptr) {
        fprintf(stderr, "[TILING] ERROR: null storage shape\n");
        return GRAPH_FAILED;
    }

    if (qStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS ||
        kStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS ||
        vStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS) {
        fprintf(stderr, "[TILING] ERROR: wrong dim num\n");
        return GRAPH_FAILED;
    }

    uint32_t bs = qStorageShape->GetStorageShape().GetDim(0);
    uint32_t s = qStorageShape->GetStorageShape().GetDim(1);
    uint32_t n = qStorageShape->GetStorageShape().GetDim(2);
    uint32_t d = qStorageShape->GetStorageShape().GetDim(3);

    tilingData->batchSize = bs;
    tilingData->seqLen = s;
    tilingData->headNum = n;
    tilingData->dim = d;

    // get attribute
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        fprintf(stderr, "[TILING] ERROR: attrs is nullptr\n");
        return ge::GRAPH_FAILED;
    }

    auto rankId = attrs->GetAttrPointer<int>(0);
    auto rankSize = attrs->GetAttrPointer<int>(1);
    auto maskType = attrs->GetAttrPointer<int>(2);
    auto maxSeqLen = attrs->GetAttrPointer<int>(3);
    auto siluScale = attrs->GetAttrPointer<float>(4);
    const gert::StorageShape* biasTensorShape = context->GetInputShape(4);

    tilingData->rankId = (rankId != nullptr) ? *rankId : 0;
    tilingData->rankSize = (rankSize != nullptr) ? *rankSize : 1;
    tilingData->maskType = (maskType != nullptr) ? *maskType : 0;
    tilingData->maxSeqLen = (maxSeqLen != nullptr) ? *maxSeqLen : 0;
    tilingData->siluScale = (siluScale != nullptr) ? *siluScale : 1.0f;
    tilingData->enableBias = (biasTensorShape == nullptr) ? 0 : 1;

    // set tiling core
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(coreNum);

    // high level api tiling
    matmul_tiling::DataType dataType;
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        dataType = matmul_tiling::DataType::DT_FLOAT;
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    } else {
        dataType = matmul_tiling::DataType::DT_BFLOAT16;
    }

    // workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * d * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem) * sizeof(float);
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize;

    // apply qk
    matmul_tiling::MatmulApiTiling qkMatmul(ascendcPlatform);
    qkMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    tilingData->blockHeight = BLOCK_HEIGHT;

    qkMatmul.SetOrgShape(BLOCK_HEIGHT, BLOCK_HEIGHT, d);
    qkMatmul.SetShape(BLOCK_HEIGHT, BLOCK_HEIGHT, d);
    qkMatmul.SetBias(false);
    qkMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling svMatmul(ascendcPlatform);
    svMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    svMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    svMatmul.SetOrgShape(BLOCK_HEIGHT, d, BLOCK_HEIGHT);
    svMatmul.SetShape(BLOCK_HEIGHT, d, BLOCK_HEIGHT);
    svMatmul.SetBias(false);
    svMatmul.SetBufferSpace(-1, -1, -1);

    if (qkMatmul.GetTiling(tilingData->qkMatmul) == -1 || svMatmul.GetTiling(tilingData->svMatmul) == -1) {
        fprintf(stderr, "[TILING] ERROR: GetTiling failed\n");
        return GRAPH_FAILED;
    }

    tilingData->qkBaseM = tilingData->qkMatmul.baseM;
    tilingData->qkBaseN = tilingData->qkMatmul.baseN;
    tilingData->svBaseM = tilingData->svMatmul.baseM;
    tilingData->svBaseN = tilingData->svMatmul.baseN;

    // set tiling key
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        context->SetTilingKey(FLOAT_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        context->SetTilingKey(FLOAT16_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_BF16) {
        context->SetTilingKey(BF16_TILING_KEY);
    } else {
        fprintf(stderr, "[TILING] ERROR: unsupported data type\n");
        return GRAPH_FAILED;
    }

    // communication (mc2 allgather)
    auto group = attrs->GetAttrPointer<char>(5);
    uint32_t opType = 6;  // allgather
    std::string algConfig = "AllGather=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    return ge::GRAPH_SUCCESS;
}

struct HstuDenseForwardCompileInfo {};
ge::graphStatus TilingParseForHstuDenseForward(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HstuDenseForward)
    .Tiling(HstuDenseForwardTilingFunc)
    .TilingParse<HstuDenseForwardCompileInfo>(TilingParseForHstuDenseForward);
}  // namespace optiling

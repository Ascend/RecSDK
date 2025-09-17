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
#include "matmul_check.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_normal_v200.h"
using namespace MatmulTilingCheck;

namespace HstuDenseForward {
REGISTER_POLICY(LAYOUT_TYPE::NORMALV200, std::make_shared<TilingPolicyNormalv200>());
bool TilingPolicyNormalv200::GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim)
{
    static const ShapeRange seqRange(128, 4096, BLOCK_HEIGHT, "seq size");
    static const ShapeRange batchRange(1, MAX_BATCH_SIZE, 1, "batch size");
    static const ShapeRange dimRange(16, 128, 16, "dim size");
    static const ShapeRange headRange(1, 8, 1, "head num");

    if (!seqRange.Check(seqLen)) {
        return false;
    }

    if (!batchRange.Check(batchSize)) {
        return false;
    }

    if (!headRange.Check(headNum)) {
        return false;
    }

    if (!dimRange.Check(dim)) {
        return false;
    }

    return true;
}

bool TilingPolicyNormalv200::TilingHeighLevelApi(gert::TilingContext* context,
                                                 optiling::HstuDenseForwardTilingData &tiling)
{
    int64_t dim = tiling.get_dim();
    matmul_tiling::DataType dataType;
    OPS_LOG_E_IF_NULL("query", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * dim * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem) * sizeof(float);
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize;

    // apply qk
    matmul_tiling::MatmulApiTiling qkMatmul(ascendPlatform);
    qkMatmul.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
    qkMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    tiling.set_blockHeight(BLOCK_HEIGHT);

    qkMatmul.SetOrgShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetBias(false);
    qkMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling svMatmul(ascendPlatform);
    svMatmul.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    svMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    svMatmul.SetOrgShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    svMatmul.SetShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    svMatmul.SetBias(false);
    svMatmul.SetBufferSpace(-1, -1, -1);

    if (qkMatmul.GetTiling(tiling.qkMatmul) == -1 || svMatmul.GetTiling(tiling.svMatmul) == -1) {
        return false;
    }

    auto findResult = matmul_tiling::DTYPE_BYTE_TAB.find(dataType);
    if (findResult == matmul_tiling::DTYPE_BYTE_TAB.end()) {
        OPS_LOG_E("", "dataType not in DTYPE_BYTE_TAB");
        return ge::GRAPH_FAILED;
    }
    int dataTypeLength = findResult->second;
    if (!CheckBaseMNK(tiling.qkMatmul, dataTypeLength, dataTypeLength) ||
        !CheckBaseMNK(tiling.svMatmul, dataTypeLength, sizeof(float))) {
        return ge::GRAPH_FAILED;
    }

    int qkTransLength = tiling.qkMatmul.get_transLength();
    int svTransLength = tiling.svMatmul.get_transLength();
    int transLength = qkTransLength > svTransLength ? qkTransLength : svTransLength;
    tiling.set_tmpUbSize(transLength);

    tiling.set_qkBaseM(tiling.qkMatmul.get_baseM());
    tiling.set_qkBaseN(tiling.qkMatmul.get_baseN());

    tiling.set_svBaseM(tiling.svMatmul.get_baseM());
    tiling.set_svBaseN(tiling.svMatmul.get_baseN());

    return true;
}

bool TilingPolicyNormalv200::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    OPS_LOG_E_IF_NULL("QShape", context->GetInputTensor(0), return false);
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT16) {
        context->SetTilingKey(FLOAT16_TILING_KEY);
    } else {
        OPS_LOG_E("", "invalid datatype, only support fp16.\n");
        return false;
    }

    return true;
}
}
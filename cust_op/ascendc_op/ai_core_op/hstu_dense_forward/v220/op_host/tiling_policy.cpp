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
#include "matmul_check.h"
#include "tiling_policy.h"
using namespace MatmulTilingCheck;

namespace HstuDenseForward {

ShapeRange::ShapeRange(int64_t lbound, int64_t ubound, int64_t mutiple, const char *name)
{
    this->lbound = lbound;
    this->ubound = ubound;
    this->mutiple = mutiple;
    this->name = name;
}

bool ShapeRange::Check(int64_t val) const
{
    OPS_CHECK((val < lbound || val > ubound || val % mutiple != 0),
              OPS_LOG_E("", "%s must meet range[%lld %lld] and mutiple of [%lld]. but get value %lld\n", name, lbound,
                        ubound, mutiple, val),
              return false);
    return true;
}

bool QKVShapeCheck(gert::TilingContext* context, int qkvDim)
{
    OPS_LOG_E_IF_NULL("QShape", context->GetInputShape(INDEX_T::INDEX_0), return false);
    OPS_LOG_E_IF_NULL("KShape", context->GetInputShape(INDEX_T::INDEX_1), return false);
    OPS_LOG_E_IF_NULL("VShape", context->GetInputShape(INDEX_T::INDEX_2), return false);

    auto QShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();
    auto KShape = context->GetInputShape(INDEX_T::INDEX_1)->GetStorageShape();
    auto VShape = context->GetInputShape(INDEX_T::INDEX_2)->GetStorageShape();
    int dim = QShape.GetDimNum();
    bool sameShape = (QShape == KShape && KShape == VShape);

    OPS_CHECK(!sameShape, OPS_LOG_E("", "QKV shape not same."), return false);
    OPS_CHECK(dim != qkvDim, OPS_LOG_E("", "Jagged QKV dim should be %d, but got %d", qkvDim, dim), return false);
    return true;
}

ge::graphStatus TilingPolicy::InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *queryShape = context->GetInputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(queryShape, return ge::GRAPH_FAILED);

    gert::Shape *attenOutputShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(attenOutputShape, return ge::GRAPH_FAILED);
    *attenOutputShape = *queryShape;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::InferDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::TilingProcess(gert::TilingContext *context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);

    optiling::HstuDenseForwardTilingData tiling;

    // step0: check platform is support
    OPS_CHECK(!CheckIsSupport(context), OPS_LOG_E("", "CheckIsSupport is failed."), return ge::GRAPH_FAILED);

    // step1: get attribute
    OPS_CHECK(!TilingAttribute(context, tiling), OPS_LOG_E("", "TilingAttribute is failed.\n"),
              return ge::GRAPH_FAILED);

    // step2: get key shape form input
    OPS_CHECK(!TilingShape(context, tiling), OPS_LOG_E("", "TilingShape is failed.\n"), return ge::GRAPH_FAILED);

    // step3: tiling core
    OPS_CHECK(!TilingCore(context, tiling), OPS_LOG_E("", "TilingCore is failed.\n"), return ge::GRAPH_FAILED);

    // step4: hight level api tiling
    OPS_CHECK(!TilingHeighLevelApi(context, tiling), OPS_LOG_E("", "TilingHeight is failed.\n"),
              return ge::GRAPH_FAILED);

    // step5: set tiling key
    OPS_CHECK(!TilingKeySet(context, tiling), OPS_LOG_E("", "TilingKeySet is failed.\n"), return ge::GRAPH_FAILED);

    // step6: tiling save to buffer
    OPS_CHECK(!TilingSaveToBuffer(context, tiling), OPS_LOG_E("", "TilingSaveToBuffer is failed.\n"),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool TilingPolicy::TilingSaveToBuffer(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling)
{
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return true;
}

bool TilingPolicy::CheckIsSupport(gert::TilingContext *context)
{
    return true;
}

bool TilingPolicy::TilingShape(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling)
{
    // base unrealized
    return false;
}

bool TilingPolicy::GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim)
{
    static const ShapeRange SEQ_RANGE(1, 20480, 1, "seq size");
    static const ShapeRange BATCH_RANGE(1, MAX_BATCH_SIZE, 1, "batch size");
    static const ShapeRange DIM_RANGE(16, 512, 16, "dim size");
    static const ShapeRange HEAD_RANGE(2, 8, 2, "head num");

    if (!SEQ_RANGE.Check(seqLen)) {
        return false;
    }

    if (!BATCH_RANGE.Check(batchSize)) {
        return false;
    }

    if (!HEAD_RANGE.Check(headNum)) {
        return false;
    }

    if (!DIM_RANGE.Check(dim)) {
        return false;
    }

    return true;
}

bool TilingPolicy::TilingAttribute(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const uint32_t *maskType = attrs->GetAttrPointer<uint32_t>(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(maskType, return false);

    const uint32_t *maxSeqLen = attrs->GetAttrPointer<uint32_t>(INDEX_T::INDEX_1);
    OPS_CHECK_PTR_NULL(maxSeqLen, return false);

    const float *siluScale = attrs->GetAttrPointer<float>(INDEX_T::INDEX_2);
    OPS_CHECK_PTR_NULL(siluScale, return false);

    auto biasTensor = context->GetOptionalInputTensor(INDEX_T::INDEX_4);
    if (biasTensor == nullptr) {
        tiling.set_enableBias(0);
    } else {
        tiling.set_enableBias(1);
    }

    tiling.set_maskType(*maskType);
    tiling.set_siluScale(*siluScale);
    tiling.set_maxSeqLen(*maxSeqLen);
    return true;
}

bool TilingPolicy::TilingHeighLevelApi(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling)
{
    int64_t dim = tiling.get_dim();

    matmul_tiling::DataType dataType;
    OPS_LOG_E_IF_NULL("query", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        dataType = matmul_tiling::DataType::DT_FLOAT;
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    } else {
        dataType = matmul_tiling::DataType::DT_BFLOAT16;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
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
    qkMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    tiling.set_blockHeight(BLOCK_HEIGHT);

    qkMatmul.SetOrgShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetBias(false);
    qkMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling svMatmul(ascendPlatform);
    svMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
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

    tiling.set_qkBaseM(tiling.qkMatmul.get_baseM());
    tiling.set_qkBaseN(tiling.qkMatmul.get_baseN());

    tiling.set_svBaseM(tiling.svMatmul.get_baseM());
    tiling.set_svBaseN(tiling.svMatmul.get_baseN());

    return true;
}

bool TilingPolicy::TilingCore(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(coreNum);
    return true;
}

bool TilingPolicy::TilingKeySet(gert::TilingContext *context, optiling::HstuDenseForwardTilingData &tiling)
{
    // base unrealized
    return false;
}

void TilingPolicy::DumpTiling(optiling::HstuDenseForwardTilingData &tiling)
{
    OPS_LOG_D("batchSize = %ld\n", tiling.get_batchSize());
    OPS_LOG_D("seqLen = %ld\n", tiling.get_seqLen());
    OPS_LOG_D("headNum = %ld\n", tiling.get_headNum());
    OPS_LOG_D("dim = %ld\n", tiling.get_dim());

    OPS_LOG_D("enableBias = %d\n", tiling.get_enableBias());
    OPS_LOG_D("maskType = %d\n", tiling.get_maskType());
    OPS_LOG_D("maxSeqLen = %d\n", tiling.get_maxSeqLen());
    OPS_LOG_D("siluScale = %f\n", tiling.get_siluScale());
}

}
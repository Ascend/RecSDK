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
#include "../op_kernel/hstu_kernel_tiling_key.h"
#include "tiling_policy.h"


using namespace MatmulTilingCheck;

namespace HstuDenseForward {

ShapeRange::ShapeRange(int64_t lbound, int64_t ubound, int64_t multiple, const char* name)
{
    this->lbound = lbound;
    this->ubound = ubound;
    this->multiple = multiple;
    this->name = name;
}

bool ShapeRange::Check(int64_t val) const
{
    OPS_CHECK((val < lbound || val > ubound || val % multiple != 0),
              OPS_LOG_E("", "%s must meet range[%lld %lld] and multiple of [%lld]. but get value %lld\n", name, lbound,
                        ubound, multiple, val),
              return false);
    return true;
}

ge::graphStatus TilingPolicy::InferShape(gert::InferShapeContext* context)
{
    const gert::Shape *queryShape = context->GetInputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(queryShape, return ge::GRAPH_FAILED);

    gert::Shape *attenOutputShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(attenOutputShape, return ge::GRAPH_FAILED);
    *attenOutputShape = *queryShape;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::TilingProcess(gert::TilingContext* context)
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
    // step7: set workspace
    OPS_CHECK(!TilingWorkSpace(context, tiling), OPS_LOG_E("", "Set workspace size is failed.\n"),
    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool TilingPolicy::TilingSaveToBuffer(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return false);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return true;
}

bool TilingPolicy::CheckIsSupport(gert::TilingContext* context)
{
    return true;
}

bool TilingPolicy::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    // base unrealized
    return false;
}

bool TilingPolicy::GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim, bool dimAlign)
{
    auto dimStep = dimAlign ? 16 : 1; // 16 means align to C0_size 1 means not align
    const ShapeRange SEQ_RANGE(1, 20480, 1, "seq size");
    const ShapeRange BATCH_RANGE(1, MAX_BATCH_SIZE, 1, "batch size");
    const ShapeRange DIM_RANGE(dimStep, 512, dimStep, "dim size"); // 非对齐情况下dim范围[1,512]
    const ShapeRange HEAD_RANGE(1, 16, 1, "head num");

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

bool TilingPolicy::TilingAttribute(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const uint32_t *maskType = attrs->GetAttrPointer<uint32_t>(ATTR_INDEX_T::MASKTYPE_INDEX);
    OPS_CHECK_PTR_NULL(maskType, return false);

    const uint32_t *maxSeqLen = attrs->GetAttrPointer<uint32_t>(ATTR_INDEX_T::MAX_SEQ_Q_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLen, return false);

    const uint32_t *maxSeqLenk = attrs->GetAttrPointer<uint32_t>(ATTR_INDEX_T::MAX_SEQ_K_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLenk, return false);

    const float *siluScale = attrs->GetAttrPointer<float>(ATTR_INDEX_T::SILU_SCALE_INDEX);
    OPS_CHECK_PTR_NULL(siluScale, return false);

    const float *alpha = attrs->GetAttrPointer<float>(ATTR_INDEX_T::ALPHA_INDEX);
    OPS_CHECK_PTR_NULL(alpha, return false);

    const bool *deterministic = attrs->GetAttrPointer<bool>(ATTR_INDEX_T::DETERMINISTIC_INDEX);
    OPS_CHECK_PTR_NULL(deterministic, return false);

    const uint32_t *targetGroupSize = attrs->GetAttrPointer<uint32_t>(ATTR_INDEX_T::TARGET_GROUP_SIZE_INDEX);
    OPS_CHECK_PTR_NULL(targetGroupSize, return false);

    auto biasTensor = context->GetOptionalInputTensor(INPUT_INDEX_T::ATTN_BIAS_INDEX);
    if (biasTensor == nullptr) {
        tiling.set_enableBias(0);
    } else {
        tiling.set_enableBias(1);
    }

    tiling.set_maskType(*maskType);
    tiling.set_siluScale(*siluScale);
    tiling.set_alpha(*alpha);
    tiling.set_maxSeqLen(*maxSeqLen);
    tiling.set_maxSeqLenq(*maxSeqLen);
    tiling.set_maxSeqLenk(*maxSeqLenk);
    tiling.set_targetGroupSize(*targetGroupSize);
    tiling.set_deterministic(*deterministic);
    return true;
}

bool TilingPolicy::TilingWorkSpace(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return false);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * MAX_DIM * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem) * sizeof(float);
    auto socVersion = ascendPlatform.GetSocVersion();
    // A5平台需要额外的FP8 workspace
    if (socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        workspaceSize += oneCoreMidElem * 1;
    }
    int64_t syncSize = coreNum * VCORE_NUM_IN_ONE_AIC * DATA_ALIGN_BYTES;
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize + syncSize;
    return true;
}

bool TilingPolicy::TilingHeighLevelApi(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    tiling.set_blockHeight(BLOCK_HEIGHT);

    matmul_tiling::DataType dataType;
    OPS_LOG_E_IF_NULL("query", context->GetInputTensor(INPUT_INDEX_T::Q_INDEX), return false);
    ge::DataType qTypeGe = context->GetInputTensor(INPUT_INDEX_T::Q_INDEX)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        dataType = matmul_tiling::DataType::DT_FLOAT;
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    } else {
        dataType = matmul_tiling::DataType::DT_BFLOAT16;
    }

    return true;
}

bool TilingPolicy::TilingCore(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(coreNum);
    return true;
}

bool TilingPolicy::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    // base unrealized
    return false;
}

bool TilingPolicy::TilingKeySetImpl(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling,
    uint32_t typeTilingKey)
{
    // A2/A3默认为false，在A5上可以选择qk结果是否使用ub，通过(tiling.get_blockHeight() == BLOCK_HEIGHT)判断
    bool isQkUseUb = false;
    bool enableBias = tiling.get_enableBias();
    bool enableDeteministic = tiling.get_deterministic();
    uint32_t maskType = tiling.get_maskType();
    uint32_t maskedType = maskType & 0x3;
    // 组合tiling key：
    
    const uint64_t tilingKey = GET_TPL_TILING_KEY(maskedType, enableBias, isQkUseUb,
                                                  typeTilingKey, enableDeteministic);
    context->SetTilingKey(tilingKey);

    return true;
}

void TilingPolicy::DumpTiling(optiling::HstuDenseForwardTilingData& tiling)
{
    OPS_LOG_D("batchSize = %ld\n", tiling.get_batchSize());
    OPS_LOG_D("seqLen = %ld\n", tiling.get_seqLen());
    OPS_LOG_D("headNum = %ld\n", tiling.get_headNum());
    OPS_LOG_D("dim = %ld\n", tiling.get_dim());

    OPS_LOG_D("enableBias = %d\n", tiling.get_enableBias());
    OPS_LOG_D("maskType = %d\n", tiling.get_maskType());
    OPS_LOG_D("maxSeqLen = %d\n", tiling.get_maxSeqLen());
    OPS_LOG_D("siluScale = %f\n", tiling.get_siluScale());
    OPS_LOG_D("alpha = %f\n", tiling.get_alpha());
    OPS_LOG_D("deterministic = %f\n", tiling.get_deterministic());
}

}

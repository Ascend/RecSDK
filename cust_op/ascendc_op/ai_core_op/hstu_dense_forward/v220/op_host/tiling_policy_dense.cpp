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
#include "tiling_policy_dense.h"

using namespace HstuForward;

namespace HstuDenseForward {

ge::graphStatus TilingPolicyDense::TilingProcess(gert::TilingContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);

    optiling::HstuDenseForwardTilingData tiling;

    // step1: get attribute
    OPS_CHECK(!TilingAttribute(context, tiling), OPS_LOG_E("", "TilingAttribute is failed.\n"),
              return ge::GRAPH_FAILED);

    // step2: get key shape form input
    OPS_CHECK(!TilingShape(context, tiling), OPS_LOG_E("", "TilingShape is failed.\n"), return ge::GRAPH_FAILED);

    // step3: tiling core
    OPS_CHECK(!TilingCore(context), OPS_LOG_E("", "TilingCore is failed.\n"), return ge::GRAPH_FAILED);

    // step4: set tiling key
    OPS_CHECK(!TilingKeySet(context, tiling), OPS_LOG_E("", "TilingKeySet is failed.\n"), return ge::GRAPH_FAILED);

    // step5: tiling save to buffer
    OPS_CHECK(!TilingSaveToBuffer(context, tiling), OPS_LOG_E("", "TilingSaveToBuffer is failed.\n"),
              return ge::GRAPH_FAILED);

    // step6: set workspace
    OPS_CHECK(!TilingWorkSpace(context), OPS_LOG_E("", "Set workspace size is failed.\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool TilingPolicyDense::TilingAttribute(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const uint32_t *maskType = attrs->GetAttrPointer<uint32_t>(DENSE_ATTR_INDEX_T::MASKTYPE_INDEX);
    OPS_CHECK_PTR_NULL(maskType, return false);

    const uint32_t *maxSeqLen = attrs->GetAttrPointer<uint32_t>(DENSE_ATTR_INDEX_T::MAX_SEQ_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLen, return false);

    const float *siluScale = attrs->GetAttrPointer<float>(DENSE_ATTR_INDEX_T::SILU_SCALE_INDEX);
    OPS_CHECK_PTR_NULL(siluScale, return false);

    auto biasTensor = context->GetOptionalInputTensor(DENSE_INPUT_INDEX_T::ATTN_BIAS_INDEX);
    bool enableBias = (biasTensor != nullptr);
    tiling.set_enableBias(enableBias);

    tiling.set_maskType(*maskType);
    tiling.set_maxSeqLen(*maxSeqLen);
    tiling.set_siluScale(*siluScale);
    tiling.set_blockHeight(BLOCK_HEIGHT);
    return true;
}

bool TilingPolicyDense::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    auto qShape = context->GetInputShape(DENSE_INPUT_INDEX_T::Q_INDEX)->GetStorageShape();
    auto kShape = context->GetInputShape(DENSE_INPUT_INDEX_T::K_INDEX)->GetStorageShape();
    auto vShape = context->GetInputShape(DENSE_INPUT_INDEX_T::V_INDEX)->GetStorageShape();

    OPS_CHECK(qShape.GetDimNum() != NORMAL_DIM_NUM,
              OPS_LOG_E("", "Normal QKV should have 4 dimensions, but get %d", qShape.GetDimNum()), return false);
    OPS_CHECK(!(qShape == kShape && kShape == vShape), OPS_LOG_E("", "Q, K, V shape mismatch"), return false);
    
    // Q: [b, s, n, d]
    int64_t batchSize = qShape.GetDim(0);
    tiling.set_batchSize(batchSize);
    int64_t seqLen = qShape.GetDim(1);
    tiling.set_seqLen(seqLen);
    tiling.set_maxSeqLenK(seqLen);
    int64_t headNum = qShape.GetDim(2);
    tiling.set_headNum(headNum);
    int64_t dim = qShape.GetDim(3);
    int64_t headNumK = kShape.GetDim(2);
    tiling.set_headNumK(headNumK);
    tiling.set_dim(dim);
    tiling.set_vDim(dim);

    OPS_CHECK(!DenseGeneralShapeCheck(batchSize, seqLen, headNum, dim),
        OPS_LOG_E("", "QK Shape Check failed"), return false);
    return true;
}

bool TilingPolicyDense::DenseGeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim)
{
    static const ShapeRange seqRange(1, 20480, 1, "seq size");
    static const ShapeRange batchRange(1, MAX_BATCH_SIZE, 1, "batch size");
    static const ShapeRange dimRange(16, 512, 16, "dim size");
    static const ShapeRange headRange(1, 16, 1, "head num");

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

bool TilingPolicyDense::TilingHeighLevelApi(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    return true;
}

bool TilingPolicyDense::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    uint32_t typeTilingKey = NORMAL_TILING_KEY & 0x3;
    TilingKeyParam param {
        .enableBias = tiling.get_enableBias(),
        .deterministic = false,
        .maskType = tiling.get_maskType(),
        .dimQ = tiling.get_dim(),
        .dimV = tiling.get_dim(),
        .maxSeqLenQ = tiling.get_maxSeqLen(),
        .maxSeqLenK = tiling.get_maxSeqLen()
    };
    return TilingKeySetImpl(context, param, typeTilingKey);
}

bool TilingPolicyDense::TilingSaveToBuffer(gert::TilingContext* context, optiling::HstuDenseForwardTilingData& tiling)
{
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return false);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return true;
}
}  // namespace HstuDenseForward

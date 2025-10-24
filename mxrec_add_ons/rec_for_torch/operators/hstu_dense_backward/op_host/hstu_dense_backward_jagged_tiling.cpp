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


#include <vector>
#include <numeric>

#include "../../../common/common_host.h"
#include "register/op_def_registry.h"
#include "hstu_dense_backward_jagged_tiling.h"


namespace optiling {
ge::graphStatus GetJaggedAttrsInfo(const gert::RuntimeAttrs *attrs, HstuDenseBackwardTilingData &tiling)
{
    const int32_t *maskType = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MASK_TYPE_INDEX);
    OPS_CHECK_PTR_NULL(maskType, return ge::GRAPH_FAILED);

    const int32_t *maxSeqLen = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MAX_SEQ_LEN_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLen, return ge::GRAPH_FAILED);

    const float *siluScale = attrs->GetAttrPointer<float>(ATTR_INDEX_T::SILU_SCALE_INDEX);
    OPS_CHECK_PTR_NULL(siluScale, return ge::GRAPH_FAILED);

    const auto targetGroupSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::TARGET_GROUP_SIZE_INDEX);
    if (targetGroupSizePtr != nullptr) {
        tiling.set_targetGroupSize(*targetGroupSizePtr);
    } else {
        tiling.set_targetGroupSize(0);
    }

    const float *alpha = attrs->GetAttrPointer<float>(ATTR_INDEX_T::REAL_ALPHA_INDEX);
    if (alpha != nullptr) {
        tiling.set_alpha(*alpha);
    } else {
        tiling.set_alpha(1.0);
    }

    tiling.set_maskType(*maskType);
    tiling.set_maxSeqLen(*maxSeqLen);
    tiling.set_siluScale(*siluScale);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetJaggedBasicShapeInfo(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t maxSeqLen = tiling.get_maxSeqLen();

    OPS_LOG_E_IF_NULL("grad", context->GetInputShape(INPUT_INDEX_T::GRAD_INDEX), return ge::GRAPH_FAILED);
    auto gradShape = context->GetInputShape(INPUT_INDEX_T::GRAD_INDEX)->GetStorageShape();
    OPS_CHECK(gradShape.GetDimNum() != JAGGED_GRAD_DIM_NUM,
              OPS_LOG_E("", "hstu jagged backward only support input with dim %d\n", JAGGED_GRAD_DIM_NUM),
              return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("seqOffset", context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_INDEX), return ge::GRAPH_FAILED);
    auto seqOffsetShape = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_INDEX)->GetStorageShape();
    OPS_CHECK(seqOffsetShape.GetDimNum() != 1,
              OPS_LOG_E("", "hstu jagged backward only support seqOffset with dim 1\n"), return ge::GRAPH_FAILED);

    int64_t batchSize = seqOffsetShape.GetDim(INDEX_T::INDEX_0) - 1;
    OPS_CHECK((batchSize < 1 || batchSize > MAX_BATCH_SIZE),
              OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize),
              return ge::GRAPH_FAILED);

    // gradShape(bs, n, d)
    int64_t seqLen = gradShape.GetDim(INDEX_T::INDEX_0);
    int64_t headNum = gradShape.GetDim(INDEX_T::INDEX_1);
    int64_t headDim = gradShape.GetDim(INDEX_T::INDEX_2);
    int64_t biasGradSeqLen = 0;
    auto attnBiasGradShape = context->GetOutputShape(OUTPUT_INDEX_T::ATTN_BIAS_GRAD_INDEX);
    if (attnBiasGradShape != nullptr) {
        biasGradSeqLen = attnBiasGradShape->GetStorageShape().GetDim(INDEX_T::INDEX_2);
        OPS_CHECK(biasGradSeqLen < maxSeqLen, OPS_LOG_E("", "attnBiasGrad get seqLen less than maxSeqLen\n"),
                  return ge::GRAPH_FAILED);
    } else {
        biasGradSeqLen = AlignUp(maxSeqLen, static_cast<int64_t>(BLOCK_256));
        OPS_CHECK((biasGradSeqLen == 0), OPS_LOG_E("", "attnBiasGrad get seqLen error\n"), return ge::GRAPH_FAILED);
    }
    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_headNum(headNum);
    tiling.set_headDim(headDim);
    tiling.set_biasGradSeqLen(biasGradSeqLen);
    tiling.set_isNormal(0);

    OPS_CHECK(!BasicShapeCheck(batchSize, maxSeqLen, headNum, headDim), OPS_LOG_E("", "jagged shape check failed\n"),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitJaggedTilingKey(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t dataTypeLength = 0;
    ge::DataType gradType = context->GetInputTensor(INPUT_INDEX_T::GRAD_INDEX)->GetDataType();
    if (gradType == ge::DataType::DT_FLOAT) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT;
        context->SetTilingKey(JAGGED_FLOAT_TILING_KEY);
        tiling.set_blockHeight(BLOCK_128);
    } else if (gradType == ge::DataType::DT_FLOAT16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(JAGGED_FLOAT16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else if (gradType == ge::DataType::DT_BF16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(JAGGED_BF16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else {
        OPS_LOG_E("", "invalid datatype, only support float/fp16/bf16");
        return ge::GRAPH_FAILED;
    }
    tiling.set_dataTypeLength(dataTypeLength);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCore(gert::TilingContext *context,
                           HstuDenseBackwardTilingData &tiling)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingJaggedFunc(gert::TilingContext *context,
                                 const gert::RuntimeAttrs *attrs,
                                 HstuDenseBackwardTilingData &tiling)
{
    OPS_CHECK(GetJaggedAttrsInfo(attrs, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling GetJaggedAttrsInfo failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(GetJaggedBasicShapeInfo(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling GetJaggedBasicShapeInfo failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(CheckMaskTypeAndBias(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling CheckMaskTypeAndBias failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(InitJaggedTilingKey(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling InitJaggedTilingKey failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(TilingCore(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling TilingCore failed\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus JaggedInferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(INPUT_INDEX_T::Q_INDEX);
    OPS_CHECK_PTR_NULL(qShape, return ge::GRAPH_FAILED);

    // q_grad、k_grad、v_grad的shape与q一致
    gert::Shape *qGradShape = context->GetOutputShape(OUTPUT_INDEX_T::Q_GRAD_INDEX);
    OPS_CHECK_PTR_NULL(qGradShape, return ge::GRAPH_FAILED);
    qGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *kGradShape = context->GetOutputShape(OUTPUT_INDEX_T::K_GRAD_INDEX);
    OPS_CHECK_PTR_NULL(kGradShape, return ge::GRAPH_FAILED);
    kGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *vGradShape = context->GetOutputShape(OUTPUT_INDEX_T::V_GRAD_INDEX);
    OPS_CHECK_PTR_NULL(vGradShape, return ge::GRAPH_FAILED);
    vGradShape->SetDimNum(qShape->GetDimNum());

    for (size_t i = 0; i < qShape->GetDimNum(); i++) {
        qGradShape->SetDim(i, qShape->GetDim(i));
        kGradShape->SetDim(i, qShape->GetDim(i));
        vGradShape->SetDim(i, qShape->GetDim(i));
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return ge::GRAPH_FAILED);
    const int32_t *maxSeqLen = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MAX_SEQ_LEN_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLen, return ge::GRAPH_FAILED);

    const gert::Shape *seqOffsetShape = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_INDEX);
    OPS_CHECK_PTR_NULL(seqOffsetShape, return ge::GRAPH_FAILED);
    int64_t batchSize = seqOffsetShape->GetDim(INDEX_T::INDEX_0) - 1;

    gert::Shape *attnBiasGradShape = context->GetOutputShape(OUTPUT_INDEX_T::ATTN_BIAS_GRAD_INDEX);
    if (attnBiasGradShape != nullptr) {
        attnBiasGradShape->SetDimNum(BIAS_DIM_NUM);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_0, batchSize);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_1, qShape->GetDim(1));
        attnBiasGradShape->SetDim(INDEX_T::INDEX_2, *maxSeqLen);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_3, *maxSeqLen);
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

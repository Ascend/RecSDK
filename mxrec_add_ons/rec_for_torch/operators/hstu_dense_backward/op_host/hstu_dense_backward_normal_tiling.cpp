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


#include "register/op_def_registry.h"

#include "hstu_dense_backward_normal_tiling.h"

namespace optiling {
ge::graphStatus GetNormalAttrsInfo(const gert::RuntimeAttrs *attrs, HstuDenseBackwardTilingData &tiling)
{
    const int32_t *maskType = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_1);
    OPS_CHECK_PTR_NULL(maskType, return ge::GRAPH_FAILED);
    tiling.set_maskType(*maskType);

    const int32_t *maxSeqLen = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_2);
    OPS_CHECK_PTR_NULL(maxSeqLen, return ge::GRAPH_FAILED);
    tiling.set_maxSeqLen(*maxSeqLen);

    const float *siluScale = attrs->GetAttrPointer<float>(INDEX_T::INDEX_3);
    OPS_CHECK_PTR_NULL(siluScale, return ge::GRAPH_FAILED);
    tiling.set_siluScale(*siluScale);

    const auto targetGroupSizePtr = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_5);
    if (targetGroupSizePtr != nullptr)
        tiling.set_targetGroupSize(*targetGroupSizePtr);
    else
        tiling.set_targetGroupSize(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetNormalBasicShapeInfo(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int32_t maxSeqLen = tiling.get_maxSeqLen();
    auto gradShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();
    auto attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3)->GetStorageShape();
    OPS_CHECK(gradShape.GetDimNum() != GRAD_DIM_NUM,
                OPS_LOG_E("", "hstu normal backward only support input with dim %d\n", GRAD_DIM_NUM),
                return ge::GRAPH_FAILED);

    int64_t batchSize = gradShape.GetDim(INDEX_T::INDEX_0);
    int64_t seqLen = gradShape.GetDim(INDEX_T::INDEX_1);
    int64_t headNum = gradShape.GetDim(INDEX_T::INDEX_2);
    int64_t headDim = gradShape.GetDim(INDEX_T::INDEX_3);
    int32_t biasGradSeqLen = attnBiasGradShape.GetDim(INDEX_T::INDEX_2);

    OPS_CHECK(biasGradSeqLen < maxSeqLen,
                OPS_LOG_E("", "attnBiasGrad get seqLen less than maxSeqLen\n"),
                return ge::GRAPH_FAILED);

    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_headNum(headNum);
    tiling.set_headDim(headDim);
    tiling.set_biasGradSeqLen(biasGradSeqLen);
    tiling.set_isNormal(1);

    OPS_CHECK(!BasicShapeCheck(batchSize, seqLen, headNum, headDim),
        OPS_LOG_E("", "normal shape check failed\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMaskTypeAndBias(gert::TilingContext *context,
                                     HstuDenseBackwardTilingData &tiling)
{
    auto batchSize = tiling.get_batchSize();
    auto headNum = tiling.get_headNum();
    auto maxSeqLen = tiling.get_maxSeqLen();
    auto maskType = tiling.get_maskType();

    auto attnBias = context->GetOptionalInputTensor(INDEX_T::INDEX_5);
    if (attnBias == nullptr) {
        tiling.set_enableBias(0);
    } else {
        tiling.set_enableBias(1);

        auto attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3)->GetStorageShape();
        auto attnBiasShape = context->GetInputShape(INDEX_T::INDEX_5)->GetStorageShape();
        OPS_CHECK(!IsSameShape(attnBiasShape, attnBiasGradShape, BIAS_DIM_NUM),
                    OPS_LOG_E("", "attnBias shape not equal with attnBiasGrad\n"),
                    return ge::GRAPH_FAILED);
    }

    auto contextMask = context->GetOptionalInputTensor(INDEX_T::INDEX_6);
    if (contextMask == nullptr) {
        tiling.set_enableContextMask(0);
    } else {
        tiling.set_enableContextMask(1);
    }

    auto targetMask = context->GetOptionalInputTensor(INDEX_T::INDEX_6);
    if (contextMask == nullptr) {
        tiling.set_enableTargetMask(0);
    } else {
        tiling.set_enableTargetMask(1);
    }

    if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
        auto mask = context->GetOptionalInputTensor(INDEX_T::INDEX_4);
        OPS_CHECK(mask == nullptr,
                    OPS_LOG_E("", "mask can't be none when maskType is MASK_CUSTOM\n"),
                    return ge::GRAPH_FAILED);

        auto maskShape = context->GetInputShape(INDEX_T::INDEX_4)->GetStorageShape();
        OPS_CHECK(maskShape.GetDimNum() != MASK_DIM_NUM,
                    OPS_LOG_E("", "mask dim num is not %d\n", MASK_DIM_NUM),
                    return ge::GRAPH_FAILED);

        OPS_CHECK(maskShape.GetDim(INDEX_T::INDEX_0) != batchSize ||
                    maskShape.GetDim(INDEX_T::INDEX_1) != headNum ||
                    maskShape.GetDim(INDEX_T::INDEX_2) != maxSeqLen ||
                    maskShape.GetDim(INDEX_T::INDEX_3) != maxSeqLen,
                    OPS_LOG_E("", "mask shape must be {batchSize, headNum, seqLen, seqLen}\n"),
                    return ge::GRAPH_FAILED);
    } else if (IfMask(maskType, MaskType::MASK_TRIL) ||
               IfMask(maskType, MaskType::MASK_NONE)) {
        // do nothing
    } else if (IfMask(maskType, MaskType::MASK_TRIU)) {
        OPS_LOG_E("", "maskType:MASK_TRIU is not support yet\n");
        return ge::GRAPH_FAILED;
    } else {
        OPS_LOG_E("", "supported maskType list is [MASK_TRIL, MASK_NONE, MASK_CUSTOM]\n");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitNormalTilingKey(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t dataTypeLength = 0;
    ge::DataType gradType = context->GetInputTensor(INDEX_T::INDEX_0)->GetDataType();
    if (gradType == ge::DataType::DT_FLOAT) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT;
        context->SetTilingKey(FLOAT_TILING_KEY);
        tiling.set_blockHeight(BLOCK_128);
    } else if (gradType == ge::DataType::DT_FLOAT16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(FLOAT16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else if (gradType == ge::DataType::DT_BF16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(BF16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else {
        OPS_LOG_E("", "invalid datatype, only support float/fp16/bf16");
        return ge::GRAPH_FAILED;
    }
    tiling.set_dataTypeLength(dataTypeLength);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingNormalFunc(gert::TilingContext *context,
                                 const gert::RuntimeAttrs *attrs,
                                 HstuDenseBackwardTilingData &tiling)
{
    OPS_CHECK(GetNormalAttrsInfo(attrs, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "NormalTiling GetNormalAttrsInfo failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(GetNormalBasicShapeInfo(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "NormalTiling GetNormalBasicShapeInfo failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(CheckMaskTypeAndBias(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "NormalTiling CheckMaskTypeAndBias failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(InitNormalTilingKey(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "NormalTiling InitNormalTilingKey failed\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormalInferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(INDEX_T::INDEX_1);
    OPS_CHECK_PTR_NULL(qShape, return ge::GRAPH_FAILED);

    // q_grad、k_grad、v_grad的shape与q一致
    gert::Shape *qGradShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(qGradShape, return ge::GRAPH_FAILED);
    qGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *kGradShape = context->GetOutputShape(INDEX_T::INDEX_1);
    OPS_CHECK_PTR_NULL(kGradShape, return ge::GRAPH_FAILED);
    kGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *vGradShape = context->GetOutputShape(INDEX_T::INDEX_2);
    OPS_CHECK_PTR_NULL(vGradShape, return ge::GRAPH_FAILED);
    vGradShape->SetDimNum(qShape->GetDimNum());

    for (size_t i = 0; i < qShape->GetDimNum(); i++) {
        qGradShape->SetDim(i, qShape->GetDim(i));
        kGradShape->SetDim(i, qShape->GetDim(i));
        vGradShape->SetDim(i, qShape->GetDim(i));
    }

    gert::Shape *attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3);
    OPS_CHECK_PTR_NULL(attnBiasGradShape, return ge::GRAPH_FAILED);
    attnBiasGradShape->SetDimNum(BIAS_DIM_NUM);
    attnBiasGradShape->SetDim(INDEX_T::INDEX_0, qShape->GetDim(INDEX_T::INDEX_0));
    attnBiasGradShape->SetDim(INDEX_T::INDEX_1, qShape->GetDim(INDEX_T::INDEX_2));
    attnBiasGradShape->SetDim(INDEX_T::INDEX_2, qShape->GetDim(INDEX_T::INDEX_1));
    attnBiasGradShape->SetDim(INDEX_T::INDEX_3, qShape->GetDim(INDEX_T::INDEX_1));

    return ge::GRAPH_SUCCESS;
}
}
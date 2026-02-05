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

#include "check_util.h"
#include "register/op_def_registry.h"
#include "hstu_dense_backward_jagged_tiling.h"

namespace optiling {
template <bool V>
bool BasicJaggedShapeCheck(int64_t batchSize, int64_t maxSeqLen, int64_t headNum, int64_t dim)
{
    static const ShapeRange batchRange(MIN_BATCH_SIZE, MAX_BATCH_SIZE, NO_INT_MULT_REQUIRE, "batch size");
    static const ShapeRange maxSeqRange(MIN_SEQ_LENS, MAX_SEQ_LENS, NO_INT_MULT_REQUIRE, "seq size");
    static const ShapeRange headRange(MIN_HEAD_NUM, MAX_HEAD_NUM, NO_INT_MULT_REQUIRE, "head num");

    int64_t minDim;
    int64_t mulDim;
    if constexpr (V) {
        minDim = MIN_HEAD_DIM_V;
        mulDim = MULT_OF_16;
    } else {
        minDim = MIN_HEAD_DIM_QK;
        mulDim = NO_INT_MULT_REQUIRE;
    }
    static const ShapeRange dimRange(minDim, MAX_HEAD_DIM, mulDim, "dim size");

    if (!batchRange.Check(batchSize)) {
        return false;
    }

    if (!maxSeqRange.Check(maxSeqLen)) {
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

ge::graphStatus GetJaggedAttrsInfo(const gert::RuntimeAttrs *attrs, HstuJaggedBackwardTilingData &tiling)
{
    const int32_t *maskType = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MASK_TYPE_INDEX);
    OPS_CHECK_PTR_NULL(maskType, return ge::GRAPH_FAILED);
    tiling.set_maskType(*maskType);

    const int32_t *maxSeqLenQ = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MAX_SEQLEN_Q_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLenQ, return ge::GRAPH_FAILED);
    tiling.set_maxSeqLenQ(*maxSeqLenQ);

    const int32_t *maxSeqLenK = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MAX_SEQLEN_K_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLenK, return ge::GRAPH_FAILED);
    tiling.set_maxSeqLenK(*maxSeqLenK);

    const float *siluScale = attrs->GetAttrPointer<float>(ATTR_INDEX_T::SILU_SCALE_INDEX);
    OPS_CHECK_PTR_NULL(siluScale, return ge::GRAPH_FAILED);
    tiling.set_siluScale(*siluScale);

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
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetJaggedBasicShapeInfo(gert::TilingContext *context, HstuJaggedBackwardTilingData &tiling)
{
    int64_t maxSeqLenQ = tiling.get_maxSeqLenQ();
    int64_t maxSeqLenK = tiling.get_maxSeqLenK();

    OPS_LOG_E_IF_NULL("grad", context->GetInputShape(INPUT_INDEX_T::GRAD_INDEX), return ge::GRAPH_FAILED);
    auto gradShape = context->GetInputShape(INPUT_INDEX_T::GRAD_INDEX)->GetStorageShape();
    OPS_CHECK(gradShape.GetDimNum() != JAGGED_GRAD_DIM_NUM,
              OPS_LOG_E("", "hstu jagged backward only support input with dim %d\n", JAGGED_GRAD_DIM_NUM),
              return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("Q", context->GetInputShape(INPUT_INDEX_T::Q_INDEX), return ge::GRAPH_FAILED);
    auto qShape = context->GetInputShape(INPUT_INDEX_T::Q_INDEX)->GetStorageShape();
    OPS_CHECK(qShape.GetDimNum() != JAGGED_GRAD_DIM_NUM,
              OPS_LOG_E("", "hstu jagged backward only support input with dim %d\n", JAGGED_GRAD_DIM_NUM),
              return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("K", context->GetInputShape(INPUT_INDEX_T::K_INDEX), return ge::GRAPH_FAILED);
    auto kShape = context->GetInputShape(INPUT_INDEX_T::K_INDEX)->GetStorageShape();
    OPS_CHECK(kShape.GetDimNum() != JAGGED_GRAD_DIM_NUM,
              OPS_LOG_E("", "hstu jagged backward only support input with dim %d\n", JAGGED_GRAD_DIM_NUM),
              return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("V", context->GetInputShape(INPUT_INDEX_T::V_INDEX), return ge::GRAPH_FAILED);
    auto vShape = context->GetInputShape(INPUT_INDEX_T::V_INDEX)->GetStorageShape();
    OPS_CHECK(vShape.GetDimNum() != JAGGED_GRAD_DIM_NUM,
              OPS_LOG_E("", "hstu jagged backward only support input with dim %d\n", JAGGED_GRAD_DIM_NUM),
              return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("seqOffsetQ", context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX), return ge::GRAPH_FAILED);
    auto seqOffsetShapeQ = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX)->GetStorageShape();
    OPS_CHECK(seqOffsetShapeQ.GetDimNum() != 1,
              OPS_LOG_E("", "hstu jagged backward only support seqOffset with dim 1\n"), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("seqOffsetK", context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_K_INDEX), return ge::GRAPH_FAILED);
    auto seqOffsetShapeK = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_K_INDEX)->GetStorageShape();
    OPS_CHECK(seqOffsetShapeQ != seqOffsetShapeK,
              OPS_LOG_E("", "seqOffsetQ.shape != seqOffsetK.shape\n"), return ge::GRAPH_FAILED);

    int64_t batchSize = seqOffsetShapeQ.GetDim(INDEX_T::INDEX_0) - 1;
    OPS_CHECK((batchSize < 1 || batchSize > MAX_BATCH_SIZE),
              OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize),
              return ge::GRAPH_FAILED);

    // qShape(bs, n, d)
    int64_t seqLenQ = qShape.GetDim(INDEX_T::INDEX_0);
    int64_t seqLenK = kShape.GetDim(INDEX_T::INDEX_0);
    int64_t headNum = qShape.GetDim(INDEX_T::INDEX_1);
    int64_t headDimQK = qShape.GetDim(INDEX_T::INDEX_2);
    int64_t headDimV = gradShape.GetDim(INDEX_T::INDEX_2);

    tiling.set_batchSize(batchSize);
    tiling.set_seqLenQ(seqLenQ);
    tiling.set_seqLenK(seqLenK);
    tiling.set_headNum(headNum);
    tiling.set_headDimQK(headDimQK);
    tiling.set_headDimV(headDimV);
    tiling.set_isNormal(0);

    OPS_CHECK(!BasicJaggedShapeCheck<false>(batchSize, maxSeqLenQ, headNum, headDimQK),
              OPS_LOG_E("", "Q jagged shape check failed\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!BasicJaggedShapeCheck<false>(batchSize, maxSeqLenK, headNum, headDimQK),
              OPS_LOG_E("", "K jagged shape check failed\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!BasicJaggedShapeCheck<true>(batchSize, maxSeqLenK, headNum, headDimV),
              OPS_LOG_E("", "V jagged shape check failed\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitJaggedTilingKey(gert::TilingContext *context, HstuJaggedBackwardTilingData &tiling)
{
    int64_t dataTypeLength = 0;
    ge::DataType gradType = context->GetInputTensor(INPUT_INDEX_T::GRAD_INDEX)->GetDataType();
    if (gradType == ge::DataType::DT_FLOAT) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT;
        tiling.set_blockHeight(BLOCK_128);
    } else if (gradType == ge::DataType::DT_FLOAT16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        tiling.set_blockHeight(BLOCK_256);
    } else if (gradType == ge::DataType::DT_BF16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        tiling.set_blockHeight(BLOCK_256);
    } else {
        OPS_LOG_E("", "invalid datatype, only support float/fp16/bf16");
        return ge::GRAPH_FAILED;
    }
    tiling.set_dataTypeLength(dataTypeLength);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCore(gert::TilingContext *context,
                           HstuJaggedBackwardTilingData &tiling)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingJaggedFunc(gert::TilingContext *context,
                                 const gert::RuntimeAttrs *attrs,
                                 HstuJaggedBackwardTilingData &tiling)
{
    OPS_CHECK(GetJaggedAttrsInfo(attrs, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling GetJaggedAttrsInfo failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(GetJaggedBasicShapeInfo(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling GetJaggedBasicShapeInfo failed\n"), return ge::GRAPH_FAILED);

    auto maxSeqLenQ = tiling.get_maxSeqLenQ();
    auto maxSeqLenK = tiling.get_maxSeqLenK();
    OPS_CHECK(
        CheckMaskTypeAndBias<HstuJaggedBackwardTilingData>(context, tiling, maxSeqLenQ, maxSeqLenK) == ge::GRAPH_FAILED,
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

    const gert::Shape *kShape = context->GetInputShape(INPUT_INDEX_T::K_INDEX);
    OPS_CHECK_PTR_NULL(kShape, return ge::GRAPH_FAILED);

    const gert::Shape *vShape = context->GetInputShape(INPUT_INDEX_T::V_INDEX);
    OPS_CHECK_PTR_NULL(vShape, return ge::GRAPH_FAILED);

    gert::Shape *qGradShape = context->GetOutputShape(OUTPUT_INDEX_T::Q_GRAD_INDEX);
    OPS_CHECK_PTR_NULL(qGradShape, return ge::GRAPH_FAILED);
    qGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *kGradShape = context->GetOutputShape(OUTPUT_INDEX_T::K_GRAD_INDEX);
    OPS_CHECK_PTR_NULL(kGradShape, return ge::GRAPH_FAILED);
    kGradShape->SetDimNum(kShape->GetDimNum());

    gert::Shape *vGradShape = context->GetOutputShape(OUTPUT_INDEX_T::V_GRAD_INDEX);
    OPS_CHECK_PTR_NULL(vGradShape, return ge::GRAPH_FAILED);
    vGradShape->SetDimNum(vShape->GetDimNum());

    for (size_t i = 0; i < qShape->GetDimNum(); i++) {
        qGradShape->SetDim(i, qShape->GetDim(i));
        kGradShape->SetDim(i, kShape->GetDim(i));
        vGradShape->SetDim(i, vShape->GetDim(i));
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return ge::GRAPH_FAILED);
    const int32_t *maxSeqLenQ = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MAX_SEQLEN_Q_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLenQ, return ge::GRAPH_FAILED);
    const int32_t *maxSeqLenK = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_T::MAX_SEQLEN_K_INDEX);
    OPS_CHECK_PTR_NULL(maxSeqLenK, return ge::GRAPH_FAILED);

    const gert::Shape *seqOffsetShape = context->GetInputShape(INPUT_INDEX_T::SEQ_OFFSET_Q_INDEX);
    OPS_CHECK_PTR_NULL(seqOffsetShape, return ge::GRAPH_FAILED);
    int64_t batchSize = seqOffsetShape->GetDim(INDEX_T::INDEX_0) - 1;

    gert::Shape *attnBiasGradShape = context->GetOutputShape(OUTPUT_INDEX_T::ATTN_BIAS_GRAD_INDEX);
    if (attnBiasGradShape != nullptr) {
        attnBiasGradShape->SetDimNum(BIAS_DIM_NUM);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_0, batchSize);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_1, qShape->GetDim(1));
        attnBiasGradShape->SetDim(INDEX_T::INDEX_2, *maxSeqLenQ);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_3, *maxSeqLenK);
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

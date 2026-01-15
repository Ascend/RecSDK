#ifndef CHECK_UTIL_H
#define CHECK_UTIL_H

#include "register/op_def_registry.h"
#include "hstu_dense_backward_tiling_common.h"
namespace optiling {

template <typename TilingData>
ge::graphStatus CheckMaskTypeAndBias(gert::TilingContext* context, TilingData& tiling)
{
    auto batchSize = tiling.get_batchSize();
    auto headNum = tiling.get_headNum();
    auto maxSeqLen = tiling.get_maxSeqLen();
    auto maskType = tiling.get_maskType();

    auto attnBias = context->GetOptionalInputTensor(INPUT_INDEX_T::ATTN_BIAS_INDEX);
    if (attnBias == nullptr) {
        tiling.set_enableBias(0);
    } else {
        tiling.set_enableBias(1);

        auto attnBiasGradShape = context->GetOutputShape(OUTPUT_INDEX_T::ATTN_BIAS_GRAD_INDEX)->GetStorageShape();
        auto attnBiasShape = context->GetInputShape(INPUT_INDEX_T::ATTN_BIAS_INDEX)->GetStorageShape();
        OPS_CHECK(!IsSameShape(attnBiasShape, attnBiasGradShape, BIAS_DIM_NUM),
                  OPS_LOG_E("", "attnBias shape not equal with attnBiasGrad\n"), return ge::GRAPH_FAILED);
    }

    auto contextMask = context->GetOptionalInputTensor(INPUT_INDEX_T::NUM_CONTEXT_INDEX);
    if (contextMask == nullptr) {
        tiling.set_enableContextMask(0);
    } else {
        tiling.set_enableContextMask(1);
    }

    auto targetMask = context->GetOptionalInputTensor(INPUT_INDEX_T::NUM_TARGET_INDEX);
    if (targetMask == nullptr) {
        tiling.set_enableTargetMask(0);
    } else {
        tiling.set_enableTargetMask(1);
    }

    if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
        auto mask = context->GetOptionalInputTensor(INPUT_INDEX_T::MASK_INDEX);
        OPS_CHECK(mask == nullptr, OPS_LOG_E("", "mask can't be none when maskType is MASK_CUSTOM\n"),
                  return ge::GRAPH_FAILED);

        auto maskShape = context->GetInputShape(INPUT_INDEX_T::MASK_INDEX)->GetStorageShape();
        OPS_CHECK(maskShape.GetDimNum() != MASK_DIM_NUM, OPS_LOG_E("", "mask dim num is not %d\n", MASK_DIM_NUM),
                  return ge::GRAPH_FAILED);

        OPS_CHECK(maskShape.GetDim(INDEX_T::INDEX_0) != batchSize || maskShape.GetDim(INDEX_T::INDEX_1) != headNum ||
                      maskShape.GetDim(INDEX_T::INDEX_2) != maxSeqLen ||
                      maskShape.GetDim(INDEX_T::INDEX_3) != maxSeqLen,
                  OPS_LOG_E("", "mask shape must be {batchSize, headNum, seqLen, seqLen}\n"), return ge::GRAPH_FAILED);
    } else if (IfMask(maskType, MaskType::MASK_TRIL) || IfMask(maskType, MaskType::MASK_NONE)) {
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
}  // namespace optiling

#endif
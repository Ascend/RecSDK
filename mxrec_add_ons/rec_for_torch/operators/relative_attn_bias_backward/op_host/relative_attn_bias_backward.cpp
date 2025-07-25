/**
 * @file relative_attn_bias_backward.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#include <cmath>
#include "relative_attn_bias_backward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../common/ops_log.h"

constexpr int32_t RESERVER_UB_SIZE = (20 * 1024);
constexpr int32_t DATA_ALIGN_BYTES = 32;
constexpr uint8_t NUM_BUFFER = 2;

// input index
constexpr int INPUT_GRAD_INDEX = 0;
constexpr int BUCKET_TIMESTAMPS_INDEX = 1;
// output index
constexpr int TIMESTAMPS_WEIGHTS_GRAD_INDEX = 0;
// attr index
constexpr int NUM_BUCKET_INDEX = 0;
// output dim
constexpr int TSW_GRAD_OUT_DIM = 2;
constexpr int RAB_TIME_GRAD_DIM = 4;
constexpr int BUCKET_TIMESTAMPS_DIM = 3;

constexpr int DIM0 = 0;
constexpr int DIM1 = 1;
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;

namespace optiling {
static ge::graphStatus TimeTilingFunc(RelativeAttnBiasBackwardTilingData& tilingData, gert::TilingContext* context)
{
    // 获取、校验必要shape数据
    auto gradShape = context->GetInputShape(TIMESTAMPS_WEIGHTS_GRAD_INDEX)->GetStorageShape();  // grad(n, b, 2s, 2s)
    auto indexShape = context->GetInputShape(BUCKET_TIMESTAMPS_INDEX)->GetStorageShape();       // grad(b, 2s, 2s)

    int numBuckets = *context->GetAttrs()->GetInt(NUM_BUCKET_INDEX);
    int numLayer = gradShape.GetDim(DIM0);
    int batchsize = gradShape.GetDim(DIM1);
    int s = gradShape.GetDim(DIM2);
    int s2 = gradShape.GetDim(DIM3);

    int indexBatchsize = indexShape.GetDim(DIM0);
    int indexS1 = indexShape.GetDim(DIM1);
    int indexS2 = indexShape.GetDim(DIM2);

    OPS_CHECK(gradShape.GetDimNum() != RAB_TIME_GRAD_DIM, OPS_LOG_E("Tiling Debug", "Grad shape is invalid."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(indexShape.GetDimNum() != BUCKET_TIMESTAMPS_DIM,
              OPS_LOG_E("Tiling Debug", "bucket_timestamps shape is invalid."), return ge::GRAPH_FAILED);
    OPS_CHECK(numBuckets <= 0, OPS_LOG_E("Tiling Debug", "NumBuckets is invalid."), return ge::GRAPH_FAILED);
    OPS_CHECK(numLayer <= 0, OPS_LOG_E("Tiling Debug", "Numlayer is invalid."), return ge::GRAPH_FAILED);
    OPS_CHECK(batchsize <= 0 || batchsize != indexBatchsize, OPS_LOG_E("Tiling Debug", "Batchsize is invalid."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(s <= 0 || s != s2 || s != indexS1 || s != indexS2, OPS_LOG_E("Tiling Debug", "Sequence len is invalid."),
              return ge::GRAPH_FAILED);

    tilingData.set_numBuckets(numBuckets);
    tilingData.set_numLayer(numLayer);
    tilingData.set_bs(batchsize);
    tilingData.set_s(s);
    // 获取计算中使用的步长等数据
    uint64_t ub;
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;
    // 获取数据类型
    auto gradDataType = context->GetInputTensor(TIMESTAMPS_WEIGHTS_GRAD_INDEX)->GetDataType();
    auto indexDataType = context->GetInputTensor(BUCKET_TIMESTAMPS_INDEX)->GetDataType();
    int gradSize = ge::GetSizeByDataType(gradDataType);
    int indexSize = ge::GetSizeByDataType(indexDataType);
    OPS_CHECK(gradSize == 0 || indexSize == 0, OPS_LOG_E("Tiling Debug", "Invalid data type."),
              return ge::GRAPH_FAILED);
    // 去除tswGrad所需ub
    ub = ub - numBuckets * numLayer * sizeof(float);
    // 计算单次处理的block大小
    int stride;
    if (gradDataType == ge::DataType::DT_FLOAT16) {
        stride = ub / (indexSize + gradSize + sizeof(float));  // 申请额外内存做cast
    } else {
        stride = ub / (indexSize + sizeof(float));
    }
    tilingData.set_gradDataType(gradDataType);
    tilingData.set_indexDataType(indexDataType);
    tilingData.set_timeStride(stride);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("rabTimeGrad", context->GetInputShape(TIMESTAMPS_WEIGHTS_GRAD_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("bucketTimestamps", context->GetInputShape(BUCKET_TIMESTAMPS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E("Tiling Debug", "Core num is 0."), return ge::GRAPH_FAILED);
    RelativeAttnBiasBackwardTilingData tilingData;
    auto ret = TimeTilingFunc(tilingData, context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    context->SetBlockDim(coreNum);
    auto rowTilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("GetRawTilingData", rowTilingData, return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rowTilingData->GetData(), rowTilingData->GetCapacity());
    rowTilingData->SetDataSize(tilingData.GetDataSize());
    return ret;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("timestampsWeightsGrad", context->GetOutputShape(TIMESTAMPS_WEIGHTS_GRAD_INDEX), \
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("rabTimeGrad", context->GetInputShape(INPUT_GRAD_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    gert::Shape* tswGradOutShape = context->GetOutputShape(TIMESTAMPS_WEIGHTS_GRAD_INDEX);
    const gert::Shape* tsGradShape = context->GetInputShape(INPUT_GRAD_INDEX);  // (n, b, 2s, 2s)
    int n = tsGradShape->GetDim(DIM0);
    int numBuckets = *context->GetAttrs()->GetInt(NUM_BUCKET_INDEX);

    tswGradOutShape->SetDimNum(TSW_GRAD_OUT_DIM);
    tswGradOutShape->SetDim(DIM0, n);
    tswGradOutShape->SetDim(DIM1, numBuckets);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RelativeAttnBiasBackward : public OpDef {
public:
    explicit RelativeAttnBiasBackward(const char* name) : OpDef(name)
    {
        this->Input("rab_time_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bucket_timestamps")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("timestamps_weights_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_buckets").Int();

        this->SetInferShape(ge::InferShape);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910", aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(RelativeAttnBiasBackward);

}  // namespace ops
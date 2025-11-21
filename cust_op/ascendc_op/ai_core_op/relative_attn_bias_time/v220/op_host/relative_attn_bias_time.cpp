/**
 * @file relative_attn_bias_time.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#include <cmath>

#include "relative_attn_bias_time_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

constexpr int32_t RESERVER_UB_SIZE = (20 * 1024);
constexpr int32_t DATA_ALIGN_BYTES = 32;

// input index
constexpr int TIMESTAMPS_INDEX = 0;
constexpr int TIMESTAMPS_WEIGHTS_INDEX = 1;
// output index
constexpr int RAB_TIME_INDEX = 0;
// attr index
constexpr int BUCKET_DIV_INDEX = 0;
// input/output dim
constexpr int TIMESTAMPS_DIM = 2;
constexpr int TIMESTAMPS_WEIGHTS_DIM = 2;
constexpr int RAB_TIME_OUT_DIM = 6;
constexpr int DIM_PLACE_HOLDER = 1;
constexpr int DIM0 = 0;
constexpr int DIM1 = 1;
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;
constexpr int DIM4 = 4;
constexpr int DIM5 = 5;
// constrain of params
constexpr int MAX_NUM_LAYER = 20;
constexpr int MAX_NUM_BUCKET = 128;
constexpr int MAX_SEQ_LEN = 4300;

namespace optiling {
static ge::graphStatus TimeTilingFunc(RelativeAttnBiasTimeTilingData& tilingData, gert::TilingContext* context)
{
    auto tsShape = context->GetInputShape(TIMESTAMPS_INDEX)->GetStorageShape();           // (b, s)
    auto tswShape = context->GetInputShape(TIMESTAMPS_WEIGHTS_INDEX)->GetStorageShape();  // (num_layer, num_buckets)

    int64_t batchsize = tsShape.GetDim(DIM0);    // (b, s)
    int64_t s = tsShape.GetDim(DIM1);            // (b, s)
    int64_t numLayer = tswShape.GetDim(DIM0);    // (num_layer, num_buckets)
    int64_t numBuckets = tswShape.GetDim(DIM1);  // (num_layer, num_buckets)
    float divs = *context->GetAttrs()->GetFloat(BUCKET_DIV_INDEX);
    float clampMax = exp((numBuckets - 1) * divs);

    OPS_CHECK(tsShape.GetDimNum() != TIMESTAMPS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for timestamps is %d, but the actual dim is %d.",
                        TIMESTAMPS_DIM, tsShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(tswShape.GetDimNum() != TIMESTAMPS_WEIGHTS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for timestamps_weights is %d, but the actual dim is %d.",
                        TIMESTAMPS_WEIGHTS_DIM, tswShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(s <= 0 || s > MAX_SEQ_LEN,
              OPS_LOG_E("Tiling Debug",
                        "Sequence len expects a value in the range [1, %d], but the actual value is %lld.",
                        MAX_SEQ_LEN, s),
              return ge::GRAPH_FAILED);
    OPS_CHECK(batchsize <= 0 || batchsize > MAX_BATCH_SIZE,
              OPS_LOG_E("Tiling Debug",
                        "Batchsize expects a value in the range [1, %d], but the actual value is %lld.",
                        MAX_BATCH_SIZE, batchsize),
              return ge::GRAPH_FAILED);
    OPS_CHECK(numLayer <= 0 || numLayer > MAX_NUM_LAYER,
              OPS_LOG_E("Tiling Debug",
                        "num_layer expects a value in the range [1, %d], but the actual value is %lld.",
                        MAX_NUM_LAYER, numLayer),
              return ge::GRAPH_FAILED);
    OPS_CHECK(numBuckets <= 0 || numBuckets > MAX_NUM_BUCKET + 1,
              OPS_LOG_E("Tiling Debug",
                        "num_buckets expects a value in the range [1, %d], but the actual value is %lld.",
                        MAX_NUM_BUCKET + 1, numBuckets),
              return ge::GRAPH_FAILED);

    tilingData.set_bs(batchsize);
    tilingData.set_s(s);
    tilingData.set_numLayer(numLayer);
    tilingData.set_numBuckets(numBuckets);
    tilingData.set_bucketDivisor(divs);
    tilingData.set_clampMax(clampMax);

    // 计算stride、buff
    // 获取ub
    uint64_t ub;
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;
    // 获取数据类型
    auto tswType = context->GetInputTensor(TIMESTAMPS_WEIGHTS_INDEX)->GetDataType();
    auto tsType = context->GetInputTensor(TIMESTAMPS_INDEX)->GetDataType();
    int tswSize = ge::GetSizeByDataType(tswType);
    int tsSize = ge::GetSizeByDataType(tsType);
    OPS_CHECK(tswSize == 0, OPS_LOG_E("Tiling Debug", "Invalid data type of timestamps_weights."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(tsSize == 0, OPS_LOG_E("Tiling Debug", "Invalid data type of timestamps."), return ge::GRAPH_FAILED);
    tilingData.set_tswType(tswType);
    tilingData.set_tsType(tsType);
    // 计算不含buff的stride长度
    ub -= numBuckets * numLayer * tswSize + numLayer * DATA_ALIGN_BYTES;  // 减去tsw预留ub
    uint32_t alignSeqLen = (s * tswSize + DATA_ALIGN_BYTES - 1) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES / tswSize;
    uint32_t stride = ub / (sizeof(float) + tsSize) / alignSeqLen;

    // 计算clamp buff所需空间
    std::vector<int64_t> shape_vec = {stride * alignSeqLen};
    ge::Shape shape(shape_vec);
    uint32_t maxBuff = 0;
    uint32_t minBuff = 0;
    AscendC::GetClampMaxMinTmpSize(shape, sizeof(float), false, maxBuff, minBuff);
    tilingData.set_buffSize(maxBuff);

    // 重新计算stride长度
    stride = (ub - maxBuff) / (sizeof(float) + tsSize) / alignSeqLen;
    tilingData.set_stride(stride);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("timestampShape", context->GetInputShape(TIMESTAMPS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("timestampShape", context->GetInputTensor(TIMESTAMPS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("tswShape", context->GetInputShape(TIMESTAMPS_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("tswShape", context->GetInputTensor(TIMESTAMPS_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E("Tiling Debug", "Core num is 0."), return ge::GRAPH_FAILED);

    RelativeAttnBiasTimeTilingData tilingData;
    auto ret = TimeTilingFunc(tilingData, context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    context->SetBlockDim(coreNum);
    auto rowTilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("GetRawTilingData", rowTilingData, return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rowTilingData->GetData(), rowTilingData->GetCapacity());
    rowTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("timestampShape", context->GetInputShape(TIMESTAMPS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("tswShape", context->GetInputShape(TIMESTAMPS_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    const gert::Shape* tsShape = context->GetInputShape(TIMESTAMPS_INDEX);
    const gert::Shape* tswShape = context->GetInputShape(TIMESTAMPS_WEIGHTS_INDEX);
    gert::Shape* rabTimeOutShape = context->GetOutputShape(RAB_TIME_INDEX);
    int64_t bs = tsShape->GetDim(DIM0);
    int64_t s = tsShape->GetDim(DIM1);
    int64_t numLayers = tswShape->GetDim(DIM1);

    rabTimeOutShape->SetDimNum(RAB_TIME_OUT_DIM);
    rabTimeOutShape->SetDim(DIM0, numLayers);
    rabTimeOutShape->SetDim(DIM1, bs);
    rabTimeOutShape->SetDim(DIM2, s);
    rabTimeOutShape->SetDim(DIM3, DIM_PLACE_HOLDER);
    rabTimeOutShape->SetDim(DIM4, s);
    rabTimeOutShape->SetDim(DIM5, DIM_PLACE_HOLDER);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RelativeAttnBiasTime : public OpDef {
public:
    explicit RelativeAttnBiasTime(const char* name) : OpDef(name)
    {
        this->Input("timestamps")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("timestamps_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rab_time")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("bucket_timestamps")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("bucket_divisor").Float();

        this->SetInferShape(ge::InferShape);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend310p", aicore_config);
        this->AICore().AddConfig("ascend910_95", aicore_config);
    }
};

OP_ADD(RelativeAttnBiasTime);

}  // namespace ops
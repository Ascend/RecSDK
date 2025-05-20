/**
* @file relative_attn_bias.cpp
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
*/

#include <cmath>
#include "relative_attn_bias_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../common/ops_log.h"

constexpr int32_t RESERVER_UB_SIZE = (5 * 1024);
constexpr int32_t DATA_ALIGN_BYTES = 32;
constexpr uint8_t NUM_BUFFER = 2;

// input index
constexpr int IDENTITY_INDEX = 1;
constexpr int TIMESTAMPS_INDEX = 2;
constexpr int TIMESTAMPS_WEIGHTS_INDEX = 3;
// output index
constexpr int RAB_POSITION_INDEX = 0;
constexpr int RAB_TIME_INDEX = 1;
// attr index
constexpr int PAST_VALID_LENS_INDEX = 0;
constexpr int BUCKET_DIV_INDEX = 1;
// output dim
constexpr int RAB_POS_OUT_DIM = 3;
constexpr int RAB_TIME_OUT_DIM = 6;
constexpr int DIM_PLACE_HOLDER = 1;
constexpr int DIM0 = 0;
constexpr int DIM1 = 1;
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;
constexpr int DIM4 = 4;
constexpr int DIM5 = 5;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("identityShape", context->GetInputShape(IDENTITY_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("timestampShape", context->GetInputShape(TIMESTAMPS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("tswShape", context->GetInputShape(TIMESTAMPS_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0,
              OPS_LOG_E("Tiling Debug", "Core num is 0."),
              return ge::GRAPH_FAILED);

    RelativeAttnBiasTilingData tilingData;

    auto timeShape = context->GetInputShape(TIMESTAMPS_INDEX)->GetStorageShape();  // timestamps(b, s)
    // 获取batchsize
    int bs = timeShape.GetDim(0);
    OPS_CHECK(bs <= 0,
              OPS_LOG_E("Tiling Debug", "Batchsize is invalid."),
              return ge::GRAPH_FAILED);
    tilingData.set_bs(bs);
    // 获取序列长度大小
    int s = timeShape.GetDim(1);
    OPS_CHECK(s > 4300,
              OPS_LOG_E("Tiling Debug", "Input table larger than (4300, 4300)."),
              return ge::GRAPH_FAILED);
    tilingData.set_s(s);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto pastValidLensPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PAST_VALID_LENS_INDEX);
    OPS_LOG_E_IF_NULL("past_valid_len", pastValidLensPtr, return ge::GRAPH_FAILED);
    int bsValid = pastValidLensPtr->GetSize();
    OPS_CHECK(bsValid != bs,
              OPS_LOG_E("Tiling Debug", "mismatch batchsize of past_valid_len and timestamps."),
              return ge::GRAPH_FAILED);

    auto *pastValidLensData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(pastValidLensPtr->GetData()));
    uint32_t pastValidLens[MAX_BATCH_SIZE];
    for (auto i = 0; i < bs; ++i) {
        pastValidLens[i] = pastValidLensData[i];
    }
    tilingData.set_pastValidLens(pastValidLens);

    // 获取ts_w(num_layer, num_buckets+1)
    auto tswShape = context->GetInputShape(TIMESTAMPS_WEIGHTS_INDEX)->GetStorageShape();
    int numLayer = tswShape.GetDim(0);
    int numBuckets = tswShape.GetDim(1);
    tilingData.set_numBuckets(numBuckets);
    tilingData.set_numLayer(numLayer);

    float divs = *context->GetAttrs()->GetFloat(BUCKET_DIV_INDEX);
    float clampMax = exp((numBuckets - 1) * divs);
    tilingData.set_bucketDivisor(divs);
    tilingData.set_clampMax(clampMax);

    // 获取ub
    uint64_t ub;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;
    // 获取数据类型
    auto floatType = context->GetInputTensor(IDENTITY_INDEX)->GetDataType();
    auto intType = context->GetInputTensor(TIMESTAMPS_INDEX)->GetDataType();
    int floatSize = ge::GetSizeByDataType(floatType);
    int intSize = ge::GetSizeByDataType(intType);
    tilingData.set_floatType(floatType);
    tilingData.set_intType(intType);
    OPS_CHECK(floatSize == 0,
              OPS_LOG_E("Tiling Debug", "Invalid data type."),
              return ge::GRAPH_FAILED);

    // 计算一次处理的窗口大小(stride)
    int stride = ub / (NUM_BUFFER * 3 * floatSize);
    tilingData.set_positionStride(stride);

    // 计算不含buff的stride长度
    ub -= numBuckets * numLayer * floatSize + numLayer * DATA_ALIGN_BYTES;  // 减去tsw预留ub
    uint32_t alignSeqLen = (s * floatSize + DATA_ALIGN_BYTES - 1) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES / floatSize;
    stride = ub / (sizeof(float) + intSize) / alignSeqLen;

    // 计算clamp buff所需空间
    std::vector<int64_t> shape_vec = {stride * alignSeqLen};
    ge::Shape shape(shape_vec);
    uint32_t maxBuff = 0;
    uint32_t minBuff = 0;
    AscendC::GetClampMaxMinTmpSize(shape, sizeof(float), false, maxBuff, minBuff);
    tilingData.set_buffSize(maxBuff);

    // 重新计算stride长度
    stride = (ub - maxBuff) / (sizeof(float) + intSize) / alignSeqLen;
    tilingData.set_timeStride(stride);

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
    gert::Shape* rabPosOutShape = context->GetOutputShape(RAB_POSITION_INDEX);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto pastValidLensPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PAST_VALID_LENS_INDEX);
    int bs = pastValidLensPtr->GetSize();
    const gert::Shape* identityShape = context->GetInputShape(IDENTITY_INDEX);
    int s = identityShape->GetDim(DIM0);  // identityShape(2s, 2s)

    rabPosOutShape->SetDimNum(RAB_POS_OUT_DIM);
    rabPosOutShape->SetDim(DIM0, bs);
    rabPosOutShape->SetDim(DIM1, s);
    rabPosOutShape->SetDim(DIM2, s);

    const gert::Shape* tShape = context->GetInputShape(TIMESTAMPS_INDEX);
    const gert::Shape* tswShape = context->GetInputShape(TIMESTAMPS_WEIGHTS_INDEX);
    gert::Shape* rabTimeOutShape = context->GetOutputShape(RAB_TIME_INDEX);
    int numLayers = tswShape->GetDim(DIM1);

    rabTimeOutShape->SetDimNum(RAB_TIME_OUT_DIM);
    rabPosOutShape->SetDim(DIM0, numLayers);
    rabPosOutShape->SetDim(DIM1, bs);
    rabPosOutShape->SetDim(DIM2, s);
    rabPosOutShape->SetDim(DIM3, DIM_PLACE_HOLDER);
    rabPosOutShape->SetDim(DIM4, s);
    rabPosOutShape->SetDim(DIM5, DIM_PLACE_HOLDER);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RelativeAttnBias : public OpDef {
public:
    explicit RelativeAttnBias(const char* name) : OpDef(name)
    {
        this->Input("rel_pos_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("identity")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
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
        this->Output("rab_pos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rab_time")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("past_valid_lens").ListInt();
        this->Attr("bucket_divisor").Float();

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
        this->AICore().AddConfig("ascend310p", aicore_config);
    }
};

OP_ADD(RelativeAttnBias);

}  // namespace ops
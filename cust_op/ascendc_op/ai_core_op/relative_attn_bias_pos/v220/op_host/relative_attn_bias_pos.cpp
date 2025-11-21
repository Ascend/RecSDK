/**
* @file relative_attn_bias_pos.cpp
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
*/

#include <cmath>
#include "relative_attn_bias_pos_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

constexpr int32_t RESERVER_UB_SIZE = (20 * 1024);
constexpr uint8_t NUM_BUFFER = 2;
constexpr int SEQ_EXPAND = 2;  // rab_pos中序列长度为原本输入的两倍

// input index
constexpr int REL_POS_BIAS_INDEX = 0;
constexpr int IDENTITY_INDEX = 1;
// output index
constexpr int RAB_POSITION_INDEX = 0;
// attr index
constexpr int PAST_VALID_LENS_INDEX = 0;
// output dim
constexpr int REL_POS_BIAS_DIM = 2;
constexpr int IDENTITY_DIM = 2;
constexpr int RAB_POS_OUT_DIM = 3;
constexpr int DIM0 = 0;
constexpr int DIM1 = 1;
constexpr int DIM2 = 2;
constexpr int64_t DOUBLE_MAX_S = 8600;

namespace optiling {
static ge::graphStatus PosTilingFunc(RelativeAttnBiasPosTilingData& tilingData, gert::TilingContext* context)
{
    // 设置past_valid_len
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto pastValidLensPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PAST_VALID_LENS_INDEX);
    int batchsize = pastValidLensPtr->GetSize();
    OPS_CHECK(batchsize <= 0 || batchsize > MAX_BATCH_SIZE,
              OPS_LOG_E("Tiling Debug", "batchsize of past_valid_lens is expected in [1, %d], but got %d",
                        MAX_BATCH_SIZE, batchsize),
              return ge::GRAPH_FAILED);
    tilingData.set_bs(batchsize);

    auto *pastValidLensData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(pastValidLensPtr->GetData()));
    uint32_t pastValidLens[MAX_BATCH_SIZE];
    for (auto i = 0; i < batchsize; ++i) {
        pastValidLens[i] = pastValidLensData[i];
    }
    tilingData.set_pastValidLens(pastValidLens);
    // 校验s
    auto biasShape = context->GetInputShape(REL_POS_BIAS_INDEX)->GetStorageShape();  // (2s, 2s)
    auto identityShape = context->GetInputShape(IDENTITY_INDEX)->GetStorageShape();  // (2s, 2s)

    int64_t biasSeqLen = biasShape.GetDim(DIM0);
    int64_t biasSeqLen2 = biasShape.GetDim(DIM1);
    int64_t idSeqLen = identityShape.GetDim(DIM0);
    int64_t idSeqLen2 = identityShape.GetDim(DIM1);

    OPS_CHECK(biasShape.GetDimNum() != REL_POS_BIAS_DIM,
              OPS_LOG_E("Tiling Debug", "Invalid rel_pos_bias shape."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(identityShape.GetDimNum() != IDENTITY_DIM,
              OPS_LOG_E("Tiling Debug", "Invalid identity shape."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(biasSeqLen != biasSeqLen2 || biasSeqLen != idSeqLen || biasSeqLen != idSeqLen2,
              OPS_LOG_E("Tiling Debug", "Mismatch sequence len of rel_pos_bias and identity."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(biasSeqLen <= 0 || biasSeqLen > DOUBLE_MAX_S,
              OPS_LOG_E("Tiling Debug", "Invalid sequence len of rel_pos_bias."),
              return ge::GRAPH_FAILED);
    tilingData.set_s(biasSeqLen / SEQ_EXPAND);

    // 获取ub
    uint64_t ub;
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;
    // 获取数据类型
    auto identityType = context->GetInputTensor(IDENTITY_INDEX)->GetDataType();
    auto biasType = context->GetInputTensor(REL_POS_BIAS_INDEX)->GetDataType();

    int biasDataSize = ge::GetSizeByDataType(biasType);
    OPS_CHECK(identityType != biasType,
              OPS_LOG_E("Tiling Debug", "Mismatch data type of identity and rel_pos_bias."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(biasDataSize < 1,
              OPS_LOG_E("Tiling Debug", "Invalid data type."),
              return ge::GRAPH_FAILED);
    tilingData.set_dataType(biasType);

    // 计算一次处理的窗口大小(stride)
    int stride = ub / (NUM_BUFFER * 3 * biasDataSize);  // 需要3份ub暂存计算结果
    tilingData.set_stride(stride);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("relPosBiasShape", context->GetInputShape(REL_POS_BIAS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("relPosBiasShape", context->GetInputTensor(REL_POS_BIAS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("identityShape", context->GetInputShape(IDENTITY_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("identityShape", context->GetInputTensor(IDENTITY_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto pastValidLensPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PAST_VALID_LENS_INDEX);
    OPS_LOG_E_IF_NULL("past_valid_len", pastValidLensPtr, return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0,
              OPS_LOG_E("Tiling Debug", "Core num is 0."),
              return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);

    RelativeAttnBiasPosTilingData tilingData;
    auto ret = PosTilingFunc(tilingData, context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
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
    OPS_LOG_E_IF_NULL("relPosBiasShape", context->GetInputShape(REL_POS_BIAS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("identityShape", context->GetInputShape(IDENTITY_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("rabPosOutShape", context->GetOutputShape(RAB_POSITION_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    gert::Shape* rabPosOutShape = context->GetOutputShape(RAB_POSITION_INDEX);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto pastValidLensPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PAST_VALID_LENS_INDEX);
    int64_t bs = pastValidLensPtr->GetSize();
    const gert::Shape* identityShape = context->GetInputShape(IDENTITY_INDEX);
    int64_t s = identityShape->GetDim(DIM0);  // identityShape(2s, 2s)

    rabPosOutShape->SetDimNum(RAB_POS_OUT_DIM);
    rabPosOutShape->SetDim(DIM0, bs);
    rabPosOutShape->SetDim(DIM1, s);
    rabPosOutShape->SetDim(DIM2, s);

    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RelativeAttnBiasPos : public OpDef {
public:
    explicit RelativeAttnBiasPos(const char* name) : OpDef(name)
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
        this->Output("rab_pos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("past_valid_lens").ListInt();

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

OP_ADD(RelativeAttnBiasPos);

}  // namespace ops
/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
*/

#include <cstdio>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "disetangle_attention_tiling.h"
#include "register/op_def_registry.h"
#include "ops_log.h"

constexpr uint32_t ACC_S = 256;  // 基本加速块的大小
constexpr uint32_t LONG_ACC_S = ACC_S * 2;
constexpr uint32_t UB_BUFFER_NUM = 1;
constexpr uint32_t UB_LOOP_PROC_N = 48 / UB_BUFFER_NUM;
constexpr uint32_t MUTIPLE_OF_DIM = 16;           // 当前DIM 必须为16的整数倍
constexpr uint32_t KBYTES = 1024;
constexpr uint32_t UB_REV_BYTE_SIZE = 20 * KBYTES;  // ub 预留20K空间


namespace optiling {

static ge::graphStatus TilingShape(gert::TilingContext *context, DisetangleAttentionTilingData &tiling)
{
    auto query_shape = context->GetInputShape(0)->GetStorageShape();

    tiling.set_batchSize(query_shape.GetDim(0)); // 0 means tensor dim 0
    tiling.set_headNum(query_shape.GetDim(1)); // 1 means tensor dim 1
    tiling.set_seqLen(query_shape.GetDim(2)); // 2 means tensor dim 2
    tiling.set_headDim(query_shape.GetDim(3)); // 3 means tensor dim 3
    tiling.set_accS(ACC_S);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCore(gert::TilingContext *context, DisetangleAttentionTilingData &tiling)
{
    auto batchSize = tiling.get_batchSize();
    auto headNum = tiling.get_headNum();

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t aivCoreNum = ascendPlatform.GetCoreNumAiv();
    OPS_LOG_E_IF(aivCoreNum == 0, context, return ge::GRAPH_FAILED, "aiv core num cant be zero");

    auto accSN = batchSize * headNum;  // 基本加速块的个数
    auto useCoreNum = (accSN > aivCoreNum) ? aivCoreNum : accSN;
    auto splitNextCoreAccSN = accSN / useCoreNum;
    auto splitPrevCoreAccSN = splitNextCoreAccSN + 1;
    auto splitCoreIdx = accSN % useCoreNum;
    size_t aicCoreNum = ascendPlatform.GetCoreNumAic();
    OPS_LOG_E_IF(aicCoreNum == 0, context, return ge::GRAPH_FAILED, "aic core num cant be zero");
    context->SetBlockDim(aicCoreNum);

    tiling.set_accS(ACC_S);
    tiling.set_useCoreNum(useCoreNum);
    tiling.set_aivCoreNum(aivCoreNum);
    tiling.set_splitNextCoreAccSN(splitNextCoreAccSN);
    tiling.set_splitPrevCoreAccSN(splitPrevCoreAccSN);
    tiling.set_splitCoreIdx(splitCoreIdx);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingMatmulImpl(gert::TilingContext *context, uint32_t m, uint32_t n, uint32_t k,
    uint32_t ori_k, bool is_trans_n, TCubeTiling &tiling_mm)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling MM(ascendPlatform);
    MM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    MM.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, is_trans_n);
    MM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    MM.SetOrgShape(m, n, k, ori_k);
    MM.SetShape(m, n, k);
    MM.SetBias(false);
    MM.SetBufferSpace(-1, -1, -1);

    OPS_LOG_E_IF(MM.GetTiling(tiling_mm) == -1, context, return ge::GRAPH_FAILED, "matmul tiling failed");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingMatmul(gert::TilingContext *context, DisetangleAttentionTilingData &tiling)
{
    auto nums = tiling.get_headNum();
    auto dim = tiling.get_headDim();
    if (ge::GRAPH_FAILED == TilingMatmulImpl(context, ACC_S, LONG_ACC_S, dim, dim * nums, true, tiling.MM)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_FAILED == TilingMatmulImpl(context, ACC_S, ACC_S, dim, dim, true, tiling.QK_MM)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_FAILED == TilingMatmulImpl(context, ACC_S, dim, ACC_S, ACC_S, false, tiling.SV_MM)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingWorkspace(gert::TilingContext *context, DisetangleAttentionTilingData &tiling)
{
    auto useCoreNum = tiling.get_useCoreNum();
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();

    int64_t workspaceSize = useCoreNum * 2 * ACC_S * LONG_ACC_S * sizeof(int16_t) + ACC_S * ACC_S * sizeof(int32_t);
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingAttrs(gert::TilingContext *context, DisetangleAttentionTilingData &tiling)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const uint32_t *pos_attr_ptr = attrs->GetAttrPointer<uint32_t>(0);
    OPS_LOG_E_IF_NULL("pos_attr_ptr", pos_attr_ptr, return ge::GRAPH_FAILED);

    const float *score_scale = attrs->GetAttrPointer<float>(1);
    OPS_LOG_E_IF_NULL("score_scale", score_scale, return ge::GRAPH_FAILED);

    context->SetTilingKey(static_cast<uint64_t>(*pos_attr_ptr));
    tiling.set_scoreScale(*score_scale);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingSoftmax(gert::TilingContext *context, DisetangleAttentionTilingData &tiling)
{
    int64_t local_workspace_byte_size = static_cast<int64_t>(UB_LOOP_PROC_N) * KBYTES;

    std::vector<int64_t> shape_vec{UB_LOOP_PROC_N, ACC_S};
    ge::Shape src_shape(shape_vec);

    uint32_t min_value = AscendC::GetSoftMaxMinTmpSize(src_shape, sizeof(int16_t), false);

    OPS_LOG_E_IF(min_value > local_workspace_byte_size, context, return ge::GRAPH_FAILED, "softmax need more ub size");

    AscendC::SoftMaxTilingFunc(src_shape, sizeof(int16_t), local_workspace_byte_size, tiling.SFT);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DisetangleAttentionTilingData tiling;
    if (ge::GRAPH_SUCCESS != TilingShape(context, tiling)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != TilingCore(context, tiling)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != TilingMatmul(context, tiling)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != TilingSoftmax(context, tiling)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != TilingWorkspace(context, tiling)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != TilingAttrs(context, tiling)) {
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape *query_shape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("query_shape", query_shape, return ge::GRAPH_FAILED);

    const gert::Shape *key_shape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL("key_shape", key_shape, return ge::GRAPH_FAILED);

    gert::Shape *atten_outputs = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("atten_outputs", atten_outputs, return ge::GRAPH_FAILED);

    gert::Shape *atten_probs = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL("atten_probs", atten_probs, return ge::GRAPH_FAILED);

    gert::Shape *atten_weights = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL("atten_weights", atten_weights, return ge::GRAPH_FAILED);

    auto batchSize = query_shape->GetDim(0);
    auto headNum = query_shape->GetDim(1);
    auto qSeqLen = query_shape->GetDim(2);
    auto kSeqLen = key_shape->GetDim(2);
    auto headDim = query_shape->GetDim(3);

    bool isDynamic = (qSeqLen == -1);
    OPS_LOG_E_IF(!isDynamic && qSeqLen != ACC_S, context, return ge::GRAPH_FAILED, "current seqLens must equal 256");
    OPS_LOG_E_IF(!isDynamic && qSeqLen != kSeqLen,
        context,
        return ge::GRAPH_FAILED,
        "current only support qSeqLen equal kSeqLen");
    OPS_LOG_E_IF(!isDynamic && ((headDim % MUTIPLE_OF_DIM) != 0),
        context,
        return ge::GRAPH_FAILED,
        "current headDim must mutiple of 16");

    atten_outputs->SetDimNum(4); // 4 means tensor have 4 dims
    atten_outputs->SetDim(0, batchSize);  // 0 means tensor 0 dim
    atten_outputs->SetDim(1, headNum);    // 1 means tensor 1 dim
    atten_outputs->SetDim(2, qSeqLen);    // 2 means tensor 2 dim
    atten_outputs->SetDim(3, headDim);    // 3 means tensor 3 dim

    atten_probs->SetDimNum(4); // 4 means tensor have 4 dims
    atten_probs->SetDim(0, batchSize); // 0 means tensor 0 dim
    atten_probs->SetDim(1, headNum); // 1 means tensor 1 dim
    atten_probs->SetDim(2, qSeqLen); // 2 means tensor 2 dim
    atten_probs->SetDim(3, qSeqLen); // 3 means tensor 3 dim

    atten_weights->SetDimNum(4); // 4 means tensor have 4 dims
    atten_weights->SetDim(0, batchSize); // 0 means tensor 0 dim
    atten_weights->SetDim(1, headNum); // 1 means tensor 1 dim
    atten_weights->SetDim(2, qSeqLen); // 2 means tensor 2 dim
    atten_weights->SetDim(3, qSeqLen); // 3 means tensor 3 dim
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    auto data_type = context->GetInputDataType(0);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, data_type)) { // 0 means tensor 0 dim
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(1, data_type)) { // 1 means tensor 1 dim
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(2, data_type)) { // 2 means tensor 2 dim
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
class DisetangleAttention : public OpDef {
public:
    explicit DisetangleAttention(const char *name) : OpDef(name)
    {
        this->Input("query_layer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_layer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_layer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pos_key_layer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pos_query_layer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("relative_pos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("atten_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("atten_outpus")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("atten_probs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("atten_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("pos_attr_type").AttrType(OPTIONAL).Int(0);
        this->Attr("score_scale").Float(1.0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DisetangleAttention);
}  // namespace ops
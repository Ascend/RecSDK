/**
 * @file index_select_for_rank1_backward.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#include "index_select_for_rank1_backward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

constexpr int DATA_ALIGN_TARGET = 32;
constexpr int RESERVER_UB_SIZE = (20 * 1024);
// input index
constexpr int GRAD_IDX = 0;
constexpr int X_IDX = 1;
constexpr int INDEX_IDX = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gradShape", context->GetInputShape(GRAD_IDX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(X_IDX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexShape", context->GetInputShape(INDEX_IDX), return ge::GRAPH_FAILED);

    auto gradShape = context->GetInputShape(GRAD_IDX)->GetStorageShape();
    auto xShape = context->GetInputShape(X_IDX)->GetStorageShape();
    auto indexShape = context->GetInputShape(INDEX_IDX)->GetStorageShape();
    OPS_CHECK(
        xShape.GetDimNum() != 1,
        OPS_LOG_E("Tiling Debug", "IndexSectForRank1Backward is only used for input-1 with dim 0 but x.dim is %ld",
                  xShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        gradShape.GetDimNum() != 1,
        OPS_LOG_E("Tiling Debug", "IndexSectForRank1Backward is only used for input-1 with dim 0 but grad.dim is %ld",
                  gradShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        indexShape.GetDimNum() != 1,
        OPS_LOG_E("Tiling Debug", "IndexSectForRank1Backward is only used for input-1 with dim 0 but index.dim is %ld",
                  indexShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E("Tiling Debug", "No available aicore"), return ge::GRAPH_FAILED);

    int64_t totalLen = indexShape.GetShapeSize();
    int64_t xDim0 = xShape.GetShapeSize();
    int64_t baseLen = indexShape.GetShapeSize() / coreNum;
    int64_t tailSplitIndex = indexShape.GetShapeSize() % coreNum;

    IndexSelectForRank1BackwardTilingData tiling;
    tiling.set_totalLen(totalLen);
    tiling.set_xDim0(xDim0);
    tiling.set_baseLen(baseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);

    auto gradType = context->GetInputTensor(GRAD_IDX)->GetDataType();
    auto indexType = context->GetInputTensor(INDEX_IDX)->GetDataType();
    int gradTypeSize = ge::GetSizeByDataType(gradType);
    int indexTypeSize = ge::GetSizeByDataType(indexType);
    int maxTypeSize = gradTypeSize > indexTypeSize ? gradTypeSize : indexTypeSize;
    OPS_CHECK(gradTypeSize <= 0 || indexTypeSize <= 0, OPS_LOG_E("Tiling Debug", "Invalid data type."),
              return ge::GRAPH_FAILED);

    uint64_t ub;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE - xDim0 * gradTypeSize;
    uint32_t stride = ub / (gradTypeSize + indexTypeSize);
    stride = (stride * maxTypeSize + DATA_ALIGN_TARGET - 1) / DATA_ALIGN_TARGET * DATA_ALIGN_TARGET / maxTypeSize;
    tiling.set_stride(stride);

    context->SetBlockDim(coreNum);
    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(optiling::X_IDX);
    const gert::Shape* indexShape = context->GetInputShape(optiling::INDEX_IDX);

    gert::Shape* gradXShape = context->GetOutputShape(optiling::GRAD_IDX);

    OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexShape", indexShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gradXShape", gradXShape, return ge::GRAPH_FAILED);

    *gradXShape = *xShape;
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class IndexSelectForRank1Backward : public OpDef {
public:
    explicit IndexSelectForRank1Backward(const char* name) : OpDef(name)
    {
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(IndexSelectForRank1Backward);
}  // namespace ops

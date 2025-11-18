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

#include <cstdint>
#include <cstdio>

#include "register/op_def_registry.h"
#include "split_embedding_codegen_forward_unweighted_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace optiling {

constexpr int DATA_TYPE_FLOAT32 = 0;
constexpr int DATA_TYPE_INT64 = 1;

constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int UB_ALIGN = 32;
constexpr int NUM_QUEUE = 32;

constexpr int POOL_MODE_MEAN = 0;
constexpr int POOL_MODE_SUM = 1;
constexpr int POOL_MODE_NOBAG = 2;

constexpr int DEV_WEIGHTS_INDEX = 0;
constexpr int UVM_WEIGHTS_INDEX = 1;
constexpr int LXU_CACHE_WEIGHTS_INDEX = 2;
constexpr int WEIGHTS_PLACEMENTS_INDEX = 3;
constexpr int WEIGHTS_OFFSETS_INDEX = 4;
constexpr int D_OFFSETS_INDEX = 5;
constexpr int INDICES_INDEX = 6;
constexpr int OFFSETS_INDEX = 7;
constexpr int LXU_CACHE_LOCATIONS_INDEX = 8;
constexpr int HASH_INDICES_INDEX = 9;
constexpr int ROWS_PER_TABLE_INDEX = 11;
constexpr int POOL_MODE_INDEX = 2;
constexpr int MAX_D_INDEX = 1;
constexpr int EC_KEY = 1;
constexpr int EBC_KEY = 2;

static ge::graphStatus ShapeTilingCheckFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("devWeights shape", context->GetInputShape(DEV_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("weightsOffset shape", context->GetInputShape(WEIGHTS_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("dOffsets shape", context->GetInputShape(D_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indices shape", context->GetInputShape(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsets shape", context->GetInputShape(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShapeTilingFunc(gert::TilingContext* context,
                                       SplitEmbeddingCodegenForwardUnweightedTilingData& tilingData)
{
    if (ShapeTilingCheckFunc(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
  
    int64_t devWeightsDim0 = context->GetInputShape(DEV_WEIGHTS_INDEX)->GetStorageShape().GetDim(0);
    int64_t weightsOffsetsDim0 = context->GetInputShape(WEIGHTS_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t dOffsetsDim0 = context->GetInputShape(D_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t indicesDim0 = context->GetInputShape(INDICES_INDEX)->GetStorageShape().GetDim(0);
    int64_t offsetsDim0 = context->GetInputShape(OFFSETS_INDEX)->GetStorageShape().GetDim(0);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    OPS_CHECK(weightsOffsetsDim0 < 1,
              OPS_LOG_E("", "The length of D_offsets must be at least 1, but the actual value is %d.",
                        weightsOffsetsDim0),
              return ge::GRAPH_FAILED);
    OPS_CHECK(dOffsetsDim0 <= 1,
              OPS_LOG_E("", "The length of D_offsets must be at least 2, but the actual value is %d.",
                        dOffsetsDim0),
              return ge::GRAPH_FAILED);

    auto rowsPerTable = context->GetOptionalInputTensor(ROWS_PER_TABLE_INDEX);
    if (rowsPerTable == nullptr) {
        tilingData.set_enableRowsPerTable(0);
    } else {
        tilingData.set_enableRowsPerTable(1);
        OPS_LOG_E_IF_NULL("rowsPerTable shape", context->GetInputShape(ROWS_PER_TABLE_INDEX), return ge::GRAPH_FAILED);
        int64_t rowsPerTableDim0 = context->GetInputShape(ROWS_PER_TABLE_INDEX)->GetStorageShape().GetDim(0);
        OPS_CHECK(rowsPerTableDim0 != weightsOffsetsDim0,
                  OPS_LOG_E("", "Len mismatch between rows_per_table(%d) and weights_offsets(%d).",
                            rowsPerTableDim0, weightsOffsetsDim0),
                  return ge::GRAPH_FAILED);
    }

    auto hashIndices = context->GetOptionalInputTensor(HASH_INDICES_INDEX);
    if (hashIndices == nullptr) {
        tilingData.set_enableHash(0);
    } else {
        tilingData.set_enableHash(1);
        OPS_LOG_E_IF_NULL("hashIndices shape", context->GetInputShape(HASH_INDICES_INDEX), return ge::GRAPH_FAILED);
        indicesDim0 = context->GetInputShape(HASH_INDICES_INDEX)->GetStorageShape().GetDim(0);
    }

    int64_t poolMode = *attrs->GetInt(POOL_MODE_INDEX);
    OPS_CHECK(poolMode < POOL_MODE_MEAN || poolMode > POOL_MODE_NOBAG,
              OPS_LOG_E("", "Invalid pooling mode %d: supported modes are MEAN(0), SUM(1), NONE(2)", poolMode),
              return ge::GRAPH_FAILED);
    // bag output shape[batchsize, totalD]
    int64_t outDim0 = (offsetsDim0 - 1) / weightsOffsetsDim0;
    int64_t outDim1 = *attrs->GetInt(0);
    // no bag output shape [indicesNum, maxD]
    if (poolMode == POOL_MODE_NOBAG) {
        outDim0 = indicesDim0;
        outDim1 = *attrs->GetInt(MAX_D_INDEX);
    }
    context->SetTilingKey(poolMode);

    int64_t bytesOfDataType = sizeof(float);
    int64_t offsetDataType = DATA_TYPE_INT64;
    int64_t maxD = *attrs->GetInt(MAX_D_INDEX);
    
    tilingData.set_maxD(maxD);
    tilingData.set_poolMode(poolMode);
    tilingData.set_devWeightsDim0(devWeightsDim0);
    tilingData.set_weightsOffsetsDim0(weightsOffsetsDim0);
    tilingData.set_dOffsetsDim0(dOffsetsDim0);
    tilingData.set_indicesDim0(indicesDim0);
    tilingData.set_offsetsDim0(offsetsDim0);
    tilingData.set_outDim0(outDim0);
    tilingData.set_outDim1(outDim1);
    tilingData.set_bytesOfDataType(bytesOfDataType);
    tilingData.set_offsetDataType(offsetDataType);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    SplitEmbeddingCodegenForwardUnweightedTilingData tiling;
    // Shape and dType
    if (ShapeTilingFunc(context, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Tiling
    size_t coreNum = ascnedPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E("", "Core num is 0."), return ge::GRAPH_FAILED);

    int64_t splitBaseLen = (tiling.get_offsetsDim0() - 1) / coreNum;
    int64_t tailSplitIndex = (tiling.get_offsetsDim0() - 1) % coreNum;

    tiling.set_splitBaseLen(splitBaseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);

    uint64_t ubCanUsed;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    tiling.set_ubCanUsed(ubCanUsed);

    context->SetBlockDim(coreNum);
    context->SetNeedAtomic(true);

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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("x1_shape", x1_shape, return ge::GRAPH_FAILED);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("y_shape", y_shape, return ge::GRAPH_FAILED);

    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SplitEmbeddingCodegenForwardUnweighted : public OpDef {
public:
    explicit SplitEmbeddingCodegenForwardUnweighted(const char* name) : OpDef(name)
    {
        this->Input("dev_weights")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("uvm_weights")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lxu_cache_weights")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights_placements")
            .ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights_offsets")
            .ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("D_offsets")
            .ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lxu_cache_locations")
            .ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("hash_indices")
            .ParamType(OPTIONAL).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offset_per_key")
            .ParamType(OPTIONAL).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rows_per_table")
            .ParamType(OPTIONAL).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("total_D").Int();
        this->Attr("max_D").Int();
        this->Attr("pool_mode").Int();
        this->Attr("output_dtype").Int();
        this->Attr("is_experimental").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SplitEmbeddingCodegenForwardUnweighted);
}  // namespace ops

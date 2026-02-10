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
#include <algorithm>

#include "register/op_def_registry.h"
#include "int_nbit_split_embedding_codegen_lookup_function_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
    constexpr int BAG_INT32_KEY = 0;
    constexpr int BAG_INT64_KEY = 1;
    constexpr int NOBAG_KEY = 2;

    constexpr int RESERVER_UB_SIZE = 20 * 1024;
    constexpr int UB_ALIGN = 32;

    constexpr int POOL_MODE_SUM = 0;
    constexpr int POOL_MODE_MEAN = 1;
    constexpr int POOL_MODE_NOBAG = 2;

    // Input indices
    constexpr int DEV_WEIGHTS_INDEX = 0;
    constexpr int UVM_WEIGHTS_INDEX = 1;
    constexpr int LXU_CACHE_WEIGHTS_INDEX = 2;
    constexpr int WEIGHTS_PLACEMENTS_INDEX = 3;
    constexpr int WEIGHTS_OFFSETS_INDEX = 4;
    constexpr int WEIGHTS_TYS_INDEX = 5;
    constexpr int D_OFFSETS_INDEX = 6;
    constexpr int INDICES_INDEX = 7;
    constexpr int OFFSETS_INDEX = 8;
    constexpr int LXU_CACHE_LOCATIONS_INDEX = 9;
    constexpr int OFFSET_PER_KEY_INDEX = 10;
    constexpr int INDICE_WEIGHTS_INDEX = 11;

    // Attribute indices
    constexpr int TOTAL_D_INDEX = 0;
    constexpr int MAX_D_INDEX = 1;
    constexpr int MAX_INT2_D_INDEX = 2;
    constexpr int MAX_INT4_D_INDEX = 3;
    constexpr int MAX_INT8_D_INDEX = 4;
    constexpr int MAX_FLOAT16_D_INDEX = 5;
    constexpr int MAX_FLOAT32_D_INDEX = 6;
    constexpr int MAX_FLOAT8_D_INDEX = 7;
    constexpr int POOL_MODE_INDEX = 8;
    constexpr int OUTPUT_DTYPE_INDEX = 9;
    constexpr int ROW_ALIGNMENT_INDEX = 10;
    constexpr int FP8_EXPONENT_BITS_INDEX = 11;
    constexpr int FP8_EXPONENT_BIAS_INDEX = 12;

}

namespace optiling {

static ge::graphStatus ShapeTilingCheckFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("devWeights shape", context->GetInputShape(DEV_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("weightsOffsets shape", context->GetInputShape(WEIGHTS_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("weightsTys shape", context->GetInputShape(WEIGHTS_TYS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("dOffsets shape", context->GetInputShape(D_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indices shape", context->GetInputShape(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsets shape", context->GetInputShape(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetPerKey shape", context->GetInputShape(OFFSET_PER_KEY_INDEX), return ge::GRAPH_FAILED);
    // 条件检查：只有当indice_weights存在时才检查
    if (context->GetInputShape(INDICE_WEIGHTS_INDEX) != nullptr) {
        OPS_LOG_E_IF_NULL("indice_weights shape",
                          context->GetInputShape(INDICE_WEIGHTS_INDEX),
                          return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShapeTilingFunc(gert::TilingContext* context,
                                       IntNbitSplitEmbeddingCodegenLookupFunctionTilingData& tilingData)
{
    if (ShapeTilingCheckFunc(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t devWeightsDim0 = context->GetInputShape(DEV_WEIGHTS_INDEX)->GetStorageShape().GetDim(0);
    int64_t weightsOffsetsDim0 = context->GetInputShape(WEIGHTS_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t weightsTysDim0 = context->GetInputShape(WEIGHTS_TYS_INDEX)->GetStorageShape().GetDim(0);
    int64_t dOffsetsDim0 = context->GetInputShape(D_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t indicesDim0 = context->GetInputShape(INDICES_INDEX)->GetStorageShape().GetDim(0);
    int64_t offsetsDim0 = context->GetInputShape(OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t offsetPerKeyDim0 = context->GetInputShape(OFFSET_PER_KEY_INDEX)->GetStorageShape().GetDim(0);

    // 条件获取：只有当indice_weights存在时才获取
    bool isWeighted = false;
    if (context->GetInputShape(INDICE_WEIGHTS_INDEX) != nullptr) {
        int64_t indiceWeightsDim0 = context->GetInputShape(INDICE_WEIGHTS_INDEX)->GetStorageShape().GetDim(0);
        isWeighted = true;

        // 校验：indice_weights的长度必须等于indices的长度（一一对应）
        OPS_CHECK(indiceWeightsDim0 != indicesDim0,
                  OPS_LOG_E("", "Len mismatch: indice_weights(%ld) must equal indices(%ld).",
                            indiceWeightsDim0, indicesDim0),
                  return ge::GRAPH_FAILED);
    }

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    // Validate weights_offsets size
    OPS_CHECK(weightsOffsetsDim0 < 1,
              OPS_LOG_E("", "The length of weights_offsets must be at least 1, but the actual value is %ld.",
                        weightsOffsetsDim0),
              return ge::GRAPH_FAILED);

    // Validate D_offsets size
    OPS_CHECK(dOffsetsDim0 <= 1,
              OPS_LOG_E("", "The length of D_offsets must be at least 2, but the actual value is %ld.",
                        dOffsetsDim0),
              return ge::GRAPH_FAILED);

    OPS_CHECK(dOffsetsDim0 != weightsOffsetsDim0 + 1,
              OPS_LOG_E("", "The length of D_offsets must equal weights_offsets(%ld) + 1.",
                        dOffsetsDim0),
              return ge::GRAPH_FAILED);

    // Validate weights_tys size matches weights_offsets
    OPS_CHECK(weightsTysDim0 != weightsOffsetsDim0,
              OPS_LOG_E("", "Len mismatch: weights_tys(%ld) must equal weights_offsets(%ld).",
                        weightsTysDim0, weightsOffsetsDim0),
              return ge::GRAPH_FAILED);

    // Validate offset_per_key size matches weights_offsets
    OPS_CHECK(offsetPerKeyDim0 != dOffsetsDim0,
              OPS_LOG_E("", "Len mismatch: offset_per_key(%ld) must equal D_offsets(%ld).",
                        offsetPerKeyDim0, weightsOffsetsDim0),
              return ge::GRAPH_FAILED);

    // Get attributes
    int64_t totalD = *attrs->GetInt(TOTAL_D_INDEX);
    int64_t maxD = *attrs->GetInt(MAX_D_INDEX);
    int64_t maxInt2D = *attrs->GetInt(MAX_INT2_D_INDEX);
    int64_t maxInt4D = *attrs->GetInt(MAX_INT4_D_INDEX);
    int64_t maxInt8D = *attrs->GetInt(MAX_INT8_D_INDEX);
    int64_t maxFloat8D = *attrs->GetInt(MAX_FLOAT8_D_INDEX);
    int64_t poolMode = *attrs->GetInt(POOL_MODE_INDEX);
    int64_t outputDtype = *attrs->GetInt(OUTPUT_DTYPE_INDEX);
    int64_t rowAlignment = *attrs->GetInt(ROW_ALIGNMENT_INDEX);
    int64_t fp8ExponentBits = *attrs->GetInt(FP8_EXPONENT_BITS_INDEX);
    int64_t fp8ExponentBias = *attrs->GetInt(FP8_EXPONENT_BIAS_INDEX);

    // Validate pooling mode
    OPS_CHECK(poolMode < POOL_MODE_SUM || poolMode > POOL_MODE_NOBAG,
              OPS_LOG_E("", "Invalid pooling mode %ld: supported modes are MEAN(0), SUM(1), NONE(2)", poolMode),
              return ge::GRAPH_FAILED);

    int64_t bytesOfDataType = sizeof(uint8_t);
    ge::DataType indicesDataType = context->GetInputDesc(INDICES_INDEX)->GetDataType();
    ge::DataType offsetsDataType = context->GetInputDesc(OFFSETS_INDEX)->GetDataType();
    ge::DataType offsetPerKeyDataType = context->GetInputDesc(OFFSET_PER_KEY_INDEX)->GetDataType();
    OPS_CHECK(indicesDataType != offsetsDataType,
              OPS_LOG_E("", "indices dtype(%d) must match offsets dtype(%d).",
                        static_cast<int>(indicesDataType), static_cast<int>(offsetsDataType)),
              return ge::GRAPH_FAILED);

    // Calculate output shape based on pooling mode
    int64_t outDim0;
    int64_t outDim1;
    int64_t offsetDataType;
    if (poolMode == POOL_MODE_NOBAG) {
        // nobag mode: output shape [indices_num, max_D]
        outDim0 = indicesDim0;
        outDim1 = maxD;
        context->SetTilingKey(NOBAG_KEY);
    } else {
        // bag mode: output shape [batch_size, total_D]
        int64_t batchSize = (offsetsDim0 - 1) / weightsOffsetsDim0;
        OPS_CHECK(batchSize < 0,
                  OPS_LOG_E("", "Invalid batch size %ld calculated from offsets(%ld) and weights_offsets(%ld).",
                            batchSize, offsetsDim0, weightsOffsetsDim0),
                  return ge::GRAPH_FAILED);
        outDim0 = batchSize;
        outDim1 = totalD;

        if (indicesDataType == ge::DT_INT32) {
            offsetDataType = BAG_INT32_KEY;
            context->SetTilingKey(BAG_INT32_KEY);
        } else if (indicesDataType == ge::DT_INT64) {
            offsetDataType = BAG_INT64_KEY;
            context->SetTilingKey(BAG_INT64_KEY);
        } else {
            OPS_LOG_E("", "indices dtype %d not supported, expect int32/int64.", static_cast<int>(indicesDataType));
            return ge::GRAPH_FAILED;
        }
    }

    // Calculate batch size (B) - same for both bag and nobag modes
    int64_t T = weightsOffsetsDim0;
    int64_t B = (offsetsDim0 - 1) / T;

    // Set tiling data
    tilingData.set_maxD(maxD);
    tilingData.set_totalD(totalD);
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
    tilingData.set_rowAlignment(rowAlignment);
    tilingData.set_fp8ExponentBits(fp8ExponentBits);
    tilingData.set_fp8ExponentBias(fp8ExponentBias);
    tilingData.set_maxFloat8D(maxFloat8D);
    tilingData.set_isWeighted(isWeighted);
    tilingData.set_outputDtype(outputDtype);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    IntNbitSplitEmbeddingCodegenLookupFunctionTilingData tiling;
    // Shape and dType
    if (ShapeTilingFunc(context, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Get UB size for kernel
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    tiling.set_ubCanUsed(ubCanUsed);

    // Calculate SIMD tiling parameters
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E("", "Core num is 0."), return ge::GRAPH_FAILED);

    int64_t totalTasks = tiling.get_offsetsDim0() - 1;
    if (tiling.get_poolMode() != POOL_MODE_NOBAG) {
        coreNum = totalTasks < coreNum ? totalTasks : coreNum;
    }
    context->SetBlockDim(coreNum);

    // tiling data
    int64_t splitBaseLen = totalTasks / coreNum;
    int64_t tailSplitIndex = totalTasks % coreNum;
    tiling.set_splitBaseLen(splitBaseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);

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

    // Get input shapes
    const gert::Shape* weightsOffsetsShape = context->GetInputShape(WEIGHTS_OFFSETS_INDEX);
    OPS_LOG_E_IF_NULL("weightsOffsetsShape", weightsOffsetsShape, return ge::GRAPH_FAILED);

    const gert::Shape* indicesShape = context->GetInputShape(INDICES_INDEX);
    OPS_LOG_E_IF_NULL("indicesShape", indicesShape, return ge::GRAPH_FAILED);

    const gert::Shape* offsetsShape = context->GetInputShape(OFFSETS_INDEX);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);

    // Get output shape
    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    // Get attributes
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    int64_t totalD = *attrs->GetInt(TOTAL_D_INDEX);
    int64_t maxD = *attrs->GetInt(MAX_D_INDEX);
    int64_t poolMode = *attrs->GetInt(POOL_MODE_INDEX);

    // Calculate output dimensions based on pooling mode
    int64_t weightsOffsetsDim0 = weightsOffsetsShape->GetDim(0);
    int64_t indicesDim0 = indicesShape->GetDim(0);
    int64_t offsetsDim0 = offsetsShape->GetDim(0);

    if (poolMode == POOL_MODE_NOBAG) {
        // nobag mode: output shape [indices_num, max_D]
        outputShape->SetDimNum(2);
        outputShape->SetDim(0, indicesDim0);
        outputShape->SetDim(1, maxD);
    } else {
        // bag mode: output shape [batch_size, total_D]
        int64_t batchSize = (offsetsDim0 - 1) / weightsOffsetsDim0;
        outputShape->SetDimNum(2);
        outputShape->SetDim(0, batchSize);
        outputShape->SetDim(1, totalD);
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class IntNbitSplitEmbeddingCodegenLookupFunction : public OpDef {
public:
    explicit IntNbitSplitEmbeddingCodegenLookupFunction(const char* name) : OpDef(name)
    {
        // Input definitions
        this->Input("dev_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});

        this->Input("uvm_weights")
            .ParamType(REQUIRED)
            .Follow("dev_weights", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Input("lxu_cache_weights")
            .ParamType(REQUIRED)
            .Follow("dev_weights", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Input("weights_placements")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Input("weights_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});

        this->Input("weights_tys")
            .ParamType(REQUIRED)
            .Follow("dev_weights", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Input("D_offsets")
            .ParamType(REQUIRED)
            .Follow("weights_placements", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Input("indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});

        this->Input("offsets")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Input("lxu_cache_locations")
            .ParamType(REQUIRED)
            .Follow("weights_placements", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Input("offset_per_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Input("indice_weights")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        // Output definition
        this->Output("out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});

        // Attribute definitions
        this->Attr("total_D").Int();
        this->Attr("max_D").Int();
        this->Attr("max_int2_D").Int();
        this->Attr("max_int4_D").Int();
        this->Attr("max_int8_D").Int();
        this->Attr("max_float16_D").Int();
        this->Attr("max_float32_D").Int();
        this->Attr("max_float8_D").Int();
        this->Attr("pooling_mode").Int();
        this->Attr("output_dtype").Int();
        this->Attr("row_alignment").Int();
        this->Attr("fp8_exponent_bits").Int();
        this->Attr("fp8_exponent_bias").Int();

        // Set infer shape function
        this->SetInferShape(ge::InferShape);

        // Set tiling function
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(IntNbitSplitEmbeddingCodegenLookupFunction);
}  // namespace ops

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

#include "dense_embedding_codegen_lookup_function_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t TILING_CONSTANT_0 = 0;
constexpr int64_t INT64_BYTES = sizeof(int64_t);
constexpr int64_t INT32_BYTES = sizeof(int32_t);
constexpr int64_t FLOAT32_BYTES = sizeof(float);
constexpr int32_t DATA_COPY_PAD_MAX_LEN = 4064;
constexpr int32_t RESERVED_UB_SIZE = 2048;
constexpr int32_t BASIC_PROCESS_UNIT_SIZE = 32;

constexpr int32_t DEV_WEIGHTS_INDEX = 0;
constexpr int32_t WEIGHTS_OFFSETS_INDEX = 1;
constexpr int32_t D_OFFSETS_INDEX = 2;
constexpr int32_t HASH_SIZE_CUMSUM_INDEX = 3;
constexpr int32_t INDICES_INDEX = 4;
constexpr int32_t OFFSETS_INDEX = 5;

constexpr int32_t MAX_D_ATTR_INDEX = 1;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("devWeights shape", context->GetInputShape(DEV_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("weightsOffsets shape", context->GetInputShape(WEIGHTS_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indices shape", context->GetInputShape(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsets shape", context->GetInputShape(OFFSETS_INDEX), return ge::GRAPH_FAILED);

    DenseEmbeddingCodegenLookupFunctionTilingData tiling;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OPS_CHECK(aivNum == 0, OPS_LOG_E("", "aivNum is zero."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* max_D_ptr = attrs->GetInt(MAX_D_ATTR_INDEX);
    OPS_LOG_E_IF_NULL("max_D_ptr", max_D_ptr, return ge::GRAPH_FAILED);

    int32_t embed_dim_length = static_cast<int32_t>(*max_D_ptr);

    OPS_CHECK(embed_dim_length == 0, OPS_LOG_E("", "embed_dim_length must be greater than 0"),
              return ge::GRAPH_FAILED);

    OPS_CHECK(context->GetInputShape(DEV_WEIGHTS_INDEX)->GetStorageShape().GetDimNum() <= 0,
              OPS_LOG_E("", "dev_weights dimension number must be greater than 0"),
              return ge::GRAPH_FAILED);

    int32_t dev_weights_length = context->GetInputShape(DEV_WEIGHTS_INDEX)->GetStorageShape().GetDim(0) /
        embed_dim_length;
    int32_t weights_offsets_length = context->GetInputShape(WEIGHTS_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int32_t indices_length = context->GetInputShape(INDICES_INDEX)->GetStorageShape().GetDim(0);
    int32_t offsets_length = context->GetInputShape(OFFSETS_INDEX)->GetStorageShape().GetDim(0);

    OPS_CHECK(weights_offsets_length == 0, OPS_LOG_E("", "weights_offsets_length must be greater than 0"),
              return ge::GRAPH_FAILED);

    int32_t batch_size = (offsets_length - 1) / weights_offsets_length;
    auto indices_dataType = context->GetInputDesc(INDICES_INDEX)->GetDataType();
    int32_t alignedEmbedDimLength = (embed_dim_length + BLOCK_SIZE / FLOAT32_BYTES - 1) / (BLOCK_SIZE / FLOAT32_BYTES) *
        (BLOCK_SIZE / FLOAT32_BYTES);

    OPS_CHECK(alignedEmbedDimLength == 0, OPS_LOG_E("", "alignedEmbedDimLength must be greater than 0"),
              return ge::GRAPH_FAILED);

    int32_t formerCoreNum = 0;
    int32_t formerCoreLength = 0;
    int32_t formerTileNum = 0;
    int32_t formerTileLength = 0;
    int32_t formerLastTileLength = 0;

    int32_t tailCoreNum = 0;
    int32_t tailCoreLength = 0;
    int32_t tailTileNum = 0;
    int32_t tailTileLength = 0;
    int32_t tailLastTileLength = 0;

    context->SetTilingKey(0);
    // 检查 alignedEmbedDimLength * FLOAT32_BYTES 是否溢出
    OPS_CHECK(alignedEmbedDimLength > INT32_MAX / FLOAT32_BYTES,
        OPS_LOG_E("", "alignedEmbedDimLength * FLOAT32_BYTES would overflow"),
        return ge::GRAPH_FAILED);

    // 检查 weights_offsets_length * INT64_BYTES 是否溢出
    OPS_CHECK(weights_offsets_length > INT32_MAX / INT64_BYTES,
        OPS_LOG_E("", "weights_offsets_length * INT64_BYTES would overflow"),
        return ge::GRAPH_FAILED);

    // 检查 offsets_length * INT64_BYTES 是否溢出
    OPS_CHECK(offsets_length > INT32_MAX / INT64_BYTES,
        OPS_LOG_E("", "offsets_length * INT64_BYTES would overflow"),
        return ge::GRAPH_FAILED);

    // 检查 alignedEmbedDimLength * FLOAT32_BYTES 是否溢出
    OPS_CHECK(alignedEmbedDimLength > INT32_MAX / FLOAT32_BYTES,
        OPS_LOG_E("", "alignedEmbedDimLength * FLOAT32_BYTES would overflow in denominator"),
        return ge::GRAPH_FAILED);

    // 检查分母是否为零
    int32_t denominator = alignedEmbedDimLength * FLOAT32_BYTES + INT64_BYTES;
    OPS_CHECK(denominator == 0,
        OPS_LOG_E("", "denominator is zero"),
        return ge::GRAPH_FAILED);

    int32_t tileLength = (ubSize - alignedEmbedDimLength * FLOAT32_BYTES - weights_offsets_length * INT64_BYTES -
        offsets_length * INT64_BYTES - RESERVED_UB_SIZE) /
        denominator;
    if (indices_dataType == ge::DT_INT32) {
        // 检查 weights_offsets_length * INT32_BYTES 是否溢出
        OPS_CHECK(weights_offsets_length > INT32_MAX / INT32_BYTES,
            OPS_LOG_E("", "weights_offsets_length * INT32_BYTES would overflow"),
            return ge::GRAPH_FAILED);

        // 检查 offsets_length * INT32_BYTES 是否溢出
        OPS_CHECK(offsets_length > INT32_MAX / INT32_BYTES,
            OPS_LOG_E("", "offsets_length * INT32_BYTES would overflow"),
            return ge::GRAPH_FAILED);

        // 检查分母部分是否会导致溢出
        int32_t denominator_int32 = alignedEmbedDimLength * FLOAT32_BYTES + INT32_BYTES;
        OPS_CHECK(denominator_int32 == 0,
            OPS_LOG_E("", "denominator is zero for int32 case"),
            return ge::GRAPH_FAILED);

        tileLength = (ubSize - alignedEmbedDimLength * FLOAT32_BYTES - weights_offsets_length * INT32_BYTES -
            offsets_length * INT32_BYTES - RESERVED_UB_SIZE) /
            denominator_int32;
        context->SetTilingKey(1);
    }
    tileLength = std::min(tileLength, DATA_COPY_PAD_MAX_LEN);

    // 检查除零风险
    OPS_CHECK(tileLength == 0, OPS_LOG_E("", "tileLength is zero."), return ge::GRAPH_FAILED);

    if (indices_length <= aivNum) {
        context->SetBlockDim(indices_length);
        formerCoreNum = indices_length;
        formerCoreLength = 1;
        formerTileNum = 1;
        formerTileLength = 1;
        formerLastTileLength = 1;
    } else {
        context->SetBlockDim(aivNum);
        formerCoreNum = indices_length % aivNum;
        if (formerCoreNum == 0) {
            formerCoreNum = aivNum;
            formerCoreLength = (indices_length + aivNum - 1) / aivNum;
            formerTileNum = (formerCoreLength + tileLength - 1) / tileLength;
            formerTileLength = tileLength;
            formerLastTileLength = formerCoreLength - (formerTileNum - 1) * tileLength;
        } else {
            formerCoreLength = (indices_length + aivNum - 1) / aivNum;
            formerTileNum = (formerCoreLength + tileLength - 1) / tileLength;
            formerTileLength = tileLength;
            formerLastTileLength = formerCoreLength - (formerTileNum - 1) * tileLength;
            tailCoreNum = aivNum - formerCoreNum;
            tailCoreLength = indices_length / aivNum;
            tailTileNum = (tailCoreLength + tileLength - 1) / tileLength;
            tailTileLength = tileLength;
            tailLastTileLength = tailCoreLength - (tailTileNum - 1) * tileLength;
        }
    }

    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_formerCoreLength(formerCoreLength);
    tiling.set_formerTileNum(formerTileNum);
    tiling.set_formerTileLength(formerTileLength);
    tiling.set_formerLastTileLength(formerLastTileLength);

    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_tailCoreLength(tailCoreLength);
    tiling.set_tailTileNum(tailTileNum);
    tiling.set_tailTileLength(tailTileLength);
    tiling.set_tailLastTileLength(tailLastTileLength);

    tiling.set_weightsOffsetsLength(weights_offsets_length);
    tiling.set_embedDimLength(embed_dim_length);
    tiling.set_batchSize(batch_size);
    tiling.set_indicesAllLength(indices_length);
    tiling.set_devWeightsLength(dev_weights_length);
    tiling.set_alignedEmbedDimLength(alignedEmbedDimLength);

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    size_t usrSize = 0;
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape *input_shape_indices = context->GetInputShape(INDICES_INDEX);
    OPS_LOG_E_IF_NULL("input_shape_indices", input_shape_indices, return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* max_D_ptr = attrs->GetInt(MAX_D_ATTR_INDEX);
    OPS_LOG_E_IF_NULL("max_D_ptr", max_D_ptr, return ge::GRAPH_FAILED);

    int32_t max_D = static_cast<int32_t>(*max_D_ptr);
    gert::Shape *output_shape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("output_shape", output_shape, return ge::GRAPH_FAILED);

    output_shape->SetDimNum(2);
    output_shape->SetDim(0, input_shape_indices->GetDim(0));
    output_shape->SetDim(1, max_D);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto data_type = context->GetInputDataType(0);
    if (context->SetOutputDataType(0, data_type) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
}


namespace ops {
class DenseEmbeddingCodegenLookupFunction : public OpDef {
public:
    explicit DenseEmbeddingCodegenLookupFunction(const char *name) : OpDef(name)
    {
        this->Input("dev_weights")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("weights_offsets")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("D_offsets")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("hash_size_cumsum")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("indice_weights")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("B_offset")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("vbe_output_offsets_feature_rank")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("vbe_B_offsets_rank_per_feature")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_INT32, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("total_D").Int();
        this->Attr("max_D").Int();
        this->Attr("total_hash_size_bits").Int();
        this->Attr("pooling_mode").Int();
        this->Attr("feature_requires_grad").AttrType(OPTIONAL).Bool(true);
        this->Attr("output_dtype").AttrType(OPTIONAL).String("float32");
        this->Attr("max_B").AttrType(OPTIONAL).Int(0);
        this->Attr("max_B_feature_rank").AttrType(OPTIONAL).Int(0);
        this->Attr("vbe_output_size").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(DenseEmbeddingCodegenLookupFunction);
}
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include "lccl_all_uss_tiling.h"
#include "lccl_comm_def.h"

#include "register/op_def_registry.h"
#include "ops_log.h"

static int g_magic = 10;
static int g_blockDim = 32;
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LcclAllUssTilingData tiling;

    auto* attrs = context->GetAttrs();
    const auto* dim_ = attrs->GetAttrPointer<int64_t>(2);
    const auto* rank_ = attrs->GetAttrPointer<int64_t>(0);
    const auto* rank_Size_ = attrs->GetAttrPointer<int64_t>(1);
    int dim = static_cast<int>(*dim_);
    int rank = static_cast<int>(*rank_);
    int rankSize = static_cast<int>(*rank_Size_);

    const gert::StorageShape* rev_shape = context->GetInputShape(4);  // Get embedding row num.
    
    OPS_CHECK(rankSize <= 0 || rankSize > LCAL_MAX_RANK_SIZE,
        OPS_LOG_E("[ERROR]", "Invalid rankSize: %d\n"
        "Valid range: must be at least 1 and not exceed %d (maximum supported rank size).",
        rankSize, LCAL_MAX_RANK_SIZE),
        return ge::GRAPH_FAILED);

    OPS_CHECK(rank < 0 || rank >= rankSize,
        OPS_LOG_E("[ERROR]", "Invalid rank: %d\n"
        "Valid range: must be at least 0 and not exceed rank size %d.",
        rank, rankSize),
        return ge::GRAPH_FAILED
    )

    tiling.set_rank(rank);
    tiling.set_dim(dim);
    tiling.set_rankSize(rankSize);

    auto outShape = context->GetInputShape(2)->GetStorageShape();
    tiling.set_outShape(outShape.GetDim(0));

    int deterministic = 0;
    const char* envDeterministic = getenv("LCCL_DETERMINISTIC");
    if (envDeterministic != nullptr) {
        std::string envValue(envDeterministic);
        if (envValue == "0") {
            deterministic = 0;
        } else if (envValue == "1") {
            deterministic = 1;
        } else {
            OPS_LOG_W(context->GetNodeName(),
                      "LCCL_DETERMINISTIC environment variable must be 0 or 1, got '%s'. Using default value 0.",
                      envDeterministic);
            deterministic = 0;
        }
    }
    tiling.set_deterministic(deterministic);
    tiling.set_magic(g_magic);

    context->SetBlockDim(g_blockDim);

    uint32_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* y_shape = context->GetOutputShape(0);
    const gert::Shape* rev_shape = context->GetInputShape(2);
    const gert::Shape* table_shape = context->GetInputShape(0);

    y_shape->SetDim(0, rev_shape->GetDim(0));
    y_shape->SetDim(1, table_shape->GetDim(1));
    y_shape->SetDim(2, 1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LcclAllUss : public OpDef {
public:
    explicit LcclAllUss(const char* name) : OpDef(name)
    {
        this->Input("send_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("send_count_matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("shape_vec")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("peer_mem")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("restore")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("recv_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("rank").Int();
        this->Attr("rank_size").Int();
        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(LcclAllUss);
}  // namespace ops

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

#include "lccl_all_to_all_tiling.h"

#include "register/op_def_registry.h"

static int g_magic = 9;
static int g_blockDim = 32;
namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        LcclAllToAllTilingData tiling;

        auto sendBuff = context->GetInputTensor(0);
        auto* attrs = context->GetAttrs();
        const auto* rank_ = attrs->GetAttrPointer<int64_t>(0);
        const auto* rankSize_ = attrs->GetAttrPointer<int64_t>(1);
        int rank = static_cast<int>(*rank_);
        int rankSize = static_cast<int>(*rankSize_);

        if (rankSize <= 0) {
            printf("rankSize cannot be smaller than 1\n");
            return ge::GRAPH_FAILED;
        }
        tiling.set_rank(rank);
        tiling.set_rankSize(rankSize);

        tiling.set_magic(g_magic);

        context->SetBlockDim(g_blockDim);

        // 参考官网默认值 设置workSpace大小
        uint32_t sysWorkspaceSize = 16 * 1024 * 1024;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = sysWorkspaceSize;

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        const gert::Shape* x3_shape = context->GetInputShape(2);
        gert::Shape* y_shape = context->GetOutputShape(0);

        y_shape->SetDim(0, x3_shape->GetDim(0));
        y_shape->SetDim(1, x1_shape->GetDim(1));
        y_shape->SetDim(2, 1);

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}


namespace ops {
    class LcclAllToAll : public OpDef {
    public:
        explicit LcclAllToAll(const char* name) : OpDef(name)
        {
            this->Input("send_data")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("send_count_matrix")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT64})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("shape_vec")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("peer_mem")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT64})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("recv_data")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("rank").Int();
            this->Attr("rank_size").Int();
            this->Attr("dim").Int();

            this->SetInferShape(ge::InferShape);
            this->SetInferDataType(ge::InferDataType);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(LcclAllToAll);
}

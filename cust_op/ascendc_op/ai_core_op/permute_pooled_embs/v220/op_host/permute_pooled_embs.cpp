/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#include "permute_pooled_embs_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"
namespace optiling {
    constexpr int NUM_QUEUE = 4;
    constexpr int UB_ALIGN = 32;
    constexpr int POOLED_EMBS_INDEX = 0;
    constexpr int OFFSET_DIM_LIST_INDEX = 1;
    constexpr int PERMUTE_LIST_INDEX = 2;
    constexpr int INV_OFFSET_DIM_LIST_INDEX = 3;

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        PermutePooledEmbsTilingData tiling;
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

        OPS_LOG_E_IF_NULL("pooledEmbsShape", context->GetInputShape(POOLED_EMBS_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("offsetDimListShape", context->GetInputShape(OFFSET_DIM_LIST_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("permuteListShape", context->GetInputShape(PERMUTE_LIST_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("invOffsetDimListShape", context->GetInputShape(INV_OFFSET_DIM_LIST_INDEX),
            return ge::GRAPH_FAILED);

        auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        gert::Shape pooledEmbsShape = context->GetInputShape(POOLED_EMBS_INDEX)->GetStorageShape();
        gert::Shape offsetDimListShape = context->GetInputShape(OFFSET_DIM_LIST_INDEX)->GetStorageShape();
        gert::Shape permuteListShape = context->GetInputShape(PERMUTE_LIST_INDEX)->GetStorageShape();
        gert::Shape invOffsetDimListShape = context->GetInputShape(INV_OFFSET_DIM_LIST_INDEX)->GetStorageShape();
        // shape check
        if (pooledEmbsShape.GetDimNum() < 2) {
            OPS_LOG_E("", "[ERROR]pooled_embs must be at least 2-D");
            return ge::GRAPH_FAILED;
        }
        if (offsetDimListShape.GetDimNum() != 1 || permuteListShape.GetDimNum() != 1 ||
            invOffsetDimListShape.GetDimNum() != 1) {
            OPS_LOG_E("", "[ERROR]offset_dim_list and permute_list must be 1-D");
            return ge::GRAPH_FAILED;
        }

        // set data dim
        int64_t batchSize = pooledEmbsShape.GetDim(0);  // batch size
        tiling.set_batchSize(batchSize);
        int64_t totalDim = pooledEmbsShape.GetDim(1);  // total embedding dimension
        tiling.set_totalDim(totalDim);
        int64_t totalFeatureNum = permuteListShape.GetDim(0);  // number of features
        tiling.set_totalFeatureNum(totalFeatureNum);

        // Validate offset_dim_list has T+1 elements
        if (offsetDimListShape.GetDim(0) != totalFeatureNum + 1 ||
            invOffsetDimListShape.GetDim(0) != totalFeatureNum + 1) {
            OPS_LOG_E("", "[ERROR]offset_dim_list and inv_offset_dim_list must have totalFeatureNum+1 elements");
            return ge::GRAPH_FAILED;
        }

        // set coreNum
        size_t coreNum = ascendPlatform.GetCoreNumAiv();
        coreNum = coreNum < totalDim ? coreNum : totalDim;
        if (coreNum == 0) {
            return ge::GRAPH_FAILED;
        }

        // set baseBatchLen, tailSplitIndex
        int64_t baseBatchLen = totalDim / coreNum;
        tiling.set_baseBatchLen(baseBatchLen);
        int64_t tailSplitIndex = totalDim % coreNum;
        tiling.set_tailSplitIndex(tailSplitIndex);

        // set ub
        uint64_t ubCanUsed;
        ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
        ubCanUsed = ubCanUsed / UB_ALIGN / NUM_QUEUE * UB_ALIGN * NUM_QUEUE;
        tiling.set_ubCanUsed(ubCanUsed);

        // apply workspace
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
        OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
        currentWorkspace[0] = systemWorkspacesSize;

        context->SetBlockDim(coreNum);

        OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* pooledEmbsShape = context->GetInputShape(optiling::POOLED_EMBS_INDEX);
    OPS_LOG_E_IF_NULL("pooledEmbsShape", pooledEmbsShape, return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(optiling::POOLED_EMBS_INDEX);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    // Output shape is same as input shape
    *outputShape = *pooledEmbsShape;

    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class PermutePooledEmbs : public OpDef {
public:
    explicit PermutePooledEmbs(const char* name) : OpDef(name)
    {
        this->Input("pooled_embs")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("offset_dim_list")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("permute_list")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("inv_offset_dim_list")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Output("permuted_pooled_embs")
            .ParamType(REQUIRED)
            .Follow("pooled_embs", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(PermutePooledEmbs);
}  // namespace ops

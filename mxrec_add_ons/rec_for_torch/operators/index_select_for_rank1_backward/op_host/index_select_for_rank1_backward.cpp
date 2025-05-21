/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "index_select_for_rank1_backward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../common/ops_log.h"

namespace optiling {

constexpr int GM_ALIGN = 64;
constexpr int FLOAT_BYTESIZE = 4;
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

    if (xShape.GetDimNum() != 1) {
        printf("IndexSectForRank1Backward is only used for input-1 with dim 0 but x.dim is %ld",
               xShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (gradShape.GetDimNum() != 1) {
        printf("IndexSectForRank1Backward is only used for input-1 with dim 0 but grad.dim is %ld",
               gradShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (indexShape.GetDimNum() != 1) {
        printf("IndexSectForRank1Backward is only used for input-1 with dim 0 but index.dim is %ld",
               indexShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        printf("[ERROR]No available aicore\n");
        return ge::GRAPH_FAILED;
    }
    int64_t totalLen = indexShape.GetShapeSize();
    int64_t xDim0 = xShape.GetShapeSize();
    int64_t baseLen = indexShape.GetShapeSize() / coreNum;
    int64_t tailSplitIndex = indexShape.GetShapeSize() % coreNum;

    IndexSelectForRank1BackwardTilingData tiling;
    tiling.set_totalLen(totalLen);
    tiling.set_xDim0(xDim0);
    tiling.set_baseLen(baseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);

    context->SetBlockDim(coreNum);
    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(optiling::X_IDX);
    const gert::Shape* indexShape = context->GetInputShape(optiling::INDEX_IDX);

    gert::Shape* gradXShape = context->GetOutputShape(optiling::GRAD_IDX);
    gert::Shape* gradIndexShape = context->GetOutputShape(optiling::INDEX_IDX);

    OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexShape", indexShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gradXShape", gradXShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gradIndexShape", gradIndexShape, return ge::GRAPH_FAILED);

    *gradXShape = *xShape;
    *gradIndexShape = *indexShape;
    return GRAPH_SUCCESS;
}
}

namespace ops {
class IndexSelectForRank1Backward : public OpDef {
public:
    explicit IndexSelectForRank1Backward(const char* name) : OpDef(name)
    {
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("grad_x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("grad_index")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(IndexSelectForRank1Backward);
}

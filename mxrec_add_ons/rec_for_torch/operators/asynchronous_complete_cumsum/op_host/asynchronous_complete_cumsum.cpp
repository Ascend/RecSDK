/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <limits>
#include "asynchronous_complete_cumsum_tiling.h"
#include "register/op_def_registry.h"

#include "../../../common/ops_log.h"

namespace {
    constexpr int32_t EMBEDDING_TYPE_INT64 = 0;
    constexpr int32_t EMBEDDING_TYPE_INT32 = 1;
}

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        AsynchronousCompleteCumsumTilingData tiling;
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

        auto inputTensor0 = context->GetInputTensor(0);
        OPS_LOG_E_IF_NULL("inputTensor0", inputTensor0, return ge::GRAPH_FAILED);

        auto inputShape0 =  context->GetInputShape(0);
        OPS_LOG_E_IF_NULL("inputShape0", inputShape0, return ge::GRAPH_FAILED);

        int64_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
        uint32_t dimNum = context->GetInputShape(0)->GetOriginShape().GetDimNum();

        OPS_CHECK(totalLength >= std::numeric_limits<int32_t>::max(),
            OPS_LOG_E("", "AsynchronousCompleteCumsum only handles up to INT_MAX elements, but got %u.", totalLength),
            return ge::GRAPH_FAILED);

        ge::DataType inputDatatype = inputTensor0->GetDataType();
        uint32_t embeddingType;
        if (inputDatatype == ge::DT_INT64) {
            embeddingType = EMBEDDING_TYPE_INT64;
        } else if (inputDatatype == ge::DT_INT32) {
            embeddingType = EMBEDDING_TYPE_INT32;
        } else {
            OPS_LOG_E("", "Invalid data type. AsynchronousCompleteCumsum only support int64 and int32.");
            return ge::GRAPH_FAILED;
        }

        tiling.set_totalLength(totalLength);
        tiling.set_dimNum(dimNum);
        tiling.set_inputType(embeddingType);

        if (dimNum != 1) {
            OPS_LOG_E("", "AsynchronousCompleteCumsum required the dim of input-0 is 1 but %ld ", dimNum);
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(1);
        OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

        const gert::Shape* xShape = context->GetInputShape(0);
        OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);

        gert::Shape* yShape = context->GetOutputShape(0);
        OPS_LOG_E_IF_NULL("yShape", yShape, return ge::GRAPH_FAILED);

        int64_t inputLength = xShape->GetShapeSize();
        yShape->SetDimNum(1);
        yShape->SetDim(0, inputLength+1);
        return GRAPH_SUCCESS;
    }
}


namespace ops {
    class AsynchronousCompleteCumsum : public OpDef {
    public:
        explicit AsynchronousCompleteCumsum(const char* name) : OpDef(name)
        {
            this->Input("x")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT64, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT64, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);

            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910_93");
            this->AICore().AddConfig("ascend310p");
        }
    };

    OP_ADD(AsynchronousCompleteCumsum);
}
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

#include "permute2d_sparse_data_tilling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../../../common/ops_log.h"
namespace optiling {

    constexpr int GM_ALIGN = 64;
    constexpr int RESERVER_UB_SIZE = 20 * 1024;
    constexpr int DATA_TYPE_INT64 = 8;
    constexpr int DATA_TYPE_INT32 = 4;
    constexpr int DATA_TYPE_FLOAT32 = 4;
    constexpr int NUM_QUEUE = 4;
    constexpr int UB_ALIGN = 32;
    constexpr int SUPPORT_EMBEDDING_DIM_NUM = 2;
    constexpr int PERMUTE_INDEX = 0;
    constexpr int LENGTH_INDEX = 1;
    constexpr int VALUES_INDEX = 2;
    constexpr int WEIGHTS_INDEX = 3;

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        Permute2dSparseDataTilingData tiling;
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("context->GetAttrs", context->GetAttrs(), return ge::GRAPH_FAILED);

        bool enableWeights = (context->GetOptionalInputTensor(WEIGHTS_INDEX) != nullptr);
        tiling.set_enableWeights(enableWeights);

        OPS_LOG_E_IF_NULL("permuteShape", context->GetInputShape(PERMUTE_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("lengthsShape", context->GetInputShape(LENGTH_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("valuesShape", context->GetInputShape(VALUES_INDEX), return ge::GRAPH_FAILED);
        if (enableWeights) {
            OPS_LOG_E_IF_NULL("weightsShape", context->GetInputShape(WEIGHTS_INDEX), return ge::GRAPH_FAILED);
        }

        auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        gert::Shape permuteShape = context->GetInputShape(PERMUTE_INDEX)->GetStorageShape();
        gert::Shape lengthsShape = context->GetInputShape(LENGTH_INDEX)->GetStorageShape();
        gert::Shape valuesShape = context->GetInputShape(VALUES_INDEX)->GetStorageShape();
        gert::Shape weightsShape;
        if (enableWeights) {
            weightsShape = context->GetInputShape(WEIGHTS_INDEX)->GetStorageShape();
        }

        // shape check
        if ((permuteShape.GetDimNum() != 1) || (lengthsShape.GetDimNum() != SUPPORT_EMBEDDING_DIM_NUM))  {
            OPS_LOG_E("", "[ERROR]permute shape or lengths shape is error. ");
            return ge::GRAPH_FAILED;
        }
        if (enableWeights && (valuesShape != weightsShape || valuesShape.GetDimNum() != 1)) {
            OPS_LOG_E("", "[ERROR]values shape or weights shape is error. values.size() = %d, weights.size() = %d\n",
                      valuesShape.GetDim(0), weightsShape.GetDim(0));
            return ge::GRAPH_FAILED;
        }

        // set data dim
        int64_t permuteDim0 = permuteShape.GetDim(0);  // permute[T]
        tiling.set_permuteDim0(permuteDim0);
        int64_t lengthsT = lengthsShape.GetDim(0);  // lengths[T + T', B]
        tiling.set_lengthsT(lengthsT);
        int64_t lengthsB = lengthsShape.GetDim(1);  // lengths[T + T', B]
        tiling.set_lengthsB(lengthsB);
        int64_t valuesDim = valuesShape.GetDim(0);  // values[L]
        tiling.set_valuesDim(valuesDim);
        int64_t valuesOutDim = *context->GetAttrs()->GetInt(0);
        tiling.set_valuesOutDim(valuesOutDim);

        // set coreNUm
        size_t coreNum = ascendPlatform.GetCoreNumAiv();
        OPS_CHECK(coreNum == 0,
                  OPS_LOG_E("Tiling Debug", "Core num is 0."),
                  return ge::GRAPH_FAILED);
        tiling.set_coreNum(coreNum);

        // tiling core
        int64_t totalBatch = permuteDim0;
        tiling.set_totalBatch(totalBatch);
        int64_t baseBatchLen = permuteDim0 / coreNum;
        tiling.set_baseBatchLen(baseBatchLen);
        int64_t tailSplitIndex = permuteDim0 % coreNum;
        tiling.set_tailSplitIndex(tailSplitIndex);

        // set ub
        uint64_t ubCanUsed;
        ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
        ubCanUsed = (ubCanUsed - RESERVER_UB_SIZE) / UB_ALIGN / NUM_QUEUE * UB_ALIGN * NUM_QUEUE;
        tiling.set_ubCanUsed(ubCanUsed);

        // apply workspace
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
        // 使用workspace共享lengths.sum(dim=1) + 各core计算的offsets结果
        // 为保证workspace同步成功需要保证首地址的32位对齐,因此乘以64
        size_t userWorkspacesSize = (lengthsT + 1) * GM_ALIGN * (coreNum + 1);
        currentWorkspace[0] = systemWorkspacesSize + userWorkspacesSize;

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
  
    const gert::Shape* permuteShape = context->GetInputShape(optiling::PERMUTE_INDEX);
    const gert::Shape* lengthsShape = context->GetInputShape(optiling::LENGTH_INDEX);
    const gert::Shape* valuesShape = context->GetInputShape(optiling::VALUES_INDEX);

    gert::Shape* outPermutedLengths = context->GetOutputShape(optiling::PERMUTE_INDEX);
    gert::Shape* outPermutedValues = context->GetOutputShape(optiling::LENGTH_INDEX);

    OPS_LOG_E_IF_NULL("permuteShape", permuteShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lengthsShape", lengthsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesShape", valuesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outPermutedLengths", outPermutedLengths, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outPermutedValues", outPermutedValues, return ge::GRAPH_FAILED);

    int dimSize = 2;
    outPermutedLengths->SetDimNum(dimSize);
    outPermutedLengths->SetDim(0, lengthsShape->GetDim(0));
    outPermutedLengths->SetDim(1, lengthsShape->GetDim(1));

    outPermutedValues->SetDimNum(1);
    outPermutedValues->SetDim(0, valuesShape->GetDim(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Permute2dSparseData : public OpDef {
public:
    explicit Permute2dSparseData(const char* name) : OpDef(name)
    {
        this->Input("permute")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("lengths")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Output("permuted_lengths")
            .ParamType(REQUIRED)
            .Follow("lengths", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("permuted_values")
            .ParamType(REQUIRED)
            .Follow("values", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("permuted_weights")
            .ParamType(OPTIONAL)
            .Follow("weights", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Attr("permuted_sum").Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Permute2dSparseData);
}  // namespace ops

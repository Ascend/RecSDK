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

#include <cstdint>
#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"
#include "expand_into_jagged_permute_tiling.h"

constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int DCACHE_SIZE = 128 * 1024;

// input index
constexpr int PERMUTE_INDEX = 0;
constexpr int INPUT_OFFSETS_INDEX = 1;
constexpr int OUTPUT_OFFSETS_INDEX = 2;
// attr index
constexpr int OUTPUT_SIZE_INDEX = 0;
// input/output dim
constexpr int PERMUTE_DIM = 1;
constexpr int OFFSETS_DIM = 1;

namespace optiling {

static ge::graphStatus ShapeTilingFunc(gert::TilingContext* context,
                                       ExpandIntoJaggedPermuteTilingData& tilingData)
{
    OPS_LOG_E_IF_NULL("permuteShape", context->GetInputShape(PERMUTE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("permuteTensor", context->GetInputTensor(PERMUTE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputoffsetShape", context->GetInputShape(INPUT_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputoffsetTensor", context->GetInputTensor(INPUT_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outputoffsetShape", context->GetInputShape(OUTPUT_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outputoffsetTensor", context->GetInputTensor(OUTPUT_OFFSETS_INDEX), return ge::GRAPH_FAILED);

    const gert::StorageShape* permuteShape = context->GetInputShape(PERMUTE_INDEX);
    const gert::StorageShape* inputoffsetShape = context->GetInputShape(INPUT_OFFSETS_INDEX);
    const gert::StorageShape* outputoffsetShape = context->GetInputShape(OUTPUT_OFFSETS_INDEX);

    auto permuteStorageShape = permuteShape->GetStorageShape();
    auto inputoffsetStorageShape = inputoffsetShape->GetStorageShape();
    auto outputoffsetStorageShape = outputoffsetShape->GetStorageShape();

    int64_t permuteLen = permuteStorageShape.GetDim(0);
    int64_t inputoffsetLen = inputoffsetStorageShape.GetDim(0);
    int64_t outputoffsetLen = outputoffsetStorageShape.GetDim(0);

    // 获取属性
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const int64_t* outputSizePtr = attrs->GetAttrPointer<int64_t>(OUTPUT_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("outputSizePtr", outputSizePtr, return ge::GRAPH_FAILED);
    int64_t outputSize = static_cast<int64_t>(*outputSizePtr);

    // 获取数据类型
    auto permuteType = context->GetInputDesc(PERMUTE_INDEX)->GetDataType();
    auto inputoffsetType = context->GetInputDesc(INPUT_OFFSETS_INDEX)->GetDataType();
    auto outputoffsetType = context->GetInputDesc(OUTPUT_OFFSETS_INDEX)->GetDataType();

    // 检查数据类型
    OPS_CHECK(permuteType != inputoffsetType || permuteType != outputoffsetType,
              OPS_LOG_E("Tiling Debug",
                        "The datatype of permute, inputoffset and outputoffset must be the same."),
              return ge::GRAPH_FAILED);

    OPS_CHECK(permuteType != ge::DT_INT32 && permuteType != ge::DT_INT64,
              OPS_LOG_E("Tiling Debug",
                        "Invalid data type. ExpandIntoJaggedPermute only support int64 and int32."),
              return ge::GRAPH_FAILED);

    // 检查维度
    OPS_CHECK(permuteStorageShape.GetDimNum() != PERMUTE_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for permute is %d, but the actual dim is %d.",
                        PERMUTE_DIM, permuteStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(inputoffsetStorageShape.GetDimNum() != OFFSETS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for inputoffset is %d, but the actual dim is %d.",
                        OFFSETS_DIM, inputoffsetStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(outputoffsetStorageShape.GetDimNum() != OFFSETS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for outputoffset is %d, but the actual dim is %d.",
                        OFFSETS_DIM, outputoffsetStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);

    // 检查长度关系
    OPS_CHECK(permuteLen + 1 != outputoffsetLen || permuteLen + 1 != inputoffsetLen,
              OPS_LOG_E("Tiling Debug",
                        "outputoffsetShape and inputoffsetShape should be one more than permuteShape. "
                        "permuteLen=%lld, inputoffsetLen=%lld, outputoffsetLen=%lld.",
                        permuteLen, inputoffsetLen, outputoffsetLen),
              return ge::GRAPH_FAILED);

    OPS_CHECK(permuteLen <= 0,
              OPS_LOG_E("Tiling Debug",
                        "permute length must be greater than 0, got %lld.",
                        permuteLen),
              return ge::GRAPH_FAILED);

    OPS_CHECK(outputSize <= 0,
              OPS_LOG_E("Tiling Debug",
                        "output_size must be greater than 0, got %lld.",
                        outputSize),
              return ge::GRAPH_FAILED);

    tilingData.set_outputSize(outputSize);
    tilingData.set_permuteLen(permuteLen);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    ExpandIntoJaggedPermuteTilingData tiling;
    // Shape and dType
    if (ShapeTilingFunc(context, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Get UB size for kernel
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    tiling.set_ubCanUsed(ubCanUsed);

    // tiling data
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0,
              OPS_LOG_E("", "Core num is 0."),
              return ge::GRAPH_FAILED);

    int64_t totalTasks = tiling.get_permuteLen();
    int64_t actualCoreNum = totalTasks < coreNum ? totalTasks : coreNum;
    int64_t splitBaseLen = totalTasks / actualCoreNum;
    int64_t tailSplitIndex = totalTasks % actualCoreNum;
    tiling.set_splitBaseLen(splitBaseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);

    context->SetBlockDim(actualCoreNum);
    context->SetLocalMemorySize(DCACHE_SIZE);

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
    // Get output shape
    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    const int64_t* outputSizePtr = attrs->GetAttrPointer<int64_t>(OUTPUT_SIZE_INDEX);
    int64_t outputSize = static_cast<int64_t>(*outputSizePtr);

    outputShape->SetDimNum(1);
    outputShape->SetDim(0, outputSize);

    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ExpandIntoJaggedPermute : public OpDef {
public:
    explicit ExpandIntoJaggedPermute(const char* name) : OpDef(name)
    {
        this->Input("permute")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("input_offsets")
            .ParamType(REQUIRED)
            .Follow("permute", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("output_offsets")
            .ParamType(REQUIRED)
            .Follow("permute", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Output("output_permute")
            .ParamType(REQUIRED)
            .Follow("permute", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Attr("output_size").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(ExpandIntoJaggedPermute);

}  // namespace ops

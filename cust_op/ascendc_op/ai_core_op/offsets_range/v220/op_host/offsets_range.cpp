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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "offsets_range_tiling.h"

namespace optiling {
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int BUFFER_PARAM = 2048;
constexpr int64_t MAX_OFFSETS_LEN = 1LL << 17;
constexpr int64_t MAX_RANGE_SIZE = 1LL << 32;

static inline int64_t ComputeLutSize(int64_t rangeSize, int64_t totalRows, uint64_t availableSpace, bool isInt64)
{
    // 每个 LUT 元素占用的字节数
    const uint64_t elemSize = isInt64 ? sizeof(int64_t) : sizeof(int32_t);

    if (totalRows <= 0 || rangeSize <= 0) {
        return 1;
    }

    // buffer size 优化
    long double ratio = (static_cast<long double>(BUFFER_PARAM) / static_cast<long double>(totalRows)) *
                        static_cast<long double>(rangeSize);
    uint64_t lutRaw = static_cast<uint64_t>(std::floor(std::sqrt(static_cast<double>(ratio)))) + 1;  // 避免除 0

    // 32B 对齐
    const uint64_t elemsPer32B = isInt64 ? 4ULL : 8ULL;
    const uint64_t elemsPer32BMask = elemsPer32B - 1;
    const uint64_t dataAlignMask = DATA_ALIGN_BYTES - 1;
    uint64_t lutAligned = (lutRaw + elemsPer32BMask) & ~elemsPer32BMask;  // 向上对齐
    uint64_t lutMax = (availableSpace & ~dataAlignMask) / elemSize;       // 向下对齐

    // 上限裁剪
    lutAligned = std::min(lutAligned, lutMax);

    return static_cast<int64_t>(lutAligned);
}

static inline size_t ComputeBlockDim(int64_t rangeSize, int64_t lutSize, size_t coreNum)
{
    if (rangeSize <= 0 || lutSize <= 0 || coreNum == 0) {
        return 0;
    }

    // 每个 core 至少处理一个 LUT 大小的数据，避免小输入时拉起过多空闲 core。
    const int64_t safeLutSize = std::max<int64_t>(lutSize, 1);
    int64_t blockDim = (rangeSize + safeLutSize - 1) / safeLutSize;
    blockDim = std::max<int64_t>(1, blockDim);
    blockDim = std::min<int64_t>(blockDim, static_cast<int64_t>(coreNum));
    return static_cast<size_t>(blockDim);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OffsetsRangeTilingData tiling;

    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsets", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("rangeSize", context->GetAttrs()->GetInt(0), return ge::GRAPH_FAILED);

    auto offsetsShape = context->GetInputShape(0)->GetStorageShape();
    OPS_LOG_E_IF(offsetsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange requires offsets to be a 1D tensor.");
    int64_t offsetsLen = offsetsShape.GetDim(0);
    OPS_LOG_E_IF(offsetsLen <= 0 || offsetsLen > MAX_OFFSETS_LEN, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange offsets length must be in [1, %lld], got %lld.", MAX_OFFSETS_LEN, offsetsLen);
    int64_t rangeSize = *context->GetAttrs()->GetInt(0);
    OPS_LOG_E_IF(rangeSize <= 0 || rangeSize > MAX_RANGE_SIZE, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange rangeSize must be in [1, %lld], got %lld.", MAX_RANGE_SIZE, rangeSize);

    ge::DataType inputDataType = context->GetInputTensor(0)->GetDataType();
    bool isInt64 = false;
#ifdef SUPPORT_950
    OPS_LOG_E_IF(inputDataType != ge::DT_INT32 && inputDataType != ge::DT_INT64, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange only supports int32 and int64 offsets.");
    isInt64 = (inputDataType == ge::DT_INT64);
#else
    OPS_LOG_E_IF(inputDataType != ge::DT_INT32, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange only supports int32 offsets on this build.");
#endif

    const int64_t elemSize = isInt64 ? static_cast<int64_t>(sizeof(int64_t)) : static_cast<int64_t>(sizeof(int32_t));
    OPS_LOG_E_IF(rangeSize > std::numeric_limits<int64_t>::max() / elemSize, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange output byte size overflows int64.");

    // platform
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_LOG_E_IF(coreNum == 0, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange requires at least one available ai core.");

    // workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    int64_t totalRows = offsetsLen;

    // UB space
    uint64_t ubSize;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t reservedSpace = static_cast<uint64_t>(DATA_ALIGN_BYTES) * 2ULL;
    if (ubSize <= reservedSpace) {
        OPS_LOG_E("[ERROR]", "UB space is not enough, current UB space = %llu, less than reserved space = %llu", ubSize,
                  reservedSpace);
        return ge::GRAPH_FAILED;
    }
    uint64_t availableSpace = ubSize - reservedSpace;

    // LUT size
    int64_t lutSize = ComputeLutSize(rangeSize, totalRows, availableSpace / 3, isInt64);  // 1/3 LUT, 2/3 double buffer
    OPS_LOG_E_IF(lutSize <= 0, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange computed an invalid lutSize: %lld.", lutSize);

    size_t blockDim = ComputeBlockDim(rangeSize, lutSize, coreNum);
    OPS_LOG_E_IF(blockDim == 0, context, return ge::GRAPH_FAILED, "[ERROR] OffsetsRange computed an invalid blockDim.");

    tiling.set_offsetsLen(offsetsLen);
    tiling.set_totalRows(totalRows);
    tiling.set_rangeSize(rangeSize);
    tiling.set_lutSize(lutSize);

    context->SetBlockDim(blockDim);

    OPS_LOG_E_IF_NULL("context->GetRawTilingData(0)", context->GetRawTilingData(), return ge::GRAPH_FAILED);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("rangeSizeAttr", context->GetAttrs()->GetInt(0), return ge::GRAPH_FAILED);

    int64_t rangeSize = *context->GetAttrs()->GetInt(0);
    OPS_LOG_E_IF(rangeSize <= 0 || rangeSize > optiling::MAX_RANGE_SIZE, context, return ge::GRAPH_FAILED,
                 "[ERROR] OffsetsRange rangeSize must be in [1, %lld], got %lld.", optiling::MAX_RANGE_SIZE, rangeSize);

    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->SetDimNum(1);
    outShape->SetDim(0, rangeSize);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class OffsetsRange : public OpDef {
public:
    explicit OffsetsRange(const char* name) : OpDef(name)
    {
#ifdef SUPPORT_950
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
#else
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
#endif
        this->Attr("rangeSize").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(OffsetsRange);
}  // namespace ops

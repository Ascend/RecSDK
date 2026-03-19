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

#include "init_address_lookup_tiling.h"
#include "register/op_def_registry.h"
#include "ops_log.h"

namespace optiling {
constexpr int32_t MIN_BLOCK_SIZE = 32;  // UB空间的数据都要按照32字节对齐

template <typename T>
static ge::graphStatus CheckNullPointer(T* pointer, const char* errorMessage)
{
    if (pointer == nullptr) {
        printf("%s nullptr\n", errorMessage);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (CheckNullPointer(context, "Tiling context") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 动态获取系统参数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    const uint32_t UB_LIMIT = ub_size;

    auto bufferOffsetsTensor = context->GetInputTensor(0);
    if (CheckNullPointer(bufferOffsetsTensor, "bufferOffsetsTensor") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto embSizesTensor = context->GetInputTensor(1);
    if (CheckNullPointer(embSizesTensor, "embSizesTensor") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto addressLookupsTensor = context->GetInputTensor(2);
    if (CheckNullPointer(addressLookupsTensor, "addressLookupsTensor") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 根据 emb_sizes 的数据类型设置 TilingKey
    // TilingKey 0 = int64, TilingKey 1 = int32
    auto embSizesDataType = embSizesTensor->GetDataType();
    int64_t elementSize;
    if (embSizesDataType == ge::DT_INT32) {
        elementSize = sizeof(int32_t);  // 4
    } else {
        elementSize = sizeof(int64_t);  // 8
    }

    // num_tables = buffer_offsets.size() - 1
    int64_t bufferOffsetsLen = bufferOffsetsTensor->GetShapeSize();
    int64_t numTables = bufferOffsetsLen - 1;

    if (numTables <= 0) {
        printf("buffer_offsets must have at least 2 elements\n");
        return ge::GRAPH_FAILED;
    }

    // 获取总行数：从输出tensor的shape获取（调用方已正确设置）
    int64_t totalRows = addressLookupsTensor->GetShapeSize();
    if (totalRows <= 0) {
        totalRows = 1;  // 边界情况
    }

    // UB空间可容纳的元素数
    int64_t reservedSpace = MIN_BLOCK_SIZE * 2;
    int64_t availableSpace = UB_LIMIT - reservedSpace;
    int64_t elemsPerBlock = MIN_BLOCK_SIZE / elementSize;  // 32字节对齐：int64=4, int32=8

    // 向量化模式：UB需同时容纳LUT缓冲和输出缓冲，各占一半
    int64_t lutSize = (availableSpace / (2 * elementSize));
    lutSize = (lutSize / elemsPerBlock) * elemsPerBlock;
    if (lutSize <= 0) {
        lutSize = elemsPerBlock;
    }

    size_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E("", "aivNum is zero."), return ge::GRAPH_FAILED);

    // 表内多核切分：blockDim 由总行数和 Kernel 侧对齐策略共同决定
    int64_t alignElems = (MIN_BLOCK_SIZE * 4) / elementSize;  // 128字节对齐的元素数
    int64_t maxActiveCores = totalRows / alignElems;
    if (maxActiveCores <= 0) {
        maxActiveCores = 1;
    }
    size_t blockDim = (static_cast<size_t>(maxActiveCores) < coreNum) ? static_cast<size_t>(maxActiveCores) : coreNum;
    if (blockDim == 0) {
        blockDim = 1;
    }

    InitAddressLookupTilingData tiling;
    tiling.set_num_tables(numTables);
    tiling.set_total_rows(totalRows);
    tiling.set_core_num(static_cast<int64_t>(blockDim));
    tiling.set_lut_size(lutSize);

    if (CheckNullPointer(context->GetRawTilingData(), "RawTilingData") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (optiling::CheckNullPointer(context, "context") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (optiling::CheckNullPointer(outputShape, "outputShape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 输出shape等于输入的address_lookups的shape（inplace操作）
    auto* addressLookupsTensor = context->GetInputShape(2);
    if (optiling::CheckNullPointer(addressLookupsTensor, "addressLookupsTensor") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 复制输入shape到输出
    outputShape->SetDimNum(addressLookupsTensor->GetDimNum());
    for (size_t i = 0; i < addressLookupsTensor->GetDimNum(); i++) {
        outputShape->SetDim(i, addressLookupsTensor->GetDim(i));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (optiling::CheckNullPointer(context, "context") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 输出类型由 address_lookups（第3个输入）的类型推断（inplace操作，输出与输入buffer类型一致）
    auto addressLookupsDataType = context->GetInputDataType(2);
    context->SetOutputDataType(0, addressLookupsDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class InitAddressLookup : public OpDef {
public:
    explicit InitAddressLookup(const char* name) : OpDef(name)
    {
        this->Input("buffer_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("emb_sizes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("address_lookups")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("address_lookups_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false);

        this->AICore().AddConfig("ascend950", aicConfig);
    }
};

OP_ADD(InitAddressLookup);
}  // namespace ops

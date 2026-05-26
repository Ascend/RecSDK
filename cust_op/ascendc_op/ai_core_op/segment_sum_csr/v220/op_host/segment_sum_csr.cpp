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

#include "segment_sum_csr_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

constexpr int64_t StartIndexByInt64 = 4;
constexpr int64_t StartIndexByInt32 = 0;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("csrSeg", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("values", context->GetInputTensor(1), return ge::GRAPH_FAILED);

    // platform info
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }

    // 获取UB大小
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    // UB 预留已由 REDUCE_BUF_SIZE + ALIGNMENT_MARGIN 覆盖，无需额外固定预留

    // 输入shape
    OPS_LOG_E_IF_NULL("x1Shape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("x2Shape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    const gert::StorageShape* x1Shape = context->GetInputShape(0);
    const gert::Shape shape1 = x1Shape->GetStorageShape();
    const auto segmentNums = shape1.GetDim(0) - 1;
    const gert::StorageShape* x2Shape = context->GetInputShape(1);
    const gert::Shape shape2 = x2Shape->GetStorageShape();
    const auto totalLength = shape2.GetDim(0);

    // 输入类型
    ge::DataType csrType = context->GetInputTensor(0)->GetDataType();
    ge::DataType valuesType = context->GetInputTensor(1)->GetDataType();

    int64_t baseCoreSegments = segmentNums / coreNum;
    int64_t remainedSegments = segmentNums % coreNum;
    int64_t formerCoreSegments = baseCoreSegments + 1;

    // 读取 batchSize
    int64_t batchSizeVal = 0;
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    if (csrType == ge::DataType::DT_INT32) {
        const int32_t* batchSize = context->GetAttrs()->GetAttrPointer<int32_t>(0);
        batchSizeVal = static_cast<int64_t>(*batchSize);
    } else {
        const int64_t* batchSize = context->GetAttrs()->GetAttrPointer<int64_t>(0);
        batchSizeVal = *batchSize;
    }

    constexpr int64_t REDUCE_BUF_SIZE = sizeof(float) * 1024LL;  // reducesumTmpBuf 固定 4KB
    constexpr int64_t ALIGNMENT_MARGIN = 2048LL;                 // 对齐/管理余量 2KB
    int64_t outQueueSize = 0;
    int64_t perElementCost = 0;

    if (valuesType == ge::DataType::DT_FLOAT) {
        outQueueSize = sizeof(float) * segmentNums;
        perElementCost = sizeof(float) + sizeof(float);  // valuesBuf(4B) + castTmpBuf(4B)
    } else {
        // fp16 / bf16: valuesBuf(2B) + castTmpBuf(4B)
        outQueueSize = (sizeof(float) / 2) * static_cast<uint64_t>(segmentNums);
        perElementCost = (sizeof(float) / 2) + sizeof(float);  // 2 + 4 = 6
    }

    int64_t maxSegFromUB = 0;
    if (ubCanUsed > outQueueSize + REDUCE_BUF_SIZE + ALIGNMENT_MARGIN) {
        int64_t remainingUB = ubCanUsed - outQueueSize - REDUCE_BUF_SIZE - ALIGNMENT_MARGIN;
        maxSegFromUB = remainingUB / perElementCost;
    }

    int64_t maxSegmentLen = maxSegFromUB > totalLength ? totalLength : maxSegFromUB;

    // 设置 TilingKey，其中根据csrType的类型设置起始的TilingKey，用于kernel侧针对不同的参数类型组合（<csrType, valueType>）创建对象，
    // 组合如下：
    // tilingKey = 0 -> <int32_t, float>
    // tilingKey = 1 -> <int32_t, half>
    // tilingKey = 2 -> <int32_t, bfloat16_t>
    // tilingKey = 4 -> <int64_t, float>
    // tilingKey = 5 -> <int64_t, half>
    // tilingKey = 6 -> <int64_t, bfloat16_t>
    int64_t tilingKey = (csrType == ge::DataType::DT_INT64 ? StartIndexByInt64 : StartIndexByInt32);
    if (valuesType == ge::DataType::DT_FLOAT16) {
        tilingKey += 1;
    } else if (valuesType == ge::DataType::DT_BF16) {
        tilingKey += 2;
    }

    // 设置 tiling 参数
    SegmentSumCsrTilingData tiling;
    tiling.set_coreNum(coreNum);
    tiling.set_segmentNums(segmentNums);
    tiling.set_csrSegLength(shape1.GetDim(0));
    tiling.set_totalLength(totalLength);
    tiling.set_baseCoreSegments(baseCoreSegments);
    tiling.set_remainedSegments(remainedSegments);
    tiling.set_formerCoreSegments(formerCoreSegments);
    tiling.set_batchSize(batchSizeVal);
    tiling.set_maxSegmentLen(maxSegmentLen);

    context->SetBlockDim(coreNum);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* csrSegshape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    int64_t input_dim0 = csrSegshape->GetDim(0);
    yShape->SetDimNum(1);
    yShape->SetDim(0, input_dim0 - 1);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SegmentSumCsr : public OpDef {
public:
    explicit SegmentSumCsr(const char* name) : OpDef(name)
    {
        this->Input("csrSeg")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("batchSize").Int();
        this->SetInferShape(ge::InferShape);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend310p", aicore_config);
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950", aicore_config);
#endif
    }
};

OP_ADD(SegmentSumCsr);
}  // namespace ops

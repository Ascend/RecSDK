/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#include "in_linear_silu_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

using namespace std;

namespace {
static const uint64_t BLOCK_SIZE = 32;
static const uint64_t HALF_SIZE = 2;
static const uint64_t FLOAT_SIZE = 4;
static const uint64_t GM_ALIGN = 512;
}  // namespace

namespace optiling {
constexpr size_t SPLIT_ARG_LIST_SIZE = 4;
ge::graphStatus DoLibApiTiling(InLinearSiluTilingData& tiling, gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    matmul_tiling::DataType dataType = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType cDataType = matmul_tiling::DataType::DT_FLOAT;
    if (context->GetTilingKey() == 0) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
        cDataType = matmul_tiling::DataType::DT_FLOAT16;
    } else if (context->GetTilingKey() == 2) {
        dataType = matmul_tiling::DataType::DT_BF16;
        cDataType = matmul_tiling::DataType::DT_BF16;
    }
    matmul_tiling::MatmulApiTiling cubeTiling;
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, cDataType);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(tiling.get_formerSingleLen(), tiling.get_nShape(),
                        tiling.get_kShape());  // 设置单核计算的M、N、K大小
    cubeTiling.SetOrgShape(tiling.get_mShape(), tiling.get_nShape(),
                           tiling.get_kShape());  // 设置原始输入M、N、K大小
    cubeTiling.EnableBias(true);                  // 设置matmul计算包含bias
    cubeTiling.SetBufferSpace(-1, -1, -1);        // 设定允许使用的空间， 缺省使用该AI处理器所有空间
    if (cubeTiling.GetTiling(tiling.cubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("wShape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xInputDesc", context->GetInputDesc(0), return ge::GRAPH_FAILED);
    InLinearSiluTilingData tiling;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // vector tiling分核计算
    // 切分思路, x[m, k] y[n, k] 主要保存nk矩阵， 对m进行40核切分
    const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape wShape = context->GetInputShape(1)->GetStorageShape();
    uint32_t m = xShape.GetDim(0);
    uint32_t k = xShape.GetDim(1);
    uint32_t n = wShape.GetDim(0);
    ge::DataType xDtype = context->GetInputDesc(0)->GetDataType();
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    auto list = context->GetAttrs()->GetListInt(0);
    OPS_LOG_E_IF_NULL("list", list, return ge::GRAPH_FAILED);
    size_t size = list->GetSize();
    OPS_LOG_E_IF(size != SPLIT_ARG_LIST_SIZE, context, return ge::GRAPH_FAILED,
                 "[ERROR]InLinearSilu attrs split_arg_list only support 4");
    const int64_t* data = list->GetData();
    OPS_LOG_E_IF_NULL("data", data, return ge::GRAPH_FAILED);
    uint32_t uLength = data[0];
    uint32_t vLength = data[1];
    uint32_t qLength = data[2];
    uint32_t kLength = data[3];

    uint64_t dtypeSize = HALF_SIZE;
    context->SetTilingKey(0);
    if (xDtype == ge::DT_FLOAT) {
        dtypeSize = FLOAT_SIZE;
        context->SetTilingKey(1);
    } else if (xDtype == ge::DT_BF16) {
        context->SetTilingKey(2);
        dtypeSize = FLOAT_SIZE;
    }

    // 单核ub支持最大m长度
    uint32_t ubLength = ubSize / dtypeSize / 2;  // 需要用到两个local tensor silu不支持源地址目标地址重叠
    OPS_LOG_E_IF(n == 0, context, return ge::GRAPH_FAILED, "[ERROR] n must be greater than 0");
    uint32_t mMax = ubLength / n;
    OPS_LOG_E_IF(mMax == 0, context, return ge::GRAPH_FAILED, "[ERROR] mMax must be greater than 0");

    uint32_t formerNum = 0;
    uint32_t tailNum = 0;
    uint32_t formerSingleLen = 0;
    uint32_t tailSingleLen = 0;
    uint32_t formerPerLoopLen = 0;
    uint32_t tailPerLoopLen = 0;
    uint32_t formerLoop = 0;
    uint32_t tailLoop = 0;
    uint32_t formerRemain = 0;
    uint32_t tailRemain = 0;

    // 计算vector的核数, context是aic cube的核数， 真正vector数是2 * blockDim
    uint32_t blockDim = aivNum / 2;
    if (m <= aivNum) {
        blockDim = (m + 1) / 2;
        aivNum = m;
    }
    OPS_LOG_E_IF(aivNum == 0, context, return ge::GRAPH_FAILED, "[ERROR] aivNum must be greater than 0");
    formerNum = m % aivNum;
    if (formerNum == 0) {
        formerNum = aivNum;
        formerSingleLen = m / aivNum;
    } else {
        tailNum = aivNum - formerNum;
        // 大核需要考虑向上取整
        formerSingleLen = (m + aivNum - 1) / aivNum;
    }

    // 大核单核内切分策略
    if (formerSingleLen > mMax) {
        formerLoop = formerSingleLen / mMax;
        // 大块单核一次计算的大小
        formerPerLoopLen = mMax;
        // 大块m大于ub时候的多余长度
        formerRemain = formerSingleLen % mMax;
    } else {
        formerLoop = 1;
        formerPerLoopLen = formerSingleLen;
    }

    // 存在小块
    if (tailNum > 0) {
        // 小块单核m长度
        tailSingleLen = m / aivNum;
        if (tailSingleLen > mMax) {
            tailLoop = tailSingleLen / mMax;
            // 大块单核一次计算的大小
            tailPerLoopLen = mMax;
            // 大块m大于ub时候的多余长度
            tailRemain = tailSingleLen % mMax;
        } else {
            tailPerLoopLen = tailSingleLen;
            tailLoop = 1;
        }
    }

    tiling.set_mShape(m);
    tiling.set_kShape(k);
    tiling.set_nShape(n);
    tiling.set_uLength(uLength);
    tiling.set_vLength(vLength);
    tiling.set_qLength(qLength);
    tiling.set_kLength(kLength);
    tiling.set_formerNum(formerNum);
    tiling.set_tailNum(tailNum);
    tiling.set_formerSingleLen(formerSingleLen);
    tiling.set_tailSingleLen(tailSingleLen);
    tiling.set_formerLoop(formerLoop);
    tiling.set_tailLoop(tailLoop);
    tiling.set_formerPerLoopLen(formerPerLoopLen);
    tiling.set_tailPerLoopLen(tailPerLoopLen);
    tiling.set_formerRemain(formerRemain);
    tiling.set_tailRemain(tailRemain);
    tiling.set_blockDim(aivNum);

    // cube tiling计算
    OPS_LOG_E_IF(DoLibApiTiling(tiling, context) != ge::GRAPH_SUCCESS, context, return ge::GRAPH_FAILED,
                 "[ERROR]InLinearSilu DoLibApiTiling failed");

    context->SetBlockDim(blockDim);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);

    // GM双缓存mmout数据
    int32_t bufferSize = (ubLength * dtypeSize + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    tiling.set_bufferSize(bufferSize);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t userSize = static_cast<size_t>(aivNum * bufferSize * 2);
    currentWorkspace[0] = userSize + sysWorkspaceSize;
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("x1_shape", x1_shape, return ge::GRAPH_FAILED);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("y_shape", y_shape, return ge::GRAPH_FAILED);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class InLinearSilu : public OpDef {
public:
    explicit InLinearSilu(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("user")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("split_arg_list").AttrType(OPTIONAL).ListInt();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(InLinearSilu);
}  // namespace ops
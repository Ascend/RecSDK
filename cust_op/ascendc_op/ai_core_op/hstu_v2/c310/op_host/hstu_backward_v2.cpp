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
============================================================================== */

#include <cstdint>
#include <algorithm>
#include <array>

#include "register/op_def_registry.h"

#include "ops_log.h"
#include "hstu_backward_v2_define.h"
#include "hstu_backward_v2_tiling.h"
#include "../op_kernel/catlass_hstu/kernel/bwd/backward_kernel_tiling.hpp"

namespace optiling {

/**
 * @brief 解析输入张量的 Shape 信息
 * @param context Tiling 上下文，包含输入张量 shape 信息
 * @param tilingData Tiling 数据结构，用于存储解析结果
 * @return bool 解析成功返回 true，失败返回 false
 * @description 从输入张量的 shape 中提取批次大小、头数、维度、序列长度等信息，
 *              并进行合法性校验（如头数范围、维度对齐等）
 */
static bool ParseShape(gert::TilingContext* context, HstuBackwardV2TilingData& tilingData)
{
    static constexpr std::array<int64_t, 2> supportHeadRange = {1, 16};

    const auto qShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::Q))->GetStorageShape();
    const auto vShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::V))->GetStorageShape();
    const auto seqOffsetShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::SEQ_OFFSET_Q))->GetStorageShape();

    auto totalSeqLenQ = qShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO));
    auto totalSeqLenK = vShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO));
    auto batch = seqOffsetShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO)) - 1;
    auto headQ = qShape.GetDim(static_cast<size_t>(DIM_INDEX::ONE));
    auto headK = vShape.GetDim(static_cast<size_t>(DIM_INDEX::ONE));
    auto dimQK = qShape.GetDim(static_cast<size_t>(DIM_INDEX::TWO));
    auto dimGV = vShape.GetDim(static_cast<size_t>(DIM_INDEX::TWO));

    auto dimChecker = [](int64_t dim) -> bool {
        return ((dim % 16) == 0);
    };

    auto headChecker = [](int64_t head) -> bool {
        return head >= supportHeadRange[0] && head <= supportHeadRange[1];
    };

    OPS_CHECK(headQ != headK, OPS_LOG_E("", "tiling failed current only support MHA.\n"), return false);
    OPS_CHECK(!dimChecker(dimQK), OPS_LOG_E("", "tiling failed dimQK must mutiple of 16.\n"), return false);
    OPS_CHECK(!dimChecker(dimGV), OPS_LOG_E("", "tiling failed dimGV must mutiple of 16.\n"), return false);
    OPS_CHECK(!headChecker(headQ), OPS_LOG_E("", "tiling failed headQ is not in support range[1~16].\n"), return false);

    tilingData.set_batch(batch);
    tilingData.set_heads(headQ);
    tilingData.set_dimQK(dimQK);
    tilingData.set_dimGV(dimGV);
    tilingData.set_totalSeqLenQ(totalSeqLenQ);
    tilingData.set_totalSeqLenK(totalSeqLenK);
    return true;
}

/**
 * @brief 解析算子属性参数
 * @param context Tiling 上下文，包含算子属性信息
 * @param tilingData Tiling 数据结构，用于存储解析结果
 * @return bool 解析成功返回 true，失败返回 false
 * @description 从算子属性中提取 max_seqlen_q, max_seqlen_k, scale, target_group_size, alpha 等参数
 */
static bool ParseAttr(gert::TilingContext* context, HstuBackwardV2TilingData& tilingData)
{
    const auto* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, false);

    const auto* maxSeqQ = attrs->GetAttrPointer<uint32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_Q));
    OPS_CHECK_PTR_NULL(maxSeqQ, false);

    const auto* maxSeqK = attrs->GetAttrPointer<uint32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_K));
    OPS_CHECK_PTR_NULL(maxSeqK, false);

    const auto* targetGroupSize = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::TARGET_GROUP_SIZE));
    OPS_CHECK_PTR_NULL(targetGroupSize, return false);

    const auto* scale = attrs->GetAttrPointer<float>(static_cast<size_t>(ATTR_INDEX::SCALE));
    OPS_CHECK_PTR_NULL(scale, false);

    const auto* alpha = attrs->GetAttrPointer<float>(static_cast<size_t>(ATTR_INDEX::ALPHA));
    OPS_CHECK_PTR_NULL(alpha, return false);

    const auto* winLeft = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::WINDOW_SIZE_LEFT));
    OPS_CHECK_PTR_NULL(winLeft, return false);

    const auto* winRight = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::WINDOW_SIZE_RIGHT));
    OPS_CHECK_PTR_NULL(winRight, return false);

    // window_size 语义校验：left/right 均可为 -1（无限），right=0 表示因果。
    OPS_CHECK(*winLeft < -1, OPS_LOG_E("", "tiling failed window_size_left must be >= -1.\n"), return false);
    OPS_CHECK(*winRight < -1, OPS_LOG_E("", "tiling failed window_size_right must be >= -1.\n"), return false);

    tilingData.set_maxSeqLenQ(*maxSeqQ);
    tilingData.set_maxSeqLenK(*maxSeqK);
    tilingData.set_targetGroupSize(*targetGroupSize);
    tilingData.set_scale(*scale);
    tilingData.set_alpha(*alpha);
    tilingData.set_windowSizeLeft(*winLeft);
    tilingData.set_windowSizeRight(*winRight);
    return true;
}

/**
 * @brief 设置 Tiling Key
 * @param context Tiling 上下文
 * @param tilingData Tiling 数据结构
 * @return bool 设置成功返回 true
 * @description 根据是否有 RAB (Relative Attention Bias) 和维度大小设置 Tiling Key，
 *              决定后续 Kernel 的分支选择（128 或 256 tiling 维度）
 */
static bool TilingKeySet(gert::TilingContext* context, HstuBackwardV2TilingData& tilingData)
{
    const auto* numContext = context->GetInputTensor(static_cast<size_t>(IN_INDEX::NUM_CONTEXT));
    const auto* numTarget = context->GetInputTensor(static_cast<size_t>(IN_INDEX::NUM_TARGET));
    const auto* rab = context->GetInputTensor(static_cast<size_t>(IN_INDEX::RAB));
    auto winLeft = tilingData.get_windowSizeLeft();
    auto winRight = tilingData.get_windowSizeRight();

    bool isCausal = (winLeft == -1 && winRight == 0);
    bool isContext = (nullptr != numContext);
    bool isTarget = (nullptr != numTarget);
    // 暂不支持Local mask\Arbitrary mask
    bool isLocal = false;
    bool isArbitrary = false;

    bool hasRab = (nullptr != rab);

    auto dimQK = tilingData.get_dimQK();
    auto dimGV = tilingData.get_dimGV();

    uint32_t tilingDim = 128;              // 128 means dim
    if ((dimQK > 128) || (dimGV > 128)) {  // 128 means dim
        tilingDim = 256;                   // 256 means dim
    }

    ASCENDC_TPL_SEL_PARAM(context, hasRab, isLocal, isCausal, isContext, isTarget, isArbitrary, tilingDim);
    return true;
}

/**
 * @brief 设置共享内存大小
 * @param context Tiling 上下文
 * @param tilingData Tiling 数据结构
 * @description 计算并设置算子所需的共享内存大小，主要用于存储 K 序列的 Block 计数信息
 */
static void ShareMemorySet(gert::TilingContext* context, HstuBackwardV2TilingData& tilingData)
{
    auto dimQK = tilingData.get_dimQK();
    auto dimGV = tilingData.get_dimGV();

    uint32_t BLOCK_N = 128;                // 128 means dim
    if ((dimQK > 128) || (dimGV > 128)) {  // 128 means dim
        BLOCK_N = 64;                      // 64 means block n
    }

    auto totalKSeqLens = tilingData.get_batch() * tilingData.get_heads() * tilingData.get_maxSeqLenK();
    auto totalKBlockCnt = (totalKSeqLens + BLOCK_N - 1) / BLOCK_N;
    int64_t shareMemorySizes = totalKBlockCnt * sizeof(uint32_t);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = shareMemorySizes + systemWorkspaceSize;
}

/**
 * @brief Tiling 主函数
 * @param context Tiling 上下文，包含输入输出信息、属性等
 * @return ge::graphStatus 成功返回 GRAPH_SUCCESS，失败返回 GRAPH_FAILED
 * @description 算子 Tiling 的入口函数，依次调用 ParseShape、ParseAttr、TilingKeySet、ShareMemorySet，
 *              完成 Tiling 数据的解析、设置和保存
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HstuBackwardV2TilingData tilingData;

    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    OPS_CHECK(!ParseShape(context, tilingData), OPS_LOG_E("", "parse shape failed.\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!ParseAttr(context, tilingData), OPS_LOG_E("", "parse attr failed.\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!TilingKeySet(context, tilingData), OPS_LOG_E("", "tiling keyset failed.\n"), return ge::GRAPH_FAILED);
    ShareMemorySet(context, tilingData);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(coreNum);

    OPS_LOG_E_IF_NULL("RawTilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {

/**
 * @brief 推导 Q/K/V 梯度的 Shape
 * @param context Shape 推导上下文
 * @return ge::graphStatus 成功返回 GRAPH_SUCCESS，失败返回 GRAPH_FAILED
 * @description 根据输入 Q、K、V 的 shape 推导对应的梯度输出 shape，保持与输入相同的维度
 */
static ge::graphStatus InferShapeGrad(gert::InferShapeContext* context)
{
    const auto* qShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::Q));
    OPS_CHECK_PTR_NULL(qShape, return ge::GRAPH_FAILED);

    const auto* kShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::K));
    OPS_CHECK_PTR_NULL(kShape, return ge::GRAPH_FAILED);

    const auto* vShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::V));
    OPS_CHECK_PTR_NULL(vShape, return ge::GRAPH_FAILED);

    gert::Shape* qGradShape = context->GetOutputShape(static_cast<size_t>(OUT_INDEX::Q_GRAD));
    OPS_CHECK_PTR_NULL(qGradShape, return ge::GRAPH_FAILED);
    qGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape* kGradShape = context->GetOutputShape(static_cast<size_t>(OUT_INDEX::K_GRAD));
    OPS_CHECK_PTR_NULL(kGradShape, return ge::GRAPH_FAILED);
    kGradShape->SetDimNum(kShape->GetDimNum());

    gert::Shape* vGradShape = context->GetOutputShape(static_cast<size_t>(OUT_INDEX::V_GRAD));
    OPS_CHECK_PTR_NULL(vGradShape, return ge::GRAPH_FAILED);
    vGradShape->SetDimNum(vShape->GetDimNum());

    for (size_t i = 0; i < qShape->GetDimNum(); i++) {
        qGradShape->SetDim(i, qShape->GetDim(i));
        kGradShape->SetDim(i, kShape->GetDim(i));
        vGradShape->SetDim(i, vShape->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导 RAB 梯度的 Shape
 * @param context Shape 推导上下文
 * @return ge::graphStatus 成功返回 GRAPH_SUCCESS，失败返回 GRAPH_FAILED
 * @description 当 RAB (Relative Attention Bias) 作为输入时，推导其梯度的输出 shape，
 *              shape 为 [batch, heads, max_seq_len_q, max_seq_len_k]
 */
static ge::graphStatus InferShapeRabGrad(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::Q));
    OPS_CHECK_PTR_NULL(qShape, return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return ge::GRAPH_FAILED);

    const auto* maxSeqQ = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_Q));
    OPS_CHECK_PTR_NULL(maxSeqQ, return ge::GRAPH_FAILED);

    const auto* maxSeqK = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_K));
    OPS_CHECK_PTR_NULL(maxSeqK, return ge::GRAPH_FAILED);

    const auto* seqOffsetShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::SEQ_OFFSET_Q));
    OPS_CHECK_PTR_NULL(seqOffsetShape, return ge::GRAPH_FAILED);

    auto batch = seqOffsetShape->GetDim(static_cast<size_t>(DIM_INDEX::ZERO)) - 1;

    auto* rabGradShape = context->GetOutputShape(static_cast<size_t>(OUT_INDEX::RAB_GRAD));
    OPS_CHECK_PTR_NULL(rabGradShape, return ge::GRAPH_FAILED);

    rabGradShape->SetDimNum(4);  // 4 means dim num
    rabGradShape->SetDim(static_cast<size_t>(DIM_INDEX::ZERO), batch);
    rabGradShape->SetDim(static_cast<size_t>(DIM_INDEX::ONE), qShape->GetDim(static_cast<size_t>(DIM_INDEX::ONE)));
    rabGradShape->SetDim(static_cast<size_t>(DIM_INDEX::TWO), *maxSeqQ);
    rabGradShape->SetDim(static_cast<size_t>(DIM_INDEX::THREE), *maxSeqK);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导所有输出张量的 Shape
 * @param context Shape 推导上下文
 * @return ge::graphStatus 成功返回 GRAPH_SUCCESS，失败返回 GRAPH_FAILED
 * @description 调用 InferShapeGrad 和 InferShapeRabGrad 推导所有输出张量的 shape
 */
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // step1: 推导Q_GRAD, K_GRAD, V_GRAD的输入输出shape
    auto result = InferShapeGrad(context);
    OPS_CHECK(result == ge::GRAPH_FAILED, OPS_LOG_E("", "InferShapeGrad failed.\n"), return ge::GRAPH_FAILED);

    // step2: 推导RAB_GRAD的输出shape 如果RAB是合法输入的话
    if (nullptr != context->GetInputTensor(static_cast<size_t>(IN_INDEX::RAB))) {
        result = InferShapeRabGrad(context);
        OPS_CHECK(result == ge::GRAPH_FAILED, OPS_LOG_E("", "InferShapeRabGrad failed.\n"), return ge::GRAPH_FAILED);
    }

    return result;
}

/**
 * @brief 推导输出张量的数据类型
 * @param context 数据类型推导上下文
 * @return ge::graphStatus 成功返回 GRAPH_SUCCESS，失败返回 GRAPH_FAILED
 * @description 根据输入梯度 GRAD 的数据类型，设置所有输出张量的数据类型与其保持一致
 */
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto dataType = context->GetInputDataType(static_cast<size_t>(IN_INDEX::GRAD));

    context->SetOutputDataType(static_cast<size_t>(OUT_INDEX::Q_GRAD), dataType);
    context->SetOutputDataType(static_cast<size_t>(OUT_INDEX::K_GRAD), dataType);
    context->SetOutputDataType(static_cast<size_t>(OUT_INDEX::V_GRAD), dataType);
    context->SetOutputDataType(static_cast<size_t>(OUT_INDEX::RAB_GRAD), dataType);

    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {

/**
 * @brief HSTU Backward V2 算子定义类
 * @description 继承自 OpDef，定义算子的输入输出、属性、推理函数和 Tiling 策略
 *              实现 Hierarchical Sparse Transformer Unit 的反向传播算子，
 *              支持 Q、K、V 的梯度计算以及可选的相对位置注意力偏置 (RAB) 梯度
 */
class HstuBackwardV2 : public OpDef {
public:
    /**
     * @brief 构造函数
     * @param name 算子名称
     * @description 初始化算子的输入输出定义、属性配置和 AI Core 调度策略
     */
    explicit HstuBackwardV2(const char* name) : OpDef(name)
    {
        this->Input("grad").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("q").ParamType(REQUIRED).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("rab").ParamType(OPTIONAL).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_k")
            .ParamType(REQUIRED)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("num_context")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("num_target")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("q_share").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        // 可选: flash_attn_metadata 分核输出(int32,HEAD+FA+FD 布局)。未传 → kernel 收到 nullptr →
        // 旧设备现算分核(零回归)
        this->Input("metadata").ParamType(OPTIONAL).DataType({ge::DT_INT32, ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("q_grad").ParamType(REQUIRED).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Output("k_grad").ParamType(REQUIRED).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Output("v_grad").ParamType(REQUIRED).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Output("rab_grad").ParamType(OPTIONAL).Follow("grad", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Attr("max_seqlen_q").Int();
        this->Attr("max_seqlen_k").Int();
        this->Attr("scale").Float();
        this->Attr("target_group_size").AttrType(OPTIONAL).Int(1);
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0);
        this->Attr("window_size_left").Int(-1);
        this->Attr("window_size_right").Int(-1);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(HstuBackwardV2);
}  // namespace ops

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

#include <cmath>
#include <cstdint>

#include "backward_codegen_adagrad_unweighted_exact_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {
constexpr int DATA_TYPE_FLOAT32 = 0;
constexpr int DATA_TYPE_INT64 = 1;

constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr uint64_t UB_ALIGN = 32;
constexpr int NUM_QUEUE = 32;
// input index
constexpr int GRAD_OUTPUT_INDEX = 0;
constexpr int DEV_WEIGHTS_INDEX = 1;
constexpr int UVM_WEIGHTS_INDEX = 2;
constexpr int LXU_CACHE_WEIGHTS_INDEX = 3;
constexpr int WEIGHTS_PLACEMENTS_INDEX = 4;
constexpr int WEIGHTS_OFFSETS_INDEX = 5;
constexpr int D_OFFSETS_INDEX = 6;
constexpr int INDICES_INDEX = 8;
constexpr int OFFSETS_INDEX = 9;
constexpr int LXU_CACHE_LOCATIONS_INDEX = 10;
constexpr int MOMENTUM1_DEV_INDEX = 11;
constexpr int HASH_INDICES_INDEX = 19;
constexpr int UNIQUE_ID_INDEX = 20;
constexpr int UNIQUE_HASH_SIZE_INDEX = 21;
constexpr int UNIQUE_INVERSE_INDEX = 22;
// attribute index
constexpr int MAX_D_INDEX = 0;
constexpr int TOTAL_HASH_SIZE_BITS = 1;
constexpr int POOL_MODE_INDEX = 2;

constexpr int OPTIM_TYPE_INDEX = 10;
constexpr int EPS_INDEX = 11;
constexpr int LEARNING_RATE_INDEX = 12;
constexpr int BETA1_INDEX = 13;
constexpr int BETA2_INDEX = 14;
constexpr int ITER_INDEX = 15;
constexpr int USE_OPTIMIZE_INDEX = 16;

// tilling key index
constexpr int NORMAL_ADAGRAD = 1;
constexpr int UNIQUE_ADAGRAD = 4;
constexpr int NORMAL_ADAM = 2;
constexpr int UNIQUE_ADAM = 5;
constexpr int NORMAL_SGD = 3;
constexpr int UNIQUE_SGD = 6;
constexpr int NORMAL_ROWWISE_ADAGRAD = 7;
// optimize type
constexpr int ADAGRAD = 1;
constexpr int ADAM = 2;
constexpr int SGD = 3;
constexpr int ROWWISE_ADAGRAD = 7;

static ge::graphStatus UniqueTilingFunc(gert::TilingContext* context,
                                        BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
{
    auto uniqueOffset = context->GetOptionalInputTensor(UNIQUE_HASH_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("uniqueOffset", uniqueOffset, return ge::GRAPH_FAILED);
    auto uniqueInverse = context->GetOptionalInputTensor(UNIQUE_INVERSE_INDEX);
    OPS_LOG_E_IF_NULL("uniqueInverse", uniqueInverse, return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("uniqueHashSizeShape", context->GetInputShape(UNIQUE_HASH_SIZE_INDEX), return ge::GRAPH_FAILED);
    int64_t uniqueHashDim0 = context->GetInputShape(UNIQUE_HASH_SIZE_INDEX)->GetStorageShape().GetDim(0);
    tilingData.set_uniqueHashDim0(uniqueHashDim0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NormalAdamTilingFunc(gert::TilingContext* context,
                                            BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
{
    float beta1 = *context->GetAttrs()->GetFloat(BETA1_INDEX);
    float beta2 = *context->GetAttrs()->GetFloat(BETA2_INDEX);
    int64_t iter = *context->GetAttrs()->GetInt(ITER_INDEX);

    OPS_CHECK(beta1 == 1.0,
              OPS_LOG_E("Tiling Debug", "beta1 can not be 1.0."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(beta2 == 1.0,
              OPS_LOG_E("Tiling Debug", "beta2 can not be 1.0."),
              return ge::GRAPH_FAILED);

    float _beta1 = (1 - pow(beta1, iter));
    float _beta2 = (1 - pow(beta2, iter));
    float _beta2sqrt = sqrt(_beta2) / _beta1;
    _beta1 = 1 / _beta1;
    _beta2 = 1 / _beta2;

    tilingData.set_beta1(beta1);
    tilingData.set_beta2(beta2);
    tilingData.set_beta1pow(_beta1);
    tilingData.set_beta2pow(_beta2);
    tilingData.set_beta2sqrt(_beta2sqrt);
    tilingData.set_iter(iter);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus UniqueTilingKey(gert::TilingContext* context, const int &optimType)
{
    if (optimType == ADAM) {
        context->SetTilingKey(UNIQUE_ADAM);
    } else if (optimType == ADAGRAD) {
        context->SetTilingKey(UNIQUE_ADAGRAD);
    } else if (optimType == SGD) {
        context->SetTilingKey(UNIQUE_SGD);
    } else {
        OPS_LOG_E("Tiling Debug", "Unsupported optimtype!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NormalTilingKey(gert::TilingContext* context, const int &optimType)
{
    if (optimType == ADAM) {
        context->SetTilingKey(NORMAL_ADAM);
    } else if (optimType == ADAGRAD) {
        context->SetTilingKey(NORMAL_ADAGRAD);
    } else if (optimType == SGD) {
        context->SetTilingKey(NORMAL_SGD);
    } else if (optimType == ROWWISE_ADAGRAD) {
        context->SetTilingKey(NORMAL_ROWWISE_ADAGRAD);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShapeTilingFunc(gert::TilingContext* context,
                                       BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
{
    OPS_LOG_E_IF_NULL("gradOutputIndexShape", context->GetInputShape(GRAD_OUTPUT_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("devWeightsIndexShape", context->GetInputShape(DEV_WEIGHTS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("weightsOffsetsIndexShape", context->GetInputShape(WEIGHTS_OFFSETS_INDEX),
              return ge::GRAPH_FAILED);

    int64_t gradOutputDim0 = context->GetInputShape(GRAD_OUTPUT_INDEX)->GetStorageShape().GetDim(0);
    int64_t gradOutputDim1 = context->GetInputShape(GRAD_OUTPUT_INDEX)->GetStorageShape().GetDim(1);
    int64_t devWeightsDim0 = context->GetInputShape(DEV_WEIGHTS_INDEX)->GetStorageShape().GetDim(0);
    int64_t weightsOffsetsDim0 = context->GetInputShape(WEIGHTS_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    OPS_CHECK(weightsOffsetsDim0 == 0, OPS_LOG_E("Tiling Debug", "weightsOffsets shape is invalid."),
              return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("dOffsetsIndexShape", context->GetInputShape(D_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    int64_t dOffsetsDim0 = context->GetInputShape(D_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    OPS_CHECK(dOffsetsDim0 <= 1, OPS_LOG_E("Tiling Debug", "dOffsets shape is invalid."), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("indicesIndexShape", context->GetInputShape(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsIndexShape", context->GetInputShape(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    int64_t indicesDim0 = context->GetInputShape(INDICES_INDEX)->GetStorageShape().GetDim(0);
    int64_t offsetsDim0 = context->GetInputShape(OFFSETS_INDEX)->GetStorageShape().GetDim(0);

    int64_t outDim0 = devWeightsDim0;

    int64_t offsetDataType = DATA_TYPE_INT64;

    auto hashIndices = context->GetOptionalInputTensor(HASH_INDICES_INDEX);
    if (hashIndices == nullptr) {
        tilingData.set_enableHash(0);
    } else {
        tilingData.set_enableHash(1);
        OPS_LOG_E_IF_NULL("hashIndicesIndexShape", context->GetInputShape(HASH_INDICES_INDEX), return ge::GRAPH_FAILED);
        indicesDim0 = context->GetInputShape(HASH_INDICES_INDEX)->GetStorageShape().GetDim(0);
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;

    int optimType = *context->GetAttrs()->GetInt(OPTIM_TYPE_INDEX);
    auto uniqueId = context->GetOptionalInputTensor(UNIQUE_ID_INDEX);

    if (optimType == ROWWISE_ADAGRAD) {
        OPS_LOG_E_IF_NULL("momentum1DevIndexShape",
            context->GetInputShape(MOMENTUM1_DEV_INDEX), return ge::GRAPH_FAILED);
        int64_t momentumDim0 = context->GetInputShape(MOMENTUM1_DEV_INDEX)->GetStorageShape().GetDim(0);
        tilingData.set_momentumDim0(momentumDim0);
    } else {
        tilingData.set_momentumDim0(outDim0);
    }

    tilingData.set_gradOutputDim0(gradOutputDim0);
    tilingData.set_gradOutputDim1(gradOutputDim1);
    tilingData.set_devWeightsDim0(devWeightsDim0);
    tilingData.set_weightsOffsetsDim0(weightsOffsetsDim0);
    tilingData.set_dOffsetsDim0(dOffsetsDim0);
    tilingData.set_indicesDim0(indicesDim0);
    tilingData.set_offsetsDim0(offsetsDim0);
    tilingData.set_outDim0(outDim0);
    tilingData.set_offsetDataType(offsetDataType);

    if (optimType == ADAM) {
        ret = NormalAdamTilingFunc(context, tilingData);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    if (uniqueId != nullptr) {
        ret = UniqueTilingFunc(context, tilingData);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        return UniqueTilingKey(context, optimType);
    }
    return NormalTilingKey(context, optimType);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);
    BackwardCodegenAdagradUnweightedExactTilingData tiling;

    OPS_LOG_E_IF_NULL("GetAttrs()", context->GetAttrs(), return ge::GRAPH_FAILED);
    int64_t total_hash_size_bits = *context->GetAttrs()->GetInt(TOTAL_HASH_SIZE_BITS);

    // Shape and dType
    ge::graphStatus ret = ShapeTilingFunc(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();

    int bitNum = 2;
    currentWorkspace[0] = std::pow(bitNum, total_hash_size_bits) + systemWorkspacesSize;

    // Tiling
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E("Tiling Debug", "Core num is 0."), return ge::GRAPH_FAILED);

    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    uint64_t flagUb = UB_ALIGN * 2;  // queFlagIn、queFlagOut两个标志位
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE - flagUb;
    tiling.set_ubCanUsed(ubCanUsed);

    int64_t poolMode = *context->GetAttrs()->GetInt(POOL_MODE_INDEX);
    int64_t maxD = *context->GetAttrs()->GetInt(MAX_D_INDEX);
    float eps = *context->GetAttrs()->GetFloat(EPS_INDEX);
    float learningRate = *context->GetAttrs()->GetFloat(LEARNING_RATE_INDEX);

    tiling.set_poolMode(poolMode);
    tiling.set_maxD(maxD);

    tiling.set_eps(eps);
    tiling.set_learningRate(learningRate);
    bool useOptimize = *context->GetAttrs()->GetBool(USE_OPTIMIZE_INDEX);
    tiling.set_useOptimize(useOptimize);

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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BackwardCodegenAdagradUnweightedExact : public OpDef {
public:
    explicit BackwardCodegenAdagradUnweightedExact(const char* name) : OpDef(name)
    {
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("dev_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("uvm_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lxu_cache_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights_placements")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("D_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("hash_size_cumsum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lxu_cache_locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum1_dev")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum1_uvm")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum1_placements")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum1_offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum2_dev")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum2_uvm")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum2_placements")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("momentum2_offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("hash_indices")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("unique_id")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("unique_hash_size")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("unique_inverse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indice_size_cumsum")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("momentum1_dev_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("momentum2_dev_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("weights_dev_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("max_D").Int();
        this->Attr("total_hash_size_bits").Int();
        this->Attr("pool_mode").Int();
        this->Attr("BT_block_size").Int();
        this->Attr("max_segment_length_per_warp").Int();
        this->Attr("stochastic_rounding").Int();
        this->Attr("info_B_num_bits").Int();
        this->Attr("info_B_mask_int64").Int();
        this->Attr("use_uniq_cache_locations").Int();
        this->Attr("use_homogeneous_placements").Int();
        this->Attr("optim_type").Int();
        this->Attr("eps").Float();
        this->Attr("learning_rate").Float();
        this->Attr("beta1").Float();
        this->Attr("beta2").Float();
        this->Attr("iter").Int();
        this->Attr("use_optimize").Bool();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BackwardCodegenAdagradUnweightedExact);
}  // namespace ops

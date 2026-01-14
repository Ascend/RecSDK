/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include "user_item_flash_attention_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
const char* K_INNER_DEBUG = "UserItemFlashAttention Tiling Debug";
constexpr uint32_t QUERY_ITEM_INPUT_INDEX = 0;
constexpr uint32_t KEY_USER_INPUT_INDEX = 1;
constexpr uint32_t VALUE_USER_INPUT_INDEX = 2;
constexpr uint32_t KEY_ITEM_INPUT_INDEX = 4;
constexpr uint32_t VALUE_ITEM_INPUT_INDEX = 5;
constexpr uint32_t BASE_M = 128;
constexpr uint32_t BASE_N = 256;
constexpr uint32_t BATCH_SEQ_BLOCK = 16;
constexpr uint32_t KV_SEQ_BLOCK = 512;
constexpr uint32_t VEC_SOFTMAX_BATCH_BLOCK = 16;
constexpr uint32_t VEC_AGGR_BATCH_BLOCK = 16;
}  // namespace

namespace optiling {

uint32_t ceil_div(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

uint32_t align_to(uint32_t x, uint32_t y)
{
    return ceil_div(x, y) * y;
}

static void SetShapeInfo(gert::TilingContext* context, UserItemFlashAttentionTilingData& tiling)
{
    const auto& query_shape = context->GetInputShape(QUERY_ITEM_INPUT_INDEX)->GetStorageShape();
    const auto& key_user_shape = context->GetInputShape(KEY_USER_INPUT_INDEX)->GetStorageShape();

    uint32_t item_batch = query_shape.GetDim(0);
    uint32_t item_seq_len = query_shape.GetDim(1);
    uint32_t q_head_num = query_shape.GetDim(2);
    uint32_t head_dim = query_shape.GetDim(3);

    uint32_t user_batch = key_user_shape.GetDim(0);
    uint32_t user_seq_len = key_user_shape.GetDim(1);
    uint32_t kv_head_num = key_user_shape.GetDim(2);

    tiling.shape_info.set_item_batch(item_batch);
    tiling.shape_info.set_q_head_num(q_head_num);
    tiling.shape_info.set_kv_head_num(kv_head_num);
    tiling.shape_info.set_head_dim(head_dim);
    tiling.shape_info.set_item_seq_len(item_seq_len);
    tiling.shape_info.set_user_seq_len(user_seq_len);
    tiling.shape_info.set_item_batch_stride(q_head_num * head_dim);
    tiling.shape_info.set_item_head_num_stride(head_dim);
    tiling.shape_info.set_user_seq_stride(kv_head_num * head_dim);
    tiling.shape_info.set_user_head_num_stride(head_dim);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // get hardware info
    uint64_t ub_size = 0;
    uint64_t l1_size = 0;
    uint64_t l0c_size = 0;
    uint32_t aivec_num = 0;
    uint32_t aicube_num = 0;
    auto info = context->GetPlatformInfo();
    auto platform_info = platform_ascendc::PlatformAscendC(info);
    aivec_num = platform_info.GetCoreNumAiv();
    aicube_num = platform_info.GetCoreNumAic();
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);

    const gert::StorageShape* key_item_shape = context->GetOptionalInputShape(KEY_ITEM_INPUT_INDEX);
    const gert::StorageShape* value_item_shape = context->GetOptionalInputShape(VALUE_ITEM_INPUT_INDEX);

    UserItemFlashAttentionTilingData tiling;
    SetShapeInfo(context, tiling);

    if (key_item_shape != nullptr && value_item_shape != nullptr) {
        tiling.set_has_item_kv(1);
    } else if (key_item_shape == nullptr && value_item_shape == nullptr) {
        tiling.set_has_item_kv(0);
    } else {
        std::cout << "unsupport input shape" << std::endl;
        return ge::GRAPH_FAILED;
    }

    uint32_t item_batch_block = BATCH_SEQ_BLOCK;
    uint32_t user_kv_seq_block = KV_SEQ_BLOCK;
    uint32_t vec_softmax_batch_block = VEC_SOFTMAX_BATCH_BLOCK;
    uint32_t vec_aggr_batch_block = VEC_AGGR_BATCH_BLOCK;
    tiling.set_item_batch_block(item_batch_block);
    tiling.set_user_kv_seq_block(user_kv_seq_block);
    tiling.set_vec_softmax_batch_block(vec_softmax_batch_block);
    tiling.set_vec_aggr_batch_block(vec_aggr_batch_block);

    uint32_t item_batch_block_num = ceil_div(tiling.shape_info.get_item_batch(), item_batch_block);
    uint32_t total_task_num = item_batch_block_num * tiling.shape_info.get_q_head_num();
    auto block_dim = platform_info.CalcTschBlockDim(std::min(aivec_num, total_task_num), aicube_num, aivec_num);
    uint32_t used_aiv_num = block_dim * 2;
    uint32_t task_num_per_core = ceil_div(total_task_num, used_aiv_num);
    tiling.set_total_task_num(total_task_num);
    tiling.set_item_batch_block_num(item_batch_block_num);
    tiling.set_task_num_per_core(task_num_per_core);
    tiling.set_used_aiv_num(used_aiv_num);

    // 输入矩阵的数据类型
    auto input0_ptr = context->GetInputDesc(0);
    if (input0_ptr == nullptr) {
        std::cout << "input0 desc is nullptr" << std::endl;
        return ge::GRAPH_FAILED;
    }
    auto input_dtype = input0_ptr->GetDataType();
    uint32_t elm_size;
    matmul_tiling::DataType matmul_dtype = matmul_tiling::DataType::DT_FLOAT;
    if (input_dtype == ge::DT_FLOAT) {
        elm_size = sizeof(float);
    } else if (input_dtype == ge::DT_FLOAT16) {
        elm_size = sizeof(int16_t);
        matmul_dtype = matmul_tiling::DataType::DT_FLOAT16;
    } else if (input_dtype == ge::DT_BF16) {
        elm_size = sizeof(int16_t);
        matmul_dtype = matmul_tiling::DataType::DT_BFLOAT16;
    } else {
        std::cout << "unsupport dtype" << std::endl;
        return ge::GRAPH_FAILED;
    }
    // 设置gemm_qk gemm_pv tiling
    matmul_tiling::MatmulApiTiling gemm_qk_tiling;
    gemm_qk_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    gemm_qk_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    gemm_qk_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);

    gemm_qk_tiling.SetBias(false);
    gemm_qk_tiling.SetOrgShape(item_batch_block, user_kv_seq_block, tiling.shape_info.get_item_batch_stride(),
                               tiling.shape_info.get_user_seq_stride());
    gemm_qk_tiling.SetShape(item_batch_block, user_kv_seq_block, tiling.shape_info.get_head_dim());
    gemm_qk_tiling.SetFixSplit(std::min(BASE_M, item_batch_block), std::min(BASE_N, user_kv_seq_block), -1);
    gemm_qk_tiling.SetBufferSpace(l1_size, l0c_size);

    if (gemm_qk_tiling.GetTiling(tiling.gemm_qk_tiling) == -1) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling gemm_pv_tiling;
    gemm_pv_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    gemm_pv_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    gemm_pv_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);

    gemm_pv_tiling.SetBias(false);
    gemm_pv_tiling.SetShape(item_batch_block, tiling.shape_info.get_head_dim(), user_kv_seq_block);
    gemm_pv_tiling.SetFixSplit(std::min(BASE_M, item_batch_block), tiling.shape_info.get_head_dim(), -1);
    gemm_pv_tiling.SetBufferSpace(l1_size, l0c_size);

    if (gemm_pv_tiling.GetTiling(tiling.gemm_pv_tiling) == -1) {
        return ge::GRAPH_FAILED;
    }

    // 计算softmax临时空间
    auto softmax_shape = ge::Shape({tiling.get_vec_softmax_batch_block(), tiling.get_user_kv_seq_block()});
    uint32_t softmax_api_buf_size_test =
        AscendC::GetSoftMaxFlashV2MaxTmpSize(softmax_shape, sizeof(float), sizeof(float), true, true);
    uint32_t softmax_api_buf_size = 32 * 1024;
    tiling.set_softmax_buf_size(softmax_api_buf_size);
    tiling.set_score_scale(1 / std::sqrt(static_cast<float>(tiling.shape_info.get_head_dim())));
    auto src_shape = ge::Shape({tiling.get_item_batch_block(), tiling.get_user_kv_seq_block()});
    AscendC::SoftMaxFlashV2TilingFunc(src_shape, sizeof(float), sizeof(float), softmax_api_buf_size,
                                      tiling.softmax_tiling, true);

    // 计算核外临时空间
    constexpr uint32_t cube_block_bytes = 512;
    size_t* workspace = context->GetWorkspaceSizes(1);
    uint32_t gemm_qk_res_size = align_to(item_batch_block * user_kv_seq_block * sizeof(float), cube_block_bytes);
    uint32_t gemm_pv_res_size =
        align_to(item_batch_block * tiling.shape_info.get_head_dim() * sizeof(float), cube_block_bytes);
    uint32_t gemm_aggr_res_size = gemm_pv_res_size;
    tiling.set_gemm_qk_res_size(gemm_qk_res_size);
    tiling.set_gemm_pv_res_size(gemm_pv_res_size);
    tiling.set_gemm_aggr_res_size(gemm_aggr_res_size);

    // cube双发，cube和vector之间做Double buffer
    workspace[0] = platform_info.GetLibApiWorkSpaceSize() +
                   ((gemm_qk_res_size + gemm_pv_res_size) * 2 + gemm_aggr_res_size) * used_aiv_num;

    context->SetBlockDim(block_dim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // attention结果shape与query相同
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    // attention结果数据类型与query相同
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class UserItemFlashAttention : public OpDef {
public:
    explicit UserItemFlashAttention(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_user")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value_user")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask_len")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_item")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value_item")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(UserItemFlashAttention);
}  // namespace ops

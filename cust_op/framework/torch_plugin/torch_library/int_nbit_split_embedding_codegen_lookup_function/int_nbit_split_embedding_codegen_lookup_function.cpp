/**
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <limits>
#include "torch/extension.h"
#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

using Tensor = at::Tensor;
using namespace at;

// PoolingMode枚举定义
enum class PoolingMode {
    SUM = 0,
    MEAN = 1,
    NONE = 2
};

enum class SparseType {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
    BF16 = 5,
    FP8 = 6,
    INVALID = 7
};

namespace fbgemm_npu_lookups {

static inline at::ScalarType sparse_type_to_scalar_type(int64_t output_dtype)
{
    switch (static_cast<SparseType>(output_dtype)) {
        case SparseType::FP32:
            return at::kFloat;
        case SparseType::FP16:
            return at::kHalf;
        case SparseType::INT8:
            return at::kByte;
        case SparseType::BF16:
            return at::kBFloat16;
        default:
            TORCH_CHECK(false, "Unsupported output_dtype: ", output_dtype,
                        ". Supported: 0(FP32), 1(FP16), 2(INT8), 5(BF16)");
    }
}

// 统一接口实现（与CUDA接口完全一致）
at::Tensor int_nbit_split_embedding_codegen_lookup_function_npu(
    at::Tensor dev_weights,
    at::Tensor uvm_weights,
    at::Tensor weights_placements,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor D_offsets,
    const int64_t total_D,
    const int64_t max_int2_D,
    const int64_t max_int4_D,
    const int64_t max_int8_D,
    const int64_t max_float16_D,
    const int64_t max_float32_D,
    at::Tensor indices,
    at::Tensor offsets,
    const int64_t pooling_mode,
    std::optional<at::Tensor> indice_weights,
    const int64_t output_dtype,
    std::optional<at::Tensor> lxu_cache_weights,
    std::optional<at::Tensor> lxu_cache_locations,
    const std::optional<int64_t> row_alignment,
    const std::optional<int64_t> max_float8_D,
    const std::optional<int64_t> fp8_exponent_bits,
    const std::optional<int64_t> fp8_exponent_bias)
{
    const at::OptionalDeviceGuard guard(device_of(dev_weights));

    // 1. 参数获取
    int64_t featCnt = weights_offsets.size(0);
    int32_t totalLen = indices.numel();
    const int64_t maxFloat8DValue = max_float8_D.value_or(0);
    const int64_t rowAlignmentValue = row_alignment.value_or(16);
    const int64_t fp8ExponentBitsValue = fp8_exponent_bits.value_or(-1);
    const int64_t fp8ExponentBiasValue = fp8_exponent_bias.value_or(-1);

    Tensor lxu_cache_weights_tensor =
        lxu_cache_weights.value_or(at::empty({0, 0}, dev_weights.options().dtype(at::kByte)));
    Tensor lxu_cache_locations_tensor =
        lxu_cache_locations.value_or(at::empty({0}, dev_weights.options().dtype(at::kInt)));
    std::vector<int64_t> max_D_list{
        max_int2_D,
        max_int4_D,
        max_int8_D,
        maxFloat8DValue,
        max_float16_D,
        max_float32_D
    };
    int64_t max_D = *std::max_element(max_D_list.begin(), max_D_list.end());

    c10::optional<at::Tensor> indice_weights_tensor = c10::nullopt;
    if (indice_weights && indice_weights->defined() && indice_weights->numel() > 0) {
        indice_weights_tensor = indice_weights->to(at::kFloat).contiguous();
    }

    // 2. offsets类型转换与计算 offset_per_key：每张表在offsets中的起始位置
    Tensor offsets_processed = offsets;
    if (offsets.scalar_type() != indices.scalar_type()) {
        offsets_processed = offsets.toType(indices.scalar_type());
    }

    // 3. 基础检查
    TORCH_CHECK(featCnt > 0, "weights_offsets size must be great than 0.");
    TORCH_CHECK(totalLen > 0, "indices can not be empty tensor.");
    TORCH_CHECK(offsets_processed.size(0) > 1, "offsets dim_0 must be great than 1.");
    TORCH_CHECK(weights_tys.size(0) == featCnt, "weights_tys size must equal weights_offsets size.");

    int64_t batchs = (offsets_processed.numel() - 1) / featCnt;
    Tensor table_offsets = torch::arange(D_offsets.size(0), offsets_processed.device()) * batchs;
    Tensor offset_per_key = offsets_processed.index_select(0, table_offsets).to(at::kInt);

    if (static_cast<SparseType>(output_dtype) == SparseType::INT8) {
        auto weights_tys_cpu = weights_tys.to(at::kCPU);
        bool hasNonInt8 = weights_tys_cpu.ne(static_cast<uint8_t>(SparseType::INT8)).any().item<bool>();
        TORCH_CHECK(!hasNonInt8, "int8 output_dtype requires all tables to use int8 weights.");
    }

    // nobag 分支
    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
        // 计算输出形状（根据output_dtype动态创建对应类型）
        at::ScalarType output_scalar_type = sparse_type_to_scalar_type(output_dtype);
        Tensor output = at::zeros({totalLen, max_D}, dev_weights.options().dtype(output_scalar_type));

        Tensor indices_int32 = indices.to(at::kInt);
        Tensor offsets_int32 = offsets_processed.to(at::kInt);

        EXEC_NPU_CMD(aclnnIntNbitSplitEmbeddingCodegenLookupFunction, dev_weights, uvm_weights,
                     lxu_cache_weights_tensor, weights_placements, weights_offsets, weights_tys, D_offsets,
                     indices_int32, offsets_int32, lxu_cache_locations_tensor, offset_per_key, indice_weights_tensor,
                     total_D, max_D, max_int2_D, max_int4_D, max_int8_D, max_float16_D, max_float32_D, maxFloat8DValue,
                     pooling_mode, output_dtype, rowAlignmentValue, fp8ExponentBitsValue, fp8ExponentBiasValue, output);
        return output;
    }

    int64_t batchSizeRes = (offsets_processed.size(0) - 1) % featCnt;
    TORCH_CHECK(batchSizeRes == 0, "offset size = ", offsets_processed.size(0),
                " is incorrect for feature count = ", featCnt);
    int64_t batchSize = (offsets_processed.size(0) - 1) / featCnt;

    // 根据output_dtype动态创建对应类型
    at::ScalarType output_scalar_type = sparse_type_to_scalar_type(output_dtype);
    Tensor output = at::zeros({batchSize, total_D}, dev_weights.options().dtype(output_scalar_type));

    if (indice_weights_tensor && indice_weights_tensor->numel() == 0) {
        indice_weights_tensor = c10::nullopt;
    }

    EXEC_NPU_CMD(aclnnIntNbitSplitEmbeddingCodegenLookupFunction, dev_weights, uvm_weights, lxu_cache_weights_tensor,
                 weights_placements, weights_offsets, weights_tys, D_offsets, indices, offsets_processed,
                 lxu_cache_locations_tensor, offset_per_key, indice_weights_tensor, total_D, max_D, max_int2_D,
                 max_int4_D, max_int8_D, max_float16_D, max_float32_D, maxFloat8DValue, pooling_mode, output_dtype,
                 rowAlignmentValue, fp8ExponentBitsValue, fp8ExponentBiasValue, output);
    return output;
}

};  // namespace fbgemm_npu_lookups

// 复用fbgemm已有schema，只在PrivateUse1注册实现
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("int_nbit_split_embedding_codegen_lookup_function",
           TORCH_FN(fbgemm_npu_lookups::int_nbit_split_embedding_codegen_lookup_function_npu));
}

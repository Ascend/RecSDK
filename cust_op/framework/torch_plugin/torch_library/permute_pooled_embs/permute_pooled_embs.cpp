/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;


void validate_permute_pooled_embs_inputs(
    const at::Tensor& pooled_embs,
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list)
{
    // ============= 维度检查 =============
    TORCH_CHECK(pooled_embs.dim() >= 2, "pooled_embs must be at least 2-D");
    TORCH_CHECK(offset_dim_list.scalar_type() == at::ScalarType::Long, "offset_dim_list must be int64");
    TORCH_CHECK(permute_list.scalar_type() == at::ScalarType::Long, "permute_list must be int64");
    TORCH_CHECK(inv_offset_dim_list.scalar_type() == at::ScalarType::Long, "inv_offset_dim_list must be int64");
    TORCH_CHECK(inv_permute_list.scalar_type() == at::ScalarType::Long, "inv_permute_list must be int64");

    const auto T = permute_list.numel();
    TORCH_CHECK(offset_dim_list.numel() == T + 1, "offset_dim_list must have T+1 elements");
    TORCH_CHECK(inv_offset_dim_list.numel() == T + 1, "inv_offset_dim_list must have T+1 elements");
    TORCH_CHECK(inv_permute_list.numel() == T, "inv_permute_list must have T elements");

    // ============= 类型检查 =============
    TORCH_CHECK(pooled_embs.scalar_type() == at::ScalarType::Float ||
                pooled_embs.scalar_type() == at::ScalarType::Half ||
                pooled_embs.scalar_type() == at::ScalarType::BFloat16,
                "pooled_embs must be float32, float16, or bfloat16");

    // ============= 空值检查 =============
    check_tensor_non_empty(pooled_embs, "pooled_embs");
    check_tensor_non_empty(offset_dim_list, "offset_dim_list");
    check_tensor_non_empty(permute_list, "permute_list");
    check_tensor_non_empty(inv_offset_dim_list, "inv_offset_dim_list");
    check_tensor_non_empty(inv_permute_list, "inv_permute_list");

    // ============= NPU设备检查 =============
    std::vector<Tensor> tensors = {pooled_embs, offset_dim_list, permute_list, inv_offset_dim_list, inv_permute_list};
    std::vector<std::string> names = {"pooled_embs",
                                      "offset_dim_list",
                                      "permute_list",
                                      "inv_offset_dim_list",
                                      "inv_permute_list"};
    check_tensor_npu_device(tensors, names);
}

at::Tensor permute_pooled_embs_impl_npu(
    const at::Tensor& pooled_embs,
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list)
{
    validate_permute_pooled_embs_inputs(pooled_embs,
                                        offset_dim_list,
                                        permute_list,
                                        inv_offset_dim_list,
                                        inv_permute_list);

    auto pooled_embs_conti = pooled_embs.contiguous();
    auto offset_dim_list_conti = offset_dim_list.contiguous();
    auto permute_list_conti = permute_list.contiguous();
    auto inv_offset_dim_list_conti = inv_offset_dim_list.contiguous();
    auto inv_permute_list_conti = inv_permute_list.contiguous();

    auto output = at::empty_like(pooled_embs_conti);
    int64_t T = permute_list_conti.numel();
    int64_t total_D = offset_dim_list_conti[T].item<int64_t>();
    std::vector<int64_t> cols;
    cols.reserve(total_D);

    for (int64_t i = 0; i < T; i++) {
        int64_t p = permute_list_conti[i].item<int64_t>();
        TORCH_CHECK(p >= 0 && p < T, "permute_list must be a permutation of 0 to ", T, " but got ", p);
        int64_t start = offset_dim_list_conti[p].item<int64_t>();
        int64_t end = offset_dim_list_conti[p + 1].item<int64_t>();
        for (int64_t j = start; j < end; j++) {
            cols.push_back(j);
        }
    }
    auto cols_tensor = at::tensor(cols, permute_list_conti.options());
    output = pooled_embs_conti.index_select(1, cols_tensor);

    return output;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m) {
    m.def("permute_pooled_embs(Tensor pooled_embs, "
          "                    Tensor offset_dim_list, "
          "                    Tensor permute_list, "
          "                    Tensor inv_offset_dim_list, "
          "                    Tensor inv_permute_list) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m) {
    m.impl("permute_pooled_embs", &permute_pooled_embs_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m) {
    m.impl("permute_pooled_embs", &permute_pooled_embs_impl_npu);
}

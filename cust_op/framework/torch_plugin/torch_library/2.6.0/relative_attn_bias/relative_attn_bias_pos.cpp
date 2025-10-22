/**
 * @file relative_attn_bias_pos.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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
using namespace at;
using namespace std;

Tensor relative_attn_bias_pos_forward(const Tensor& relPosBias, const Tensor& identity,
                                      const at::IntArrayRef pastValidLens)
{
    auto relPosBiasConti = relPosBias.contiguous();
    auto identityConti = identity.contiguous();

    check_tensor_non_empty(relPosBias, "relPosBias");
    check_tensor_non_empty(identity, "identity");

    // 检查NPU设备且设备ID一致
    std::vector<Tensor> tensors = {relPosBias, identity};
    std::vector<std::string> names = {"relPosBias", "identity"};
    check_tensor_npu_device(tensors, names);

    const int64_t bs = pastValidLens.size();
    const int64_t sx2 = relPosBias.size(0);  // relPosBias(2s, 2s)

    at::Tensor rabPosOut = at::zeros({bs, sx2, sx2}, relPosBiasConti.options());

    EXEC_NPU_CMD(aclnnRelativeAttnBiasPos, relPosBiasConti, identityConti, pastValidLens, rabPosOut);
    return rabPosOut;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("relative_attn_bias_pos(Tensor rel_pos_bias, "
          "                       Tensor identity, "
          "                       int[] past_valid_lens"
          "                       ) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("relative_attn_bias_pos", &relative_attn_bias_pos_forward);
}

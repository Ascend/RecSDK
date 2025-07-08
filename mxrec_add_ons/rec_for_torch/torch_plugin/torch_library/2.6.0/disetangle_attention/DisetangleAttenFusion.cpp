/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
*/

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

bool pos_attr_safe_get(const std::string pos_attr_type, int &pos_attr)
{
    if (pos_attr_type == std::string("c2p")) {
        pos_attr = 0; // 0 means c2p
    } else if (pos_attr_type == std::string("p2c")) {
        pos_attr = 1; // 1 means p2c
    } else if (pos_attr_type == std::string("c2p|p2c")) {
        pos_attr = 2; // 2 means c2p|p2c
    } else {
        printf("pos_attr is invalid must in [c2p, p2c, c2p|p2c]");
        return false;
    }
    return true;
}

// 为NPU设备注册前向实现
std::tuple<at::Tensor, at::Tensor, at::Tensor> DisetangleAttentionPTA(const at::Tensor &query_layer,
    const at::Tensor &key_layer, const at::Tensor &value_layer, const at::Tensor &pos_key_layer,
    const at::Tensor &pos_query_layer, const at::Tensor &relative_pos, const at::Tensor &attn_mask,
    const std::string pos_attr_type, const double score_scale)
{
    auto query_layer_conti = query_layer.contiguous();
    auto key_layer_conti = key_layer.contiguous();
    auto value_layer_conti = value_layer.contiguous();
    auto pos_key_layer_conti = pos_key_layer.contiguous();
    auto pos_query_layer_conti = pos_query_layer.contiguous();
    auto relative_pos_conti = relative_pos.contiguous();
    auto mask_conti = attn_mask.contiguous();

    auto batch = query_layer.size(0);
    auto head = query_layer.size(1);
    auto seq = query_layer.size(2);
    auto dim = query_layer.size(3);

    at::Tensor attn_output = at::empty_like(query_layer_conti);
    at::Tensor attn_probs = at::empty({batch, head, seq, seq}, query_layer_conti.options());
    at::Tensor attn_weight = at::empty({batch, head, seq, seq}, query_layer_conti.options());
    int pos_attr = 0;
    if (!pos_attr_safe_get(pos_attr_type, pos_attr)) {
        return std::make_tuple(attn_output, attn_probs, attn_weight);
    }

    EXEC_NPU_CMD(aclnnDisetangleAttention,
        query_layer_conti,
        key_layer_conti,
        value_layer_conti,
        pos_key_layer_conti,
        pos_query_layer_conti,
        relative_pos_conti,
        mask_conti,
        pos_attr,
        score_scale,
        attn_output,
        attn_probs,
        attn_weight);
    return std::make_tuple(attn_output, attn_probs, attn_weight);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> DisetangleAttentionMeta(const at::Tensor &query_layer,
    const at::Tensor &key_layer, const at::Tensor &value_layer, const at::Tensor &pos_key_layer,
    const at::Tensor &pos_query_layer, const at::Tensor &relative_pos, const at::Tensor &attn_mask,
    const std::string pos_attr_type, const double score_scale)
{
    auto outSize = query_layer.sym_sizes();
    return std::make_tuple(at::empty_like(query_layer),
        at::empty_symint({outSize[0], outSize[1], outSize[2], outSize[2]}, query_layer.options()),
        at::empty_symint({outSize[0], outSize[1], outSize[2], outSize[2]}, query_layer.options()));
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("disentangle_attention( \
        Tensor query_layer, \
        Tensor key_layer, \
        Tensor value_layer, \
        Tensor pos_key_layer, \
        Tensor pos_query_layer, \
        Tensor relative_pos, \
        Tensor attn_mask, \
        str pos_attr_type, \
        float score_scale) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("disentangle_attention", &DisetangleAttentionPTA);
}

TORCH_LIBRARY_IMPL(mxrec, Meta, m)
{
    m.impl("disentangle_attention", &DisetangleAttentionMeta);
}
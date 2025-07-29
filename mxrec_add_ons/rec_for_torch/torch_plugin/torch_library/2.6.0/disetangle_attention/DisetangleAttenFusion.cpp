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

constexpr uint32_t CONST_1 = 1;
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t MAX_BATCH = 100;
constexpr uint32_t MAX_NUM = 64;
constexpr uint32_t MAX_DIM = 512;

bool pos_attr_safe_get(const std::string pos_attr_type, int& pos_attr)
{
    if (pos_attr_type == std::string("c2p")) {
        pos_attr = 0;  // 0 means c2p
    } else if (pos_attr_type == std::string("p2c")) {
        pos_attr = 1;  // 1 means p2c
    } else if (pos_attr_type == std::string("c2p|p2c")) {
        pos_attr = 2;  // 2 means c2p|p2c
    } else {
        printf("pos_attr is invalid must in [c2p, p2c, c2p|p2c]");
        return false;
    }
    return true;
}

void tensor_format_check(const at::Tensor& query_layer, const at::Tensor& key_layer, const at::Tensor& value_layer,
                         const at::Tensor& pos_key_layer, const at::Tensor& pos_query_layer,
                         const at::Tensor& relative_pos, const at::Tensor& attn_mask)
{
    TORCH_CHECK(query_layer.dim() == CONST_4, "query_layer expect 4 dim but get ", query_layer.dim());
    TORCH_CHECK(key_layer.dim() == CONST_4, "key_layer expect 4 dim but get ", key_layer.dim());
    TORCH_CHECK(value_layer.dim() == CONST_4, "value_layer expect 4 dim but get ", value_layer.dim());
    TORCH_CHECK(pos_key_layer.dim() == CONST_3, "pos_key_layer expect 3 dim but get ", pos_key_layer.dim());
    TORCH_CHECK(pos_query_layer.dim() == CONST_3, "pos_query_layer expect 3 dim but get ", pos_query_layer.dim());
    TORCH_CHECK(relative_pos.dim() == CONST_3 || relative_pos.dim() == CONST_4 || relative_pos.dim() == CONST_2,
                "relative_pos expect [2,3,4] dim but get ", pos_query_layer.dim());
    TORCH_CHECK(attn_mask.dim() == CONST_4 || attn_mask.dim() == CONST_3, "attn_mask expect [3,4] dim but get ",
                attn_mask.dim());

    auto batch = query_layer.size(0);
    auto head = query_layer.size(1);
    auto seq = query_layer.size(2);
    auto dim = query_layer.size(3);

    TORCH_CHECK(value_layer.sizes().equals(query_layer.sizes()), "query_layer format must equal value_layer format");
    TORCH_CHECK(key_layer.sizes().equals(query_layer.sizes()), "query_layer format must equal key_layer format");
    TORCH_CHECK(
        (pos_key_layer.size(0) == seq * CONST_2) && (pos_key_layer.size(1) == head) && (pos_key_layer.size(2) == dim),
        "pos_key_layer format must [2s,n,d]");
    TORCH_CHECK(pos_key_layer.sizes().equals(pos_query_layer.sizes()),
                "pos_key_layer format must equal pos_query_layer format");

    if (relative_pos.dim() == CONST_4) {
        TORCH_CHECK(relative_pos.size(0) == CONST_1, "[1, 1, s, s] relative_pos.size(0) must 1");
        TORCH_CHECK(relative_pos.size(1) == CONST_1, "[1, 1, s, s] relative_pos.size(1) must 1");
        TORCH_CHECK(relative_pos.size(2) == seq, "[1, 1, s, s] relative_pos.size(2) must ", seq);
        TORCH_CHECK(relative_pos.size(3) == seq, "[1, 1, s, s] relative_pos.size(3) must ", seq);
    } else if (relative_pos.dim() == CONST_3) {
        TORCH_CHECK(relative_pos.size(0) == CONST_1, "[1, s, s] relative_pos.size(0) must 1");
        TORCH_CHECK(relative_pos.size(1) == seq, "[1, s, s] relative_pos.size(1) must ", seq);
        TORCH_CHECK(relative_pos.size(2) == seq, "[1, s, s] relative_pos.size(2) must ", seq);
    } else {
        TORCH_CHECK(relative_pos.size(0) == seq, "[s, s] relative_pos.size(0) must ", seq);
        TORCH_CHECK(relative_pos.size(1) == seq, "[s, s] relative_pos.size(1) must ", seq);
    }

    if (attn_mask.dim() == CONST_3) {
        TORCH_CHECK(attn_mask.size(0) == batch, "[b, s, s] attn_mask.size(0) must ", batch);
        TORCH_CHECK(attn_mask.size(1) == seq, "[b, s, s] attn_mask.size(1) must ", seq);
        TORCH_CHECK(attn_mask.size(2) == seq, "[b, s, s] attn_mask.size(2) must ", seq);
    } else {
        TORCH_CHECK(attn_mask.size(0) == batch, "[b, 1, s, s] attn_mask.size(0) must ", batch);
        TORCH_CHECK(attn_mask.size(1) == CONST_1, "[b, 1, s, s] attn_mask.size(1) must ", CONST_1);
        TORCH_CHECK(attn_mask.size(2) == seq, "[b, 1, s, s] attn_mask.size(2) must ", seq);
        TORCH_CHECK(attn_mask.size(3) == seq, "[b, 1, s, s] attn_mask.size(3) must ", seq);
    }
}

// 为NPU设备注册前向实现
std::tuple<at::Tensor, at::Tensor, at::Tensor> DisetangleAttentionPTA(
    const at::Tensor& query_layer, const at::Tensor& key_layer, const at::Tensor& value_layer,
    const at::Tensor& pos_key_layer, const at::Tensor& pos_query_layer, const at::Tensor& relative_pos,
    const at::Tensor& attn_mask, const std::string pos_attr_type, const double score_scale)
{
    auto query_layer_conti = query_layer.contiguous();
    auto key_layer_conti = key_layer.contiguous();
    auto value_layer_conti = value_layer.contiguous();
    auto pos_key_layer_conti = pos_key_layer.contiguous();
    auto pos_query_layer_conti = pos_query_layer.contiguous();
    auto relative_pos_conti = relative_pos.contiguous();
    auto mask_conti = attn_mask.contiguous();

    tensor_format_check(query_layer, key_layer, value_layer, pos_key_layer, pos_query_layer, relative_pos, attn_mask);

    auto batch = query_layer.size(0);
    auto head = query_layer.size(1);
    auto seq = query_layer.size(2);
    auto dim = query_layer.size(3);
    TORCH_CHECK(batch <= MAX_BATCH, "current max batch is 100 but get", batch);
    TORCH_CHECK(head <= MAX_NUM, "current max head is 64 but get", head);
    TORCH_CHECK(dim <= MAX_DIM, "current max dim is 512 but get", dim);

    at::Tensor attn_output = at::empty_like(query_layer_conti);
    at::Tensor attn_probs = at::empty({batch, head, seq, seq}, query_layer_conti.options());
    at::Tensor attn_weight = at::empty({batch, head, seq, seq}, query_layer_conti.options());
    int pos_attr = 0;
    if (!pos_attr_safe_get(pos_attr_type, pos_attr)) {
        return std::make_tuple(attn_output, attn_probs, attn_weight);
    }

    EXEC_NPU_CMD(aclnnDisetangleAttention, query_layer_conti, key_layer_conti, value_layer_conti, pos_key_layer_conti,
                 pos_query_layer_conti, relative_pos_conti, mask_conti, pos_attr, score_scale, attn_output, attn_probs,
                 attn_weight);
    return std::make_tuple(attn_output, attn_probs, attn_weight);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> DisetangleAttentionMeta(
    const at::Tensor& query_layer, const at::Tensor& key_layer, const at::Tensor& value_layer,
    const at::Tensor& pos_key_layer, const at::Tensor& pos_query_layer, const at::Tensor& relative_pos,
    const at::Tensor& attn_mask, const std::string pos_attr_type, const double score_scale)
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
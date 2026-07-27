/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

#include <string>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"

using namespace at;

namespace {
// metadata 内存布局常量，与 op_kernel_aicpu/hstu_attn_metadata.h 保持一致：
//   每个 section 内 AIC/AIV core 各占 16 个 int32；head 段额外 16 个 int32。
constexpr int64_t AIC_CORE_NUM = 36;
constexpr int64_t AIV_CORE_NUM = 72;
constexpr int64_t METADATA_STRIDE = 16;
constexpr int64_t METADATA_ALIGN = 4096;

int64_t DeriveBatchSize(int64_t batch_size, const c10::optional<at::Tensor>& seqused_q,
                        const c10::optional<at::Tensor>& cu_seqlens_q)
{
    if (batch_size > 0) {
        return batch_size;
    }
    if (seqused_q.has_value() && seqused_q->defined() && seqused_q->numel() > 0) {
        return seqused_q->size(0);
    }
    if (cu_seqlens_q.has_value() && cu_seqlens_q->defined() && cu_seqlens_q->numel() > 1) {
        return cu_seqlens_q->size(0) - 1;
    }
    return -1;
}

c10::optional<at::Tensor> ContiguousOpt(const c10::optional<at::Tensor>& t)
{
    if (t.has_value() && t->defined()) {
        return c10::optional<at::Tensor>(t->contiguous());
    }
    return c10::nullopt;
}
}  // namespace

at::Tensor hstu_attn_metadata_impl_npu(const c10::optional<at::Tensor>& cu_seqlens_q,
                                       const c10::optional<at::Tensor>& cu_seqlens_kv,
                                       const c10::optional<at::Tensor>& seqused_q,
                                       const c10::optional<at::Tensor>& seqused_kv, int64_t batch_size,
                                       int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t num_heads_q,
                                       int64_t num_heads_kv, int64_t head_dim, int64_t mask_mode, int64_t win_left,
                                       int64_t win_right, std::string layout_q, std::string layout_kv,
                                       std::string layout_out)
{
    TORCH_CHECK(num_heads_q > 0 && num_heads_kv > 0, "num_heads_q/num_heads_kv must be > 0");
    TORCH_CHECK(num_heads_q % num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv");

    int64_t batch = DeriveBatchSize(batch_size, seqused_q, cu_seqlens_q);
    TORCH_CHECK(batch > 0,
                "hstu_attn_metadata: batch_size must be > 0, or provide seqused_q / cu_seqlens_q to derive it.");

    // 输出 metadata 大小：((AIC+AIV) * batch * num_heads_kv + 1) * 16，再按 4096 对齐。
    int64_t elems = ((AIC_CORE_NUM + AIV_CORE_NUM) * batch * num_heads_kv + 1) * METADATA_STRIDE;
    int64_t aligned = ((elems + METADATA_ALIGN - 1) / METADATA_ALIGN) * METADATA_ALIGN;

    auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kInt);
    at::Tensor metadata = at::empty({aligned}, options);

    auto cu_q = ContiguousOpt(cu_seqlens_q);
    auto cu_kv = ContiguousOpt(cu_seqlens_kv);
    auto used_q = ContiguousOpt(seqused_q);
    auto used_kv = ContiguousOpt(seqused_kv);

    // ConvertTypes 以非常量左值引用接收参数，const char* 必须先落到左值再传入。
    const char* layout_q_c = layout_q.c_str();
    const char* layout_kv_c = layout_kv.c_str();
    const char* layout_out_c = layout_out.c_str();

    EXEC_NPU_CMD(aclnnHstuAttnMetadata, cu_q, cu_kv, used_q, used_kv, batch, max_seqlen_q, max_seqlen_kv, num_heads_q,
                 num_heads_kv, head_dim, mask_mode, win_left, win_right, layout_q_c, layout_kv_c, layout_out_c,
                 metadata);
    return metadata;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_attn_metadata(Tensor? cu_seqlens_q, Tensor? cu_seqlens_kv, Tensor? seqused_q, Tensor? seqused_kv, "
          "int batch_size, int max_seqlen_q, int max_seqlen_kv, int num_heads_q, int num_heads_kv, int head_dim, "
          "int mask_mode, int win_left, int win_right, str layout_q, str layout_kv, str layout_out) -> Tensor");
}

// 该算子的 4 个张量入参均为可选，可能全部为 None（仅靠 batch_size 等属性驱动）。
// 此时分发器无法从参数推断出 PrivateUse1(NPU) key，故注册为 CompositeExplicitAutograd
// 全后端兜底 kernel：无论入参是否带 NPU 张量都会路由到本实现（内部固定在 NPU 上分配/执行）。
TORCH_LIBRARY_IMPL(mxrec, CompositeExplicitAutograd, m)
{
    m.impl("hstu_attn_metadata", &hstu_attn_metadata_impl_npu);
}

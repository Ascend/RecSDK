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

/*!
 * \file hstu_attn_metadata_backward.cpp
 * \brief HstuAttnMetadataBackward（纯 AI CPU 算子）的 PyTorch 绑定。
 *
 * 对外接口：torch.ops.mxrec.hstu_attn_metadata_backward(cu_seqlens_q, cu_seqlens_kv, num_heads, head_dim,
 *                                                        mask_mode=0, num_contexts=None, num_targets=None,
 *                                                        target_group_size=0, full_cnt=None, mask_cnt=None)
 * num_contexts / num_targets / target_group_size 为 causal（mask_mode=1）专属输入，布局 [batch]
 * int32/int64（契约见 kernel 侧 mask/causal_mask_predictor.h）；full_cnt / mask_cnt 为 arbitrary
 * （mask_mode=2）专属输入，可选但 af 下必选，逻辑布局 [batch, maxBlkCntK] int32（契约见 kernel 侧
 * mask/arbitrary_mask_predictor.h）。槽位分段固定：非当前模式的段必须保持 None。
 *
 * 与普通 AscendC 算子不同：本算子的 aclnn 接口在**独立 vendor** 的 libcust_opapi.so 里，
 * 由 EXEC_NPU_CMD 通过 dlopen 动态加载，因此运行前必须先编译并安装 AICPU kernel + aclnn
 * vendor 包（详见同目录 README_backward.md）。
 *
 * 输出 metadata 由本绑定**自行分配**：算子把结果 in-place 写进 metadata 输出张量，
 * 容量必须由调用方按 sectionNum 上界备足（kernel 侧会校验 metadataLen >= required）。
 */

#include <cstdint>
#include <limits>

#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "hstu_attn_metadata_layout.h"

using namespace at;

namespace {

// head_dim 白名单，与 host 侧 HstuAttnMetadataBackwardCheck::CheckAttr 一致。
constexpr int64_t HEAD_DIM_32 = 32;
constexpr int64_t HEAD_DIM_64 = 64;
constexpr int64_t HEAD_DIM_128 = 128;
constexpr int64_t HEAD_DIM_256 = 256;

// mask_mode 取值域，与 host 侧 CheckAttr / kernel 侧 MaskMode 枚举同源（0=nomask，1=causal，2=arbitrary）。
constexpr int64_t MASK_MODE_NO_MASK = 0;
constexpr int64_t MASK_MODE_CAUSAL = 1;
constexpr int64_t MASK_MODE_ARBITRARY = 2;

int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

/*! \brief q/k offsets 张量校验：存在、一维、长度 >= 2、dtype 为 int32/int64。 */
void CheckOffsetTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " should be provided, but got undefined tensor");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1D, but got ", tensor.dim(), " dims");
    TORCH_CHECK(tensor.size(0) >= 2, name, " must have at least 2 elements (batch + 1), but got ", tensor.size(0));
    TORCH_CHECK(tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong, name,
                " must be int32 or int64, but got ", tensor.scalar_type());
}

/*! \brief causal 专属张量校验：未提供（None/undefined）合法；提供时必须 1D、len == batch、int32/int64。 */
void CheckCausalTensor(const c10::optional<at::Tensor>& t, const char* name, int64_t mask_mode, int64_t batch)
{
    const bool provided = t.has_value() && t->defined();
    if (!provided) {
        return;  // null / 空 = 未提供，任何 mask 模式都接受（causal 下 = history-only / 无 target 封顶）
    }
    TORCH_CHECK(mask_mode == MASK_MODE_CAUSAL, name, " is causal-mode-only input, but mask_mode is ", mask_mode);
    TORCH_CHECK(t->dim() == 1, name, " must be 1D, but got ", t->dim(), " dims");
    TORCH_CHECK(t->size(0) == batch, name, " length must equal batch size ", batch, ", but got ", t->size(0));
    TORCH_CHECK(t->scalar_type() == at::kInt || t->scalar_type() == at::kLong, name,
                " must be int32 or int64, but got ", t->scalar_type());
}

/*! \brief 可选张量的 contiguous 化：None / undefined 统一折叠成 nullopt（→ aclnn nullptr）。 */
c10::optional<at::Tensor> ContiguousOpt(const c10::optional<at::Tensor>& t)
{
    if (t.has_value() && t->defined()) {
        return c10::optional<at::Tensor>(t->contiguous());
    }
    return c10::nullopt;
}

/*!
 * \brief 入参前置校验：与 host 侧 HstuAttnMetadataBackwardCheck 同口径，但在 torch 这一层
 * 先拦一道，让非法入参报出可读的 PyTorch 错误，而不是等到 aclnn 的 plog 里去翻。
 * \return batch（= cu_seqlens 长度 - 1，已保证 > 0 且 batch * num_heads 不溢出 uint32）
 */
int64_t CheckInputs(const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_kv, int64_t num_heads,
                    int64_t head_dim, int64_t mask_mode, const c10::optional<at::Tensor>& num_contexts,
                    const c10::optional<at::Tensor>& num_targets, int64_t target_group_size,
                    const c10::optional<at::Tensor>& full_cnt, const c10::optional<at::Tensor>& mask_cnt)
{
    CheckOffsetTensor(cu_seqlens_q, "cu_seqlens_q");
    CheckOffsetTensor(cu_seqlens_kv, "cu_seqlens_kv");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_kv.size(0),
                "cu_seqlens_q and cu_seqlens_kv must have the same length, but got ", cu_seqlens_q.size(0), " and ",
                cu_seqlens_kv.size(0));
    TORCH_CHECK(num_heads > 0, "num_heads must be greater than 0, but got ", num_heads);
    TORCH_CHECK(
        head_dim == HEAD_DIM_32 || head_dim == HEAD_DIM_64 || head_dim == HEAD_DIM_128 || head_dim == HEAD_DIM_256,
        "head_dim only supports 32, 64, 128, 256, but got ", head_dim);
    // mask_mode 与 host CheckAttr 分层校验同口径：三类（no-mask / causal / arbitrary）均已实现，
    // 各自专属段的入参校验见下。
    TORCH_CHECK(mask_mode >= MASK_MODE_NO_MASK && mask_mode <= MASK_MODE_ARBITRARY, "mask_mode must be in [",
                MASK_MODE_NO_MASK, ", ", MASK_MODE_ARBITRARY, "], but got ", mask_mode);

    // target_group_size 是 causal 的 target 项组大小：负值拒绝（<= 0 在 kernel 侧语义为「无 target」），
    // 其余 mask 模式忽略该值。
    TORCH_CHECK(target_group_size >= 0, "target_group_size must be non-negative, but got ", target_group_size);

    // num_contexts / num_targets 是 causal 专属输入：causal 下独立可选（None = history-only /
    // 无 target 封顶），非 causal 模式必须缺席（详见 CheckCausalTensor）。
    const int64_t batch = cu_seqlens_q.size(0) - 1;
    TORCH_CHECK(batch > 0, "batch (cu_seqlens length - 1) must be greater than 0, but got ", batch);
    CheckCausalTensor(num_contexts, "num_contexts", mask_mode, batch);
    CheckCausalTensor(num_targets, "num_targets", mask_mode, batch);

    // full_cnt / mask_cnt 是 arbitrary 专属输入：必须成对出现，且只在 af 下出现；
    // dtype / numel（batch * maxBlkCntK）等校验由 host 与 kernel 分层把关。
    const bool has_full_cnt = full_cnt.has_value() && full_cnt->defined();
    const bool has_mask_cnt = mask_cnt.has_value() && mask_cnt->defined();
    TORCH_CHECK(has_full_cnt == has_mask_cnt, "full_cnt and mask_cnt must be provided in pairs");
    if (mask_mode == MASK_MODE_ARBITRARY) {
        TORCH_CHECK(has_full_cnt, "mask_mode ", MASK_MODE_ARBITRARY, " (arbitrary) requires full_cnt and mask_cnt");
    } else {
        TORCH_CHECK(!has_full_cnt, "full_cnt / mask_cnt are arbitrary-mode-only inputs, but mask_mode is ", mask_mode);
    }

    // 下面的 sectionNum 上界要经过 uint32 传给 RequiredMetadataElements，这里挡掉溢出。
    TORCH_CHECK(batch <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) / num_heads,
                "batch * num_heads is too large: ", batch, " * ", num_heads);
    return batch;
}

}  // namespace

at::Tensor hstu_attn_metadata_backward_impl_npu(const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_kv,
                                                int64_t num_heads, int64_t head_dim, int64_t mask_mode,
                                                const c10::optional<at::Tensor>& num_contexts,
                                                const c10::optional<at::Tensor>& num_targets, int64_t target_group_size,
                                                const c10::optional<at::Tensor>& full_cnt,
                                                const c10::optional<at::Tensor>& mask_cnt)
{
    const int64_t batch = CheckInputs(cu_seqlens_q, cu_seqlens_kv, num_heads, head_dim, mask_mode, num_contexts,
                                      num_targets, target_group_size, full_cnt, mask_cnt);

    // ---- 输出容量：按 sectionNum 的上界备足 ----
    // 每个 BN2（= batch * num_heads）最多被切成一个 section，故 batch * num_heads 是 sectionNum 的上界，
    // 实际 sectionNum 通常更小 —— 每 section 一条 FA + 一条 FD，多出来的尾部容量不会被算子触碰。
    const int64_t section_num_max = batch * num_heads;
    const int64_t elems = hstu_meta::HstuMetadataCapacityElems(section_num_max);
    const int64_t aligned = AlignUp(elems, hstu_meta::METADATA_ALIGN);

    auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kInt);
    at::Tensor metadata = at::empty({aligned}, options);

    auto cu_q = cu_seqlens_q.contiguous();
    auto cu_kv = cu_seqlens_kv.contiguous();
    auto ctx = ContiguousOpt(num_contexts);
    auto tgt = ContiguousOpt(num_targets);
    auto full = ContiguousOpt(full_cnt);
    auto mask = ContiguousOpt(mask_cnt);

    EXEC_NPU_CMD(aclnnHstuAttnMetadataBackward, cu_q, cu_kv, ctx, tgt, num_heads, head_dim, mask_mode,
                 target_group_size, full, mask, metadata);
    return metadata;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    // mask_mode 给默认值 0（no mask）：已有调用方（hstu_v2 backend、既有用例）都按 no-mask 使用，
    // 给默认值可避免它们被迫改动。num_contexts / num_targets / target_group_size 是 causal
    // （mask_mode=1）专属输入，full_cnt / mask_cnt 是 arbitrary（mask_mode=2）专属输入，
    // 其余模式必须保持 None / 0（槽位分段固定，非当前模式的段拒收非空值）。
    m.def("hstu_attn_metadata_backward(Tensor cu_seqlens_q, Tensor cu_seqlens_kv, "
          "int num_heads, int head_dim, int mask_mode=0, Tensor? num_contexts=None, Tensor? num_targets=None, "
          "int target_group_size=0, Tensor? full_cnt=None, Tensor? mask_cnt=None) -> Tensor");
}

// 两个 offsets 张量都是必选入参且必须是 NPU 张量，故直接挂到 PrivateUse1（NPU）分发键上，
// 与算子库中大多数适配层（ln_mul / hstu_v2 / reverse_sequence 等）的注册方式一致。
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_attn_metadata_backward", &hstu_attn_metadata_backward_impl_npu);
}

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

/**
 * @file hstu_forward_v2.cpp
 * @brief HSTU Forward V2 算子 Kernel 实现
 * @description 实现算子在 AI Core 上的执行逻辑，包括:
 *              - Cube 核上的 MMAD 主循环 (QK、PV 计算)
 *              - Vector 核上的 Epilogue 主循环 (SiLU 激活、RAB 处理、输出转置)
 *              - 支持 HAS_RAB 和 BLOCK_K 两种模板参数分支
 */

#include "kernel_operator.h"
#include "catlass_hstu/kernel/fwd/forward_kernel_tiling.hpp"
#include "catlass_hstu/kernel/fwd/forward_kernel_builder.hpp"
#include "catlass_hstu/kernel/fwd/forward_kernel_mmad_mainloop.hpp"
#include "catlass_hstu/kernel/fwd/forward_kernel_epilogue_mainloop.hpp"

/**
 * @brief HSTU Forward V2 算子 Kernel 入口函数
 * @tparam HAS_RAB 是否有相对位置偏置
 * @tparam BLOCK_K K 方向的块大小 (128)
 * @param q Query 张量
 * @param k Key 张量
 * @param v Value 张量
 * @param attnBias 注意力偏置 (RAB)
 * @param mask 注意力掩码 (可选)
 * @param seq_offset_q Query 序列偏移量
 * @param seq_offset_k Key 序列偏移量
 * @param num_context 上下文长度 (可选)
 * @param num_target 目标长度 (可选)
 * @param attnOutput 输出: 注意力输出
 * @param workSpace 工作空间
 * @param tiling Tiling 数据
 * @description 算子的主入口函数，根据当前运行的核类型 (AIC/AIV) 选择执行:
 *              - AIC (Cube 核): 执行 MMAD 主循环
 *              - AIV (Vector 核): 执行 Epilogue 主循环
 */
template <bool HAS_RAB, uint32_t BLOCK_K>
CATLASS_GLOBAL void hstu_forward_v2(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attnBias, GM_ADDR mask,
                                    GM_ADDR seq_offset_q, GM_ADDR seq_offset_k, GM_ADDR num_context, GM_ADDR num_target,
                                    GM_ADDR attnOutput, GM_ADDR workSpace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr bool HAS_MASK = false;
    using namespace Catlass::Kernel;

    using Element = DTYPE_Q;
    using ElementOffset = DTYPE_SEQ_OFFSET_Q;
    using KernelConfig = ForwardKernelConfig<Arch::Ascend950, Element, ElementOffset, BLOCK_K, HAS_RAB, HAS_MASK>;
    using KernelBuilder = ForwardKenrelBuilder<KernelConfig>;

    using QBlockScheduler = typename KernelBuilder::QBlockScheduler;
    using KBlockScheduler = typename KernelBuilder::KBlockScheduler;

    if ASCEND_IS_AIC {
        using BlockMmadQK = typename KernelBuilder::BlockMmadQK;
        using BlockMmadPV = typename KernelBuilder::BlockMmadPV;

        using MmadMainLoopKernel = ForwardMmadMainloop<BlockMmadQK, BlockMmadPV, QBlockScheduler, KBlockScheduler>;

        using MmadMainLoopParams = typename MmadMainLoopKernel::Params;

        MmadMainLoopParams params{q, k, v, seq_offset_q, seq_offset_k};
        MmadMainLoopKernel kernel(tiling);
        kernel(params);
    } else {
        using BlockEpilogueQK = typename KernelBuilder::BlockEpilogueQK;
        using BlockEpiloguePV = typename KernelBuilder::BlockEpiloguePV;

        using EpilogueMainLoopKernel =
            ForwardEpilogueMainloop<BlockEpilogueQK, BlockEpiloguePV, QBlockScheduler, KBlockScheduler>;

        using EpilogueMainLoopParams = typename EpilogueMainLoopKernel::Params;

        EpilogueMainLoopParams params{attnBias, seq_offset_q, seq_offset_k, attnOutput};
        EpilogueMainLoopKernel kernel(tiling);
        kernel(params);
    }
}

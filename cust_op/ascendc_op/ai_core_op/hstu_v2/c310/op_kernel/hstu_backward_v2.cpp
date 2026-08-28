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
 • @file hstu_backward_v2.cpp

 • @brief HSTU Backward V2 算子 Kernel 实现

 • @description 实现算子在 AI Core 上的执行逻辑，包括:

 •              - Cube 核上的 MMAD 主循环 (QK、GV、dQ、dK、dV 计算)

 •              - Vector 核上的 Epilogue 主循环 (SiLU 激活、梯度计算、RAB 处理)

 •              - 支持 HAS_RAB 和 BLOCK_K 两种模板参数分支

 */

#include "kernel_operator.h"
#include "catlass_hstu/kernel/bwd/backward_kernel_tiling.hpp"
#include "catlass_hstu/kernel/bwd/backward_kernel_builder.hpp"
#include "catlass_hstu/kernel/bwd/backward_kernel_mmad_mainloop.hpp"
#include "catlass_hstu/kernel/bwd/backward_kernel_epilogue_mainloop.hpp"

/**
 • @brief HSTU Backward V2 算子 Kernel 入口函数

 • @tparam HAS_RAB 是否有相对位置偏置

 • @tparam BLOCK_K K 方向的块大小 (128 或 256)

 • @param grad 梯度输入

 • @param q Query 张量

 • @param k Key 张量

 • @param v Value 张量

 • @param rab 相对位置偏置 (可选)

 • @param seqOffsetQ Query 序列偏移量

 • @param seqOffsetK Key 序列偏移量

 • @param numContext 上下文长度 (可选)

 • @param numTarget 目标长度 (可选)

 • @param qShare Q 共享数据 (可选)

 • @param qGrad 输出: Query 梯度

 • @param kGrad 输出: Key 梯度

 • @param vGrad 输出: Value 梯度

 • @param rabGrad 输出: RAB 梯度 (可选)

 • @param workSpace 工作空间

 • @param tiling Tiling 数据

 • @description 算子的主入口函数，根据当前运行的核类型 (AIC/AIV) 选择执行:

 •              - AIC (Cube 核): 执行 MMAD 主循环

 •              - AIV (Vector 核): 执行 Epilogue 主循环

 */
template <bool HAS_RAB, bool IS_LOCAL, bool IS_CAUSAL, bool IS_CONTEXT, bool IS_TARGET, bool IS_ARBITRARY,
          uint32_t BLOCK_K>
CATLASS_GLOBAL void hstu_backward_v2(GM_ADDR grad, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR rab, GM_ADDR seqOffsetQ,
                                     GM_ADDR seqOffsetK, GM_ADDR numContext, GM_ADDR numTarget, GM_ADDR qShare,
                                     GM_ADDR metadata, GM_ADDR arbitraryFunc, GM_ADDR sparseInfo, GM_ADDR qGrad,
                                     GM_ADDR kGrad, GM_ADDR vGrad, GM_ADDR rabGrad, GM_ADDR workSpace, GM_ADDR tiling)
{
    // 设置算子类型为CUBE:VECTOR = 1:2的MIX模式
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    using namespace Catlass::Kernel;
    using namespace Catlass::Kernel::Mask;

    using Element = DTYPE_Q;
    using ElementOffset = DTYPE_SEQ_OFFSET_Q;
    using KernelConfig = BackwardKernelConfig<Arch::Ascend950, Element, ElementOffset, BLOCK_K, HAS_RAB, IS_LOCAL,
                                              IS_CAUSAL, IS_ARBITRARY>;
    using KernelBuilder = BackwardKernelBuilder<KernelConfig>;

    using QBlockScheduler = typename KernelBuilder::QBlockScheduler;
    using KBlockScheduler = typename KernelBuilder::KBlockScheduler;

    using L1TileShape = typename KernelConfig::L1TileShape;
    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});
    using Predictor = PredictorSelector<IS_LOCAL, IS_CAUSAL, IS_ARBITRARY, BLOCK_M, BLOCK_N, ElementOffset,
                                        /*IS_FWD=*/false>;

    // metadata 驱动路径的调度器类型(与设备类型同参: 行轴块=BLOCK_N=Rk, 列轴块=BLOCK_M=Cq)
    using MetaKBlockScheduler = Catlass::Gemm::Block::MetadataRowBlockScheduler<ElementOffset, BLOCK_N, BLOCK_M>;
    using MetaQBlockScheduler = Catlass::Gemm::Block::ColumnBlockScheduler<MetaKBlockScheduler, BLOCK_N, BLOCK_M, true>;

    // mmad mainloop kernel
    if ASCEND_IS_AIC {
        using BlockMmadQK = typename KernelBuilder::BlockMmadQK;
        using BlockMmadGV = typename KernelBuilder::BlockMmadGV;
        using BlockMmadVGrad = typename KernelBuilder::BlockMmadVGrad;
        using BlockMmadKGrad = typename KernelBuilder::BlockMmadKGrad;
        using BlockMmadQGrad = typename KernelBuilder::BlockMmadQGrad;

        // 运行期择一: 传了 metadata → metadata 驱动;否则 → 旧设备现算(逐字节等价、零回归)
        if (metadata != nullptr) {
            using MmadMainLoopKernel =
                BackwardMmadMainloop<BlockMmadQK, BlockMmadGV, BlockMmadVGrad, BlockMmadKGrad, BlockMmadQGrad,
                                     MetaQBlockScheduler, MetaKBlockScheduler, ElementOffset, IS_LOCAL, IS_CAUSAL,
                                     IS_CONTEXT, IS_TARGET, IS_ARBITRARY, Predictor>;
            typename MmadMainLoopKernel::Params params{grad,          q,         k,          v,         seqOffsetQ,
                                                       seqOffsetK,    qShare,    numContext, numTarget, metadata,
                                                       arbitraryFunc, sparseInfo};
            MmadMainLoopKernel kernel(tiling);
            kernel(params);
        } else {
            using MmadMainLoopKernel =
                BackwardMmadMainloop<BlockMmadQK, BlockMmadGV, BlockMmadVGrad, BlockMmadKGrad, BlockMmadQGrad,
                                     QBlockScheduler, KBlockScheduler, ElementOffset, IS_LOCAL, IS_CAUSAL, IS_CONTEXT,
                                     IS_TARGET, IS_ARBITRARY, Predictor>;
            typename MmadMainLoopKernel::Params params{grad,          q,         k,          v,         seqOffsetQ,
                                                       seqOffsetK,    qShare,    numContext, numTarget, nullptr,
                                                       arbitraryFunc, sparseInfo};
            MmadMainLoopKernel kernel(tiling);
            kernel(params);
        }
    } else {
        // epilogue mainloop kernel
        using BlockEpilogueQK = typename KernelBuilder::BlockEpilogueQK;
        using BlockEpilogueGV = typename KernelBuilder::BlockEpilogueGV;
        using BlockEpilogueKVGrad = typename KernelBuilder::BlockEpilogueKVGrad;
        using BlockEpilogueQGrad = typename KernelBuilder::BlockEpilogueQGrad;

        if (metadata != nullptr) {
            using EpilogueMainLoopKernel =
                BackwardEpilogueMainloop<BlockEpilogueQK, BlockEpilogueGV, BlockEpilogueKVGrad, BlockEpilogueQGrad,
                                         MetaQBlockScheduler, MetaKBlockScheduler, ElementOffset, IS_LOCAL, IS_CAUSAL,
                                         IS_CONTEXT, IS_TARGET, IS_ARBITRARY, Predictor>;
            typename EpilogueMainLoopKernel::Params params{rab,      seqOffsetQ,    seqOffsetK, qGrad,      kGrad,
                                                           vGrad,    rabGrad,       qShare,     numContext, numTarget,
                                                           metadata, arbitraryFunc, sparseInfo};
            EpilogueMainLoopKernel kernel(tiling);
            kernel(params);
        } else {
            using EpilogueMainLoopKernel =
                BackwardEpilogueMainloop<BlockEpilogueQK, BlockEpilogueGV, BlockEpilogueKVGrad, BlockEpilogueQGrad,
                                         QBlockScheduler, KBlockScheduler, ElementOffset, IS_LOCAL, IS_CAUSAL,
                                         IS_CONTEXT, IS_TARGET, IS_ARBITRARY, Predictor>;
            typename EpilogueMainLoopKernel::Params params{rab,     seqOffsetQ,    seqOffsetK, qGrad,      kGrad,
                                                           vGrad,   rabGrad,       qShare,     numContext, numTarget,
                                                           nullptr, arbitraryFunc, sparseInfo};
            EpilogueMainLoopKernel kernel(tiling);
            kernel(params);
        }
    }
}

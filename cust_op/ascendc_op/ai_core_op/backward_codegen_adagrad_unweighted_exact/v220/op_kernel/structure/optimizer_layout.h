/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef BACKWARD_CODEGEN_UNWEIGHTED_EXACT_OPTIMIZER_LAYOUT_H
#define BACKWARD_CODEGEN_UNWEIGHTED_EXACT_OPTIMIZER_LAYOUT_H

using namespace AscendC;

namespace BackwardCodegenUnweightedExact {

// 优化器内存布局描述结构体
struct EngineLayout {
    // 定义内存布局类型枚举，表示不同动量数量的内存布局
    enum class MomentumLayoutType {
        LAYOUT_GRAD_ONLY = 0,               // SGD: 仅梯度，无动量
        LAYOUT_GRAD_MOMENTUM1_PER_DIM = 1,  // AdaGrad: 梯度+一阶动量（按维度）
        LAYOUT_GRAD_MOMENTUM1_ROWWISE = 2,  // RowWise AdaGrad: 梯度+一阶动量（行级）
        LAYOUT_GRAD_MOMENTUM1_MOMENTUM2 = 3 // Adam: 梯度+一阶动量+二阶动量
    };

    static constexpr int MOMENTUM_PAD_NUM = 16; // RowwiseAdagrad

    // 定义各种布局类型的输出数量常量
    static constexpr int OUTPUT_COUNT_LAYOUT_GRAD_ONLY = 1;                // SGD: 仅梯度输出
    static constexpr int OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_PER_DIM = 2;   // AdaGrad: 梯度+一阶动量
    static constexpr int OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_ROWWISE = 2;   // RowWise AdaGrad: 梯度+一阶动量
    static constexpr int OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_MOMENTUM2 = 3; // Adam: 梯度+一阶动量+二阶动量

    // 输出数量的静态查找表
    static constexpr int OUTPUT_COUNT_TABLE[] = {
        OUTPUT_COUNT_LAYOUT_GRAD_ONLY,               // LAYOUT_GRAD_ONLY
        OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_PER_DIM,  // LAYOUT_GRAD_MOMENTUM1_PER_DIM
        OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_ROWWISE,  // LAYOUT_GRAD_MOMENTUM1_ROWWISE
        OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_MOMENTUM2 // LAYOUT_GRAD_MOMENTUM1_MOMENTUM2
    };

    // 计算输出数量的模板辅助类，使用静态查表
    template <MomentumLayoutType layoutType> struct OutputCount {
        static constexpr int value = OUTPUT_COUNT_TABLE[static_cast<int>(layoutType)];
    };
};

using MomentumLayoutType = EngineLayout::MomentumLayoutType;
template <EngineLayout::MomentumLayoutType layout> using OutputCount = typename EngineLayout::OutputCount<layout>;

} // namespace BackwardCodegenUnweightedExact

#endif // BACKWARD_CODEGEN_UNWEIGHTED_EXACT_OPTIMIZER_LAYOUT_H
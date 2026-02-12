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

#ifndef ADAGRAD_OPTIMIZER_H
#define ADAGRAD_OPTIMIZER_H

#include "optimizer_interface.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenUnweightedExact {
namespace Optimizers {

// Adagrad优化器
class AdagradOptimizer : public OptimizerInterfaces::IOptimizer<AdagradOptimizer> {
public:
    __aicore__ inline void Compute(LocalTensor<float>& newInputLt, LocalTensor<float>& outLt, int64_t gradOffset,
                                   int64_t moment1Offset, int64_t embedDim, OptimizerConfig& optimizerConfig)
    {
        // Adagrad 计算逻辑
        // m_t = g_t^2 (梯度平方)
        Mul<float>(outLt[gradOffset], newInputLt[gradOffset], newInputLt[gradOffset], embedDim);
        // v_t = m_{t-1} + m_t (加上moment1)
        Add<float>(outLt[gradOffset], newInputLt[moment1Offset], outLt[gradOffset], embedDim);
        // r_t = sqrt(v_t) + ε (开根号后加上eps)
        Sqrt<float>(outLt[gradOffset], outLt[gradOffset], embedDim);
        // r_t = r_t + ε (加上eps)
        Adds<float>(outLt[gradOffset], outLt[gradOffset], optimizerConfig.eps, embedDim);
        // lr_t = -lr (设置调整后的学习率)
        Duplicate<float>(outLt[moment1Offset], optimizerConfig.learningRate, embedDim);
        // lr_t = lr_t / r_t (学习率除以sqrt结果)
        Div<float>(outLt[gradOffset], outLt[moment1Offset], outLt[gradOffset], embedDim);
        // Δw = lr_t * g_t (乘以梯度得到更新值)
        Mul<float>(outLt[gradOffset], outLt[gradOffset], newInputLt[gradOffset], embedDim);
        // Δw = -Δw (取反)
        Muls<float>(outLt[gradOffset], outLt[gradOffset], -1, embedDim);
        // moment1_{t+1} = m_t (更新moment1为当前梯度平方)
        Mul<float>(outLt[moment1Offset], newInputLt[gradOffset], newInputLt[gradOffset], embedDim);
    }
};

} // namespace Optimizers
} // namespace BackwardCodegenUnweightedExact

#endif // ADAGRAD_OPTIMIZER_H
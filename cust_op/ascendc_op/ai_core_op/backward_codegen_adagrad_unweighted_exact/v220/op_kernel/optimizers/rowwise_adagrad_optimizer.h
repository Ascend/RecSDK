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

#ifndef ROWWISE_ADAGRAD_OPTIMIZER_H
#define ROWWISE_ADAGRAD_OPTIMIZER_H

#include "optimizer_interface.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenUnweightedExact {
namespace Optimizers {

// RowwiseAdagrad优化器
class RowWiseAdagradOptimizer : public OptimizerInterfaces::IOptimizer<RowWiseAdagradOptimizer> {
public:
    __aicore__ inline void Compute(LocalTensor<float>& inputLt, LocalTensor<float>& outLt, int64_t gradOffset,
                                   int64_t momentOffset, int64_t embedDim, OptimizerConfig& optimizerConfig)
    {
        // Step 1: 计算 grad^2 → 复用 outLt[gradOffset]
        Mul(outLt[gradOffset], inputLt[gradOffset], inputLt[gradOffset], embedDim);

        // Step 2: ReduceSum → 结果暂存到 outLt[momentOffset]（scalar）
        uint32_t srcShape[2] = {1, static_cast<uint32_t>(embedDim)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(outLt[momentOffset], outLt[gradOffset], srcShape,
                                                                       false);

        // Step 3: 计算 meanSq = sumSq / D
        float sumSq = outLt.GetValue(momentOffset);
        float meanSq = sumSq / static_cast<float>(embedDim);

        // Step 4: 获取当前（历史）动量
        float currentMomentum = inputLt.GetValue(momentOffset);

        // Step 5: 临时 newMomentum = current + meanSq（用于计算 lr）
        float newMomentum = currentMomentum + meanSq;
        float adaptiveLr = optimizerConfig.learningRate / (sqrt(newMomentum) + optimizerConfig.eps);

        // Step 6: 计算最终梯度更新
        Muls(outLt[gradOffset], inputLt[gradOffset], -adaptiveLr, embedDim);

        // Step 7: 输出 meanSq（供外部累加），不是 newMomentum
        outLt.SetValue(momentOffset, meanSq);
    }
};

} // namespace Optimizers
} // namespace BackwardCodegenUnweightedExact

#endif // ROWWISE_ADAGRAD_OPTIMIZER_H
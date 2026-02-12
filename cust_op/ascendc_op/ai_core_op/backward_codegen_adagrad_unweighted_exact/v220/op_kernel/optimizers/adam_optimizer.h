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

#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "optimizer_interface.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenUnweightedExact {
namespace Optimizers {

// Adam优化器
class AdamOptimizer : public OptimizerInterfaces::IOptimizer<AdamOptimizer> {
public:
    __aicore__ inline void Compute(LocalTensor<float>& inputLt, LocalTensor<float>& outLt, int64_t gradOffset,
                                   int64_t moment1Offset, int64_t moment2Offset, int64_t embedDim,
                                   OptimizerConfig& optimizerConfig)
    {
        float oneMinusBeta1 = (1 - optimizerConfig.beta1);
        float oneMinusBeta2 = (1 - optimizerConfig.beta2);
        float minusLearningRate = -optimizerConfig.learningRate;
        float stepSize = minusLearningRate * optimizerConfig.beta2sqrt;

        // v[:] = beta1 * v + (1 - beta1) * p.grad
        Muls<float>(outLt[moment1Offset], inputLt[moment1Offset], optimizerConfig.beta1, embedDim);
        Muls<float>(outLt[gradOffset], inputLt[gradOffset], oneMinusBeta1, embedDim);
        Add<float>(outLt[moment1Offset], outLt[moment1Offset], outLt[gradOffset], embedDim);

        // s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
        Muls<float>(outLt[moment2Offset], inputLt[moment2Offset], optimizerConfig.beta2, embedDim);
        Mul<float>(outLt[gradOffset], inputLt[gradOffset], inputLt[gradOffset], embedDim);
        Muls<float>(outLt[gradOffset], outLt[gradOffset], oneMinusBeta2, embedDim);
        Add<float>(outLt[moment2Offset], outLt[moment2Offset], outLt[gradOffset], embedDim);

        // p[:] -= stepSize * v / (torch.sqrt(s) + optimizerConfig.eps)
        Sqrt<float>(inputLt[moment2Offset], outLt[moment2Offset], embedDim);
        Adds<float>(inputLt[moment2Offset], inputLt[moment2Offset], optimizerConfig.eps, embedDim);
        Div<float>(outLt[gradOffset], outLt[moment1Offset], inputLt[moment2Offset], embedDim);
        Muls<float>(outLt[gradOffset], outLt[gradOffset], stepSize, embedDim);
    }
};

} // namespace Optimizers
} // namespace BackwardCodegenUnweightedExact

#endif // ADAM_OPTIMIZER_H
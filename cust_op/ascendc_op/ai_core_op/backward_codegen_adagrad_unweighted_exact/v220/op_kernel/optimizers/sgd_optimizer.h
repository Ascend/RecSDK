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

#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include "optimizer_interface.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenUnweightedExact {
namespace Optimizers {

// SGD优化器
class SgdOptimizer : public OptimizerInterfaces::IOptimizer<SgdOptimizer> {
public:
    __aicore__ inline void Compute(LocalTensor<float>& inputLt, LocalTensor<float>& outLt, int64_t gradOffset,
                                   int64_t embedDim, OptimizerConfig& optimizerConfig)
    {
        // SGD: Δθ = -η * grad
        Muls<float>(outLt[gradOffset], inputLt[gradOffset], -optimizerConfig.learningRate, embedDim);
    }
};

} // namespace Optimizers
} // namespace BackwardCodegenUnweightedExact

#endif // SGD_OPTIMIZER_H
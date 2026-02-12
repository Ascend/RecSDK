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

#ifndef OPTIMIZER_INTERFACE_H
#define OPTIMIZER_INTERFACE_H

#include "kernel_operator.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenUnweightedExact {
namespace OptimizerInterfaces {

// CRTP基类
template <typename T> class IOptimizer {
public:
    __aicore__ inline void Compute(LocalTensor<float>& newInputLt, LocalTensor<float>& outLt, int64_t gradOffset,
                                   int64_t embedDim, OptimizerConfig& optimizerConfig)
    {
        static_cast<T*>(this)->Compute(newInputLt, outLt, gradOffset, embedDim, optimizerConfig);
    }

    __aicore__ inline void Compute(LocalTensor<float>& newInputLt, LocalTensor<float>& outLt, int64_t gradOffset,
                                   int64_t moment1Offset, int64_t embedDim, OptimizerConfig& optimizerConfig)
    {
        static_cast<T*>(this)->Compute(newInputLt, outLt, gradOffset, moment1Offset, embedDim, optimizerConfig);
    }

    __aicore__ inline void Compute(LocalTensor<float>& newInputLt, LocalTensor<float>& outLt, int64_t gradOffset,
                                   int64_t moment1Offset, int64_t moment2Offset, int64_t embedDim,
                                   OptimizerConfig& optimizerConfig)
    {
        static_cast<T*>(this)->Compute(newInputLt, outLt, gradOffset, moment1Offset, moment2Offset, embedDim,
                                       optimizerConfig);
    }
};

} // namespace OptimizerInterfaces
} // namespace BackwardCodegenUnweightedExact

#endif // OPTIMIZER_INTERFACE_H
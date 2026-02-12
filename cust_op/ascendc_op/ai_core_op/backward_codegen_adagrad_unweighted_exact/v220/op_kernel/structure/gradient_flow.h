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

#ifndef GRADIENT_FLOW_H
#define GRADIENT_FLOW_H

#include "kernel_operator.h"
#include "pooling_mode_enum.h"

using namespace AscendC;

// 梯度流管理：GradientFlow
struct GradientFlow {
    GlobalTensor<float> inputGradGT;      // 上游梯度输入
    GlobalTensor<float> accmulatedGradGT; // 反向展开并累加后的梯度输出
    int64_t inputGradDim0;
    int64_t inputGradDim1;
    PoolingMode poolMode;

    __aicore__ inline PoolingMode GetPoolMode() const { return poolMode; }
    __aicore__ inline GlobalTensor<float>& GetInputTensor() { return inputGradGT; }

    __aicore__ inline GlobalTensor<float>& GetOutputTensor() { return accmulatedGradGT; }

    __aicore__ inline int64_t GetInputDim0() const { return inputGradDim0; }

    __aicore__ inline int64_t GetInputDim1() const { return inputGradDim1; }

    __aicore__ inline int64_t GetOutputSize() const { return accmulatedGradGT.GetSize(); }

    // 初始化方法
    __aicore__ inline void Init(Args& args, BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        inputGradGT.SetGlobalBuffer((__gm__ float*) args.gradOutput,
                                    tilingData.gradOutputDim0 * tilingData.gradOutputDim1);
        accmulatedGradGT.SetGlobalBuffer((__gm__ float*) args.out, tilingData.outDim0);
        inputGradDim0 = tilingData.gradOutputDim0;
        inputGradDim1 = tilingData.gradOutputDim1;
        this->poolMode = static_cast<PoolingMode>(tilingData.poolMode);
    }
};

#endif // GRADIENT_FLOW_H
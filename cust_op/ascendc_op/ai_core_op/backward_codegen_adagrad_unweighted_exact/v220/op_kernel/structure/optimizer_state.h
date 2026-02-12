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

#ifndef OPTIMIZER_STATE_H
#define OPTIMIZER_STATE_H

#include "kernel_operator.h"
#include "../structure/args_struct.h"

using namespace AscendC;

// 优化器状态（in-place 语义显式表达）
struct OptimizerState {
    // 统一的输出索引枚举类
    enum class Index : int64_t { GRAD_IDX = 0, MOMENTUM1_IDX = 1, MOMENTUM2_IDX = 2 };

    GlobalTensor<float> weightsDevOutGT;

    GlobalTensor<float> momentum1InGT;
    GlobalTensor<float> momentum1OutGT; // = momentum1In（in-place */

    GlobalTensor<float> momentum2InGT;
    GlobalTensor<float> momentum2OutGT; // = momentum2In（in-place */

    __aicore__ inline void Init(Args& args, BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        weightsDevOutGT.SetGlobalBuffer((__gm__ float*) args.weightsDevOut, tilingData.outDim0);
        // Host 保证 mom1In == mom1Out（in-place）
        momentum1InGT.SetGlobalBuffer((__gm__ float*) args.momentum1Dev, tilingData.momentumDim0);
        momentum1OutGT.SetGlobalBuffer((__gm__ float*) args.momentum1DevOut, tilingData.momentumDim0);

        momentum2InGT.SetGlobalBuffer((__gm__ float*) args.momentum2Dev, tilingData.momentumDim0);
        momentum2OutGT.SetGlobalBuffer((__gm__ float*) args.momentum2DevOut, tilingData.momentumDim0);
    }

    // Get方法，用于访问内部的GlobalTensor变量
    __aicore__ inline GlobalTensor<float>& GetMomentum1InGT() { return momentum1InGT; }

    __aicore__ inline GlobalTensor<float>& GetMomentum1OutGT() { return momentum1OutGT; }

    __aicore__ inline GlobalTensor<float>& GetMomentum2InGT() { return momentum2InGT; }

    __aicore__ inline GlobalTensor<float>& GetMomentum2OutGT() { return momentum2OutGT; }

    __aicore__ inline GlobalTensor<float>& GetWeightsDevOutGT() { return weightsDevOutGT; }
};

#endif // OPTIMIZER_STATE_H
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

#ifndef OPTIMIZER_CONFIG_H
#define OPTIMIZER_CONFIG_H

#include "kernel_operator.h"
#include "../structure/args_struct.h"
#include "embedding_table_layout.h"

using namespace AscendC;

// 优化器配置（超参 + 控制信号）
struct OptimizerConfig {
    float learningRate;
    float eps;
    float beta1;
    float beta2;
    float beta2sqrt;

    // 初始化方法
    __aicore__ inline void Init(Args& args, BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        learningRate = tilingData.learningRate;
        eps = tilingData.eps;
        beta1 = tilingData.beta1;
        beta2 = tilingData.beta2;
        beta2sqrt = tilingData.beta2sqrt;
    }
};

#endif // OPTIMIZER_CONFIG_H
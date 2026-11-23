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

#ifndef TILING_PARAMS_H
#define TILING_PARAMS_H

#include "kernel_operator.h"
using namespace AscendC;

// 为当前 AI Core Block 分配数据分片范围（偏移 + 长度）
struct WorkloadSharder {
    int64_t length; // 当前 Block 处理的数据长度
    int64_t start;  // 当前 Block 数据起始偏移
    int64_t end;    // 当前 Block 数据结束偏移

    __aicore__ inline void Compute(int64_t totalLen)
    {
        int64_t base = totalLen / GetBlockNum();
        int64_t tail = totalLen % GetBlockNum();
        if (GetBlockIdx() < tail) {
            length = base + 1;
            start = GetBlockIdx() * length;
        } else {
            length = base;
            start = tail * (base + 1) + (GetBlockIdx() - tail) * base;
        }
        end = start + length;
    }
};

#endif // TILING_PARAMS_H
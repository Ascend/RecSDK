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
#ifndef SPLIT_CORE_H
#define SPLIT_CORE_H

#include <unistd.h>
#include <cstdint>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

// 分核
// mix_uvqk [seq_len, H]
// weight [dim, H]
// x [seq_len, dim]
// 分块 (seq_len / blockM) * (H / blockK) * (dim / blockK)
namespace InLinearSiluBackward {
class BlockTaskAssign {
public:
    __aicore__ inline BlockTaskAssign(uint32_t coreNum,
                                     int64_t blockM,
                                     int64_t blockK,
                                     int64_t seqLen,
                                     int64_t hiddenSize,
                                     int64_t dim)
    {
        this->coreNum = coreNum;
        this->blockM = blockM;
        this->blockK = blockK;
        this->seqLen = seqLen;
        this->hiddenSize = hiddenSize;
        this->dim = dim;
    }
    
    __aicore__ inline void SplitCoreFast(int (&result)[2], int coreId) // 按行分
    {
        int64_t totalTaskNum = (this->seqLen + this->blockM - 1) / this->blockM;
        uint32_t usedCoreNum = (this->coreNum > totalTaskNum) ? totalTaskNum : this->coreNum;
        uint32_t splitNextCoreProcNum = totalTaskNum / usedCoreNum;
        uint32_t splitPrevCoreProcNum = splitNextCoreProcNum + 1;
        uint32_t splitCoreIdx = totalTaskNum % usedCoreNum;
        if (coreId < splitCoreIdx) {
            result[0] = splitPrevCoreProcNum * coreId;
            result[1] = result[0] + splitPrevCoreProcNum;
        } else if (coreId < usedCoreNum) {
            result[0] = splitPrevCoreProcNum * splitCoreIdx +
                      (coreId - splitCoreIdx) * splitNextCoreProcNum;
            result[1] = result[0] + splitNextCoreProcNum;
        } else {
            result[0] = 0;
            result[1] = 0;
        }
    }
private:
    uint32_t coreNum;
    int64_t blockM;
    int64_t blockK;
    int64_t seqLen;
    int64_t hiddenSize;
    int64_t dim;
};
}
#endif
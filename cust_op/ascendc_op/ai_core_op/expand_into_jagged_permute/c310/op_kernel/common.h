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

#ifndef EXPAND_INTO_JAGGED_PERMUTE_COMMON_H
#define EXPAND_INTO_JAGGED_PERMUTE_COMMON_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

// 常量定义
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int SIMT_LAUNCH_BOUND = 32 * 8;
constexpr int THRESHOLD = 512;
constexpr int DIM = 256;

// Args结构体
struct Args {
    GM_ADDR permute;
    GM_ADDR input_offsets;
    GM_ADDR output_offsets;
    GM_ADDR output_permute;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

// ========== 数据拷贝函数 ==========

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    DataCopy(gt, lt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
    }
}

#endif // EXPAND_INTO_JAGGED_PERMUTE_COMMON_H

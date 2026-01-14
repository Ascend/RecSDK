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

#ifndef TRANS_TEST_H
#define TRANS_TEST_H

#include <cstdint>
#include "trans.h"
#include "kernel_operator.h"

constexpr int seqLen = 64;
constexpr int64_t headNum = 16;
constexpr uint32_t headDimPadding = 16;
constexpr int64_t headDim = 16;
constexpr int64_t blockHeightQ = 256;
constexpr int64_t blockHeightK = 256;
constexpr int64_t ubSize = 160 * 1024;
constexpr float valueScale = 1.0f;
constexpr float loss = 0.0001f;

using namespace AscendC;
using namespace HstuDenseBackward;

/**
 * @brief 初始化用于测试Matmul的矩阵输入。
 *
 * 生成【seqLen, headNum, headDim】的矩阵，并复制到workspaceGt中。矩阵的值为HeadId
 */
template <typename T>
__aicore__ inline void InitWorkSpace(LocalTensor<T>& copyInLt, GlobalTensor<T>& workspaceGt, int blockQ, int headNum,
                                     int headDim, float scale = 1.0f)
{
    int oneHeadNumel = blockQ * headDim;
    for (int i = 0; i < headNum; i++) {
        Duplicate<T>(copyInLt, (i + 1) * scale, oneHeadNumel);
        PipeBarrier<PIPE_ALL>();
        DataCopy(workspaceGt[oneHeadNumel * i], copyInLt, oneHeadNumel);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline T GetValueFromQueue(LocalTensor<T>& copyInLt, const GlobalTensor<T>& workspaceGt, int offset)
{
    PipeBarrier<PIPE_ALL>();
    DataCopy(copyInLt, workspaceGt[offset], 32);
    PipeBarrier<PIPE_ALL>();
    float value = copyInLt.GetValue(0);
    return value;
}

template <typename T>
__aicore__ inline T AbsScalar(T a, T b)
{
    if (a > b) {
        return a - b;
    } else {
        return b - a;
    }
}

__aicore__ inline void TransTest(GM_ADDR baseAddr)
{
    if (g_coreType == AIC) {
        return;
    }
    if (GetBlockIdx() != 0) {
        return;
    }
    GM_ADDR qGradTempAddr = baseAddr;
    GM_ADDR qGradAddr = qGradTempAddr + headNum * headDim * blockHeightQ * sizeof(half);
    int64_t initSize = headNum * headDim * blockHeightQ;

    GlobalTensor<half> qGradTempGt;
    GlobalTensor<half> qGradGt;
    qGradTempGt.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qGradTempAddr));
    qGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qGradAddr));

    TPipe pipe;
    TransStrideHdDKernel<half, half> transStrideHdDKernel;
    transStrideHdDKernel.Init(&pipe, headNum, headDim);

    LocalTensor<half> tempLt = {TPosition::VECIN, 0, blockHeightQ};
    InitWorkSpace<half>(tempLt, qGradTempGt, blockHeightQ, headNum, headDim, valueScale);
    AscendC::InitGlobalMemory(qGradGt, initSize, (half)0.0f);
    PipeBarrier<PIPE_ALL>();

    transStrideHdDKernel.DoTransOfStrideHeadDim(qGradTempGt, qGradGt, blockHeightQ * headDim, headDim,
                                                blockHeightQ * headDim);
    float result00 = GetValueFromQueue<half>(tempLt, qGradGt, headDim - 1);
    float result0headDim = GetValueFromQueue<half>(tempLt, qGradGt, headDim);

    int64_t endOffset = (blockHeightQ - 1) * headDim * headNum + headDim;
    float resultE0 = GetValueFromQueue<half>(tempLt, qGradGt, endOffset - 1);
    float resultEheadDim = GetValueFromQueue<half>(tempLt, qGradGt, endOffset);
    PRINTF("result00: %f, result0headDim: %f, resultE0: %f, resultEheadDim: %f\n", result00, result0headDim, resultE0,
           resultEheadDim);
    ASSERT(AbsScalar(result00, 0.0f) < loss, "Result00 ERROR \n");
    ASSERT(AbsScalar(result0headDim, 2.0f) < loss, "Result0headDim ERROR \n");
    ASSERT(AbsScalar(resultE0, 0.0f) < loss, "ResultE0 ERROR \n");
    ASSERT(AbsScalar(resultEheadDim, 2.0f) < loss, "ResultEheadDim ERROR \n");
}
#endif

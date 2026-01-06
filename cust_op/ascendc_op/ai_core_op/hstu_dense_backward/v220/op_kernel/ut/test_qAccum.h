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
#include "qAccum.h"
#include "kernel_operator.h"

constexpr int seqLen = 64;
constexpr int64_t headNum = 16;
constexpr int64_t headDim = 16;
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
        Duplicate<T>(copyInLt, (i + 1) / scale, oneHeadNumel);
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

__aicore__ inline void QAccumTest(GM_ADDR baseAddr)
{
    if (g_coreType == AIC) {
        return;
    }
    if (GetBlockIdx() != 0) {
        return;
    }
    GM_ADDR qGradTempAddr = baseAddr;
    GM_ADDR qGradAddr = qGradTempAddr + headNum * headDim * seqLen * sizeof(half);
    GM_ADDR seqOffsetsAddr = qGradAddr + headNum * headDim * seqLen * sizeof(half);
    int64_t initSize = headNum * headDim * seqLen;

    GlobalTensor<half> qGradTempGt;
    GlobalTensor<half> qGradGt;
    GlobalTensor<int32_t> seqOffsetsGt;
    qGradTempGt.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qGradTempAddr));
    qGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qGradAddr));
    seqOffsetsGt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqOffsetsAddr));

    TPipe pipe;
    qBlockAcuumKernel<half, half, int32_t> qBlockAcuumKernel;
    BaseShapeArgs baseShapeArgs = {1, headNum, headDim, seqLen};
    qBlockAcuumKernel.Init(&pipe, &baseShapeArgs, qGradTempGt, qGradGt, seqOffsetsGt, 1);

    LocalTensor<half> tempLt = {TPosition::VECIN, 0, ubSize};
    InitWorkSpace<half>(tempLt, qGradTempGt, seqLen, headNum, headDim, valueScale);
    AscendC::InitGlobalMemory(qGradGt, initSize, (half)0.0f);
    seqOffsetsGt.SetValue(0, 0);
    seqOffsetsGt.SetValue(1, seqLen);
    PipeBarrier<PIPE_ALL>();

    qBlockAcuumKernel.DoCopyQGrad();
    float result00 = GetValueFromQueue<half>(tempLt, qGradGt, headDim - 1);
    float result0headDim = GetValueFromQueue<half>(tempLt, qGradGt, headDim);

    int64_t endOffset = (seqLen - 1) * headDim * headNum + headDim;
    float resultE0 = GetValueFromQueue<half>(tempLt, qGradGt, endOffset - 1);
    float resultEheadDim = GetValueFromQueue<half>(tempLt, qGradGt, endOffset);
    printf("result00: %f, result0headDim: %f, resultE0: %f, resultEheadDim: %f\n", result00, result0headDim, resultE0,
           resultEheadDim);
    assert(AbsScalar(result00, 1.0f) < loss, "Result00 ERROR \n");
    assert(AbsScalar(result0headDim, 2.0f) < loss, "Result0headDim ERROR \n");
    assert(AbsScalar(resultE0, 1.0f) < loss, "ResultE0 ERROR \n");
    assert(AbsScalar(resultEheadDim, 2.0f) < loss, "ResultEheadDim ERROR \n");
}
#endif

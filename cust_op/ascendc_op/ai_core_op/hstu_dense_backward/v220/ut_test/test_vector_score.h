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

#ifndef VECTOR_SCORE_TEST_H
#define VECTOR_SCORE_TEST_H

#include <cstdint>
#include "vector_score.h"
#include "kernel_operator.h"

constexpr int seqLen = 64;
constexpr int64_t headNum = 16;
constexpr int64_t headDim = 16;
constexpr int64_t blockHeightQ = 256;
constexpr int64_t blockHeightK = 256;
constexpr int64_t ubSize = 160 * 1024;
constexpr float valueScale = 0.1f;
constexpr float loss = 0.0001f;

using namespace AscendC;
using namespace HstuDenseBackward;

/**
 * @brief 初始化用于测试Matmul的矩阵输入。
 *
 * 生成【seqLen, headNum, headDim】的矩阵，并复制到workspaceGt中。矩阵的值为HeadId
 */
template <typename T>
__aicore__ inline void InitWorkSpace(LocalTensor<T>& copyInLt, GlobalTensor<T>& workspaceGt, int blockQ, int blockK,
                                     float scale = 1.0f)
{
    for (int i = 0; i < blockQ; i++) {
        Duplicate<T>(copyInLt, (i + 1) * scale, blockK);
        PipeBarrier<PIPE_ALL>();
        DataCopy(workspaceGt[blockQ * i], copyInLt, blockK);
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

__aicore__ inline void VectorScoreTest(GM_ADDR baseAddr)
{
    if (g_coreType == AIC) {
        return;
    }
    if (GetBlockIdx() != 0) {
        return;
    }
    int64_t computePipeNum = 3;
    GM_ADDR qkTempAddr = baseAddr;
    GM_ADDR gvTempAddr = qkTempAddr + blockHeightQ * blockHeightK * computePipeNum * sizeof(half);
    GM_ADDR maskTempAddr = gvTempAddr + blockHeightQ * blockHeightK * computePipeNum * sizeof(half);
    GM_ADDR biasTempAddr = biasTempAddr + blockHeightQ * blockHeightK * computePipeNum * sizeof(half);

    BlockMaskParams blockMaskParam = {0, 0, seqLen, blockHeightQ, 0, 0, 0, 1.0f};
    TPipe pipe;
    LocalTensor<half> tempLt = {TPosition::VECIN, 0, blockHeightK};
    VectorScoreAttrs attrs = {1.0f, 1.0f};
    VectorScoreGtInfo gtInfo = {qkTempAddr, gvTempAddr, maskTempAddr, biasTempAddr};
    HstuM0R0VectorScore<half, blockHeightQ, blockHeightK, headDim> vectorScore;
    vectorScore.Init(&pipe, &attrs, &gtInfo);

    InitWorkSpace<half>(tempLt, vectorScore.qkTemp_, blockHeightQ, blockHeightK, valueScale);
    InitWorkSpace<half>(tempLt, vectorScore.gvTemp_, blockHeightQ, blockHeightK, valueScale);
    vectorScore.VecScoreJagged(0, 0, 0, 0, 0, seqLen, headNum, blockMaskParam);
    float qkResult00 = GetValueFromQueue<half>(tempLt, vectorScore.qkTemp_, 0);
    float gvResult00 = GetValueFromQueue<half>(tempLt, vectorScore.gvTemp_, 0);
    float qkResult01 = GetValueFromQueue<half>(tempLt, vectorScore.qkTemp_, 255);
    float gvResult01 = GetValueFromQueue<half>(tempLt, vectorScore.gvTemp_, 255);

    float qkGolden00 = 0.05249f;
    float gvGolden00 = 0.05497f;
    float qkGolden01 = 0.0f;
    float gvGolden01 = 0.0f;

    printf("qkResult00: %f, gvResult00: %f, qkGolden00: %f, gvGolden00: %f\n", qkResult00, gvResult00, qkGolden00,
           gvGolden00);
    printf("qkResult01: %f, gvResult01: %f, qkGolden01: %f, gvGolden01: %f\n", qkResult01, gvResult01, qkGolden01,
           gvGolden01);
    ASSERT(AbsScalar(qkResult00, qkGolden00) < loss, "QkResult00 ERROR \n");
    ASSERT(AbsScalar(gvResult00, gvGolden00) < loss, "GvResult00 ERROR \n");
    ASSERT(AbsScalar(qkResult01, qkGolden01) < loss, "QkResult01 ERROR \n");
    ASSERT(AbsScalar(gvResult01, gvGolden01) < loss, "GvResult01 ERROR \n");
}
#endif

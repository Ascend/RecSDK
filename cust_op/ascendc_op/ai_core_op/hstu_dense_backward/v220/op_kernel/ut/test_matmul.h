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

#ifndef MATMUL_TEST_H
#define MATMUL_TEST_H

#include <cstdint>
#include "matmul_j_f16_r0_const.h"
#include "matmul_mgmt_j_f16_r0.h"
#include "kernel_operator.h"

constexpr int seqLen = 64;
constexpr int64_t headNum = 16;
constexpr int64_t headDim = 16;
constexpr int64_t blockHeightQ = 256;
constexpr int64_t blockHeightK = 256;
constexpr int64_t ubSize = 160 * 1024;
constexpr float valueScale = 100.0f;
constexpr float loss = 0.00001f;
using namespace AscendC;
using namespace HstuDenseBackward;

/**
 * @brief 初始化用于测试Matmul的矩阵输入。
 *
 * 生成【seqLen, headNum, headDim】的矩阵，并复制到workspaceGt中。矩阵的值为HeadId
 */
__aicore__ inline void InitWorkSpace(TQue<TPosition::VECIN, 1>& queue, GlobalTensor<half>& workspaceGt, int seqLen,
                                     int headNum, int headDim, float scale = 1.0f)
{
    LocalTensor<half> copyInLt = queue.AllocTensor<half>();
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < headNum; j++) {
            int64_t offset = i * headNum * headDim + j * headDim;
            Duplicate<half>(copyInLt[offset], (j + 1) / scale, headDim);
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(workspaceGt, copyInLt, seqLen * headNum * headDim);
    PipeBarrier<PIPE_ALL>();
    queue.FreeTensor(copyInLt);
}

template <typename T>
__aicore__ inline T GetValueFromQueue(TQue<TPosition::VECIN, 1>& queue, const GlobalTensor<T>& workspaceGt, int offset)
{
    LocalTensor<T> copyInLt = queue.AllocTensor<T>();

    PipeBarrier<PIPE_ALL>();
    DataCopy(copyInLt, workspaceGt[offset], 32);
    PipeBarrier<PIPE_ALL>();
    float value = copyInLt.GetValue(0);
    queue.FreeTensor(copyInLt);
    return value;
}

template <typename T>
__aicore__ inline void CopyGM2GM(TQue<TPosition::VECIN, 1>& queue, const GlobalTensor<T>& fromGt,
                                 const GlobalTensor<T>& toGt, int size)
{
    LocalTensor<T> copyInLt = queue.AllocTensor<T>();

    PipeBarrier<PIPE_ALL>();
    DataCopy(copyInLt, fromGt, size);
    PipeBarrier<PIPE_ALL>();
    DataCopy(toGt, copyInLt, size);
    PipeBarrier<PIPE_ALL>();
    queue.FreeTensor(copyInLt);
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

struct TestTilingData {
    uint16_t headNum;
    uint16_t headDim;
};

__aicore__ inline void MmMgmtTest(GM_ADDR baseAddr)
{
    GM_ADDR tilingAddr = baseAddr;
    GM_ADDR gradAddr = baseAddr + sizeof(TestTilingData)*headNum*headDim;
    GM_ADDR qAddr = gradAddr + seqLen * headNum * headDim * sizeof(half);
    GM_ADDR kAddr = qAddr + seqLen * headNum * headDim * sizeof(half);
    GM_ADDR vAddr = kAddr + seqLen * headNum * headDim * sizeof(half);
    GM_ADDR qGradAddr = vAddr + seqLen * headNum * headDim * sizeof(half);
    GM_ADDR kGradAddr = qGradAddr + seqLen * headNum * headDim * sizeof(half);
    GM_ADDR vGradAddr = kGradAddr + seqLen * headNum * headDim * sizeof(half);
    GM_ADDR workspaceAddr = vGradAddr + seqLen * headNum * headDim * sizeof(half);

    int64_t blockSize = blockHeightQ * blockHeightK;
    int64_t tensorSize = seqLen * headNum * headDim;

    TPipe tPipe;
    TQue<TPosition::VECIN, 1> queueTemp;
    tPipe.InitBuffer(queueTemp, 1, ubSize);

    const AddrArgs addrArgs = {gradAddr, qAddr, kAddr, vAddr, qGradAddr, kGradAddr, vGradAddr, workspaceAddr};
    const BaseShapeArgs baseShape = {1, headNum, headDim, seqLen};
    MmMgmtFp16R0Jagged<half, blockHeightQ, blockHeightK, headDim, TestTilingData> mgmt;
    mgmt.Init(&addrArgs, &baseShape);

    InitWorkSpace(queueTemp, mgmt.q_, seqLen, headNum, headDim, valueScale);
    InitWorkSpace(queueTemp, mgmt.k_, seqLen, headNum, headDim, valueScale);
    InitWorkSpace(queueTemp, mgmt.v_, seqLen, headNum, headDim, valueScale);
    InitWorkSpace(queueTemp, mgmt.qGrad_, seqLen, headNum, headDim, valueScale);
    InitWorkSpace(queueTemp, mgmt.kGrad_, seqLen, headNum, headDim, valueScale);
    InitWorkSpace(queueTemp, mgmt.vGrad_, seqLen, headNum, headDim, valueScale);
    InitWorkSpace(queueTemp, mgmt.grad_, seqLen, headNum, headDim, valueScale);
    SyncAll();
    __gm__ TestTilingData* tilingData = reinterpret_cast<__gm__ TestTilingData*>(tilingAddr);
    tilingData->headNum = headNum;
    tilingData->headDim = headDim;

    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mgmt.qkOrGvMatmul_, (TCubeTiling*)nullptr, mgmt.vGradMatmul_,
                      (TCubeTiling*)nullptr, mgmt.qGradMatmul_, (TCubeTiling*)nullptr, mgmt.kGradMatmul_,
                      (TCubeTiling*)nullptr);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(tilingAddr);
    mgmt.qkOrGvMatmul_.SetUserDefInfo(tilingPtr);
    mgmt.vGradMatmul_.SetUserDefInfo(tilingPtr);
    mgmt.qGradMatmul_.SetUserDefInfo(tilingPtr);
    mgmt.kGradMatmul_.SetUserDefInfo(tilingPtr);

    if (GetBlockIdx() != 0) {
        return;
    }
    MatmulArgs matmulArgs = {0, 0, 0, 0, seqLen, seqLen};
    MatmulArgs matmulArgs1 = {1, 1, headDim, headDim, seqLen, seqLen};
    mgmt.DoQkMatmul(0, 0, 0, seqLen, seqLen);
    mgmt.DoGvMatmul(0, 0, 0, seqLen, seqLen);
    mgmt.DoQkMatmul(blockSize, headDim, headDim, seqLen, seqLen);
    mgmt.DoGvMatmul(blockSize, headDim, headDim, seqLen, seqLen);

    mgmt.QkOrGvMatmulWait();
    mgmt.QkOrGvMatmulWait();

    float result = GetValueFromQueue<half>(queueTemp, mgmt.qkTemp_, 0);
    float golden = 0.0016f;
    PRINTF("QkMatmulTest result 0: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "QkMatmulTest result 0 ERROR \n");

    result = GetValueFromQueue<half>(queueTemp, mgmt.gvTemp_, 0);
    golden = 0.0016f;
    PRINTF("GvMatmulTest result 0: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "GvMatmulTest result 0 ERROR \n");

    mgmt.DoKGradMatmul(0, 0, 0, seqLen, seqLen, true);
    mgmt.DoQGradMatmul(0, 0, 0, seqLen, seqLen);
    mgmt.DoVGradMatmul(0, 0, 0, seqLen, seqLen, true);

    mgmt.kGradMatmulWait();
    mgmt.qGradMatmulWait();
    mgmt.vGradMatmulWait();

    int64_t scoreTempOffset = blockHeightQ * blockHeightK;
    int64_t qGradTempOffset = blockHeightQ * headDim;
    int64_t kGradTempOffset = blockHeightK * headDim;
    int64_t vGradTempOffset = blockHeightK * headDim;

    result = GetValueFromQueue<float>(queueTemp, mgmt.kGradAccumTemp_, 0);
    golden = 0.001024f;
    PRINTF("KGradMatmulTest result 0: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "KGradMatmulTest result ERROR \n");

    result = GetValueFromQueue<float>(queueTemp, mgmt.qGradAccumTemp_, 0);
    golden = 0.001024f;
    PRINTF("QGradMatmulTest result 0: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "QGradMatmulTest result ERROR \n");

    result = GetValueFromQueue<float>(queueTemp, mgmt.vGradAccumTemp_, 0);
    golden = 0.001024f;
    PRINTF("VGradMatmulTest result 0: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "VGradMatmulTest result ERROR \n");

    mgmt.QkOrGvMatmulWait();
    mgmt.QkOrGvMatmulWait();

    result = GetValueFromQueue<half>(queueTemp, mgmt.qkTemp_[scoreTempOffset], 0);
    golden = 0.0064f;
    PRINTF("QkMatmulTest result 1: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "QkMatmulTest result 0 ERROR \n");

    result = GetValueFromQueue<half>(queueTemp, mgmt.gvTemp_[scoreTempOffset], 0);
    golden = 0.0064f;
    PRINTF("GvMatmulTest result 1: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "GvMatmulTest result 0 ERROR \n");

    mgmt.DoKGradMatmul(kGradTempOffset, blockSize, headDim, seqLen, seqLen, false);
    mgmt.DoQGradMatmul(qGradTempOffset, blockSize, headDim, seqLen, seqLen);
    mgmt.DoVGradMatmul(vGradTempOffset, blockSize, headDim, seqLen, seqLen, false);

    mgmt.kGradMatmulWait();
    mgmt.qGradMatmulWait();
    mgmt.vGradMatmulWait();

    result = GetValueFromQueue<float>(queueTemp, mgmt.kGradAccumTemp_[kGradTempOffset], 0);
    golden = 0.0081920f;
    PRINTF("KGradMatmulTest result 1: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "KGradMatmulTest result ERROR \n");

    result = GetValueFromQueue<float>(queueTemp, mgmt.qGradAccumTemp_[qGradTempOffset], 0);
    golden = 0.0081920f;
    PRINTF("QGradMatmulTest result 1: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "QGradMatmulTest result ERROR \n");

    result = GetValueFromQueue<float>(queueTemp, mgmt.vGradAccumTemp_[vGradTempOffset], 0);
    golden = 0.0081920f;
    PRINTF("VGradMatmulTest result 1: %f, golden: %f\n", result, golden);
    assert(AbsScalar(result, golden) < loss, "VGradMatmulTest result ERROR \n");
}
#endif

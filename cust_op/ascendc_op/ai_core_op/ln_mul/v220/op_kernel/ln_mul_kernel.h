/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#ifndef MXREC_LAYERNORM_MUL_H
#define MXREC_LAYERNORM_MUL_H

#include "kernel_operator.h"

namespace LnMulKernel {
constexpr uint32_t DATA_ALIGN_BYTES = 32;
constexpr uint32_t DATA_COPY_ALIGN_BYTES = 16;
struct Args {
    GM_ADDR inputXGm;
    GM_ADDR inputUGm;
    GM_ADDR gammGm;
    GM_ADDR betaGm;
    GM_ADDR outputGm;

    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <bool isReuseSource = false>
class KernelLayernorm {
public:
    __aicore__ inline KernelLayernorm() {}
    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        inputXGm = args.inputXGm;
        inputUGm = args.inputUGm;
        gammGm = args.gammGm;
        betaGm = args.betaGm;
        outputGm = args.outputGm;

        this->epsilon = tilingData.epsilon;
        this->aLength = tilingData.aLength;
        this->rLength = tilingData.rLength;
        this->rLengthWithPadding = tilingData.rLengthWithPadding;
        this->coreNum = tilingData.coreNum;
        this->perCoreComputeRows = tilingData.perCoreComputeRows;

        // 当前核的index
        int64_t blockIdx = AscendC::GetBlockIdx();
        // 判断当前核是前核还是尾核
        bool isTailCore = (blockIdx >= tilingData.formerCoreNums);

        // 获取当前核的行数和循环次数
        this->currentCoreRows = isTailCore ? tilingData.baseCoreRows : tilingData.formerCoreRows;
        this->currentLoopCount = isTailCore ? tilingData.loopCountTail : tilingData.loopCountFormer;
        this->currentRowLeft = isTailCore ? tilingData.tailRowLeft : tilingData.formerRowLeft;

        // 当前核的起始处理行号
        this->startRow = isTailCore ? tilingData.formerCoreNums * tilingData.formerCoreRows +
                                          (blockIdx - tilingData.formerCoreNums) * tilingData.baseCoreRows
                                    : blockIdx * tilingData.formerCoreRows;

        this->arLength = perCoreComputeRows * rLengthWithPadding;

        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputXGm), aLength * rLength);
        inputUGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputUGm), aLength * rLength);
        gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gammGm), rLength);
        betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(betaGm), rLength);
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputGm), aLength * rLength);

        pipe.InitBuffer(inQueueX, 1, sizeof(float) * arLength);
        pipe.InitBuffer(inQueueU, 1, sizeof(float) * arLength);
        pipe.InitBuffer(inQueueGamma, 1, sizeof(float) * rLengthWithPadding);
        pipe.InitBuffer(inQueueBeta, 1, sizeof(float) * rLengthWithPadding);
        pipe.InitBuffer(outQueue, 1, sizeof(float) * arLength);
        pipe.InitBuffer(meanQueue, 1, sizeof(float) * perCoreComputeRows);
        pipe.InitBuffer(rstdQueue, 1, sizeof(float) * perCoreComputeRows);
        pipe.InitBuffer(tmpQueue, 1, sizeof(float) * arLength);
        pipe.InitBuffer(oneQueue, 1, sizeof(float) * perCoreComputeRows);
    }
    __aicore__ inline void Process()
    {
        if (currentCoreRows == 0) {
            return;
        }
        for (uint32_t i = 0; i < currentLoopCount; ++i) {
            CopyIn(i, perCoreComputeRows);
            Compute(perCoreComputeRows);
            CopyOut(i, perCoreComputeRows);
        }
        // 处理剩余行
        if (currentRowLeft > 0) {
            CopyIn(currentLoopCount, currentRowLeft);
            Compute(currentRowLeft);
            CopyOut(currentLoopCount, currentRowLeft);
        }
    }
    template <typename T>
    __aicore__ inline void CpGm2Local(const AscendC::LocalTensor<T>& lt, const AscendC::GlobalTensor<T>& gt,
                                      int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        AscendC::GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
        AscendC::LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        if (alignLen != 0) {
            DataCopy(uint16Lt, uint16Gt, alignLen / 2);
        }

        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
            DataCopyPadGm2Local(uint16Lt[alignLen / 2], uint16Gt[alignLen / 2], unAlignLen / 2);
#else
            const AscendC::DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const AscendC::DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(uint16Lt[alignLen / 2], uint16Gt[alignLen / 2], dataCopyExtParams, dataCopyPadExtParams);
#endif
        }
    }

    __aicore__ inline void DataCopyPadGm2Local(const AscendC::LocalTensor<uint16_t>& lt,
                                               const AscendC::GlobalTensor<uint16_t>& gt, int64_t len)
    {
        AscendC::DataCopy<uint16_t>(lt, gt, DATA_COPY_ALIGN_BYTES);
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        AscendC::Duplicate<uint16_t>(lt, 0, mask, 1, 1, 1);
    }

    template <typename T>
    __aicore__ inline void CpLocal2Gm(const AscendC::GlobalTensor<T>& gt, const AscendC::LocalTensor<T>& lt,
                                      int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        AscendC::GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
        AscendC::LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        if (alignLen != 0) {
            DataCopy(uint16Gt, uint16Lt, alignLen / 2);
        }
        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
            DataCopyPadLocal2Gm(uint16Gt[alignLen / 2], uint16Lt[alignLen / 2], unAlignLen / 2);
#else
            const AscendC::DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const AscendC::DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(uint16Gt[alignLen / 2], uint16Lt[alignLen / 2], dataCopyExtParams);
#endif
        }
    }

    __aicore__ inline void DataCopyPadLocal2Gm(const AscendC::GlobalTensor<uint16_t>& gt,
                                               const AscendC::LocalTensor<uint16_t>& lt, int64_t len)
    {
        AscendC::SetAtomicAdd<uint16_t>();
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        AscendC::Duplicate<uint16_t>(lt, 0, mask, 1, 1, 1);
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(gt, lt, DATA_COPY_ALIGN_BYTES);
        AscendC::SetAtomicNone();
    }

private:
    __aicore__ inline void CopyIn(uint32_t iterIdx, uint32_t rowCount)
    {
        // 核内偏移
        uint32_t offsetInCore = iterIdx * rowCount * rLength;
        uint32_t copyLen = rowCount * rLength;

        AscendC::LocalTensor<float> inputXLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> inputULocal = inQueueU.AllocTensor<float>();
        AscendC::LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
        AscendC::LocalTensor<float> betaLocal = inQueueBeta.AllocTensor<float>();

        if (rLength == rLengthWithPadding) {
            CpGm2Local(inputXLocal, inputXGlobal[startRow * rLength + iterIdx * perCoreComputeRows * rLength], copyLen);
            CpGm2Local(inputULocal, inputUGlobal[startRow * rLength + iterIdx * perCoreComputeRows * rLength], copyLen);
        } else {
            for (int i = 0; i < rowCount; ++i) {
                CpGm2Local(inputXLocal[i * rLengthWithPadding],
                           inputXGlobal[(startRow + iterIdx * perCoreComputeRows + i) * rLength], rLength);
                CpGm2Local(inputULocal[i * rLengthWithPadding],
                           inputUGlobal[(startRow + iterIdx * perCoreComputeRows + i) * rLength], rLength);
            }
        }
        CpGm2Local(gammaLocal, gammGlobal, rLength);
        CpGm2Local(betaLocal, betaGlobal, rLength);

        inQueueX.EnQue(inputXLocal);
        inQueueU.EnQue(inputULocal);
        inQueueGamma.EnQue(gammaLocal);
        inQueueBeta.EnQue(betaLocal);
    }
    __aicore__ inline void Compute(uint32_t rowCount)
    {
        AscendC::LocalTensor<float> inputXLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> inputULocal = inQueueU.DeQue<float>();
        AscendC::LocalTensor<float> gammaLocal = inQueueGamma.DeQue<float>();
        AscendC::LocalTensor<float> betaLocal = inQueueBeta.DeQue<float>();

        AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> meanLocal = meanQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> rstdLocal = rstdQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> oneLocal = oneQueue.AllocTensor<float>();
        AscendC::Duplicate(oneLocal, 1.0f, rowCount);

        // shape
        uint32_t meanShape[2] = {rowCount, 1};
        uint32_t shape[2] = {rowCount, rLength};
        // 计算reducesum, isReuseSource=false不能设置为true，否则会修改inputXLocal
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(meanLocal, inputXLocal, shape, false);
        // 计算平均值mean
        AscendC::Muls(meanLocal, meanLocal, 1.0f / rLength, rowCount);
        // broadcast mean [rowCount] -> [rowCount, rLengthWithPadding]
        uint32_t trueShape[2] = {rowCount, rLengthWithPadding};
        AscendC::Broadcast<float, 2, 1>(tmpLocal, meanLocal, trueShape, meanShape);
        // 计算tmpLocal=x-mean
        AscendC::Sub(tmpLocal, inputXLocal, tmpLocal, rowCount * rLengthWithPadding);
        // 计算(x-mean) * (x-mean)
        AscendC::Mul(outputLocal, tmpLocal, tmpLocal, rowCount * rLengthWithPadding);
        // 计算var
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(rstdLocal, outputLocal, shape, false);
        AscendC::Muls(rstdLocal, rstdLocal, 1.0f / rLength, rowCount);
        // 计算rstd=1/sqrt(var+epsilon)
        AscendC::Adds(rstdLocal, rstdLocal, epsilon, rowCount);
        AscendC::Sqrt(rstdLocal, rstdLocal, rowCount);
        AscendC::Div(rstdLocal, oneLocal, rstdLocal, rowCount);
        AscendC::Broadcast<float, 2, 1>(outputLocal, rstdLocal, trueShape, meanShape);
        // 计算layernorm=rstd*(x-mean)
        AscendC::Mul(outputLocal, outputLocal, tmpLocal, rowCount * rLengthWithPadding);
        // 计算layernorm * gamma + beta
        uint32_t gammaShape[2] = {1, rLengthWithPadding};
        // 加入gamma，将gamma broadcast成{rowCount, rLengthWithPadding}
        AscendC::Broadcast<float, 2, 0>(tmpLocal, gammaLocal, trueShape, gammaShape);
        AscendC::Mul(outputLocal, outputLocal, tmpLocal, rowCount * rLengthWithPadding);
        // 加入beta，将beta broadcast成{rowCount, rLengthWithPadding}
        AscendC::Broadcast<float, 2, 0>(tmpLocal, betaLocal, trueShape, gammaShape);
        AscendC::Add(outputLocal, outputLocal, tmpLocal, rowCount * rLengthWithPadding);
        // 计算layernorm * U
        AscendC::Mul(outputLocal, outputLocal, inputULocal, rowCount * rLengthWithPadding);

        outQueue.EnQue<float>(outputLocal);

        inQueueX.FreeTensor(inputXLocal);
        inQueueU.FreeTensor(inputULocal);
        inQueueGamma.FreeTensor(gammaLocal);
        inQueueBeta.FreeTensor(betaLocal);
        meanQueue.FreeTensor(meanLocal);
        rstdQueue.FreeTensor(rstdLocal);
        tmpQueue.FreeTensor(tmpLocal);
        oneQueue.FreeTensor(oneLocal);
    }
    __aicore__ inline void CopyOut(uint32_t iterIdx, uint32_t rowCount)
    {
        // 核内偏移
        uint32_t offsetInCore = startRow * rLength + iterIdx * perCoreComputeRows * rLength;
        uint32_t copyLen = rowCount * rLength;

        AscendC::LocalTensor<float> outputLocal = outQueue.DeQue<float>();

        if (rLength == rLengthWithPadding) {
            CpLocal2Gm(outputGlobal[offsetInCore], outputLocal, copyLen);
        } else {
            for (int i = 0; i < rowCount; ++i) {
                CpLocal2Gm(outputGlobal[offsetInCore + i * rLength], outputLocal[i * rLengthWithPadding], rLength);
            }
        }

        outQueue.FreeTensor(outputLocal);
    }

private:
    GM_ADDR inputXGm;
    GM_ADDR inputUGm;
    GM_ADDR gammGm;
    GM_ADDR betaGm;
    GM_ADDR outputGm;
    AscendC::GlobalTensor<float> inputXGlobal;
    AscendC::GlobalTensor<float> inputUGlobal;
    AscendC::GlobalTensor<float> gammGlobal;
    AscendC::GlobalTensor<float> betaGlobal;
    AscendC::GlobalTensor<float> outputGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueU;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBeta;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> meanQueue;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> rstdQueue;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQueue;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> oneQueue;

    uint32_t aLength, rLength, rLengthWithPadding, arLength, coreNum, perCoreComputeRows, currentCoreRows;
    uint32_t currentLoopCount, currentRowLeft, startRow;
    float epsilon;
};
}  // namespace LnMulKernel

#endif  // MXREC_LAYERNORM_MUL_H

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

#include "kernel_operator.h"

using namespace AscendC;

namespace TokenMixing_Kernel {
constexpr int ALIGN_32 = 32;
struct TokenMixingArgs {
    GM_ADDR x;
    GM_ADDR x_t;
    GM_ADDR gamma;
    GM_ADDR beta;
    GM_ADDR y;
    float epsilon;
    uint32_t coreNum;
    uint64_t xDim0;
    uint64_t xDim1;
    uint64_t xDim2;
    uint64_t xDim2WithPadding;
    uint64_t perCoreComputeRows;
    uint64_t formerCoreRows;
    uint64_t formerLoopCount;
    uint64_t formerRemainRows;
    uint64_t tailCoreRows;
    uint64_t tailLoopCount;
    uint64_t tailRemainRows;
};

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, uint64_t len)
{
    uint64_t alignLen = len * sizeof(T) / ALIGN_32 * ALIGN_32;
    uint64_t unAlignLen = len * sizeof(T) - alignLen;

    GlobalTensor<uint16_t> uint16Gt;
    uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
    LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

    if (alignLen != 0) {
        DataCopy(uint16Lt, uint16Gt, alignLen / 2);
    }
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(unAlignLen), 0, 0, 0};
        const DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(uint16Lt[alignLen / 2], uint16Gt[alignLen / 2], dataCopyExtParams, dataCopyPadExtParams);
    }
}

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, uint64_t len)
{
    uint64_t alignLen = len * sizeof(T) / ALIGN_32 * ALIGN_32;
    uint64_t unAlignLen = len * sizeof(T) - alignLen;

    GlobalTensor<uint16_t> uint16Gt;
    uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
    LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

    if (alignLen != 0) {
        DataCopy(uint16Gt, uint16Lt, alignLen / 2);
    }
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(unAlignLen), 0, 0, 0};
        const DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(uint16Gt[alignLen / 2], uint16Lt[alignLen / 2], dataCopyExtParams);
    }
}

template <typename T>
class TokenMixing {
public:
    __aicore__ inline TokenMixing(){};

    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(TokenMixingArgs* args, TPipe* pipe)
    {
        this->args = args;
        this->pipe = pipe;
        bshLength_ = args->perCoreComputeRows * (args->xDim2WithPadding);
        bsLength_ = args->perCoreComputeRows;

        // 当前核的index
        int64_t blockIdx = GetBlockIdx();
        bool isTailCore = (blockIdx == args->coreNum - 1);
        coreRows_ = isTailCore ? args->tailCoreRows : args->formerCoreRows;
        loopCount_ = isTailCore ? args->tailLoopCount : args->formerLoopCount;
        remainRows_ = isTailCore ? args->tailRemainRows : args->formerRemainRows;
        startRow_ = blockIdx * args->formerCoreRows;
        bshLength_ = args->perCoreComputeRows * args->xDim2WithPadding;
        bsLength_ = args->perCoreComputeRows;

        xGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->x), bshLength_);
        xTGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->x_t), bshLength_);
        gammaGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->gamma), args->xDim2WithPadding);
        betaGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->beta), args->xDim2WithPadding);
        yGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->y), bshLength_);
        pipe->InitBuffer(inQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(inTQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(inQueueGamma, 1, (args->xDim2WithPadding) * sizeof(T));
        pipe->InitBuffer(inQueueBeta, 1, (args->xDim2WithPadding) * sizeof(T));
        pipe->InitBuffer(outQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(outQueueMean, 1, bsLength_ * sizeof(float));
        pipe->InitBuffer(outQueueRstd, 1, bsLength_ * sizeof(float));
        pipe->InitBuffer(addTmpQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(tmpQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(tmpMulQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(oneQueue, 1, bsLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (coreRows_ == 0) {
            return;
        }
        for (uint64_t i = 0; i < loopCount_; ++i) {
            CopyIn(i, args->perCoreComputeRows);
            Compute(args->perCoreComputeRows);
            CopyOut(i, args->perCoreComputeRows);
        }
        // 处理剩余行
        if (remainRows_ > 0) {
            CopyIn(loopCount_, remainRows_);
            Compute(remainRows_);
            CopyOut(loopCount_, remainRows_);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t iterIdx, uint64_t rowCount)
    {
        // 核内偏移
        uint64_t offsetInCore = startRow_ * args->xDim2 + iterIdx * args->perCoreComputeRows * args->xDim2;
        uint64_t copyLen = rowCount * args->xDim2;

        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        Duplicate(xLocal, 0.0f, bshLength_);
        LocalTensor<T> xTLocal = inTQueue.AllocTensor<T>();
        Duplicate(xTLocal, 0.0f, bshLength_);
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        LocalTensor<T> betaLocal = inQueueBeta.AllocTensor<T>();
        PipeBarrier<PIPE_ALL>();

        if (args->xDim2 == args->xDim2WithPadding) {
            CpGm2Local(xLocal, xGT[offsetInCore], copyLen);
            CpGm2Local(xTLocal, xTGT[offsetInCore], copyLen);
        } else {
            uint64_t xDataLen = args->xDim2 * sizeof(float);
            DataCopyExtParams params{1, static_cast<uint32_t>(xDataLen), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
            for (uint64_t i = 0; i < rowCount; ++i) {
                DataCopyPad(xLocal[i * args->xDim2WithPadding], xGT[offsetInCore + i * args->xDim2], params, padParams);
                DataCopyPad(xTLocal[i * args->xDim2WithPadding], xTGT[offsetInCore + i * args->xDim2], params,
                            padParams);
            }
        }
        CpGm2Local(gammaLocal, gammaGT, args->xDim2);
        CpGm2Local(betaLocal, betaGT, args->xDim2);
        inQueue.EnQue(xLocal);
        inTQueue.EnQue(xTLocal);
        inQueueGamma.EnQue(gammaLocal);
        inQueueBeta.EnQue(betaLocal);
    }

    __aicore__ inline void Compute(uint64_t rowCount)
    {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> xTLocal = inTQueue.DeQue<T>();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal = inQueueBeta.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        LocalTensor<float> meanLocal = outQueueMean.AllocTensor<float>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<T> addTmpLocal = addTmpQueue.AllocTensor<T>();
        LocalTensor<T> tmpLocal = tmpQueue.AllocTensor<T>();
        LocalTensor<T> tmpMulLocal = tmpMulQueue.AllocTensor<T>();
        LocalTensor<T> oneLocal = oneQueue.AllocTensor<T>();
        Duplicate(oneLocal, 1.0f, rowCount);

        Add(addTmpLocal, xLocal, xTLocal, bshLength_);
        // shape
        uint32_t meanShape[2] = {static_cast<uint32_t>(rowCount), 1};
#ifdef SUPPORT_C310
        uint32_t shape[2] = {static_cast<uint32_t>(rowCount), static_cast<uint32_t>(args->xDim2WithPadding)};
#else
        uint32_t shape[2] = {static_cast<uint32_t>(rowCount), static_cast<uint32_t>(args->xDim2)};
#endif
        // layerNorm拆分
        // 计算平均值mean
        Muls(tmpMulLocal, addTmpLocal, 1.0f / args->xDim2, bshLength_);
        PipeBarrier<PIPE_V>();
        ReduceSum<float, Pattern::Reduce::AR, false>(meanLocal, tmpMulLocal, shape, true);
        PipeBarrier<PIPE_V>();
        // broadcast mean[rowCount] -> [rowCount, xDim2WithPadding]
        uint32_t trueShape[2] = {static_cast<uint32_t>(rowCount), static_cast<uint32_t>(args->xDim2WithPadding)};
        Broadcast<float, 2, 1>(tmpLocal, meanLocal, trueShape, meanShape);
        if (args->xDim2 < args->xDim2WithPadding) {
            for (uint64_t i = 0; i < rowCount; ++i) {
                for (uint64_t j = args->xDim2; j < args->xDim2WithPadding; ++j) {
                    tmpLocal.SetValue(i * args->xDim2WithPadding + j, 0.0f);
                }
            }
        }
        // (x-mean) * (x-mean)
        Sub(tmpLocal, addTmpLocal, tmpLocal, rowCount * args->xDim2WithPadding);
        Mul(yLocal, tmpLocal, tmpLocal, rowCount * args->xDim2WithPadding);
        // 计算var
        Muls(yLocal, yLocal, 1.0f / args->xDim2, bshLength_);
        ReduceSum<float, Pattern::Reduce::AR, false>(rstdLocal, yLocal, shape, true);
        PipeBarrier<PIPE_V>();
        // 计算rstd=1/sqrt(var+epsilon)
        Adds(rstdLocal, rstdLocal, args->epsilon, rowCount);
        Sqrt(rstdLocal, rstdLocal, rowCount);
        Div(rstdLocal, oneLocal, rstdLocal, rowCount);
        Broadcast<float, 2, 1>(yLocal, rstdLocal, trueShape, meanShape);
        // 计算layernorm=(rstd*(x-mean))*gamma+beta
        Mul(yLocal, yLocal, tmpLocal, rowCount * args->xDim2WithPadding);
        uint32_t gammaShape[2] = {1, static_cast<uint32_t>(args->xDim2WithPadding)};
        Broadcast<float, 2, 0>(tmpLocal, gammaLocal, trueShape, gammaShape);
        Mul(yLocal, yLocal, tmpLocal, rowCount * args->xDim2WithPadding);
        Broadcast<float, 2, 0>(tmpLocal, betaLocal, trueShape, gammaShape);
        Add(yLocal, yLocal, tmpLocal, rowCount * args->xDim2WithPadding);

        outQueue.EnQue<T>(yLocal);

        inQueue.FreeTensor(xLocal);
        inTQueue.FreeTensor(xTLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        inQueueBeta.FreeTensor(betaLocal);
        addTmpQueue.FreeTensor(addTmpLocal);
        tmpMulQueue.FreeTensor(tmpMulLocal);
        tmpQueue.FreeTensor(tmpLocal);
        oneQueue.FreeTensor(oneLocal);
        outQueueMean.FreeTensor(meanLocal);
        outQueueRstd.FreeTensor(rstdLocal);
    }

    __aicore__ inline void CopyOut(uint64_t iterIdx, uint64_t rowCount)
    {
        // 核内偏移
        uint64_t offsetInCore = startRow_ * args->xDim2 + iterIdx * args->perCoreComputeRows * args->xDim2;
        uint64_t copyLen = rowCount * args->xDim2;
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        if (args->xDim2 == args->xDim2WithPadding) {
            CpLocal2Gm(yGT[offsetInCore], yLocal, copyLen);
        } else {
            for (int i = 0; i < rowCount; ++i) {
                CpLocal2Gm(yGT[offsetInCore + i * args->xDim2], yLocal[i * args->xDim2WithPadding], args->xDim2);
            }
        }

        outQueue.FreeTensor(yLocal);
    }

    TPipe* pipe;
    GlobalTensor<T> xGT, xTGT, gammaGT, betaGT, yGT;
    GlobalTensor<float> meanGT, rstdGT;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECIN, 1> inTQueue;
    TQue<TPosition::VECIN, 1> inQueueGamma;
    TQue<TPosition::VECIN, 1> inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECCALC, 1> outQueueMean;
    TQue<TPosition::VECCALC, 1> outQueueRstd;
    TQue<TPosition::VECCALC, 1> addTmpQueue;
    TQue<TPosition::VECCALC, 1> tmpQueue;
    TQue<TPosition::VECCALC, 1> tmpMulQueue;
    TQue<TPosition::VECCALC, 1> oneQueue;

    TokenMixingArgs* args;
    uint64_t bshLength_;
    uint64_t bsLength_;
    uint64_t coreRows_;
    uint64_t loopCount_;
    uint64_t remainRows_;
    uint64_t startRow_;
};
}  // namespace TokenMixing_Kernel

extern "C" __global__ __aicore__ void token_mixing(GM_ADDR x, GM_ADDR x_t, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TokenMixing_Kernel::TokenMixingArgs args{x,
                                             x_t,
                                             gamma,
                                             beta,
                                             y,
                                             tiling_data.epsilon,
                                             tiling_data.coreNum,
                                             tiling_data.xDim0,
                                             tiling_data.xDim1,
                                             tiling_data.xDim2,
                                             tiling_data.xDim2WithPadding,
                                             tiling_data.perCoreComputeRows,
                                             tiling_data.formerCoreRows,
                                             tiling_data.formerLoopCount,
                                             tiling_data.formerRemainRows,
                                             tiling_data.tailCoreRows,
                                             tiling_data.tailLoopCount,
                                             tiling_data.tailRemainRows};
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        TokenMixing_Kernel::TokenMixing<float> kernel;
        kernel.Init(&args, &pipe);
        kernel.Process();
    }
}

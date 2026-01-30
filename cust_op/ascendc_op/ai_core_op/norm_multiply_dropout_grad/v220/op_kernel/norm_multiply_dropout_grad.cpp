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
#include "kernel_operator.h"

using namespace AscendC;

namespace kernels {
constexpr uint32_t VEC_BLOCK_BYTES = 32;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NO_DOUBLE = 1;

template <typename T>
class NormMultiplyDropoutGrad {
public:
    __aicore__ NormMultiplyDropoutGrad(){};

    __aicore__ inline void init(GM_ADDR dOut, GM_ADDR xInput, GM_ADDR uInput, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR mean, GM_ADDR var, GM_ADDR mask, GM_ADDR dU, GM_ADDR dX, GM_ADDR dWeight,
                                GM_ADDR dBias, GM_ADDR workspace, const NormMultiplyDropoutGradTilingData& tiling)
    {
        ASSERT(GetBlockNum() != 0 && "useful core num can not be zero!!!")
        this->tiling = tiling;
        int32_t coreId = GetBlockIdx();

        bigCoreNum = tiling.bigCoreNum;
        littleCoreNum = tiling.littleCoreNum;
        avgCoreCalcRows = tiling.avgCoreCalcRows;
        xRowRount = tiling.xRowCount;
        xColCount = tiling.xColCount;
        singleBlockRows = tiling.singleBlockRows;
        useCoreNum = tiling.useCoreNum;
        eps = tiling.eps;
        dropoutRatio = tiling.dropoutRatio;

        if (bigCoreNum == 0) {
            coreOffset = coreId * avgCoreCalcRows;
            coreCalcRows = avgCoreCalcRows;
        } else if (coreId < bigCoreNum) {
            coreOffset = coreId * (avgCoreCalcRows + 1);
            coreCalcRows = avgCoreCalcRows + 1;
        } else {
            coreOffset = bigCoreNum * (avgCoreCalcRows + 1) + (coreId - bigCoreNum) * avgCoreCalcRows;
            coreCalcRows = avgCoreCalcRows;
        }

        tail = coreCalcRows % singleBlockRows;
        loop = coreCalcRows / singleBlockRows;

        dOutGM.SetGlobalBuffer((__gm__ T*)dOut);
        xGM.SetGlobalBuffer((__gm__ T*)xInput);
        uGM.SetGlobalBuffer((__gm__ T*)uInput);
        weightGM.SetGlobalBuffer((__gm__ T*)weight);
        biasGM.SetGlobalBuffer((__gm__ T*)bias);
        meanGM.SetGlobalBuffer((__gm__ float*)mean);
        varGM.SetGlobalBuffer((__gm__ float*)var);
        maskGM.SetGlobalBuffer((__gm__ uint8_t*)mask);
        dUGM.SetGlobalBuffer((__gm__ T*)dU);
        dXGM.SetGlobalBuffer((__gm__ T*)dX);
        dWeightGM.SetGlobalBuffer((__gm__ float*)dWeight);
        dBiasGM.SetGlobalBuffer((__gm__ float*)dBias);

        pipe.InitBuffer(dOutQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));  // 20k
        pipe.InitBuffer(xQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));      // 20k
        pipe.InitBuffer(uQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));      // 20k
        pipe.InitBuffer(weightQUE, NO_DOUBLE, xColCount * sizeof(float));
        pipe.InitBuffer(biasQUE, NO_DOUBLE, xColCount * sizeof(float));
        pipe.InitBuffer(meanQUE, NO_DOUBLE, singleBlockRows * sizeof(float));
        pipe.InitBuffer(varQUE, NO_DOUBLE, singleBlockRows * sizeof(float));
        pipe.InitBuffer(maskQUE, BUFFER_NUM, singleBlockRows * xColCount / 8 * sizeof(uint8_t));  // 1k
        pipe.InitBuffer(d_uQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));         // 20k
        pipe.InitBuffer(d_xQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));         // 20k

        pipe.InitBuffer(dOutcastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));       // 20k
        pipe.InitBuffer(xcastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));           // 20k
        pipe.InitBuffer(ucastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));           // 20k
        pipe.InitBuffer(weightcastQUE, NO_DOUBLE, xColCount * sizeof(T));
        pipe.InitBuffer(biascastQUE, NO_DOUBLE, xColCount * sizeof(T));
        pipe.InitBuffer(dxCastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));  // 20k
        pipe.InitBuffer(duCastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));  // 20k

        pipe.InitBuffer(normBuf, singleBlockRows * xColCount * sizeof(float));
        pipe.InitBuffer(rstdBuf, singleBlockRows * sizeof(float));
        pipe.InitBuffer(lnOutBuf, singleBlockRows * xColCount * sizeof(float));

        pipe.InitBuffer(dLnOutBuf, singleBlockRows * xColCount * sizeof(float));
        pipe.InitBuffer(dWeightBuf, singleBlockRows * xColCount * sizeof(float));
        pipe.InitBuffer(temp1Buf, singleBlockRows * xColCount * sizeof(float));
        pipe.InitBuffer(temp2Buf, singleBlockRows * sizeof(float));
        pipe.InitBuffer(temp3Buf, singleBlockRows * xColCount * sizeof(float));
        pipe.InitBuffer(temp4Buf, singleBlockRows * sizeof(float));
        pipe.InitBuffer(dropBuf, singleBlockRows * 2048);  // 20k

        pipe.InitBuffer(dWeightBuf2, xColCount * sizeof(float));
        pipe.InitBuffer(dBiasBuf2, xColCount * sizeof(float));
    }

    __aicore__ inline void process()
    {
        int32_t coreId = GetBlockIdx();
        loop *= BUFFER_NUM;
        dBiasLocal2 = dBiasBuf2.Get<float>();
        dWeightLocal2 = dWeightBuf2.Get<float>();
        Duplicate(dBiasLocal2, float(0), xColCount);
        Duplicate(dWeightLocal2, float(0), xColCount);
        for (int32_t blockId = 0; blockId < loop; blockId++) {
            copyIn(blockId, singleBlockRows);
            compute(blockId, singleBlockRows);
            copyOut(blockId, singleBlockRows);
        }
        if (tail != 0) {
            copyIn(loop, tail);
            compute(loop, tail);
            copyOut(loop, tail);
        }

        SetAtomicAdd<float>();
        DataCopy(dWeightGM, dWeightLocal2, xColCount);
        DataCopy(dBiasGM, dBiasLocal2, xColCount);
        SetAtomicNone();
    }

    __aicore__ inline void copyIn(int32_t blockId, int32_t calcRows)
    {
        dOutLocal = dOutQUE.AllocTensor<float>();
        xLocal = xQUE.AllocTensor<float>();
        uLocal = uQUE.AllocTensor<float>();
        weightLocal = weightQUE.AllocTensor<float>();
        biasLocal = biasQUE.AllocTensor<float>();
        meanLocal = meanQUE.AllocTensor<float>();
        varLocal = varQUE.AllocTensor<float>();
        maskLocal = maskQUE.AllocTensor<uint8_t>();

        dOutCast = dOutcastQUE.AllocTensor<T>();
        xCast = xcastQUE.AllocTensor<T>();
        uCast = ucastQUE.AllocTensor<T>();
        weightCast = weightcastQUE.AllocTensor<T>();
        biasCast = biascastQUE.AllocTensor<T>();

        DataCopy(dOutCast, dOutGM[(coreOffset + blockId * singleBlockRows) * xColCount],
                 calcRows * xColCount);
        DataCopy(xCast, xGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(uCast, uGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(weightCast, weightGM, xColCount);
        DataCopy(biasCast, biasGM, xColCount);
        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calcRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(meanLocal, meanGM[(coreOffset + blockId * singleBlockRows)], copyParams, padParams);
        DataCopyPad(varLocal, varGM[(coreOffset + blockId * singleBlockRows)], copyParams, padParams);
        DataCopy(maskLocal, maskGM[(coreOffset + blockId * singleBlockRows) * xColCount / 8],
                 calcRows * xColCount / 8);

        dOutcastQUE.EnQue(dOutCast);
        xcastQUE.EnQue(xCast);
        ucastQUE.EnQue(uCast);
        weightcastQUE.EnQue(weightCast);
        biascastQUE.EnQue(biasCast);
        dOutCast = dOutcastQUE.DeQue<T>();
        xCast = xcastQUE.DeQue<T>();
        uCast = ucastQUE.DeQue<T>();
        weightCast = weightcastQUE.DeQue<T>();
        biasCast = biascastQUE.DeQue<T>();

        Cast<float, T>(dOutLocal, dOutCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(xLocal, xCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(uLocal, uCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(biasLocal, biasCast, RoundMode::CAST_NONE, xColCount);
        Cast<float, T>(weightLocal, weightCast, RoundMode::CAST_NONE, xColCount);

        dOutcastQUE.FreeTensor(dOutCast);
        xcastQUE.FreeTensor(xCast);
        ucastQUE.FreeTensor(uCast);
        weightcastQUE.FreeTensor(weightCast);
        biascastQUE.FreeTensor(biasCast);

        dOutQUE.EnQue(dOutLocal);
        xQUE.EnQue(xLocal);
        uQUE.EnQue(uLocal);
        weightQUE.EnQue(weightLocal);
        biasQUE.EnQue(biasLocal);
        meanQUE.EnQue(meanLocal);
        varQUE.EnQue(varLocal);
        maskQUE.EnQue(maskLocal);
    }

    __aicore__ inline void compute(int32_t blockId, int32_t calcRows)
    {
        dOutLocal = dOutQUE.DeQue<float>();
        xLocal = xQUE.DeQue<float>();
        uLocal = uQUE.DeQue<float>();
        weightLocal = weightQUE.DeQue<float>();
        biasLocal = biasQUE.DeQue<float>();
        meanLocal = meanQUE.DeQue<float>();
        varLocal = varQUE.DeQue<float>();
        maskLocal = maskQUE.DeQue<uint8_t>();

        // dropout_grad
        AscendC::DropOutShapeInfo info;
        float probValue = 1 - dropoutRatio;
        info.firstAxis = calcRows;
        info.srcLastAxis = xColCount;
        info.maskLastAxis = xColCount / 8;

        dropTemp = dropBuf.Get<uint8_t>();
        DropOut<float, false, 3>(dOutLocal, dOutLocal, maskLocal, dropTemp, probValue, info);
        pipe_barrier(PIPE_ALL);
        // recompute
        normLocal = normBuf.Get<float>();
        rstdLocal = rstdBuf.Get<float>();
        lnOutLocal = lnOutBuf.Get<float>();

        Muls<float>(meanLocal, meanLocal, static_cast<float>(-1), calcRows);
        for (int32_t i = 0; i < calcRows; i++) {
            Adds(xLocal[i * xColCount], xLocal[i * xColCount], meanLocal.GetValue(i), xColCount);
        }

        Adds<float>(varLocal, varLocal, eps, calcRows);
        Sqrt(rstdLocal, varLocal, calcRows);

        for (int32_t i = 0; i < calcRows; i++) {
            Muls<float>(normLocal[i * xColCount], xLocal[i * xColCount], (float)1.0 / rstdLocal.GetValue(i),
                        xColCount);
        }

        for (int32_t i = 0; i < calcRows; i++) {
            Mul<float>(lnOutLocal[i * xColCount], normLocal[i * xColCount], weightLocal, xColCount);
            Add<float>(lnOutLocal[i * xColCount], lnOutLocal[i * xColCount], biasLocal, xColCount);
        }

        // mul_grad
        dLnOutLocal = dLnOutBuf.Get<float>();
        duLocal = d_uQUE.AllocTensor<float>();
        dxLocal = d_xQUE.AllocTensor<float>();
        Mul<float>(dLnOutLocal, dOutLocal, uLocal, calcRows * xColCount);
        Mul<float>(duLocal, dOutLocal, lnOutLocal, calcRows * xColCount);

        dWeightLocal = dWeightBuf.Get<float>();
        for (int32_t i = 0; i < calcRows; i++) {
            Add(dBiasLocal2, dBiasLocal2, dLnOutLocal[i * xColCount], xColCount);
        }

        Mul<float>(dWeightLocal, normLocal, dLnOutLocal, calcRows * xColCount);
        for (int32_t i = 1; i < calcRows; i++) {
            Add(dWeightLocal[0], dWeightLocal[0], dWeightLocal[i * xColCount], xColCount);
        }
        Add(dWeightLocal2, dWeightLocal2, dWeightLocal[0], xColCount);
        // dX
        //  temp1: d_ln_out * weight
        //  temp2: mean(temp1)
        //  temp3: temp1 - mean(temp1)
        temp1Local = temp1Buf.Get<float>();
        temp2Local = temp2Buf.Get<float>();
        temp3Local = temp3Buf.Get<float>();
        temp4Local = temp4Buf.Get<float>();
        reduceTemp = dropBuf.Get<float>();

        for (int32_t i = 0; i < calcRows; i++) {
            Mul<float>(temp1Local[i * xColCount], dLnOutLocal[i * xColCount], weightLocal,
                       xColCount);  // d_ln_out * gamma
            Mul<float>(temp3Local[i * xColCount], temp1Local[i * xColCount], normLocal[i * xColCount],
                       xColCount);  // d_ln_out * gamma * norm
        }

        WholeReduceSum<float>(reduceTemp, temp1Local, 256 / sizeof(float),
                              calcRows * xColCount / (256 / sizeof(float)), 1, 1, 8);
        WholeReduceSum<float>(temp2Local, reduceTemp, xColCount / (256 / sizeof(float)), calcRows, 1, 1, 1);

        WholeReduceSum<float>(reduceTemp, temp3Local, 256 / sizeof(float),
                              calcRows * xColCount / (256 / sizeof(float)), 1, 1, 8);
        WholeReduceSum<float>(temp4Local, reduceTemp, xColCount / (256 / sizeof(float)), calcRows, 1, 1, 1);

        for (int32_t i = 0; i < calcRows; i++) {
            float sumDOutWeight = temp2Local.GetValue(i);
            float meanDOutWeight = sumDOutWeight / xColCount;  // A

            float sumDOutWeightNorm = temp4Local.GetValue(i);
            float meanDOutWeightNorm = sumDOutWeightNorm / xColCount;  // B

            Muls<float>(normLocal[i * xColCount], normLocal[i * xColCount], -meanDOutWeightNorm, xColCount);
            Adds<float>(normLocal[i * xColCount], normLocal[i * xColCount], -meanDOutWeight, xColCount);

            Add<float>(dxLocal[i * xColCount], temp1Local[i * xColCount], normLocal[i * xColCount],
                       xColCount);
            Muls<float>(dxLocal[i * xColCount], dxLocal[i * xColCount], (float)1.0 / rstdLocal.GetValue(i),
                        xColCount);
        }

        dOutQUE.FreeTensor(dOutLocal);
        xQUE.FreeTensor(xLocal);
        uQUE.FreeTensor(uLocal);
        weightQUE.FreeTensor(weightLocal);
        biasQUE.FreeTensor(biasLocal);
        meanQUE.FreeTensor(meanLocal);
        varQUE.FreeTensor(varLocal);
        maskQUE.FreeTensor(maskLocal);

        d_uQUE.EnQue(duLocal);
        d_xQUE.EnQue(dxLocal);
    }

    __aicore__ inline void copyOut(int32_t blockId, int32_t calcRows)
    {
        duLocal = d_uQUE.DeQue<float>();
        dxLocal = d_xQUE.DeQue<float>();

        duCast = duCastQUE.AllocTensor<T>();
        dxCast = dxCastQUE.AllocTensor<T>();

        Cast<T, float>(duCast, duLocal, RoundMode::CAST_ROUND, calcRows * xColCount);
        Cast<T, float>(dxCast, dxLocal, RoundMode::CAST_ROUND, calcRows * xColCount);

        duCastQUE.EnQue(duCast);
        duCast = duCastQUE.DeQue<T>();
        dxCastQUE.EnQue(dxCast);
        dxCast = dxCastQUE.DeQue<T>();

        DataCopy(dUGM[(coreOffset + blockId * singleBlockRows) * xColCount], duCast, calcRows * xColCount);
        DataCopy(dXGM[(coreOffset + blockId * singleBlockRows) * xColCount], dxCast, calcRows * xColCount);

        duCastQUE.FreeTensor(duCast);
        dxCastQUE.FreeTensor(dxCast);
        d_uQUE.FreeTensor(duLocal);
        d_xQUE.FreeTensor(dxLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> dOutQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> uQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> varQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQUE;
    TQue<QuePosition::VECOUT, BUFFER_NUM> d_uQUE;
    TQue<QuePosition::VECOUT, BUFFER_NUM> d_xQUE;

    TQue<QuePosition::VECIN, BUFFER_NUM> dOutcastQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> xcastQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> ucastQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightcastQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> biascastQUE;

    TQue<QuePosition::VECOUT, BUFFER_NUM> duCastQUE;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxCastQUE;

    TBuf<TPosition::VECCALC> normBuf;
    TBuf<TPosition::VECCALC> rstdBuf;
    TBuf<TPosition::VECCALC> lnOutBuf;
    TBuf<TPosition::VECCALC> dLnOutBuf;
    TBuf<TPosition::VECCALC> dWeightBuf;
    TBuf<TPosition::VECCALC> temp1Buf;
    TBuf<TPosition::VECCALC> temp2Buf;
    TBuf<TPosition::VECCALC> temp3Buf;
    TBuf<TPosition::VECCALC> temp4Buf;
    TBuf<TPosition::VECCALC> dropBuf;

    TBuf<TPosition::VECCALC> dWeightBuf2;
    TBuf<TPosition::VECCALC> dBiasBuf2;

    GlobalTensor<T> dOutGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> uGM;
    GlobalTensor<T> weightGM;
    GlobalTensor<T> biasGM;
    GlobalTensor<float> meanGM;
    GlobalTensor<float> varGM;
    GlobalTensor<uint8_t> maskGM;
    GlobalTensor<T> dUGM;
    GlobalTensor<T> dXGM;
    GlobalTensor<float> dWeightGM;
    GlobalTensor<float> dBiasGM;

    LocalTensor<float> dOutLocal;
    LocalTensor<float> xLocal;
    LocalTensor<float> uLocal;
    LocalTensor<float> weightLocal;
    LocalTensor<float> biasLocal;
    LocalTensor<float> meanLocal;
    LocalTensor<float> varLocal;
    LocalTensor<uint8_t> maskLocal;

    LocalTensor<T> dOutCast;
    LocalTensor<T> xCast;
    LocalTensor<T> uCast;
    LocalTensor<T> weightCast;
    LocalTensor<T> biasCast;

    LocalTensor<float> duLocal;
    LocalTensor<float> dxLocal;
    LocalTensor<float> dWeightLocal;
    LocalTensor<float> dBiasLocal2;
    LocalTensor<float> dWeightLocal2;

    LocalTensor<T> duCast;
    LocalTensor<T> dxCast;

    LocalTensor<float> normLocal;
    LocalTensor<float> dLnOutLocal;
    LocalTensor<float> temp1Local;
    LocalTensor<float> temp2Local;
    LocalTensor<float> temp3Local;
    LocalTensor<float> temp4Local;
    LocalTensor<uint8_t> dropTemp;
    LocalTensor<float> reduceTemp;
    LocalTensor<float> lnOutLocal;
    LocalTensor<float> rstdLocal;

    NormMultiplyDropoutGradTilingData tiling;

    DataCopyPadExtParams<float> padParams{true, 0, 0, 0};

    int32_t bigCoreNum;
    int32_t littleCoreNum;
    int32_t avgCoreCalcRows;
    int32_t xRowRount;
    int32_t xColCount;
    int32_t singleBlockRows;
    int32_t useCoreNum;
    float eps;
    float dropoutRatio;

    int32_t loop;
    int32_t tail;
    int32_t coreOffset;
    int32_t coreCalcRows;

    AscendC::DropOutShapeInfo info;
};
}  // namespace kernels

extern "C" __global__ __aicore__ void norm_multiply_dropout_grad(GM_ADDR dOut, GM_ADDR xInput, GM_ADDR uInput,
                                                                 GM_ADDR weight, GM_ADDR bias, GM_ADDR mean,
                                                                 GM_ADDR var, GM_ADDR mask, GM_ADDR dU, GM_ADDR dX,
                                                                 GM_ADDR dWeight, GM_ADDR dBias, GM_ADDR workspace,
                                                                 GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GET_TILING_DATA(tilingData, tiling);
    kernels::NormMultiplyDropoutGrad<DTYPE_X> op;
    op.init(dOut, xInput, uInput, weight, bias, mean, var, mask, dU, dX, dWeight, dBias, workspace, tilingData);
    op.process();
}

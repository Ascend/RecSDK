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

#include "norm_multiply_dropout_backward_tilingKey.h"

using namespace AscendC;

namespace kernels {
constexpr uint32_t VEC_BLOCK_BYTES = 32;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NO_DOUBLE = 1;
constexpr int32_t REDUCE_CALC_BYTE = 256;
constexpr int32_t MASK_ELEMENTS_BITS = 8;
constexpr float NEGATIVE_ONE_F = -1.0;
constexpr float POSITIVE_ONE_F = 1.0;
constexpr int32_t DROP_TEMP_BUF = 2048;

template <typename T, bool isNeedDrop>
class NormMultiplyDropoutBackward {
public:
    __aicore__ NormMultiplyDropoutBackward(){};

    __aicore__ inline void init(GM_ADDR dOut, GM_ADDR x, GM_ADDR u, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR mean, GM_ADDR var, GM_ADDR mask, GM_ADDR dU, GM_ADDR dX, GM_ADDR dWeight,
                                GM_ADDR dBias, GM_ADDR workspace, const NormMultiplyDropoutBackwardTilingData& tiling)
    {
        ASSERT(GetBlockNum() != 0 && "useful core num can not be zero!!!")
        this->tiling = tiling;
        int32_t coreId = GetBlockIdx();

        bigCoreNum = tiling.bigCoreNum;
        littleCoreNum = tiling.littleCoreNum;
        avgCoreCalcRows = tiling.avgCoreCalcRows;
        xRowCount = tiling.xRowCount;
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
        xGM.SetGlobalBuffer((__gm__ T*)x);
        uGM.SetGlobalBuffer((__gm__ T*)u);
        weightGM.SetGlobalBuffer((__gm__ T*)weight);
        biasGM.SetGlobalBuffer((__gm__ T*)bias);
        meanGM.SetGlobalBuffer((__gm__ float*)mean);
        varGM.SetGlobalBuffer((__gm__ float*)var);
        maskGM.SetGlobalBuffer((__gm__ uint8_t*)mask);
        dUGM.SetGlobalBuffer((__gm__ T*)dU);
        dXGM.SetGlobalBuffer((__gm__ T*)dX);
        dWeightGM.SetGlobalBuffer((__gm__ float*)dWeight);
        dBiasGM.SetGlobalBuffer((__gm__ float*)dBias);

        pipe.InitBuffer(dOutQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));    // n * C * 4
        pipe.InitBuffer(xQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));       // n * C * 4
        pipe.InitBuffer(uQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));       // n * C * 4
        pipe.InitBuffer(weightQue, NO_DOUBLE, xColCount * sizeof(float));                     // C * 4
        pipe.InitBuffer(biasQue, NO_DOUBLE, xColCount * sizeof(float));                       // C * 4
        pipe.InitBuffer(meanQue, NO_DOUBLE, singleBlockRows * sizeof(float));                 // n * 4
        pipe.InitBuffer(varQue, NO_DOUBLE, singleBlockRows * sizeof(float));                  // n * 4
        if constexpr (isNeedDrop) {
            pipe.InitBuffer(maskQue, BUFFER_NUM,
                            singleBlockRows * xColCount / MASK_ELEMENTS_BITS * sizeof(uint8_t));  // n * C * 2
        }
        pipe.InitBuffer(duQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));      // n * C * 4
        pipe.InitBuffer(dxQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));      // n * C * 4

        if constexpr (!std::is_same<T, float>::value) {
            pipe.InitBuffer(dOutcastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));    // n * C * 2
            pipe.InitBuffer(xCastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));       // n * C * 2
            pipe.InitBuffer(uCastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));       // n * C * 2
            pipe.InitBuffer(weightCastQue, NO_DOUBLE, xColCount * sizeof(T));                     // C * 2
            pipe.InitBuffer(biasCastQue, NO_DOUBLE, xColCount * sizeof(T));                       // C * 2
            pipe.InitBuffer(dxCastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));      // n * C * 2
            pipe.InitBuffer(duCastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));      // n * C * 2
        }

        pipe.InitBuffer(normBuf, singleBlockRows * xColCount * sizeof(float));                // n * C * 2
        pipe.InitBuffer(stdBuf, singleBlockRows * sizeof(float));                             // n * 4
        pipe.InitBuffer(lnOutBuf, singleBlockRows * xColCount * sizeof(float));               // n * C * 4

        pipe.InitBuffer(dLnOutBuf, singleBlockRows * xColCount * sizeof(float));              // n * C * 4
        pipe.InitBuffer(dWeightBuf, singleBlockRows * xColCount * sizeof(float));             // n * C * 4
        pipe.InitBuffer(temp1Buf, singleBlockRows * xColCount * sizeof(float));               // n * C * 4
        pipe.InitBuffer(temp2Buf, singleBlockRows * sizeof(float));                           // n * 4
        pipe.InitBuffer(temp3Buf, singleBlockRows * xColCount * sizeof(float));               // n * C * 4
        pipe.InitBuffer(temp4Buf, singleBlockRows * sizeof(float));                           // n * 4
        pipe.InitBuffer(dropBuf, singleBlockRows * DROP_TEMP_BUF);                            // n * 2048

        pipe.InitBuffer(dWeightBuf2, xColCount * sizeof(float));                              // C * 4
        pipe.InitBuffer(dBiasBuf2, xColCount * sizeof(float));                                // C * 4
    }

    __aicore__ inline void process()
    {
        int32_t coreId = GetBlockIdx();
        loop *= BUFFER_NUM;
        dBiasLocal2 = dBiasBuf2.Get<float>();
        dWeightLocal2 = dWeightBuf2.Get<float>();
        Duplicate(dBiasLocal2, float(0), xColCount);
        Duplicate(dWeightLocal2, float(0), xColCount);
        if constexpr (std::is_same<T, float>::value) {
            for (int32_t blockId = 0; blockId < loop; blockId++) {
                CopyInNoCast(blockId, singleBlockRows);
                Compute(blockId, singleBlockRows);
                CopyOutNoCast(blockId, singleBlockRows);
            }
            if (tail != 0) {
                CopyInNoCast(loop, tail);
                Compute(loop, tail);
                CopyOutNoCast(loop, tail);
            }
        } else {
            for (int32_t blockId = 0; blockId < loop; blockId++) {
                CopyInWithCast(blockId, singleBlockRows);
                Compute(blockId, singleBlockRows);
                CopyOutWithCast(blockId, singleBlockRows);
            }
            if (tail != 0) {
                CopyInWithCast(loop, tail);
                Compute(loop, tail);
                CopyOutWithCast(loop, tail);
            }
        }

        SetAtomicAdd<float>();
        DataCopy(dWeightGM, dWeightLocal2, xColCount);
        DataCopy(dBiasGM, dBiasLocal2, xColCount);
        SetAtomicNone();
    }

    __aicore__ inline void CopyInNoCast(int32_t blockId, int32_t calcRows)
    {
        dOutLocal = dOutQue.AllocTensor<T>();
        xLocal = xQue.AllocTensor<T>();
        uLocal = uQue.AllocTensor<T>();
        weightLocal = weightQue.AllocTensor<T>();
        biasLocal = biasQue.AllocTensor<T>();
        meanLocal = meanQue.AllocTensor<T>();
        varLocal = varQue.AllocTensor<T>();

        if constexpr (isNeedDrop) {
            maskLocal = maskQue.AllocTensor<uint8_t>();
        }

        DataCopy(dOutLocal, dOutGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(xLocal, xGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(uLocal, uGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(weightLocal, weightGM, xColCount);
        DataCopy(biasLocal, biasGM, xColCount);
        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calcRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(meanLocal, meanGM[(coreOffset + blockId * singleBlockRows)], copyParams, padParams);
        DataCopyPad(varLocal, varGM[(coreOffset + blockId * singleBlockRows)], copyParams, padParams);
        if constexpr (isNeedDrop) {
            DataCopy(maskLocal, maskGM[(coreOffset + blockId * singleBlockRows) * xColCount / MASK_ELEMENTS_BITS],
                     calcRows * xColCount / MASK_ELEMENTS_BITS);
        }

        dOutQue.EnQue(dOutLocal);
        xQue.EnQue(xLocal);
        uQue.EnQue(uLocal);
        weightQue.EnQue(weightLocal);
        biasQue.EnQue(biasLocal);
        meanQue.EnQue(meanLocal);
        varQue.EnQue(varLocal);
        if constexpr (isNeedDrop) {
            maskQue.EnQue(maskLocal);
        }
    }

    __aicore__ inline void CopyInWithCast(int32_t blockId, int32_t calcRows)
    {
        dOutLocal = dOutQue.AllocTensor<float>();
        xLocal = xQue.AllocTensor<float>();
        uLocal = uQue.AllocTensor<float>();
        weightLocal = weightQue.AllocTensor<float>();
        biasLocal = biasQue.AllocTensor<float>();
        meanLocal = meanQue.AllocTensor<float>();
        varLocal = varQue.AllocTensor<float>();

        dOutCast = dOutcastQue.AllocTensor<T>();
        xCast = xCastQue.AllocTensor<T>();
        uCast = uCastQue.AllocTensor<T>();
        weightCast = weightCastQue.AllocTensor<T>();
        biasCast = biasCastQue.AllocTensor<T>();
        if constexpr (isNeedDrop) {
            maskLocal = maskQue.AllocTensor<uint8_t>();
        }

        DataCopy(dOutCast, dOutGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(xCast, xGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(uCast, uGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(weightCast, weightGM, xColCount);
        DataCopy(biasCast, biasGM, xColCount);
        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calcRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(meanLocal, meanGM[(coreOffset + blockId * singleBlockRows)], copyParams, padParams);
        DataCopyPad(varLocal, varGM[(coreOffset + blockId * singleBlockRows)], copyParams, padParams);
        if constexpr (isNeedDrop) {
            DataCopy(maskLocal, maskGM[(coreOffset + blockId * singleBlockRows) * xColCount / MASK_ELEMENTS_BITS],
                     calcRows * xColCount / MASK_ELEMENTS_BITS);
        }
        dOutcastQue.EnQue(dOutCast);
        xCastQue.EnQue(xCast);
        uCastQue.EnQue(uCast);
        weightCastQue.EnQue(weightCast);
        biasCastQue.EnQue(biasCast);
        dOutCast = dOutcastQue.DeQue<T>();
        xCast = xCastQue.DeQue<T>();
        uCast = uCastQue.DeQue<T>();
        weightCast = weightCastQue.DeQue<T>();
        biasCast = biasCastQue.DeQue<T>();

        Cast<float, T>(dOutLocal, dOutCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(xLocal, xCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(uLocal, uCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(weightLocal, weightCast, RoundMode::CAST_NONE, xColCount);
        Cast<float, T>(biasLocal, biasCast, RoundMode::CAST_NONE, xColCount);

        dOutQue.EnQue(dOutLocal);
        xQue.EnQue(xLocal);
        uQue.EnQue(uLocal);
        weightQue.EnQue(weightLocal);
        biasQue.EnQue(biasLocal);
        meanQue.EnQue(meanLocal);
        varQue.EnQue(varLocal);
        if constexpr (isNeedDrop) {
            maskQue.EnQue(maskLocal);
        }

        dOutcastQue.FreeTensor(dOutCast);
        xCastQue.FreeTensor(xCast);
        uCastQue.FreeTensor(uCast);
        weightCastQue.FreeTensor(weightCast);
        biasCastQue.FreeTensor(biasCast);
    }

    __aicore__ inline void Compute(int32_t blockId, int32_t calcRows)
    {
        dOutLocal = dOutQue.DeQue<float>();
        xLocal = xQue.DeQue<float>();
        uLocal = uQue.DeQue<float>();
        weightLocal = weightQue.DeQue<float>();
        biasLocal = biasQue.DeQue<float>();
        meanLocal = meanQue.DeQue<float>();
        varLocal = varQue.DeQue<float>();
        if constexpr (isNeedDrop) {
            maskLocal = maskQue.DeQue<uint8_t>();
        }

        // dropout_grad
        AscendC::DropOutShapeInfo info;
        float probValue = 1 - dropoutRatio;
        info.firstAxis = calcRows;
        info.srcLastAxis = xColCount;
        info.maskLastAxis = xColCount / MASK_ELEMENTS_BITS;

        // dropout grad: d_scaled = dropout(grad_out, mask, p)
        if constexpr (isNeedDrop) {
            dropTemp = dropBuf.Get<uint8_t>();
            DropOut<float, false, 3>(dOutLocal, dOutLocal, maskLocal, dropTemp, probValue, info);
        }
        pipe_barrier(PIPE_ALL);
        // 计算标准norm结果: x_hat = (x - mean) / sqrt(variance + eps)
        normLocal = normBuf.Get<float>();
        stdLocal = stdBuf.Get<float>();
        lnOutLocal = lnOutBuf.Get<float>();
        Muls<float>(meanLocal, meanLocal, NEGATIVE_ONE_F, calcRows);
        for (int32_t i = 0; i < calcRows; i++) {
            Adds(xLocal[i * xColCount], xLocal[i * xColCount], meanLocal.GetValue(i), xColCount);
        }
        Adds<float>(varLocal, varLocal, eps, calcRows);
        Sqrt(stdLocal, varLocal, calcRows);
        for (int32_t i = 0; i < calcRows; i++) {
            Muls<float>(normLocal[i * xColCount], xLocal[i * xColCount], POSITIVE_ONE_F / stdLocal.GetValue(i),
                        xColCount);
        }
        // 标准norm + 仿射变换后的输出：ln_out = x_hat * weight + bias
        for (int32_t i = 0; i < calcRows; i++) {
            Mul<float>(lnOutLocal[i * xColCount], normLocal[i * xColCount], weightLocal, xColCount);
            Add<float>(lnOutLocal[i * xColCount], lnOutLocal[i * xColCount], biasLocal, xColCount);
        }

        dLnOutLocal = dLnOutBuf.Get<float>();
        duLocal = duQue.AllocTensor<float>();
        dxLocal = dxQue.AllocTensor<float>();

        // d_u = d_out * ln_out
        Mul<float>(duLocal, dOutLocal, lnOutLocal, calcRows * xColCount);
        // d_ln_out = d_out * u
        Mul<float>(dLnOutLocal, dOutLocal, uLocal, calcRows * xColCount);
        // d_bias = accumulation(d_ln_out) 每一行累加
        dWeightLocal = dWeightBuf.Get<float>();
        for (int32_t i = 0; i < calcRows; i++) {
            Add(dBiasLocal2, dBiasLocal2, dLnOutLocal[i * xColCount], xColCount);
        }
        // d_weight = accumulation(x_hat * d_ln_out)
        Mul<float>(dWeightLocal, normLocal, dLnOutLocal, calcRows * xColCount);
        for (int32_t i = 1; i < calcRows; i++) {
            Add(dWeightLocal[0], dWeightLocal[0], dWeightLocal[i * xColCount], xColCount);
        }
        // 将本次计算的d_weight的值累加到dWeightLocal2中，计算完成后再搬出
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
            // temp1Local: dx_hat = d_ln_out * weight
            Mul<float>(temp1Local[i * xColCount], dLnOutLocal[i * xColCount], weightLocal, xColCount);
            // temp3Local: x_dx = dx_hat * x_hat
            Mul<float>(temp3Local[i * xColCount], temp1Local[i * xColCount], normLocal[i * xColCount], xColCount);
        }
        // reduce_dx_hat = accumulation(dx_hat)
        WholeReduceSum<float>(reduceTemp, temp1Local, reduceElements, calcRows * xColCount / reduceElements, 1, 1,
                              reduceElements / oneBLockElements);
        WholeReduceSum<float>(temp2Local, reduceTemp, xColCount / reduceElements, calcRows, 1, 1,
                              xColCount / reduceElements / oneBLockElements);
        // reduce_x_dx = accumulation(x_hat * dx_hat)
        WholeReduceSum<float>(reduceTemp, temp3Local, reduceElements, calcRows * xColCount / reduceElements, 1, 1,
                              reduceElements / oneBLockElements);
        WholeReduceSum<float>(temp4Local, reduceTemp, xColCount / reduceElements, calcRows, 1, 1,
                              xColCount / reduceElements / oneBLockElements);

        for (int32_t i = 0; i < calcRows; i++) {
            // mean_dx = reduce_dx_hat / N
            float sumDOutWeight = temp2Local.GetValue(i);
            float meanDOutWeight = sumDOutWeight / xColCount;
            // mean_x_dx = reduce_x_dx / N
            float sumDOutWeightNorm = temp4Local.GetValue(i);
            float meanDOutWeightNorm = sumDOutWeightNorm / xColCount;

            // dx = (dx_hat - mean_dx - x_hat * mean_x_dx) / std
            // 变换下公式 dx = (dx_hat + (-1) * mean_dx + x_hat * (-1) * mean_x_dx) / std
            Muls<float>(normLocal[i * xColCount], normLocal[i * xColCount], -meanDOutWeightNorm, xColCount);
            Adds<float>(normLocal[i * xColCount], normLocal[i * xColCount], -meanDOutWeight, xColCount);
            Add<float>(dxLocal[i * xColCount], temp1Local[i * xColCount], normLocal[i * xColCount], xColCount);
            Muls<float>(dxLocal[i * xColCount], dxLocal[i * xColCount], POSITIVE_ONE_F / stdLocal.GetValue(i),
                        xColCount);
        }

        duQue.EnQue(duLocal);
        dxQue.EnQue(dxLocal);

        dOutQue.FreeTensor(dOutLocal);
        xQue.FreeTensor(xLocal);
        uQue.FreeTensor(uLocal);
        weightQue.FreeTensor(weightLocal);
        biasQue.FreeTensor(biasLocal);
        meanQue.FreeTensor(meanLocal);
        varQue.FreeTensor(varLocal);
        if constexpr (isNeedDrop) {
            maskQue.FreeTensor(maskLocal);
        }
    }

    __aicore__ inline void CopyOutNoCast(int32_t blockId, int32_t calcRows)
    {
        duLocal = duQue.DeQue<T>();
        dxLocal = dxQue.DeQue<T>();
        DataCopy(dUGM[(coreOffset + blockId * singleBlockRows) * xColCount], duLocal, calcRows * xColCount);
        DataCopy(dXGM[(coreOffset + blockId * singleBlockRows) * xColCount], dxLocal, calcRows * xColCount);
        duQue.FreeTensor(duLocal);
        dxQue.FreeTensor(dxLocal);
    }

    __aicore__ inline void CopyOutWithCast(int32_t blockId, int32_t calcRows)
    {
        duLocal = duQue.DeQue<float>();
        dxLocal = dxQue.DeQue<float>();

        duCast = duCastQue.AllocTensor<T>();
        dxCast = dxCastQue.AllocTensor<T>();

        Cast<T, float>(duCast, duLocal, RoundMode::CAST_ROUND, calcRows * xColCount);
        Cast<T, float>(dxCast, dxLocal, RoundMode::CAST_ROUND, calcRows * xColCount);

        duCastQue.EnQue(duCast);
        duCast = duCastQue.DeQue<T>();
        dxCastQue.EnQue(dxCast);
        dxCast = dxCastQue.DeQue<T>();

        DataCopy(dUGM[(coreOffset + blockId * singleBlockRows) * xColCount], duCast, calcRows * xColCount);
        DataCopy(dXGM[(coreOffset + blockId * singleBlockRows) * xColCount], dxCast, calcRows * xColCount);

        duCastQue.FreeTensor(duCast);
        dxCastQue.FreeTensor(dxCast);
        duQue.FreeTensor(duLocal);
        dxQue.FreeTensor(dxLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> dOutQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> uQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> varQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> duQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxQue;

    TQue<QuePosition::VECIN, BUFFER_NUM> dOutcastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xCastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> uCastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightCastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasCastQue;

    TQue<QuePosition::VECOUT, BUFFER_NUM> duCastQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxCastQue;

    TBuf<TPosition::VECCALC> normBuf;
    TBuf<TPosition::VECCALC> stdBuf;
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
    LocalTensor<float> stdLocal;

    NormMultiplyDropoutBackwardTilingData tiling;

    DataCopyPadExtParams<float> padParams{true, 0, 0, 0};

    int32_t bigCoreNum;
    int32_t littleCoreNum;
    int32_t avgCoreCalcRows;
    int32_t xRowCount;
    int32_t xColCount;
    int32_t singleBlockRows;
    int32_t useCoreNum;
    float eps;
    float dropoutRatio;

    int32_t loop;
    int32_t tail;
    int32_t coreOffset;
    int32_t coreCalcRows;
    int32_t reduceElements = REDUCE_CALC_BYTE / sizeof(float);
    // 每个 data block 32字节，对应float个数
    int32_t oneBLockElements = 32 / sizeof(float);
    AscendC::DropOutShapeInfo info;
};
}  // namespace kernels

template<bool isNeedDrop>
__global__ __aicore__ void norm_multiply_dropout_backward(GM_ADDR dOut, GM_ADDR x, GM_ADDR u, GM_ADDR weight,
                                                          GM_ADDR bias, GM_ADDR mean, GM_ADDR var, GM_ADDR mask,
                                                          GM_ADDR dU, GM_ADDR dX, GM_ADDR dWeight, GM_ADDR dBias,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GET_TILING_DATA(tilingData, tiling);
    kernels::NormMultiplyDropoutBackward<DTYPE_X, isNeedDrop> op;
    op.init(dOut, x, u, weight, bias, mean, var, mask, dU, dX, dWeight, dBias, workspace, tilingData);
    op.process();
}

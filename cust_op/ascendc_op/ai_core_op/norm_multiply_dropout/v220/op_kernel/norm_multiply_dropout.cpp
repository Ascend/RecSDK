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
constexpr int32_t REDUCE_CALC_BYTE = 256;

template <typename T>
class NormMultiplyDropout {
public:
    __aicore__ NormMultiplyDropout(){};

    __aicore__ inline void init(GM_ADDR xInput, GM_ADDR uInput, GM_ADDR weight, GM_ADDR bias, GM_ADDR mask,
                                GM_ADDR output, GM_ADDR mean, GM_ADDR var, GM_ADDR workspace,
                                const NormMultiplyDropoutTilingData& tiling)
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
        dropTmpMem = tiling.dropTmpMem;
        eps = tiling.eps;
        dropoutRatio = tiling.dropoutRatio;

        if (bigCoreNum == 0) {
            coreOffset = coreId * avgCoreCalcRows;
            coreCalcRows = avgCoreCalcRows;
        } else if (coreId < bigCoreNum && bigCoreNum != 0) {
            coreOffset = coreId * (avgCoreCalcRows + 1);
            coreCalcRows = avgCoreCalcRows + 1;
        } else {
            coreOffset = bigCoreNum * (avgCoreCalcRows + 1) + (coreId - bigCoreNum) * avgCoreCalcRows;
            coreCalcRows = avgCoreCalcRows;
        }

        tail = coreCalcRows % singleBlockRows;
        loop = coreCalcRows / singleBlockRows;

        xGM.SetGlobalBuffer((__gm__ T*)xInput);
        uGM.SetGlobalBuffer((__gm__ T*)uInput);
        weightGM.SetGlobalBuffer((__gm__ T*)weight);
        biasGM.SetGlobalBuffer((__gm__ T*)bias);
        maskGM.SetGlobalBuffer((__gm__ uint8_t*)mask);
        outputGM.SetGlobalBuffer((__gm__ T*)output);
        meanGM.SetGlobalBuffer((__gm__ float*)mean);
        varGM.SetGlobalBuffer((__gm__ float*)var);

        pipe.InitBuffer(xcastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));       //  n * 512 * 2
        pipe.InitBuffer(ucastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));       // n * 512 * 2
        pipe.InitBuffer(weightcastQUE, NO_DOUBLE, xColCount * sizeof(T));                     // 1k
        pipe.InitBuffer(biascastQUE, NO_DOUBLE, xColCount * sizeof(T));                       // 1k

        pipe.InitBuffer(xQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));       //  n * 512 * 4
        pipe.InitBuffer(uQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));       // n * 512 *4
        pipe.InitBuffer(weightQUE, NO_DOUBLE, xColCount * sizeof(float));                     // 2k
        pipe.InitBuffer(biasQUE, NO_DOUBLE, xColCount * sizeof(float));                       // 2k
        pipe.InitBuffer(maskQUE, BUFFER_NUM,
                        singleBlockRows * xColCount / reduceRepeatStride * sizeof(uint8_t));  // n * 64
        pipe.InitBuffer(outputQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));  // n * 512 * 4
        pipe.InitBuffer(meanQUE, BUFFER_NUM, singleBlockRows * sizeof(float));                // n * 4
        pipe.InitBuffer(varQUE, BUFFER_NUM, singleBlockRows * sizeof(float));                 // n * 4

        pipe.InitBuffer(outputcastQUE, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));  // n * 512 * 4
        pipe.InitBuffer(meancastQUE, BUFFER_NUM, singleBlockRows * sizeof(T));                // n * 4
        pipe.InitBuffer(varcastQUE, BUFFER_NUM, singleBlockRows * sizeof(T));                 // n * 4

        pipe.InitBuffer(rstdBuf, singleBlockRows * sizeof(float));                            // n * 4
        pipe.InitBuffer(normBuf, singleBlockRows * xColCount * sizeof(float));                // n * 512 *4
        pipe.InitBuffer(ln_outBuf, singleBlockRows * xColCount * sizeof(float));              // n * 512 *4

        pipe.InitBuffer(meanBuf, (singleBlockRows * xColCount / (REDUCE_CALC_BYTE / sizeof(float))) * sizeof(float));
        pipe.InitBuffer(squareBuf, (singleBlockRows * xColCount / (REDUCE_CALC_BYTE / sizeof(float))) * sizeof(float));
        pipe.InitBuffer(dropBuf, singleBlockRows * dropTmpMem);  // 20k

        rstdLocal = rstdBuf.Get<float>();
        normLocal = normBuf.Get<float>();
        lnOutLocal = ln_outBuf.Get<float>();
        meanTemp0 = meanBuf.Get<float>();
        squareTemp0 = squareBuf.Get<float>();
        dropTemp = dropBuf.Get<uint8_t>();
    }

    __aicore__ inline void process()
    {
        int32_t coreId = GetBlockIdx();
        loop *= BUFFER_NUM;
        for (int32_t blockId = 0; blockId < loop; blockId++) {
            CopyIn(blockId, singleBlockRows);
            Compute(blockId, singleBlockRows);
            CopyOut(blockId, singleBlockRows);
        }
        if (tail != 0) {
            CopyIn(loop, tail);
            Compute(loop, tail);
            CopyOut(loop, tail);
        }
    }

    __aicore__ inline void CopyIn(int32_t blockId, int32_t calcRows)
    {
        xLocal = xQUE.AllocTensor<float>();
        uLocal = uQUE.AllocTensor<float>();
        weightLocal = weightQUE.AllocTensor<float>();
        biasLocal = biasQUE.AllocTensor<float>();
        maskLocal = maskQUE.AllocTensor<uint8_t>();

        xCast = xcastQUE.AllocTensor<T>();
        uCast = ucastQUE.AllocTensor<T>();
        weightCast = weightcastQUE.AllocTensor<T>();
        biasCast = biascastQUE.AllocTensor<T>();

        DataCopy(xCast, xGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(uCast, uGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(weightCast, weightGM, xColCount);
        DataCopy(biasCast, biasGM, xColCount);
        DataCopy(maskLocal, maskGM[(coreOffset + blockId * singleBlockRows) * xColCount / reduceRepeatStride],
                 calcRows * xColCount / reduceRepeatStride);

        xcastQUE.EnQue(xCast);
        ucastQUE.EnQue(uCast);
        weightcastQUE.EnQue(weightCast);
        biascastQUE.EnQue(biasCast);
        xCast = xcastQUE.DeQue<T>();
        uCast = ucastQUE.DeQue<T>();
        weightCast = weightcastQUE.DeQue<T>();
        biasCast = biascastQUE.DeQue<T>();

        Cast<float, T>(xLocal, xCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(uLocal, uCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(biasLocal, biasCast, RoundMode::CAST_NONE, xColCount);
        Cast<float, T>(weightLocal, weightCast, RoundMode::CAST_NONE, xColCount);

        xcastQUE.FreeTensor(xCast);
        ucastQUE.FreeTensor(uCast);
        weightcastQUE.FreeTensor(weightCast);
        biascastQUE.FreeTensor(biasCast);

        xQUE.EnQue(xLocal);
        uQUE.EnQue(uLocal);
        weightQUE.EnQue(weightLocal);
        biasQUE.EnQue(biasLocal);
        maskQUE.EnQue(maskLocal);
    }

    __aicore__ inline void Compute(int32_t blockId, int32_t calcRows)
    {
        xLocal = xQUE.DeQue<float>();
        uLocal = uQUE.DeQue<float>();
        weightLocal = weightQUE.DeQue<float>();
        biasLocal = biasQUE.DeQue<float>();
        maskLocal = maskQUE.DeQue<uint8_t>();

        outputLocal = outputQUE.AllocTensor<float>();
        meanLocal = meanQUE.AllocTensor<float>();
        varLocal = varQUE.AllocTensor<float>();

        // mean
        WholeReduceSum<float>(meanTemp0, xLocal, wholeReduceSumMask, calcRows * xColCount / wholeReduceSumMask, 1, 1,
                              reduceRepeatStride);
        WholeReduceSum<float>(meanLocal, meanTemp0, xColCount / wholeReduceSumMask, calcRows, 1, 1, 1);
        Muls<float>(meanLocal, meanLocal, positiveOneFloat / xColCount, calcRows);
        pipe_barrier(PIPE_ALL);

        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calcRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(meanGM[(coreOffset + blockId * singleBlockRows)], meanLocal, copyParams);
        pipe_barrier(PIPE_ALL);

        // variance
        Muls<float>(meanLocal, meanLocal, static_cast<float>(-1), calcRows);
        for (int32_t i = 0; i < calcRows; i++) {
            Adds(xLocal[i * xColCount], xLocal[i * xColCount], meanLocal.GetValue(i), xColCount);
        }
        Mul<float>(outputLocal, xLocal, xLocal, calcRows * xColCount);
        WholeReduceSum<float>(squareTemp0, outputLocal, wholeReduceSumMask, calcRows * xColCount / wholeReduceSumMask,
                              1, 1, reduceRepeatStride);
        WholeReduceSum<float>(varLocal, squareTemp0, xColCount / wholeReduceSumMask, calcRows, 1, 1, 1);
        Muls<float>(varLocal, varLocal, positiveOneFloat / xColCount, calcRows);
        pipe_barrier(PIPE_ALL);

        DataCopyPad(varGM[(coreOffset + blockId * singleBlockRows)], varLocal, copyParams);
        pipe_barrier(PIPE_ALL);
        Adds<float>(varLocal, varLocal, eps, calcRows);

        // norm
        Sqrt(rstdLocal, varLocal, calcRows);
        for (int32_t i = 0; i < calcRows; i++) {
            Muls<float>(normLocal[i * xColCount], xLocal[i * xColCount], positiveOneFloat / rstdLocal.GetValue(i),
                        xColCount);
        }
        // scale+shift
        for (int32_t i = 0; i < calcRows; i++) {
            Mul<float>(xLocal[i * xColCount], normLocal[i * xColCount], weightLocal, xColCount);
            Add<float>(xLocal[i * xColCount], xLocal[i * xColCount], biasLocal, xColCount);
        }

        //----------------------layernorm end--------------------------------//
        Mul<float>(outputLocal, xLocal, uLocal, calcRows * xColCount);
        AscendC::DropOutShapeInfo info;
        float probValue = 1 - dropoutRatio;
        info.firstAxis = calcRows;
        info.srcLastAxis = xColCount;
        info.maskLastAxis = xColCount / reduceRepeatStride;

        pipe_barrier(PIPE_V);
        DropOut<float, false, 3>(outputLocal, outputLocal, maskLocal, dropTemp, probValue, info);
        outputQUE.EnQue(outputLocal);

        xQUE.FreeTensor(xLocal);
        uQUE.FreeTensor(uLocal);
        weightQUE.FreeTensor(weightLocal);
        biasQUE.FreeTensor(biasLocal);
        maskQUE.FreeTensor(maskLocal);
    }

    __aicore__ inline void CopyOut(int32_t blockId, int32_t calcRows)
    {
        outputLocal = outputQUE.DeQue<float>();
        output_cast = outputcastQUE.AllocTensor<T>();

        Cast<T, float>(output_cast, outputLocal, RoundMode::CAST_RINT, calcRows * xColCount);
        outputcastQUE.EnQue(output_cast);
        output_cast = outputcastQUE.DeQue<T>();

        DataCopy(outputGM[(coreOffset + blockId * singleBlockRows) * xColCount], output_cast, calcRows * xColCount);

        outputcastQUE.FreeTensor(output_cast);
        outputQUE.FreeTensor(outputLocal);
        meanQUE.FreeTensor(meanLocal);
        varQUE.FreeTensor(varLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> xcastQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> ucastQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightcastQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> biascastQUE;

    TQue<QuePosition::VECIN, BUFFER_NUM> xQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> uQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQUE;
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQUE;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQUE;
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQUE;
    TQue<QuePosition::VECOUT, BUFFER_NUM> varQUE;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outputcastQUE;
    TQue<QuePosition::VECOUT, BUFFER_NUM> meancastQUE;
    TQue<QuePosition::VECOUT, BUFFER_NUM> varcastQUE;

    TBuf<TPosition::VECCALC> rstdBuf;
    TBuf<TPosition::VECCALC> normBuf;
    TBuf<TPosition::VECCALC> ln_outBuf;

    TBuf<TPosition::VECCALC> meanBuf;
    TBuf<TPosition::VECCALC> squareBuf;
    TBuf<TPosition::VECCALC> dropBuf;

    GlobalTensor<T> xGM;
    GlobalTensor<T> uGM;
    GlobalTensor<T> weightGM;
    GlobalTensor<T> biasGM;
    GlobalTensor<uint8_t> maskGM;
    GlobalTensor<T> outputGM;
    GlobalTensor<float> meanGM;
    GlobalTensor<float> varGM;

    LocalTensor<float> xLocal;
    LocalTensor<float> uLocal;
    LocalTensor<float> weightLocal;
    LocalTensor<float> biasLocal;
    LocalTensor<uint8_t> maskLocal;
    LocalTensor<float> outputLocal;
    LocalTensor<float> meanLocal;
    LocalTensor<float> varLocal;
    LocalTensor<float> rstdLocal;
    LocalTensor<float> normLocal;
    LocalTensor<float> lnOutLocal;

    LocalTensor<T> xCast;
    LocalTensor<T> uCast;
    LocalTensor<T> weightCast;
    LocalTensor<T> biasCast;
    LocalTensor<T> output_cast;

    LocalTensor<float> meanTemp0;
    LocalTensor<float> squareTemp0;
    LocalTensor<uint8_t> dropTemp;

    NormMultiplyDropoutTilingData tiling;

    int32_t bigCoreNum;
    int32_t littleCoreNum;
    int32_t avgCoreCalcRows;
    int32_t xRowCount;
    int32_t xColCount;
    int32_t singleBlockRows;
    int32_t useCoreNum;
    int32_t dropTmpMem;
    float eps;
    float dropoutRatio;

    int32_t loop;
    int32_t tail;
    int32_t coreOffset;
    int32_t coreCalcRows;

    int32_t wholeReduceSumMask = REDUCE_CALC_BYTE / sizeof(float);
    int32_t reduceRepeatStride = 8;
    float positiveOneFloat = 1.0;
    float negativeOneFloat = -1.0;

    AscendC::DropOutShapeInfo info;
};
}  // namespace kernels

extern "C" __global__ __aicore__ void norm_multiply_dropout(GM_ADDR xInput, GM_ADDR uInput, GM_ADDR weight,
                                                            GM_ADDR bias, GM_ADDR mask, GM_ADDR output, GM_ADDR mean,
                                                            GM_ADDR var, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    kernels::NormMultiplyDropout<DTYPE_X> op;
    op.init(xInput, uInput, weight, bias, mask, output, mean, var, workspace, tiling_data);
    op.process();
}

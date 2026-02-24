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
#include "norm_multiply_dropout_tilingKey.h"

using namespace AscendC;

namespace kernels {
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NO_DOUBLE = 1;
constexpr int32_t REDUCE_CALC_BYTES = 256;
constexpr uint32_t VEC_BLOCK_BYTES = 32;
constexpr uint32_t DROPOUT_MODE = 3;

template <typename T, bool isNeedDrop>
class NormMultiplyDropout {
public:
    __aicore__ NormMultiplyDropout(){};

    __aicore__ inline void init(GM_ADDR x, GM_ADDR u, GM_ADDR weight, GM_ADDR bias, GM_ADDR mask,
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

        xGM.SetGlobalBuffer((__gm__ T*)x);
        uGM.SetGlobalBuffer((__gm__ T*)u);
        weightGM.SetGlobalBuffer((__gm__ T*)weight);
        biasGM.SetGlobalBuffer((__gm__ T*)bias);
        if constexpr (isNeedDrop) {
            maskGM.SetGlobalBuffer((__gm__ uint8_t*)mask);
        }
        outputGM.SetGlobalBuffer((__gm__ T*)output);
        meanGM.SetGlobalBuffer((__gm__ float*)mean);
        varGM.SetGlobalBuffer((__gm__ float*)var);

        if constexpr (!std::is_same<T, float>::value) {
            // 输入数据原始数据类型
            pipe.InitBuffer(xCastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));       // n * C * 2
            pipe.InitBuffer(uCastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));       // n * C * 2
            pipe.InitBuffer(weightCastQue, NO_DOUBLE, xColCount * sizeof(T));                     // C * 2
            pipe.InitBuffer(biasCastQue, NO_DOUBLE, xColCount * sizeof(T));                       // C * 2
            // 输出数据转换为原始输入类型
            pipe.InitBuffer(outputCastQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(T));  // n * C * 2
        }

        // 输入数据float类型
        pipe.InitBuffer(xQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));       // n * C * 4
        pipe.InitBuffer(uQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));       // n * C * 4
        pipe.InitBuffer(weightQue, NO_DOUBLE, xColCount * sizeof(float));                     // C * 4
        pipe.InitBuffer(biasQue, NO_DOUBLE, xColCount * sizeof(float));                       // C * 4
        if constexpr (isNeedDrop) {
            pipe.InitBuffer(maskQue, BUFFER_NUM,
                            singleBlockRows * xColCount / reduceRepeatStride * sizeof(uint8_t));  // n * C / 8
        }

        // 输出数据float类型
        pipe.InitBuffer(outputQue, BUFFER_NUM, singleBlockRows * xColCount * sizeof(float));  // n * C * 4
        pipe.InitBuffer(meanQue, BUFFER_NUM, singleBlockRows * sizeof(float));                // n * 4
        pipe.InitBuffer(varQue, BUFFER_NUM, singleBlockRows * sizeof(float));                 // n * 4

        pipe.InitBuffer(stdBuf, singleBlockRows * sizeof(float));                             // n * 4
        pipe.InitBuffer(normBuf, singleBlockRows * xColCount * sizeof(float));                // n * C * 4
        pipe.InitBuffer(lnOutBuf, singleBlockRows * xColCount * sizeof(float));               // n * C * 4

        // n * C / 64 第一次规约后的数据个数
        pipe.InitBuffer(meanBuf, (singleBlockRows * xColCount / (REDUCE_CALC_BYTES / sizeof(float))) * sizeof(float));
        pipe.InitBuffer(dropBuf, singleBlockRows * dropTmpMem);  // 20k

        stdLocal = stdBuf.Get<float>();
        normLocal = normBuf.Get<float>();
        lnOutLocal = lnOutBuf.Get<float>();
        meanTemp0 = meanBuf.Get<float>();
        dropTemp = dropBuf.Get<uint8_t>();
    }

    __aicore__ inline void process()
    {
        int32_t coreId = GetBlockIdx();
        loop *= BUFFER_NUM;
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
    }

    __aicore__ inline void CopyInNoCast(int32_t blockId, int32_t calcRows)
    {
        if constexpr (isNeedDrop) {
            maskLocal = maskQue.AllocTensor<uint8_t>();
            DataCopy(maskLocal, maskGM[(coreOffset + blockId * singleBlockRows) * xColCount / oneMask2FloatNum],
                     calcRows * xColCount / oneMask2FloatNum);
        }

        xLocal = xQue.AllocTensor<T>();
        uLocal = uQue.AllocTensor<T>();
        weightLocal = weightQue.AllocTensor<T>();
        biasLocal = biasQue.AllocTensor<T>();
        DataCopy(xLocal, xGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(uLocal, uGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(weightLocal, weightGM, xColCount);
        DataCopy(biasLocal, biasGM, xColCount);

        xQue.EnQue(xLocal);
        uQue.EnQue(uLocal);
        weightQue.EnQue(weightLocal);
        biasQue.EnQue(biasLocal);
        if constexpr (isNeedDrop) {
            maskQue.EnQue(maskLocal);
        }
    }

    __aicore__ inline void CopyInWithCast(int32_t blockId, int32_t calcRows)
    {
        xCast = xCastQue.AllocTensor<T>();
        uCast = uCastQue.AllocTensor<T>();
        weightCast = weightCastQue.AllocTensor<T>();
        biasCast = biasCastQue.AllocTensor<T>();
        if constexpr (isNeedDrop) {
            maskLocal = maskQue.AllocTensor<uint8_t>();
        }

        DataCopy(xCast, xGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(uCast, uGM[(coreOffset + blockId * singleBlockRows) * xColCount], calcRows * xColCount);
        DataCopy(weightCast, weightGM, xColCount);
        DataCopy(biasCast, biasGM, xColCount);
        if constexpr (isNeedDrop) {
            DataCopy(maskLocal, maskGM[(coreOffset + blockId * singleBlockRows) * xColCount / oneMask2FloatNum],
                     calcRows * xColCount / oneMask2FloatNum);
        }

        xCastQue.EnQue(xCast);
        uCastQue.EnQue(uCast);
        weightCastQue.EnQue(weightCast);
        biasCastQue.EnQue(biasCast);

        xCast = xCastQue.DeQue<T>();
        uCast = uCastQue.DeQue<T>();
        weightCast = weightCastQue.DeQue<T>();
        biasCast = biasCastQue.DeQue<T>();

        xLocal = xQue.AllocTensor<float>();
        uLocal = uQue.AllocTensor<float>();
        weightLocal = weightQue.AllocTensor<float>();
        biasLocal = biasQue.AllocTensor<float>();

        Cast<float, T>(xLocal, xCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(uLocal, uCast, RoundMode::CAST_NONE, calcRows * xColCount);
        Cast<float, T>(weightLocal, weightCast, RoundMode::CAST_NONE, xColCount);
        Cast<float, T>(biasLocal, biasCast, RoundMode::CAST_NONE, xColCount);

        xQue.EnQue(xLocal);
        uQue.EnQue(uLocal);
        weightQue.EnQue(weightLocal);
        biasQue.EnQue(biasLocal);
        if constexpr (isNeedDrop) {
            maskQue.EnQue(maskLocal);
        }

        xCastQue.FreeTensor(xCast);
        uCastQue.FreeTensor(uCast);
        weightCastQue.FreeTensor(weightCast);
        biasCastQue.FreeTensor(biasCast);
    }

    __aicore__ inline void Compute(int32_t blockId, int32_t calcRows)
    {
        xLocal = xQue.DeQue<float>();
        uLocal = uQue.DeQue<float>();
        weightLocal = weightQue.DeQue<float>();
        biasLocal = biasQue.DeQue<float>();
        if constexpr (isNeedDrop) {
            maskLocal = maskQue.DeQue<uint8_t>();
        }

        outputLocal = outputQue.AllocTensor<float>();
        meanLocal = meanQue.AllocTensor<float>();
        varLocal = varQue.AllocTensor<float>();

        // mean
        WholeReduceSum<float>(meanTemp0, xLocal, reduceElements, calcRows * xColCount / reduceElements, 1, 1,
                              reduceElements / oneBLockElements);
        WholeReduceSum<float>(meanLocal, meanTemp0, xColCount / reduceElements, calcRows, 1, 1,
                              xColCount / reduceElements / oneBLockElements);
        Muls<float>(meanLocal, meanLocal, positiveOneFloat / xColCount, calcRows);
        pipe_barrier(PIPE_ALL);

        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calcRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(meanGM[(coreOffset + blockId * singleBlockRows)], meanLocal, copyParams);
        pipe_barrier(PIPE_ALL);

        // variance
        //   pow(x - mean)
        Muls<float>(meanLocal, meanLocal, negativeOneFloat, calcRows);
        for (int32_t i = 0; i < calcRows; i++) {
            Adds(xLocal[i * xColCount], xLocal[i * xColCount], meanLocal.GetValue(i), xColCount);
        }
        Mul<float>(outputLocal, xLocal, xLocal, calcRows * xColCount);
        //  reduce_sum(pow(x - mean)) / N
        WholeReduceSum<float>(meanTemp0, outputLocal, reduceElements, calcRows * xColCount / reduceElements,
                              1, 1, reduceElements / oneBLockElements);
        WholeReduceSum<float>(varLocal, meanTemp0, xColCount / reduceElements, calcRows, 1, 1,
                              xColCount / reduceElements / oneBLockElements);
        Muls<float>(varLocal, varLocal, positiveOneFloat / xColCount, calcRows);
        pipe_barrier(PIPE_ALL);

        DataCopyPad(varGM[(coreOffset + blockId * singleBlockRows)], varLocal, copyParams);
        pipe_barrier(PIPE_ALL);

        // norm
        // std = sqrt(var + eps)
        Adds<float>(varLocal, varLocal, eps, calcRows);
        Sqrt(stdLocal, varLocal, calcRows);
        // x_hat = (x - mean) / std
        for (int32_t i = 0; i < calcRows; i++) {
            Muls<float>(normLocal[i * xColCount], xLocal[i * xColCount], positiveOneFloat / stdLocal.GetValue(i),
                        xColCount);
        }
        // scale+shift: ln_out = x_hat * weight + bias
        for (int32_t i = 0; i < calcRows; i++) {
            Mul<float>(xLocal[i * xColCount], normLocal[i * xColCount], weightLocal, xColCount);
            Add<float>(xLocal[i * xColCount], xLocal[i * xColCount], biasLocal, xColCount);
        }
        //----------------------layer_norm end--------------------------------//

        // scaled = ln_out * u
        Mul<float>(outputLocal, xLocal, uLocal, calcRows * xColCount);
        AscendC::DropOutShapeInfo info;
        float probValue = 1 - dropoutRatio;
        info.firstAxis = calcRows;
        info.srcLastAxis = xColCount;
        info.maskLastAxis = xColCount / oneMask2FloatNum;

        pipe_barrier(PIPE_V);
        if constexpr (isNeedDrop) {
            // dropout
            DropOut<float, false, DROPOUT_MODE>(outputLocal, outputLocal, maskLocal, dropTemp, probValue, info);
        }
        outputQue.EnQue(outputLocal);

        xQue.FreeTensor(xLocal);
        uQue.FreeTensor(uLocal);
        weightQue.FreeTensor(weightLocal);
        biasQue.FreeTensor(biasLocal);
        if constexpr (isNeedDrop) {
            maskQue.FreeTensor(maskLocal);
        }
    }

    __aicore__ inline void CopyOutNoCast(int32_t blockId, int32_t calcRows)
    {
        outputLocal = outputQue.DeQue<T>();
        DataCopy(outputGM[(coreOffset + blockId * singleBlockRows) * xColCount], outputLocal, calcRows * xColCount);

        outputQue.FreeTensor(outputLocal);
        meanQue.FreeTensor(meanLocal);
        varQue.FreeTensor(varLocal);
    }

    __aicore__ inline void CopyOutWithCast(int32_t blockId, int32_t calcRows)
    {
        outputLocal = outputQue.DeQue<float>();
        outputCast = outputCastQue.AllocTensor<T>();

        Cast<T, float>(outputCast, outputLocal, RoundMode::CAST_RINT, calcRows * xColCount);
        outputCastQue.EnQue(outputCast);
        outputCast = outputCastQue.DeQue<T>();

        DataCopy(outputGM[(coreOffset + blockId * singleBlockRows) * xColCount], outputCast, calcRows * xColCount);

        outputCastQue.FreeTensor(outputCast);
        outputQue.FreeTensor(outputLocal);
        meanQue.FreeTensor(meanLocal);
        varQue.FreeTensor(varLocal);
    }

private:
    TPipe pipe;
    // 输入参数
    TQue<QuePosition::VECIN, BUFFER_NUM> xQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> uQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQue;
    // 输入参数类型转换
    TQue<QuePosition::VECIN, BUFFER_NUM> xCastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> uCastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightCastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasCastQue;
    // 输出参数
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> varQue;
    // 输出参数类型转换
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputCastQue;
    // 临时变量
    TBuf<TPosition::VECCALC> meanBuf; // 计算均值时存放第一次规约化后的结果
    TBuf<TPosition::VECCALC> stdBuf; // 归一化计算中的标准差
    TBuf<TPosition::VECCALC> normBuf; // 归一化计算结果
    TBuf<TPosition::VECCALC> lnOutBuf; // 仿射变换后的归一化结果
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
    LocalTensor<float> stdLocal;
    LocalTensor<float> normLocal;
    LocalTensor<float> lnOutLocal;

    LocalTensor<T> xCast;
    LocalTensor<T> uCast;
    LocalTensor<T> weightCast;
    LocalTensor<T> biasCast;
    LocalTensor<T> outputCast;

    LocalTensor<float> meanTemp0;
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

    // 规约操作一次迭代计算的元素个数 64
    int32_t reduceElements = REDUCE_CALC_BYTES / sizeof(float);
    // 规约操作一次迭代计算元素对应的block数
    int32_t reduceRepeatStride = 8;
    int32_t oneMask2FloatNum = 8;
    // 每个block 32字节，对应8个float
    int32_t oneBLockElements = 32 / sizeof(float);
    float positiveOneFloat = 1.0;
    float negativeOneFloat = -1.0;

    AscendC::DropOutShapeInfo info;
};
}  // namespace kernels

template<bool isNeedDrop>
__global__ __aicore__ void norm_multiply_dropout(GM_ADDR x, GM_ADDR u, GM_ADDR weight, GM_ADDR bias, GM_ADDR mask,
                                                 GM_ADDR output, GM_ADDR mean, GM_ADDR var, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    kernels::NormMultiplyDropout<DTYPE_X, isNeedDrop> op;
    op.init(x, u, weight, bias, mask, output, mean, var, workspace, tiling_data);
    op.process();
}

/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

template <typename T>
class LazyAdam {
public:
    __aicore__ inline LazyAdam() {}

    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR gradient, GM_ADDR indices, GM_ADDR inputM, GM_ADDR inputV, GM_ADDR inputVar,
                                GM_ADDR lr, GM_ADDR inputMRef, GM_ADDR inputVRef, GM_ADDR inputVarRef, float beta1,
                                float beta2, float epsilon, int32_t dim0, int32_t dim1, int32_t dim2, int32_t row,
                                int32_t indicesAllocSize, int32_t otherAllocSize, int32_t batch, int32_t loopCount,
                                int32_t rowLeft, int32_t loopCountTail, int32_t rowLeftTail, int32_t coreNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        // 属性赋值
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->epsilon = epsilon;
        // tiling 数据赋值
        this->dim0 = dim0;
        this->dim1 = dim1;
        this->dim2 = dim2;
        this->row = row;
        this->batch = batch;
        this->loopCount = loopCount;
        this->rowLeft = rowLeft;
        this->loopCountTail = loopCountTail;
        this->rowLeftTail = rowLeftTail;
        this->coreNum = coreNum;
        // 输入的 gm shape 大小
        int32_t shape = this->dim0 * this->dim2;
        int32_t shapeIndices = this->dim1 * 1;
        int32_t shapeGradient = this->dim1 * this->dim2;
        this->gmGradient.SetGlobalBuffer((__gm__ T*)gradient + this->batch * this->dim2 * get_block_idx(),
                                         shapeGradient);
        this->gmIndices.SetGlobalBuffer((__gm__ int32_t*)indices + this->batch * get_block_idx(), shapeIndices);

        this->gmInputM.SetGlobalBuffer((__gm__ T*)inputM, shape);
        this->gmInputV.SetGlobalBuffer((__gm__ T*)inputV, shape);
        this->gmInputVar.SetGlobalBuffer((__gm__ T*)inputVar, shape);

        this->gmLearningRate.SetGlobalBuffer((__gm__ T*)lr, sizeof(float));
        this->lr = this->gmLearningRate.GetValue(0);

        // 将输出地址指向输入地址
        inputMRef = inputM;
        inputVRef = inputV;
        inputVarRef = inputVar;

        // 单次循环申请的 ub 大小, 32位对齐后的大小
        this->pipe.InitBuffer(this->inQueGradient, 1, otherAllocSize);
        this->pipe.InitBuffer(this->inQueIndices, 1, indicesAllocSize);
        this->pipe.InitBuffer(this->queMSlice, 1, otherAllocSize);
        this->pipe.InitBuffer(this->queVSlice, 1, otherAllocSize);
        this->pipe.InitBuffer(this->queVarSlice, 1, otherAllocSize);

        this->pipe.InitBuffer(this->calcBufM, otherAllocSize);
        this->updateM = this->calcBufM.template Get<T>();

        this->pipe.InitBuffer(this->calcBufV, otherAllocSize);
        this->updateV = this->calcBufV.template Get<T>();

        this->pipe.InitBuffer(this->calcBufVar, otherAllocSize);
        this->updateVar = this->calcBufVar.template Get<T>();

        this->pipe.InitBuffer(this->calcBuf, otherAllocSize);
        this->temp = this->calcBuf.template Get<T>();
    }

    // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
    __aicore__ inline void Process()
    {
        if (get_block_idx() == this->coreNum - 1) {
            for (int32_t i = 0; i < this->loopCountTail; i++) {
                CopyIn(i, this->row);
                Compute(i, this->row);
            }
            // 尾块处理
            if (this->rowLeft > 0) {
                CopyIn(this->loopCountTail, this->rowLeftTail);
                Compute(this->loopCountTail, this->rowLeftTail);
            }
        } else {
            for (int32_t i = 0; i < this->loopCount; i++) {
                CopyIn(i, this->row);
                Compute(i, this->row);
            }
            // 尾块处理
            if (this->rowLeft > 0) {
                CopyIn(this->loopCount, this->rowLeft);
                Compute(this->loopCount, this->rowLeft);
            }
        }
    }

private:
    // 搬入函数，完成CopyIn阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyIn(int32_t progress, int32_t row)
    {
        LocalTensor<T> localGradient = this->inQueGradient.template AllocTensor<T>();
        uint32_t gradientDataLen = row * this->dim2 * sizeof(T);
        // 连续传输数据块个数；len:连续传输数据块长度，Byte，非对齐搬运；0, 0, 0:源/目标数据块间隔，保留字段
        DataCopyExtParams gradientParams{1, gradientDataLen, 0, 0, 0};
        // 搬运填充参数
        DataCopyPadExtParams<T> gradientPadParams{true, 0, 2, 0};
        DataCopyPad(localGradient, this->gmGradient[progress * this->row * this->dim2], gradientParams,
                    gradientPadParams);

        LocalTensor<int32_t> localIndices = this->inQueIndices.template AllocTensor<int32_t>();
        uint32_t indicesDataLen = row * sizeof(int32_t);
        DataCopyExtParams indicesParams{1, indicesDataLen, 0, 0, 0};
        DataCopyPadExtParams<int32_t> indicesPadParams{true, 0, 2, 0};
        DataCopyPad(localIndices, this->gmIndices[progress * this->row], indicesParams, indicesPadParams);

        this->inQueGradient.EnQue(localGradient);
        this->inQueIndices.EnQue(localIndices);
    }

    // 计算函数，完成Compute阶段的处理，被核心Process函数调用
    __aicore__ inline void Compute(int32_t progress, int32_t row)
    {
        LocalTensor<T> localGradient = this->inQueGradient.template DeQue<T>();
        LocalTensor<int32_t> localIndices = this->inQueIndices.template DeQue<int32_t>();
        Muls(localIndices, localIndices, this->dim2, row);
        // 根据 indices 从 inputM 中切分出来 m_slice
        LocalTensor<T> localMSlice = this->queMSlice.template AllocTensor<T>();
        LocalTensor<T> localVSlice = this->queVSlice.template AllocTensor<T>();
        LocalTensor<T> localVarSlice = this->queVarSlice.template AllocTensor<T>();

        pipe_barrier(PIPE_ALL);

        int32_t index = 0;
        for (int32_t i = 0; i < row; i++) {
            index = localIndices.GetValue(i);
            if (index >= 0) {
                DataCopy(localMSlice[i * this->dim2], gmInputM[index], this->dim2);
                DataCopy(localVSlice[i * this->dim2], gmInputV[index], this->dim2);
                DataCopy(localVarSlice[i * this->dim2], gmInputVar[index], this->dim2);
            }
        }

        this->queMSlice.EnQue(localMSlice);
        this->queVSlice.EnQue(localVSlice);
        this->queVarSlice.EnQue(localVarSlice);
        localMSlice = this->queMSlice.template DeQue<T>();
        localVSlice = this->queVSlice.template DeQue<T>();
        localVarSlice = this->queVarSlice.template DeQue<T>();

        // 计算M
        Muls(localMSlice, localMSlice, this->beta1, row * this->dim2);
        Muls(this->updateM, localGradient, (1 - this->beta1), row * this->dim2);
        this->updateM = localMSlice + this->updateM;

        // 计算V
        Muls(localVSlice, localVSlice, this->beta2, row * this->dim2);
        Mul(this->updateV, localGradient, localGradient, row * this->dim2);
        Muls(this->updateV, this->updateV, (1 - this->beta2), row * this->dim2);
        this->updateV = localVSlice + this->updateV;

        // 计算Var
        Sqrt(this->updateVar, this->updateV, row * this->dim2);
        Adds(this->updateVar, this->updateVar, this->epsilon, row * this->dim2);
        Muls(this->temp, this->updateM, -this->lr, row * this->dim2);
        Div(this->updateVar, this->temp, this->updateVar, row * this->dim2);
        Add(this->updateVar, this->updateVar, localVarSlice, row * this->dim2);

        pipe_barrier(PIPE_ALL);

        // 计算结果数据原地更新到输入tensor中
        for (int32_t i = 0; i < row; i++) {
            index = localIndices.GetValue(i);
            if (index >= 0) {
                // __GET_CODE_CHANNEL__宏的作用是防止拷贝操作被识别为matmul而报错
#ifndef __GET_CODE_CHANNEL__
                DataCopy(this->gmInputM[index], this->updateM[i * this->dim2], this->dim2);
                DataCopy(this->gmInputV[index], this->updateV[i * this->dim2], this->dim2);
                DataCopy(this->gmInputVar[index], this->updateVar[i * this->dim2], this->dim2);
#endif
            }
        }
        pipe_barrier(PIPE_ALL);

        this->inQueGradient.FreeTensor(localGradient);
        this->queMSlice.FreeTensor(localMSlice);
        this->queVSlice.FreeTensor(localVSlice);
        this->queVarSlice.FreeTensor(localVarSlice);
        this->inQueIndices.FreeTensor(localIndices);
    }

private:
    float lr, beta1, beta2, epsilon;
    int32_t dim0, dim1, dim2, row, batch, loopCount, rowLeft, loopCountTail, rowLeftTail, coreNum;
    LocalTensor<T> updateM, updateV, updateVar, temp;
    LocalTensor<int32_t> localIndices;
    GlobalTensor<T> gmGradient, gmInputM, gmInputV, gmInputVar;
    GlobalTensor<int32_t> gmIndices;
    GlobalTensor<T> gmLearningRate;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueGradient, inQueIndices;
    TQue<QuePosition::VECIN, 1> queMSlice, queVSlice, queVarSlice;
    TBuf<TPosition::VECCALC> calcBufM;
    TBuf<TPosition::VECCALC> calcBufV;
    TBuf<TPosition::VECCALC> calcBufVar;
    TBuf<TPosition::VECCALC> calcBuf;
};

extern "C" __global__ __aicore__ void lazy_adam(GM_ADDR gradient, GM_ADDR indices, GM_ADDR inputM, GM_ADDR inputV,
                                                GM_ADDR inputVar, GM_ADDR lr, GM_ADDR inputMRef, GM_ADDR inputVRef,
                                                GM_ADDR inputVarRef, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    LazyAdam<float> op32;
    op32.Init(gradient, indices, inputM, inputV, inputVar, lr, inputMRef, inputVRef, inputVarRef, tiling_data.beta1,
              tiling_data.beta2, tiling_data.epsilon, tiling_data.dim0, tiling_data.dim1, tiling_data.dim2,
              tiling_data.row, tiling_data.indicesAllocSize, tiling_data.otherAllocSize, tiling_data.batch,
              tiling_data.loopCount, tiling_data.rowLeft, tiling_data.loopCountTail, tiling_data.rowLeftTail,
              tiling_data.coreNum);
    op32.Process();
}
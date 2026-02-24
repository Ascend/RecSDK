/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#pragma once

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

namespace AdamWUpdateSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;

template <bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallInBlockDataCompute(
    __gm__ float* grads, __gm__ float* __gm__* valuesPtr, int32_t gradDim, int32_t inLength,
    float lr, float beta1, float beta2, float eps, float weightDecay, int32_t iterNum, int32_t gradDimShift)
{
    // 1. 线程信息计算
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

    int32_t blockBase = blockIdx * blockElementCapacity;

    if (blockBase >= inLength) {
        return;
    }

    int32_t elementsRemaining = inLength - blockBase;
    int32_t elementsThisBlock = (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
    if (elementsThisBlock <= 0) {
        return;
    }

    int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;
    if (threadElementBase >= inLength) {
        return;
    }

    int32_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
    int32_t elementsForThread =
        (elementsThisBlockRemaining > MAX_ELEMENTS_PER_THREAD) ? MAX_ELEMENTS_PER_THREAD : elementsThisBlockRemaining;

    // 当前梯度元素所属的嵌入向量（行）索引
    int rowIdx = isPowerOfTwo ? (threadElementBase >> gradDimShift) : (threadElementBase / gradDim);
    // 当前梯度元素所属的嵌入向量内的特征（列）索引
    int colIdx = isPowerOfTwo ? (threadElementBase & (gradDim - 1)) : (threadElementBase % gradDim);
    // 获取当前行的基地址
    __gm__ float* curRowBasePtr = reinterpret_cast<__gm__ float*>(valuesPtr[rowIdx]);
    // 部分中间变量置于循环外计算
    float oneMinusBeta1 = (1.0f - beta1);
    float oneMinusBeta2 = (1.0f - beta2);
    float mHatDenom = 1.0f - AscendC::Simt::Pow(beta1, (float)iterNum);
    float vHatDenom = 1.0f - AscendC::Simt::Pow(beta2, (float)iterNum);

    // 2. 实际计算
#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
        if (i >= elementsForThread) {
            return;
        }
        int32_t globalIdx = threadElementBase + i;
        if (globalIdx >= inLength) {
            return;
        }

        float tmpGrad = grads[globalIdx];

        int weightIdx = colIdx;
        int mIdx = weightIdx + gradDim;
        int vIdx = mIdx + gradDim;

        float tmpWeight = curRowBasePtr[weightIdx];
        float tmpM = curRowBasePtr[mIdx];
        float tmpV = curRowBasePtr[vIdx];

        // 计算新的动量
        float newM = beta1 * tmpM + oneMinusBeta1 * tmpGrad;
        // 计算新的方差
        float newV = beta2 * tmpV + oneMinusBeta2 * tmpGrad * tmpGrad;
        // 计算归一化后的动量和方差
        float mHat = newM / mHatDenom;
        float vHat = newV / vHatDenom;
        // 计算权重更新量
        float deltaW = lr * (mHat / (AscendC::Simt::Sqrt(vHat) + eps) + weightDecay * tmpWeight);
        // 更新权重
        curRowBasePtr[weightIdx] = tmpWeight - deltaW;
        // 更新动量和方差
        curRowBasePtr[mIdx] = newM;
        curRowBasePtr[vIdx] = newV;

        // --- 索引更新逻辑 ---
        colIdx++;
        if (colIdx == gradDim) {
            colIdx = 0;
            rowIdx++;
            // 只有当还有下一次循环且确实跨行时，才去全局内存加载新的行指针
            if (i < MAX_ELEMENTS_PER_THREAD - 1) {
                curRowBasePtr = reinterpret_cast<__gm__ float*>(valuesPtr[rowIdx]);
            }
        }
    }
}

template <bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ float* grads, __gm__ float* __gm__* valuesPtr, int32_t gradDim, int32_t inLength, float lr,
    float beta1, float beta2, float eps, float weightDecay, int32_t iterNum, int32_t totalBlocks,
    int32_t blockStartIdx, int32_t curBlocksCount, int32_t gradDimShift)
{
    // 1. 线程信息计算
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        if (globalBlockIdx >= totalBlocks) {
            break;
        }
        
        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= inLength) {
            break;
        }

        int32_t elementsRemaining = inLength - blockBase;
        int32_t elementsThisBlock = elementsRemaining < blockElementCapacity ?
                                    elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;
        if (threadElementBase >= inLength) {
            break;
        }

        int32_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
        int32_t elementsForThread = elementsThisBlockRemaining > MAX_ELEMENTS_PER_THREAD ?
                                    MAX_ELEMENTS_PER_THREAD : elementsThisBlockRemaining;

        // 当前梯度元素所属的嵌入向量（行）索引
        int rowIdx = isPowerOfTwo ? (threadElementBase >> gradDimShift) : (threadElementBase / gradDim);
        // 当前梯度元素所属的嵌入向量内的特征（列）索引
        int colIdx = isPowerOfTwo ? (threadElementBase & (gradDim - 1)) : (threadElementBase % gradDim);
        // 获取当前行的基地址
        __gm__ float* curRowBasePtr = reinterpret_cast<__gm__ float*>(valuesPtr[rowIdx]);
        // 部分中间变量置于循环外计算
        float oneMinusBeta1 = (1.0f - beta1);
        float oneMinusBeta2 = (1.0f - beta2);
        float mHatDenom = 1.0f - AscendC::Simt::Pow(beta1, (float)iterNum);
        float vHatDenom = 1.0f - AscendC::Simt::Pow(beta2, (float)iterNum);

        // 2. 实际计算
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
            if (i >= elementsForThread) {
                break;
            }
            int32_t globalIdx = threadElementBase + i;
            if (globalIdx >= inLength) {
                break;
            }

            float tmpGrad = grads[globalIdx];

            int weightIdx = colIdx;
            int mIdx = weightIdx + gradDim;
            int vIdx = mIdx + gradDim;
            
            float tmpWeight = curRowBasePtr[weightIdx];
            float tmpM = curRowBasePtr[mIdx];
            float tmpV = curRowBasePtr[vIdx];

            // 计算新的动量
            float newM = beta1 * tmpM + oneMinusBeta1 * tmpGrad;
            // 计算新的方差
            float newV = beta2 * tmpV + oneMinusBeta2 * tmpGrad * tmpGrad;
            // 计算归一化后的动量和方差
            float mHat = newM / mHatDenom;
            float vHat = newV / vHatDenom;
            // 计算权重更新量
            float deltaW = lr * (mHat / (AscendC::Simt::Sqrt(vHat) + eps) + weightDecay * tmpWeight);
            // 更新权重
            curRowBasePtr[weightIdx] = tmpWeight - deltaW;
            // 更新动量和方差
            curRowBasePtr[mIdx] = newM;
            curRowBasePtr[vIdx] = newV;

            // --- 索引更新逻辑 ---
            colIdx++;
            if (colIdx == gradDim) {
                colIdx = 0;
                rowIdx++;
                // 只有当还有下一次循环且确实跨行时，才去全局内存加载新的行指针
                if (i < MAX_ELEMENTS_PER_THREAD - 1) {
                    curRowBasePtr = reinterpret_cast<__gm__ float*>(valuesPtr[rowIdx]);
                }
            }
        }
    }
}

} // namespace AdamWUpdateSimt
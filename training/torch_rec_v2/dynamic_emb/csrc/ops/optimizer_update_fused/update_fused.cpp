/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <type_traits>
#include <iostream>
#include "update_fused_kernel.h"
#include "../AdamW_update/AdamW_update_kernel.h"
#include "../Adagrad_update/Adagrad_update_kernel.h"
#include "../Rowwise_adagrad_update/Rowwise_adagrad_update_kernel.h"
#include "../../optimizer_kind.h"
#include "kernel_operator.h"
#include "../ops_utils.h"

constexpr int32_t BLOCK_THREADS = UpdateFusedSimt::MAX_THREADS_PER_BLOCK;

template <typename OptimizerT, typename g_type, typename w_type>
__aicore__ inline void DispatchOptimizerUpdateFused(
    __gm__ g_type* grads, __gm__ w_type* values, __gm__ bool* founds, bool isPowerOfTwo,
    uint32_t gradDim, uint32_t valDim, int32_t inLength, float beta1, float beta2, float oneMinusBeta1,
    float oneMinusBeta2, float stepSize, float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks,
    int32_t blocksPerCore, int32_t remainderBlocks, bool isSmall, int32_t gradDimShift, int32_t coreId)
{
    OptimizerT optimizer;
    if (isSmall) {
        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<UpdateFusedSimt::SimtSmallInBlockDataCompute<true, g_type, w_type, OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDim, valDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradDimShift, optimizer);
        } else {
            AscendC::Simt::VF_CALL<UpdateFusedSimt::SimtSmallInBlockDataCompute<false, g_type, w_type, OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDim, valDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradDimShift, optimizer);
        }
    } else {
        int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);
        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<UpdateFusedSimt::SimtLargeDataCompute<true, g_type, w_type, OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDim, valDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimShift, optimizer);
        } else {
            AscendC::Simt::VF_CALL<UpdateFusedSimt::SimtLargeDataCompute<false, g_type, w_type, OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDim, valDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimShift, optimizer);
        }

    }
}

__global__ __aicore__ void update_fused(GM_ADDR grads, GM_ADDR values, GM_ADDR founds, uint32_t gradDim, uint32_t valDim, int32_t inLength,
    float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize, float invVHatDenom,
    float decayFactor, float eps, int32_t totalBlocks, int32_t blocksPerCore, int32_t remainderBlocks, bool isSmall,
    uint32_t gradTypeRaw, uint32_t weightTypeRaw, uint32_t optimizerKindRaw)
{
    int32_t coreId = AscendC::GetBlockIdx();
    bool isPowerOfTwo = (gradDim & (gradDim - 1)) == 0;

    int32_t gradDimShift = 0;
    if (isPowerOfTwo) {
        uint32_t gradDimCopy = gradDim;
        while ((gradDimCopy >>= 1) != 0) {
            gradDimShift++;
        }
    }
    dyn_emb::DataType gradType = static_cast<dyn_emb::DataType>(gradTypeRaw);
    dyn_emb::DataType weightType = static_cast<dyn_emb::DataType>(weightTypeRaw);
    OptimizerKind kind = static_cast<OptimizerKind>(optimizerKindRaw);
    FLOAT_TYPE_DISPATCH(gradType, grad_t, {
        FLOAT_TYPE_DISPATCH(weightType, weight_t, {
            __gm__ grad_t* gradsPtr = reinterpret_cast<__gm__ grad_t*>(grads);
            __gm__ weight_t* valuesPtr = reinterpret_cast<__gm__ weight_t*>(values);
            __gm__ bool* foundsPtr = reinterpret_cast<__gm__ bool*>(founds);
            switch (kind) {
                // 分支 1：AdamW 优化器
                case OptimizerKind::AdamW:
                    DispatchOptimizerUpdateFused<AdamWOptimizer,grad_t,weight_t>(gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDim, valDim,
                        inLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                        totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimShift, coreId);
                    break;
                case OptimizerKind::AdaGrad:
                    DispatchOptimizerUpdateFused<AdaGradOptimizer, grad_t, weight_t>(gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDim, valDim,
                        inLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                        totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimShift, coreId);
                    break;
                case OptimizerKind::RowWiseAdaGrad:
                    DispatchOptimizerUpdateFused<RowWiseAdaGradOptimizer, grad_t, weight_t>(gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDim, valDim,
                        inLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                        totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimShift, coreId);
                    break;
                // 分支 2：后续支持SGD优化器
                default:
                    AscendC::printf("Unsupported optimizer kind: %d\n", optimizerKindRaw);
                    return;
                }
            });
        });
 }


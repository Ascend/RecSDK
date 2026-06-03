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

#include "kernel_operator.h"

using namespace AscendC;

namespace dyn_emb_rowwise_adagrad_float2 {

constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;

__simt_callee__ inline float WarpInclusiveSum(float val, int32_t laneId)
{
#pragma unroll
    for (int32_t offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other = AscendC::Simt::WarpShflUpSync(val, static_cast<uint32_t>(offset));
        if (laneId >= offset) {
            val += other;
        }
    }
    return val;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void RowwiseAdagradUpdateFloat2Kernel(
    __gm__ float2* grads, __gm__ float* __gm__* values, __gm__ bool* founds, uint32_t gradDim, int32_t inLength,
    float lr, float eps)
{
    const uint32_t gradDimVec = gradDim >> 1;
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t laneId = tid % WARP_SIZE;
    const int32_t warpIdInBlock = tid / WARP_SIZE;
    const int32_t warpNumPerBlock = AscendC::Simt::GetThreadNum<0>() / WARP_SIZE;
    const int32_t globalWarpId = AscendC::Simt::GetBlockIdx() * warpNumPerBlock + warpIdInBlock;
    const int32_t totalWarps = AscendC::Simt::GetBlockNum() * warpNumPerBlock;
    const int32_t rowNum = inLength / static_cast<int32_t>(gradDim);

    for (int32_t rowIdx = globalWarpId; rowIdx < rowNum; rowIdx += totalWarps) {
        if (founds != nullptr && !founds[rowIdx]) {
            continue;
        }

        __gm__ float* rowBase = reinterpret_cast<__gm__ float*>(values[rowIdx]);
        if (rowBase == nullptr) {
            continue;
        }

        __gm__ float2* rowWeight = reinterpret_cast<__gm__ float2*>(rowBase);
        const int64_t gradBaseVec = static_cast<int64_t>(rowIdx) * static_cast<int64_t>(gradDimVec);

        float gradSqrSum = 0.0f;
        for (int32_t colVecIdx = laneId; colVecIdx < static_cast<int32_t>(gradDimVec); colVecIdx += WARP_SIZE) {
            const float2 g = grads[gradBaseVec + colVecIdx];
            gradSqrSum += g.x * g.x + g.y * g.y;
        }

        const float warpPrefix = WarpInclusiveSum(gradSqrSum, laneId);
        if (laneId == WARP_SIZE - 1) {
            const float oldAccum = rowBase[gradDim];
            const float newAccum = oldAccum + warpPrefix / static_cast<float>(gradDim);
            rowBase[gradDim] = newAccum;
        }
        AscendC::Simt::ThreadBarrier();

        const float rowAccum = rowBase[gradDim];
        const float denom = AscendC::Simt::Sqrt(rowAccum) + eps;
        for (int32_t colVecIdx = laneId; colVecIdx < static_cast<int32_t>(gradDimVec); colVecIdx += WARP_SIZE) {
            const float2 g = grads[gradBaseVec + static_cast<int64_t>(colVecIdx)];
            float2 w = rowWeight[colVecIdx];
            w.x -= lr * g.x / denom;
            w.y -= lr * g.y / denom;
            rowWeight[colVecIdx] = w;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void RowwiseAdagradFusedFloat2Kernel(
    __gm__ float2* grads, __gm__ float* values, uint32_t gradDim, uint32_t valDim, int32_t inLength, float lr,
    float eps)
{
    const uint32_t gradDimVec = gradDim >> 1;
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t laneId = tid % WARP_SIZE;
    const int32_t warpIdInBlock = tid / WARP_SIZE;
    const int32_t warpNumPerBlock = AscendC::Simt::GetThreadNum<0>() / WARP_SIZE;
    const int32_t globalWarpId = AscendC::Simt::GetBlockIdx() * warpNumPerBlock + warpIdInBlock;
    const int32_t totalWarps = AscendC::Simt::GetBlockNum() * warpNumPerBlock;
    const int32_t rowNum = inLength / static_cast<int32_t>(gradDim);

    for (int32_t rowIdx = globalWarpId; rowIdx < rowNum; rowIdx += totalWarps) {
        __gm__ float* rowBase = values + static_cast<int64_t>(rowIdx) * static_cast<int64_t>(valDim);
        __gm__ float2* rowWeight = reinterpret_cast<__gm__ float2*>(rowBase);
        const int64_t gradBaseVec = static_cast<int64_t>(rowIdx) * static_cast<int64_t>(gradDimVec);

        float gradSqrSum = 0.0f;
        for (int32_t colVecIdx = laneId; colVecIdx < static_cast<int32_t>(gradDimVec); colVecIdx += WARP_SIZE) {
            const float2 g = grads[gradBaseVec + colVecIdx];
            gradSqrSum += g.x * g.x + g.y * g.y;
        }

        const float warpPrefix = WarpInclusiveSum(gradSqrSum, laneId);
        if (laneId == WARP_SIZE - 1) {
            const float oldAccum = rowBase[gradDim];
            const float newAccum = oldAccum + warpPrefix / static_cast<float>(gradDim);
            rowBase[gradDim] = newAccum;
        }
        AscendC::Simt::ThreadBarrier();

        const float rowAccum = rowBase[gradDim];
        const float denom = AscendC::Simt::Sqrt(rowAccum) + eps;
        for (int32_t colVecIdx = laneId; colVecIdx < static_cast<int32_t>(gradDimVec); colVecIdx += WARP_SIZE) {
            const float2 g = grads[gradBaseVec + static_cast<int64_t>(colVecIdx)];
            float2 w = rowWeight[colVecIdx];
            w.x -= lr * g.x / denom;
            w.y -= lr * g.y / denom;
            rowWeight[colVecIdx] = w;
        }
    }
}

}  // namespace dyn_emb_rowwise_adagrad_float2

extern "C" __global__ __aicore__ void rowwise_adagrad_update_float2(GM_ADDR grads, GM_ADDR values, GM_ADDR founds,
                                                                    uint32_t gradDim, int32_t inLength, float lr,
                                                                    float eps)
{
    __gm__ float2* gradsPtr = reinterpret_cast<__gm__ float2*>(grads);
    __gm__ float* __gm__* valuesPtr = reinterpret_cast<__gm__ float * __gm__*>(values);
    __gm__ bool* foundsPtr = reinterpret_cast<__gm__ bool*>(founds);
    Simt::VF_CALL<dyn_emb_rowwise_adagrad_float2::RowwiseAdagradUpdateFloat2Kernel>(
        Simt::Dim3{dyn_emb_rowwise_adagrad_float2::MAX_THREADS_PER_BLOCK, 1, 1}, gradsPtr, valuesPtr, foundsPtr,
        gradDim, inLength, lr, eps);
}

extern "C" __global__ __aicore__ void rowwise_adagrad_fused_float2(GM_ADDR grads, GM_ADDR values, uint32_t gradDim,
                                                                   uint32_t valDim, int32_t inLength, float lr,
                                                                   float eps)
{
    __gm__ float2* gradsPtr = reinterpret_cast<__gm__ float2*>(grads);
    __gm__ float* valuesPtr = reinterpret_cast<__gm__ float*>(values);
    Simt::VF_CALL<dyn_emb_rowwise_adagrad_float2::RowwiseAdagradFusedFloat2Kernel>(
        Simt::Dim3{dyn_emb_rowwise_adagrad_float2::MAX_THREADS_PER_BLOCK, 1, 1}, gradsPtr, valuesPtr, gradDim, valDim,
        inLength, lr, eps);
}

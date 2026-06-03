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

#include "../ops_utils.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace dyn_emb_rowwise_adagrad_update {

constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_THREADS_PER_BLOCK = 2048;

template <typename T>
__simt_callee__ inline float CastToFloat(T v)
{
    return static_cast<float>(v);
}

template <typename T>
__simt_callee__ inline T CastFromFloat(float v)
{
    return static_cast<T>(v);
}

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

template <typename grad_t, typename weight_t>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void RowwiseAdagradUpdateKernel(
    __gm__ grad_t* grads, __gm__ weight_t* __gm__* values, __gm__ bool* founds, uint32_t gradDim, int32_t inLength,
    float lr, float eps)
{
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

        __gm__ weight_t* rowBase = values[rowIdx];
        if (rowBase == nullptr) {
            continue;
        }

        const int64_t gradBase = static_cast<int64_t>(rowIdx) * static_cast<int64_t>(gradDim);
        float gradSqrSum = 0.0f;
        for (int32_t colIdx = laneId; colIdx < static_cast<int32_t>(gradDim); colIdx += WARP_SIZE) {
            const float g = CastToFloat(grads[gradBase + colIdx]);
            gradSqrSum += g * g;
        }

        const float warpPrefix = WarpInclusiveSum(gradSqrSum, laneId);
        if (laneId == WARP_SIZE - 1) {
            const float oldAccum = CastToFloat(rowBase[gradDim]);
            const float newAccum = oldAccum + warpPrefix / static_cast<float>(gradDim);
            rowBase[gradDim] = CastFromFloat<weight_t>(newAccum);
        }
        AscendC::Simt::ThreadBarrier();

        const float rowAccum = CastToFloat(rowBase[gradDim]);
        const float denom = AscendC::Simt::Sqrt(rowAccum) + eps;
        for (int32_t colIdx = laneId; colIdx < static_cast<int32_t>(gradDim); colIdx += WARP_SIZE) {
            const int64_t idx = gradBase + static_cast<int64_t>(colIdx);
            const float g = CastToFloat(grads[idx]);
            const float w = CastToFloat(rowBase[colIdx]);
            rowBase[colIdx] = CastFromFloat<weight_t>(w - lr * g / denom);
        }
    }
}

template <typename grad_t, typename weight_t>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void RowwiseAdagradFusedKernel(
    __gm__ grad_t* grads, __gm__ weight_t* values, uint32_t gradDim, uint32_t valDim, int32_t inLength, float lr,
    float eps)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t laneId = tid % WARP_SIZE;
    const int32_t warpIdInBlock = tid / WARP_SIZE;
    const int32_t warpNumPerBlock = AscendC::Simt::GetThreadNum<0>() / WARP_SIZE;
    const int32_t globalWarpId = AscendC::Simt::GetBlockIdx() * warpNumPerBlock + warpIdInBlock;
    const int32_t totalWarps = AscendC::Simt::GetBlockNum() * warpNumPerBlock;
    const int32_t rowNum = inLength / static_cast<int32_t>(gradDim);

    for (int32_t rowIdx = globalWarpId; rowIdx < rowNum; rowIdx += totalWarps) {
        const int64_t gradBase = static_cast<int64_t>(rowIdx) * static_cast<int64_t>(gradDim);
        __gm__ weight_t* rowBase = values + static_cast<int64_t>(rowIdx) * static_cast<int64_t>(valDim);

        float gradSqrSum = 0.0f;
        for (int32_t colIdx = laneId; colIdx < static_cast<int32_t>(gradDim); colIdx += WARP_SIZE) {
            const float g = CastToFloat(grads[gradBase + colIdx]);
            gradSqrSum += g * g;
        }

        const float warpPrefix = WarpInclusiveSum(gradSqrSum, laneId);
        if (laneId == WARP_SIZE - 1) {
            const float oldAccum = CastToFloat(rowBase[gradDim]);
            const float newAccum = oldAccum + warpPrefix / static_cast<float>(gradDim);
            rowBase[gradDim] = CastFromFloat<weight_t>(newAccum);
        }
        AscendC::Simt::ThreadBarrier();

        const float rowAccum = CastToFloat(rowBase[gradDim]);
        const float denom = AscendC::Simt::Sqrt(rowAccum) + eps;
        for (int32_t colIdx = laneId; colIdx < static_cast<int32_t>(gradDim); colIdx += WARP_SIZE) {
            const int64_t idx = gradBase + static_cast<int64_t>(colIdx);
            const float g = CastToFloat(grads[idx]);
            const float w = CastToFloat(rowBase[colIdx]);
            rowBase[colIdx] = CastFromFloat<weight_t>(w - lr * g / denom);
        }
    }
}

}  // namespace dyn_emb_rowwise_adagrad_update

extern "C" __global__ __aicore__ void rowwise_adagrad_update(GM_ADDR grads, GM_ADDR values, GM_ADDR founds,
                                                             uint32_t gradDim, int32_t inLength, float lr, float eps,
                                                             uint32_t gradTypeRaw, uint32_t weightTypeRaw)
{
    const dyn_emb::DataType gradType = static_cast<dyn_emb::DataType>(gradTypeRaw);
    const dyn_emb::DataType weightType = static_cast<dyn_emb::DataType>(weightTypeRaw);

    FLOAT_TYPE_DISPATCH(gradType, grad_t, {
        FLOAT_TYPE_DISPATCH(weightType, weight_t, {
            __gm__ grad_t* gradsPtr = reinterpret_cast<__gm__ grad_t*>(grads);
            __gm__ weight_t* __gm__* valuesPtr = reinterpret_cast<__gm__ weight_t * __gm__*>(values);
            __gm__ bool* foundsPtr = reinterpret_cast<__gm__ bool*>(founds);
            Simt::VF_CALL<dyn_emb_rowwise_adagrad_update::RowwiseAdagradUpdateKernel<grad_t, weight_t>>(
                Simt::Dim3{dyn_emb_rowwise_adagrad_update::MAX_THREADS_PER_BLOCK, 1, 1}, gradsPtr, valuesPtr, foundsPtr,
                gradDim, inLength, lr, eps);
        });
    });
}

extern "C" __global__ __aicore__ void rowwise_adagrad_fused(GM_ADDR grads, GM_ADDR values, uint32_t gradDim,
                                                            uint32_t valDim, int32_t inLength, float lr, float eps,
                                                            uint32_t gradTypeRaw, uint32_t weightTypeRaw)
{
    const dyn_emb::DataType gradType = static_cast<dyn_emb::DataType>(gradTypeRaw);
    const dyn_emb::DataType weightType = static_cast<dyn_emb::DataType>(weightTypeRaw);

    FLOAT_TYPE_DISPATCH(gradType, grad_t, {
        FLOAT_TYPE_DISPATCH(weightType, weight_t, {
            __gm__ grad_t* gradsPtr = reinterpret_cast<__gm__ grad_t*>(grads);
            __gm__ weight_t* valuesPtr = reinterpret_cast<__gm__ weight_t*>(values);
            Simt::VF_CALL<dyn_emb_rowwise_adagrad_update::RowwiseAdagradFusedKernel<grad_t, weight_t>>(
                Simt::Dim3{dyn_emb_rowwise_adagrad_update::MAX_THREADS_PER_BLOCK, 1, 1}, gradsPtr, valuesPtr, gradDim,
                valDim, inLength, lr, eps);
        });
    });
}

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

#ifndef HSTU_DENSE_BACKWARD_KERNEL_COMMON_H
#define HSTU_DENSE_BACKWARD_KERNEL_COMMON_H

#include <cstdint>
#include <type_traits>
#include <unistd.h>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"

using namespace AscendC;

namespace HstuDenseBackward {

__aicore__ inline bool IfMask(const int32_t& maskType, MaskType maskTypeEnum)
{
    return static_cast<int32_t>(maskTypeEnum) == maskType;
}

// For QK and GV
template <typename qType>
__aicore__ inline void CopyQKA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col, int useM,
                                int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useM * useK);

    HstuDenseBackwardTilingData* tilingP = reinterpret_cast<HstuDenseBackwardTilingData*>(tilingPtr);
    int64_t headNum = tilingP->headNum;
    int64_t headDim = tilingP->headDim;

    int32_t baseM = tilingP->qkMatmul.baseM;
    int32_t baseN = tilingP->qkMatmul.baseN;
    int32_t baseK = tilingP->qkMatmul.baseK;

    uint16_t alignedUseM = AlignUp(useM, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useM, (uint16_t)useK, 0, (uint16_t)(headNum * headDim), alignedUseM, 1, 0};

    int64_t startIdx = row * baseM * headNum * headDim + col * baseK;
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

// For QK and GV
template <typename qType>
__aicore__ inline void CopyQKB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK,
                                int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseBackwardTilingData* tilingP = reinterpret_cast<HstuDenseBackwardTilingData*>(tilingPtr);
    int64_t headNum = tilingP->headNum;
    int64_t headDim = tilingP->headDim;

    int32_t baseM = tilingP->qkMatmul.baseM;
    int32_t baseN = tilingP->qkMatmul.baseN;
    int32_t baseK = tilingP->qkMatmul.baseK;

    uint16_t alignedUseN = AlignUp(useN, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useN, (uint16_t)useK, 0, (uint16_t)(headNum * headDim), alignedUseN, 1, 0};

    int64_t startIdx = col * baseN * headNum * headDim + row * baseK;
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

// For q_grad
template <typename qType>
__aicore__ inline void CopyQGradA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col,
                                   int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useM * useK);

    HstuDenseBackwardTilingData* tilingP = reinterpret_cast<HstuDenseBackwardTilingData*>(tilingPtr);
    int32_t isNormal = tilingP->isNormal;
    int32_t enableBias = tilingP->enableBias;
    int64_t copyBlockLen = (isNormal || enableBias) ? tilingP->biasGradSeqLen : tilingP->blockHeight;

    int32_t baseM = tilingP->qGradMatmul.baseM;
    int32_t baseN = tilingP->qGradMatmul.baseN;
    int32_t baseK = tilingP->qGradMatmul.baseK;

    uint16_t alignedUseM = AlignUp(useM, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useM, (uint16_t)useK, 0, (uint16_t)copyBlockLen, alignedUseM, 1, 0};

    int64_t startIdx = row * baseM * copyBlockLen + col * baseK;
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

// For k_grad
template <typename qType>
__aicore__ inline void CopyKGradA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col,
                                   int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useK * useM);

    HstuDenseBackwardTilingData* tilingP = reinterpret_cast<HstuDenseBackwardTilingData*>(tilingPtr);
    int32_t isNormal = tilingP->isNormal;
    int32_t enableBias = tilingP->enableBias;
    int64_t copyBlockLen = (isNormal || enableBias) ? tilingP->biasGradSeqLen : tilingP->blockHeight;

    int32_t baseM = tilingP->kGradMatmul.baseM;
    int32_t baseN = tilingP->kGradMatmul.baseN;
    int32_t baseK = tilingP->kGradMatmul.baseK;

    uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useM, 0, (uint16_t)copyBlockLen, alignedUseK, 1, 0};

    int64_t startIdx = col * baseK * copyBlockLen + row * baseM;
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

template <typename qType>
__aicore__ inline void CopyVGradB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col,
                                   int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseBackwardTilingData* tilingP = reinterpret_cast<HstuDenseBackwardTilingData*>(tilingPtr);
    int64_t headNum = tilingP->headNum;
    int64_t headDim = tilingP->headDim;

    int32_t baseM = tilingP->vGradMatmul.baseM;
    int32_t baseN = tilingP->vGradMatmul.baseN;
    int32_t baseK = tilingP->vGradMatmul.baseK;

    uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useN, 0, (uint16_t)(headNum * headDim), alignedUseK, 1, 0};

    int64_t startIdx = row * baseK * headNum * headDim + col * baseN;
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};
}  // namespace HstuDenseBackward

#endif
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

#ifndef HSTU_DENSE_KERNEL_PATTEN_COMMON_H
#define HSTU_DENSE_KERNEL_PATTEN_COMMON_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace HstuDenseForwardFuxi {

constexpr uint32_t MAX_BATCH_SIZE = 512;
constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int VEC_PER_PROCESS = 32;
constexpr int MAX_INDICS_ONE_BLOCK = 100;
constexpr int UB_SIZE = 170 * 1024;  // 170KB
constexpr int QUEUE_IN_NUM = 2;
constexpr int SPLIT_CORE = 2;
constexpr int ALIGN_16 = 16;

constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;

constexpr int SV_WORKSPACE_IDX = 0;
constexpr int TV_WORKSPACE_IDX = 1;
constexpr int PV_WORKSPACE_IDX = 2;

constexpr int ATTN_WORKSPACE_IDX = 0;
constexpr int TS_WORKSPACE_IDX = 1;
constexpr int POS_WORKSPACE_IDX = 2;

constexpr int OUTPUT_DIM2_TIMES3 = 3;

constexpr int INVALID_TASK_ID = -1;

struct Args {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR timestampBias;
    GM_ADDR positionBias;
    GM_ADDR mask;
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

enum class CausalMaskT {
    MASK_TRIL = 0,          // 下三角
    MASK_TRIU,              // 上三角
    MASK_NONE,              // 不使能mask
    MASK_CUSTOME,           // 用户自定义mask
};

template<typename qType, CausalMaskT maskType>
__aicore__ inline void DoCausalMask(LocalTensor<qType>& inMaskLt, int64_t maskOffset, int64_t maskLens,
    int64_t maskStride, int64_t repeatTimes, qType value)
{
    if constexpr (maskType == CausalMaskT::MASK_TRIL) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            int64_t thisIndexMask = maskOffset + i + 1;
            Duplicate<qType>(inMaskLt[i * maskStride], value, thisIndexMask);
        }
    } else if constexpr (maskType == CausalMaskT::MASK_NONE) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            Duplicate<qType>(inMaskLt[i * maskStride], value, maskOffset);
        }
    } else {
        ASCENDC_ASSERT((false), "DoCausalMask only support MASK_TRIL and MASK_NONE");
    }
}


template <typename qType>
__aicore__ inline void CopyQKA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col, int useM,
                                int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useM * useK);
    int blockLen = useM * useK;

    HstuDenseForwardFuxiTilingData* tilingP = reinterpret_cast<HstuDenseForwardFuxiTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int64_t headNum = tilingP->headNum;
    int32_t baseM = tilingP->qkMatmul.baseM;
    int32_t baseK = tilingP->qkMatmul.baseK;

    auto alignOfM = AlignUp(useM, ALIGN_16);
    Nd2NzParams param = {
        1, static_cast<uint16_t>(useM), static_cast<uint16_t>(useK), 0,
        static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfM), 1, 0
    };

    int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNum * static_cast<int64_t>(baseM) +
                         static_cast<int64_t>(col) * static_cast<int64_t>(baseK);
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType>
__aicore__ inline void CopyQKB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK,
                                int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseForwardFuxiTilingData* tilingP = reinterpret_cast<HstuDenseForwardFuxiTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int64_t headNum = tilingP->headNum;
    int32_t baseN = tilingP->qkMatmul.baseN;
    int32_t baseK = tilingP->qkMatmul.baseK;

    auto alignOfN = AlignUp(useN, ALIGN_16);
    Nd2NzParams param = {
        1, static_cast<uint16_t>(useN), static_cast<uint16_t>(useK), 0,
        static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfN), 1, 0
    };

    int64_t offsetOfGt = static_cast<int64_t>(col) * dim * headNum * static_cast<int64_t>(baseN) +
                         static_cast<int64_t>(row) * static_cast<int64_t>(baseK);
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType>
__aicore__ inline void CopySVB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK,
                                int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseForwardFuxiTilingData* tilingP = reinterpret_cast<HstuDenseForwardFuxiTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int64_t headNum = tilingP->headNum;
    int32_t baseN = tilingP->svMatmul.baseN;
    int32_t baseK = tilingP->svMatmul.baseK;
    auto alignOfK = AlignUp(useK, ALIGN_16);

    Nd2NzParams param = {
        1, static_cast<uint16_t>(useK), static_cast<uint16_t>(useN), 0,
        static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfK), 1, 0
    };

    int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNum * static_cast<int64_t>(baseK) +
                         static_cast<int64_t>(col) * static_cast<int64_t>(baseN);
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

} // namespace HstuDenseForwardFuxi

#endif
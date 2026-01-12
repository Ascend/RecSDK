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

#include <cstdint>

#include "kernel_operator.h"

#include "hstu_dense_backward_jagged_kernel.h"
#include "hstu_dense_backward_kernel.h"
#include "hstu_jagged_f16_r0_kernel.h"
#include "static_switch.h"
#include "vector_score.h"
constexpr static int64_t HEAD_NUM_4 = 4;
constexpr static int64_t HEAD_NUM_8 = 8;
constexpr static int64_t HEAD_DIM_64 = 64;
constexpr static int64_t HEAD_DIM_128 = 128;
constexpr static int64_t BLOCK_HEIGHT_256 = 256;
constexpr static int64_t BLOCK_HEIGHT_128 = 128;
constexpr static int64_t MASK_TYPE_TRIL = 0;

template <typename qType, typename seqOffsetType, uint32_t blockHeightQ, uint32_t blockHeightK, uint32_t headDimPadding, class MatmulMgmtType, class VectorScoreType>
__aicore__ inline void InvokeKernelWithRgistMM(HstuDenseBackward::Args& args)
{
    MatmulMgmtType mmMgmtCommon;
    VectorScoreType vectorScoreKernel;

    HstuDenseBackward::HstuJaggedF16R0Kernel<qType, seqOffsetType, blockHeightQ,
    blockHeightK, headDimPadding, MatmulMgmtType, VectorScoreType>
        kernel;
    REGIST_MATMUL_OBJ(&kernel.pipe, GetSysWorkSpacePtr(), mmMgmtCommon.qkOrGvMatmul_, &args.tilingDataPtr->qkMatmul, mmMgmtCommon.qGradMatmul_,
                      &args.tilingDataPtr->qGradMatmul, mmMgmtCommon.kGradMatmul_, &args.tilingDataPtr->kGradMatmul, mmMgmtCommon.vGradMatmul_,
                      &args.tilingDataPtr->vGradMatmul);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    mmMgmtCommon.qkOrGvMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.vGradMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.qGradMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.kGradMatmul_.SetUserDefInfo(tilingPtr);

    kernel.Compute(args, &mmMgmtCommon, &vectorScoreKernel); 
}

extern "C" __global__ __aicore__ void hstu_jagged_backward(GM_ADDR grad, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                           GM_ADDR attnBias, GM_ADDR seqOffset, GM_ADDR numContext,
                                                           GM_ADDR numTarget, GM_ADDR qGrad, GM_ADDR kGrad,
                                                           GM_ADDR vGrad, GM_ADDR attnBiasGrad, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const HstuDenseBackwardTilingData* __restrict tilingDataPtr = &tilingData;

    HstuDenseBackward::Args args{grad,      q,     k,     v,     mask,         attnBias,  seqOffset, numContext,
                                 numTarget, qGrad, kGrad, vGrad, attnBiasGrad, workspace, tiling,    tilingDataPtr};

    bool isConstMaskType = (tilingDataPtr->maskType == MASK_TYPE_TRIL);
    bool isConstNoBias = (tilingDataPtr->enableBias == 0);
    constexpr uint32_t HEAD_DIM_PADDING = 128;
    if (TILING_KEY_IS(5)) {
        using MMCOM = HstuDenseBackward::MmMgmtCommon<float, BLOCK_HEIGHT_128, BLOCK_HEIGHT_128, HEAD_DIM_PADDING, HstuDenseBackwardTilingData>; 
        using VsKernelType = HstuDenseBackward::HstuVectorScoreCommon<float, BLOCK_HEIGHT_128, BLOCK_HEIGHT_128>;
        InvokeKernelWithRgistMM<float, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_128, BLOCK_HEIGHT_128, HEAD_DIM_PADDING, MMCOM, VsKernelType>(args);

    } else if (TILING_KEY_IS(4)) {
        using MMCOM = HstuDenseBackward::MmMgmtCommon<bfloat16_t, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, HEAD_DIM_PADDING, HstuDenseBackwardTilingData>; 
        using VsKernelType = HstuDenseBackward::HstuVectorScoreCommon<bfloat16_t, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256>;
        InvokeKernelWithRgistMM<bfloat16_t, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, HEAD_DIM_PADDING, MMCOM, VsKernelType>(args);
    } else if (TILING_KEY_IS(3)) {
        using MMCOM = HstuDenseBackward::MmMgmtCommon<half, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, HEAD_DIM_PADDING, HstuDenseBackwardTilingData>; 
        using VsKernelType = HstuDenseBackward::HstuVectorScoreCommon<half, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256>;
        InvokeKernelWithRgistMM<half, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, HEAD_DIM_PADDING, MMCOM, VsKernelType>(args);
    } else if (TILING_KEY_IS(2)) {
        HstuDenseBackward::HstuDenseBackwardKernel<float> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(1)) {
        HstuDenseBackward::HstuDenseBackwardKernel<bfloat16_t> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(0)) {
        HstuDenseBackward::HstuDenseBackwardKernel<half> kernel;
        kernel.Compute(args);
    }
}
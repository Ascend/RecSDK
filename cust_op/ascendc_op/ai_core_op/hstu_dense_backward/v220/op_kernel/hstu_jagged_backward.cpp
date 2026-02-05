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

#include <cstdint>

#include "kernel_operator.h"

#include "hstu_jagged_kernel.h"
#include "matmul_mgmt.h"
#include "static_switch.h"
#include "vector_score.h"
constexpr static int64_t HEAD_NUM_4 = 4;
constexpr static int64_t HEAD_NUM_8 = 8;
constexpr static int64_t HEAD_DIM_64 = 64;
constexpr static int64_t HEAD_DIM_128 = 128;
constexpr static int64_t BLOCK_HEIGHT_256 = 256;
constexpr static int64_t BLOCK_HEIGHT_128 = 128;
constexpr static int64_t MASK_TYPE_TRIL = 0;

template <typename qType, typename seqOffsetType, uint32_t blockHeightQ, uint32_t blockHeightK, uint32_t headDimPadding,
          class MatmulMgmtType, class VectorScoreType>
__aicore__ inline void InvokeKernelWithRegistMM(HstuDenseBackward::Args& args)
{
    MatmulMgmtType mmMgmtCommon;
    VectorScoreType vectorScoreKernel;

    HstuDenseBackward::HstuJaggedKernel<qType, seqOffsetType, blockHeightQ, blockHeightK, headDimPadding,
                                        MatmulMgmtType, VectorScoreType>
        kernel;
    REGIST_MATMUL_OBJ(&kernel.pipe, GetSysWorkSpacePtr(), mmMgmtCommon.qkOrGvMatmul_, &args.tilingDataPtr->qkMatmul,
                      mmMgmtCommon.qGradMatmul_, &args.tilingDataPtr->qGradMatmul, mmMgmtCommon.kGradMatmul_,
                      &args.tilingDataPtr->kGradMatmul, mmMgmtCommon.vGradMatmul_, &args.tilingDataPtr->vGradMatmul);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    mmMgmtCommon.qkOrGvMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.vGradMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.qGradMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.kGradMatmul_.SetUserDefInfo(tilingPtr);

    kernel.Compute(args, &mmMgmtCommon, &vectorScoreKernel);
}

template <typename qType, typename seqOffsetType, uint32_t blockHeightQ, uint32_t blockHeightK, uint32_t headDimPadding,
          class MatmulMgmtType, class VectorScoreType>
__aicore__ inline void InvokeKernelConstMM(HstuDenseBackward::Args& args)
{
    MatmulMgmtType mmMgmtCommon;
    VectorScoreType vectorScoreKernel;

    HstuDenseBackward::HstuJaggedKernel<qType, seqOffsetType, blockHeightQ, blockHeightK, headDimPadding,
                                        MatmulMgmtType, VectorScoreType>
        kernel;
    REGIST_MATMUL_OBJ(&kernel.pipe, GetSysWorkSpacePtr(), mmMgmtCommon.qkOrGvMatmul_, (TCubeTiling*)nullptr,
                      mmMgmtCommon.qGradMatmul_, (TCubeTiling*)nullptr, mmMgmtCommon.kGradMatmul_,
                      (TCubeTiling*)nullptr, mmMgmtCommon.vGradMatmul_, (TCubeTiling*)nullptr);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    mmMgmtCommon.qkOrGvMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.vGradMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.qGradMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmtCommon.kGradMatmul_.SetUserDefInfo(tilingPtr);

    kernel.Compute(args, &mmMgmtCommon, &vectorScoreKernel);
}

extern "C" __global__ __aicore__ void hstu_jagged_backward(GM_ADDR grad, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                           GM_ADDR attnBias, GM_ADDR seqOffsetQ, GM_ADDR seqOffsetK,
                                                           GM_ADDR numContext, GM_ADDR numTarget, GM_ADDR qGrad,
                                                           GM_ADDR kGrad, GM_ADDR vGrad, GM_ADDR attnBiasGrad,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const HstuJaggedBackwardTilingData* __restrict tilingDataPtr = &tilingData;

    HstuDenseBackward::Args args{grad,         q,         k,     v,     mask,  attnBias,     seqOffsetQ, seqOffsetK,
                                 numContext,   numTarget, qGrad, kGrad, vGrad, attnBiasGrad, workspace,  tiling,
                                 tilingDataPtr};

    bool isMaskTypeTril = (tilingDataPtr->maskType == MASK_TYPE_TRIL);
    bool noBias = (tilingDataPtr->enableBias == 0);
    constexpr uint32_t HEAD_DIM_PADDING = 128;
    if constexpr (std::is_same<DTYPE_GRAD, float>::value) {
        using MMCOM = HstuDenseBackward::MmMgmtCommon<float, BLOCK_HEIGHT_128, BLOCK_HEIGHT_128, HEAD_DIM_PADDING,
                                                      HstuJaggedBackwardTilingData>;
        using VsKernelType = HstuDenseBackward::HstuVectorScoreCommon<float, BLOCK_HEIGHT_128, BLOCK_HEIGHT_128>;
        InvokeKernelWithRegistMM<float, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_128, BLOCK_HEIGHT_128, HEAD_DIM_PADDING, MMCOM,
                                 VsKernelType>(args);
    } else {
        bool isSameDim = (tilingDataPtr->headDimQK == tilingDataPtr->headDimV);
        HEAD_DIM_SWITCH(
            isMaskTypeTril && noBias && isSameDim, tilingDataPtr->headDimQK, HEAD_DIM, IS_M0R0_CONST_MODE,
            if constexpr (IS_M0R0_CONST_MODE) {
                using MMCOM = HstuDenseBackward::MmMgmtFp16R0Jagged<DTYPE_GRAD, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256,
                                                                    HEAD_DIM, HstuJaggedBackwardTilingData>;
                using VsKernelType =
                    HstuDenseBackward::HstuF16R0VectorScore<DTYPE_GRAD, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256>;
                InvokeKernelConstMM<DTYPE_GRAD, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, HEAD_DIM, MMCOM,
                                    VsKernelType>(args);
            } else {
                using MMCOM = HstuDenseBackward::MmMgmtCommon<DTYPE_GRAD, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256,
                                                              HEAD_DIM_PADDING, HstuJaggedBackwardTilingData>;
                using VsKernelType =
                    HstuDenseBackward::HstuVectorScoreCommon<DTYPE_GRAD, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256>;
                InvokeKernelWithRegistMM<DTYPE_GRAD, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_256, BLOCK_HEIGHT_256,
                                         HEAD_DIM_PADDING, MMCOM, VsKernelType>(args);
            });
    }
}
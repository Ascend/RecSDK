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

#ifndef HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_H
#define HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "hstu_dense_causal_mask.h"
#include "hstu_common_const.h"
#include "matmul_constexpr.h"
#include "hstu_traitparams.h"

using namespace AscendC;
using namespace HstuForward;

namespace HstuDenseForward {

struct DenseArgs {
    // hstu normal
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR attnBias;
    
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename TraitParams, typename TilingDataType>
class HstuDenseForwardKernelPattenBsnd {
public:
    using qType = typename TraitParams::qType;
    using oType = typename TraitParams::oType;
    static constexpr int ElementOfBlock = DATA_ALIGN_BYTES / sizeof(qType);
    // blockHeight only used in dense_hstu_forward
    static constexpr int blockHeight = BLOCK_HEIGHT_256;
    static constexpr int vectorScoreUbBlockElem = (VEC_PER_PROCESS * blockHeight) / USE_QUEUE_NUM;
    static constexpr auto qkMMCPos = TPosition::GM;
    static constexpr MatmulConfig qkMMConfig = std::is_same<qType, float>::value ?
        mmStaticConfigQKFp32 : mmStaticConfigQKFp16;
    static constexpr MatmulConfig svMMConfig = std::is_same<qType, float>::value ?
        mmStaticConfigSVFp32 : mmStaticConfigSVFp16;

    __aicore__ inline HstuDenseForwardKernelPattenBsnd() {}
    __aicore__ inline void Init(const DenseArgs& args, const HstuDenseForwardTilingData* __restrict tilingDataPtr,
                                TPipe* pipePtr)
    {
        InitArgs(args, tilingDataPtr);
        InitPipe(pipePtr);
    }

    __aicore__ inline void InitArgs(const DenseArgs& args, const HstuDenseForwardTilingData* __restrict tilingDataPtr)
    {
        q = args.q;
        k = args.k;
        v = args.v;
        attnBias = args.attnBias;
        mask = args.mask;

        attnOutput = args.attnOutput;
        workspace = args.workspace;

        // Batch Size
        batchSize = tilingDataPtr->batchSize;
        // Seq Len
        seqLen = tilingDataPtr->seqLen;
        // Head Num
        headNum = tilingDataPtr->headNum;
        // Embedding Dim
        dim = tilingDataPtr->dim;

        maxSeqLen = tilingDataPtr->maxSeqLen;

        // attr
        siluScale = tilingDataPtr->siluScale;
    }

    __aicore__ inline void InitPipe(TPipe* pipePtr)
    {
        pipe = pipePtr;

        // Gt
        qGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(q));
        kGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(k));
        vGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(v));

        if constexpr (TraitParams::enableBias) {
            attnBiasGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnBias));
        }

        if constexpr (TraitParams::maskType == CausalMaskT::MASK_CUSTOM) {
            attnMaskGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(mask));
        }

        attnOutputGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnOutput));

        const uint32_t coreNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;

        int64_t oneBlockMidElem = blockHeight * blockHeight * COMPUTE_PIPE_NUM;
        int64_t oneCoreMidElem = coreNum * oneBlockMidElem;

        int64_t oneBlockMidTransElem = blockHeight * MAX_BLOCK_DIM * TRANS_PIPE_NUM;
        int64_t oneCoreTransMidElem = coreNum * oneBlockMidTransElem;
        int64_t kvOffset = oneCoreMidElem + oneCoreTransMidElem * 3; // svResultGt midkGt midvGt

        attnScoreGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(workspace) + GetBlockIdx() * oneBlockMidElem);
        svResultGt.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace) + oneCoreMidElem + GetBlockIdx() * oneBlockMidTransElem,
            oneBlockMidTransElem);

        // Init pipe total 32K * 5 = 160K
        pipe->InitBuffer(queIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(queOut, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(tmpBuff, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(biasIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(queMaskIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
    }

    __aicore__ inline void CastQtype2Float(LocalTensor<float> distTensor, LocalTensor<qType> srcTensor,
                                           LocalTensor<qType> midTensor, int64_t len)
    {
        if constexpr (!std::is_same<qType, float>::value) {
            DataCopy<qType>(midTensor, srcTensor, len);
            Cast(distTensor, midTensor, RoundMode::CAST_NONE, len);
        }
    }

    __aicore__ inline void CastFloat2Qtype(LocalTensor<qType>& distTensor, LocalTensor<float>& srcTensor,
                                           LocalTensor<float>& midTensor, int64_t len)
    {
        if constexpr (!std::is_same<qType, float>::value) {
            DataCopy<float>(midTensor, srcTensor, len);
            Cast(distTensor, midTensor, RoundMode::CAST_RINT, len);
        }
    }

    __aicore__ inline void WaitQkMatmul()
    {
        qkMatmul.WaitIterateAll();
        qkMatmul.End();
    }

    __aicore__ inline void WaitSvMatmul()
    {
        svMatmul.WaitIterateAll();
        svMatmul.End();
    }

    __aicore__ inline void DoMaskOptional(
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& newOutLt,
        int64_t thisLen,
        bool needMask,
        float scale)
    {
        if constexpr (TraitParams::maskType != CausalMaskT::MASK_NONE) {
            queMaskIn.DeQue();
        }

        if (needMask) {
            if constexpr (TraitParams::maskType == CausalMaskT::MASK_CUSTOM) {
                inMaskLtFp32 = inMaskLt.template ReinterpretCast<float>();
                CastQtype2Float(inMaskLtFp32, inMaskLt, tmpLt, thisLen);
                Muls<float>(inMaskLtFp32, inMaskLtFp32, scale, thisLen);
            }
            Mul<float>(newOutLt, newOutLt, inMaskLtFp32, thisLen);
        } else {
            Muls<float>(newOutLt, newOutLt, scale, thisLen);
        }

        if constexpr (TraitParams::maskType != CausalMaskT::MASK_NONE) {
            queMaskIn.FreeTensor(inMaskLtFp32);
        }
    }

    __aicore__ inline void DoBiasOptional(
        LocalTensor<float>& newInLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& tmpLt,
        int64_t thisLen
    )
    {
        if constexpr (TraitParams::enableBias) {
            biasIn.DeQue();
            auto newBiasLt = biasLt.template ReinterpretCast<float>();
            CastQtype2Float(newBiasLt, biasLt, tmpLt, thisLen);
            Add<float>(newInLt, newInLt, newBiasLt, thisLen);
            biasIn.FreeTensor(biasLt);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32NoRab(
        LocalTensor<qType>& inLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& tmpLtFp32,
        bool needMask,
        int64_t thisLen,
        float scale)
    {
        if constexpr (!std::is_same<qType, float>::value) {
            queIn.DeQue();
            Cast(tmpLtFp32, inLt, RoundMode::CAST_NONE, thisLen);
            queIn.FreeTensor(inLt);

            auto biasLtFp32 = biasLt.template ReinterpretCast<float>();
            Silu<float>(biasLtFp32, tmpLtFp32, thisLen);
            DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, biasLtFp32, thisLen, needMask, scale);

            auto outLt = queOut.AllocTensor<qType>();
            Cast(outLt, biasLtFp32, RoundMode::CAST_RINT, thisLen);
            queOut.EnQue(outLt);
        } else {
            queIn.DeQue();

            auto outLt = queOut.AllocTensor<qType>();
            Silu<float>(outLt, inLt, thisLen);
            queIn.FreeTensor(inLt);

            DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, outLt, thisLen, needMask, scale);
            queOut.EnQue(outLt);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32(
        LocalTensor<qType>& inLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& tmpLtFp32,
        bool needMask,
        int64_t thisLen,
        float scale)
    {
        queIn.DeQue();
        auto newInLt = inLt.template ReinterpretCast<float>();
        CastQtype2Float(newInLt, inLt, tmpLt, thisLen);
        DoBiasOptional(newInLt, biasLt, tmpLt, thisLen);

        auto outLt = queOut.AllocTensor<qType>();
        auto newOutLt = outLt.template ReinterpretCast<float>();
        Silu<float>(newOutLt, newInLt, thisLen);

        queIn.FreeTensor(inLt);
        DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, newOutLt, thisLen, needMask, scale);
        CastFloat2Qtype(outLt, newOutLt, tmpLtFp32, thisLen);
        queOut.EnQue(outLt);
    }

    __aicore__ inline void DataCopyMayPad(
        const LocalTensor<qType>& lt, GlobalTensor<qType>& gt, uint16_t copyBlock, uint32_t blockLen,
        int64_t offset)
    {
        bool align = false;
        uint16_t alignOfN = AlignUp(blockLen, ElementOfBlock);
        align = (maxSeqLen % ElementOfBlock == 0) && (alignOfN == blockLen);

        uint16_t dstStride = (TraitParams::blockN - alignOfN) * sizeof(qType) / DATA_ALIGN_BYTES;

        if (align) {
            uint16_t copyLen = alignOfN * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t srcStride = (maxSeqLen - blockLen) * sizeof(qType) / DATA_ALIGN_BYTES;

            DataCopyParams copyParms = { copyBlock, copyLen, srcStride, dstStride };
            DataCopy(lt, gt[offset], copyParms);
        } else {
            uint16_t copyLenBytes = blockLen * sizeof(qType);
            uint16_t srcStrideBytes = (maxSeqLen - blockLen) * sizeof(qType);

            uint8_t padLens = alignOfN - blockLen;
            DataCopyParams copyParms = { copyBlock, copyLenBytes, srcStrideBytes, dstStride };
            DataCopyPadParams padParms = { true, 0, padLens, 0 };
            DataCopyPad(lt, gt[offset], copyParms, padParms);
        }
    }

    __aicore__ inline bool GenMask(
        LocalTensor<float>& inMaskLt, int causalMask, int64_t maskLen, int64_t maskOffset, float sclae)
    {
        bool needMask = false;
        if (causalMask == 1) {
            DoCausalMask<float, CausalMaskT::MASK_TRIL>(inMaskLt, maskOffset, maskLen, TraitParams::blockM,
                                                
                maskLen / TraitParams::blockM, sclae);
            needMask = true;
        }

        return needMask;
    }

    template<typename MaskInfoType>
    __aicore__ inline bool DoMaskInitOptional(
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        MaskInfoType& maskinfo,
        int64_t maskOffset,
        int64_t thisLen,
        int64_t blockOffset,
        float scale,
        uint32_t n)
    {
        bool needMask = false;
        if constexpr (TraitParams::maskType == CausalMaskT::MASK_TRIL) {
            inMaskLtFp32 = queMaskIn.AllocTensor<float>();
            needMask = GenMask(
                inMaskLtFp32,
                maskinfo,
                thisLen,
                ((maskinfo > 0) ? (blockOffset) : n),  // blockOffset为行号
                scale);

            queMaskIn.EnQue(inMaskLtFp32);
        } else if constexpr (TraitParams::maskType == CausalMaskT::MASK_CUSTOM) {
            int64_t thisMaskOffset = maskOffset + blockOffset * maxSeqLen;

            inMaskLt = queMaskIn.AllocTensor<qType>();
            DataCopyMayPad(inMaskLt, attnMaskGt,
                (uint16_t)(thisLen / TraitParams::blockM), n, thisMaskOffset);
            queMaskIn.EnQue(inMaskLt);

            needMask = true;
        }

        return needMask;
    }

    __aicore__ inline void DoBiasCopyOptional(
        LocalTensor<qType>& biasLt,
        int64_t biasOffset,
        int64_t thisLen,
        int64_t blockOffset,
        uint32_t n)
    {
        if constexpr (TraitParams::enableBias) {
            int64_t thisBiasOffset = biasOffset + blockOffset * maxSeqLen;
            biasLt = biasIn.AllocTensor<qType>();
            DataCopyMayPad(biasLt, attnBiasGt,
                (uint16_t)(thisLen / TraitParams::blockM), n, thisBiasOffset);
            biasIn.EnQue(biasLt);
        }
    }

    template<typename MaskInfoType>
    __aicore__ inline void VecScoreImpl(
        int64_t taskId,
        int64_t biasOffset,
        int64_t maskOffset,
        float scale,
        MaskInfoType& maskinfo,
        uint32_t m,
        uint32_t n)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t total = m * TraitParams::blockM;
        int64_t offset = midResultIdx * TraitParams::blockM * TraitParams::blockM;

        auto tmpLt = tmpBuff.AllocTensor<qType>();
        auto tmpLtFp32 = tmpLt.template ReinterpretCast<float>();
        LocalTensor<qType> biasLt;
        LocalTensor<qType> inMaskLt;
        LocalTensor<float> inMaskLtFp32;

        if constexpr (!TraitParams::enableBias) {
            biasLt = biasIn.AllocTensor<qType>();
        }

        int64_t remain = total;
        while (remain > 0) {
            int64_t thisLen = vectorScoreUbBlockElem;
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t thisOffset = offset + (total - remain);
            auto inLt = queIn.AllocTensor<qType>();

            DataCopy(inLt, attnScoreGt[thisOffset], thisLen);

            queIn.EnQue(inLt);

            int64_t blockOffset = (total - remain) / TraitParams::blockM;
            DoBiasCopyOptional(biasLt, biasOffset, thisLen, blockOffset, n);

            bool needMask =
                DoMaskInitOptional(inMaskLt, inMaskLtFp32, maskinfo, maskOffset, thisLen, blockOffset, scale, n);

            if constexpr (TraitParams::enableBias) {
                CalcuScoreWithFloat32(inLt, biasLt, inMaskLt, inMaskLtFp32, tmpLt, tmpLtFp32,
                    needMask, thisLen, scale);
            } else {
                CalcuScoreWithFloat32NoRab(inLt, biasLt, inMaskLt, inMaskLtFp32, tmpLt, tmpLtFp32,
                    needMask, thisLen, scale);
            }

            auto outLt = queOut.DeQue<qType>();
            DataCopy(attnScoreGt[thisOffset], outLt, thisLen);
            queOut.FreeTensor(outLt);

            remain = remain - thisLen;
        }

        if constexpr (!TraitParams::enableBias) {
            biasIn.FreeTensor(biasLt);
        }

        tmpBuff.FreeTensor(tmpLt);
    }

    template<bool isFirst = true>
    __aicore__ inline void DoQkMatmulImpl(int64_t qOffset, int64_t kOffset, uint32_t taskId, uint32_t m, uint32_t n,
                                          uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outOffset = midResultIdx * TraitParams::blockM * TraitParams::blockM;

        qkMatmul.SetTensorA(qGt[qOffset]);
        qkMatmul.SetTensorB(kGt[kOffset], true);
        qkMatmul.SetTail(m, n, k);
        qkMatmul.SetSelfDefineData(headNum); // 设置CopyQK的自定义headNum数据

        qkMatmul.template IterateAll<false>(attnScoreGt[outOffset], 0, false, true);
    }

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, uint8_t isAtomicAdd,
                                          uint32_t m, uint32_t n, uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * TraitParams::blockM * TraitParams::blockK;
        int64_t sOffset = midResultIdx * TraitParams::blockM * TraitParams::blockN;

        svMatmul.SetTensorA(attnScoreGt[sOffset]);
        svMatmul.SetTensorB(vGt[vOffset]);
        svMatmul.SetTail(m, n, k);
        svMatmul.SetSelfDefineData(headNum); // 设置CopyQK的自定义headNum数据

        svMatmul.template IterateAll<false>(svResultGt[outOffset], isAtomicAdd, false, true);
    }

    template <bool needAtomic = false>
    __aicore__ inline void DoTransSvImpl(int64_t transTaskId, int64_t outStartOffset, uint32_t m)
    {
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t inOffset = outMidIndex * TraitParams::blockM * TraitParams::blockK;

        int64_t total = m * dim;
        int64_t remain = total;

        DataCopyParams srcCopyParams;
        srcCopyParams.blockLen = dim * sizeof(float) / DATA_ALIGN_BYTES;
        srcCopyParams.srcStride = (TraitParams::blockK - dim) * sizeof(float) / DATA_ALIGN_BYTES;
        srcCopyParams.dstStride = 0;

        DataCopyParams dstCopyParams;
        dstCopyParams.blockLen = dim * sizeof(qType) / DATA_ALIGN_BYTES;
        dstCopyParams.srcStride = 0;
        dstCopyParams.dstStride = (headNum * dim - dim) * sizeof(qType) / DATA_ALIGN_BYTES;

        int64_t copyLenEachLoopAlignHeadDim = vectorScoreUbBlockElem / dim * dim;

        if constexpr (needAtomic) {
            AscendC::SetAtomicNone();
        }

        while (remain > 0) {
            int64_t thisLen = copyLenEachLoopAlignHeadDim;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t kThisOffset = inOffset + (total - remain) / dim * MAX_BLOCK_DIM;

            srcCopyParams.blockCount = static_cast<uint16_t>(thisLen / dim);
            LocalTensor<float> inLt = queIn.AllocTensor<float>();
            DataCopy(inLt, svResultGt[kThisOffset], srcCopyParams);

            queIn.EnQue(inLt);

            LocalTensor<float> newInLt = queIn.DeQue<float>();
            LocalTensor<qType> outLt = queOut.AllocTensor<qType>();
            if constexpr (std::is_same<qType, float>::value) {
                DataCopy(outLt.template ReinterpretCast<float>(), newInLt, thisLen);
            } else {
                Cast(outLt, newInLt, RoundMode::CAST_RINT, thisLen);
            }

            queOut.EnQue(outLt);
            queIn.FreeTensor(newInLt);

            LocalTensor<qType> newOutLt = queOut.DeQue<qType>();

            dstCopyParams.blockCount = static_cast<uint16_t>(thisLen / dim);
            int64_t thisLineOffset = (total - remain) / dim;
            int64_t outOffset = outStartOffset + thisLineOffset * headNum * dim;
            if constexpr (needAtomic) {
                AscendC::SetAtomicAdd<qType>();
                AscendC::SetAtomicType<qType>();
                DataCopy(attnOutputGt[outOffset], newOutLt, dstCopyParams);
                AscendC::SetAtomicNone();
            }
            else {
                DataCopy(attnOutputGt[outOffset], newOutLt, dstCopyParams);
            }
            queOut.FreeTensor(newOutLt);

            remain = remain - thisLen;
        }
    }

    // GM_ADDR
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR attnBias;
    GM_ADDR mask;

    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;

    // Shape
    int64_t batchSize;
    int64_t seqLen;
    int64_t headNum;
    int64_t dim;

    int64_t maxSeqLen;

    // Attr
    float siluScale;

    // Tpipe
    TPipe *pipe;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> biasIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queMaskIn;
    TQue<TPosition::VECCALC, USE_QUEUE_NUM> tmpBuff;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queOut;

    // Gt
    GlobalTensor<qType> qGt;
    GlobalTensor<qType> kGt;
    GlobalTensor<qType> vGt;
    GlobalTensor<qType> attnOutputGt;
    GlobalTensor<qType> attnScoreGt;
    GlobalTensor<qType> attnBiasGt;
    GlobalTensor<qType> attnMaskGt;
    GlobalTensor<float> svResultGt;

    using CopyFun = MatmulCopyFun<qType, TilingDataType>;

    using QK_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using QK_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, CopyFun::CopyQKA1, CopyFun::CopyQKB1>;
    using QK_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>;
    using QK_MM_C_T = matmul::MatmulType<qkMMCPos, CubeFormat::ND, qType, false>;
    using QK_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    static constexpr auto staticQkTilingCfg = GetMatmulApiTiling<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T>(
        qkMMConfig, MATMUL_L1_SIZE);
    matmul::Matmul<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T, staticQkTilingCfg, QK_MM_CB_T> qkMatmul;
    
    using SV_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false, LayoutMode::NONE, false,
                                        TPosition::VECOUT>;
    using SV_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using SV_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using SV_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using SV_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, nullptr, CopyFun::CopySVB1>;
    static constexpr auto staticSvTilingCfg = GetMatmulApiTiling<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T>(
        svMMConfig, MATMUL_L1_SIZE);
    matmul::Matmul<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T, staticSvTilingCfg, SV_MM_CB_T> svMatmul;
};
}  // namespace HstuDenseForward
#endif

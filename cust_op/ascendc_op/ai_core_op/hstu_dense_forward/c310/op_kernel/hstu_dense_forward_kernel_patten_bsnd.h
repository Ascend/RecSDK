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
#include "regbase_silu.h"

using namespace AscendC;
using namespace HstuForward;

namespace HstuDenseForward {

struct Args {
    // hstu normal
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR attnBias;
    // jagged
    GM_ADDR seqOffsetQ;
    GM_ADDR seqOffsetK;
    // page
    GM_ADDR seqOffsetT;
    GM_ADDR kvCache;
    GM_ADDR pageOffsets;
    GM_ADDR pageIds;
    GM_ADDR lastPageLen;
    // mask
    GM_ADDR numContext;
    GM_ADDR numTarget;

    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename qType, int blockK>
__aicore__ inline constexpr bool UseL1Cache() {
    return !std::is_same<qType, float>::value && (blockK < MAX_BLOCK_DIM);
}

template <typename qType, int blockM, int blockN, int blockK>
__aicore__ inline constexpr size_t GetL1CacheSize() {
    if constexpr (UseL1Cache<qType, blockK>()) {
        return MATMUL_L1_SIZE - 2 * sizeof(qType) * blockM * blockN;
    } else {
        return MATMUL_L1_SIZE;
    }
}

template <typename TraitParams, typename TilingDataType>
class HstuDenseForwardKernelPattenBsnd {
public:
    using qType = typename TraitParams::qType;
    using oType = typename TraitParams::oType;
    static constexpr int ElementOfBlock = DATA_ALIGN_BYTES / sizeof(qType);
    static constexpr int blockHeight = BLOCK_HEIGHT_256; // only used in dense_hstu_forward
    static constexpr auto qkMMCPos = TPosition::GM;
    static constexpr MatmulConfig qkMMConfig = lookup<sizeof(qType),
                     TraitParams::blockM, TraitParams::blockN, TraitParams::blockK>().getQKMatmulConfig();
    static constexpr MatmulConfig svMMConfig = lookup<sizeof(qType),
                     TraitParams::blockM, TraitParams::blockN, TraitParams::blockK>().getSVMatmulConfig();
    using scoreType = std::conditional_t<std::is_same<qType, fp8_e4m3fn_t>::value, float, qType>;

    __aicore__ inline HstuDenseForwardKernelPattenBsnd(int vecPerProcess = 32)
    {
        vectorScoreUbBlockElem = (vecPerProcess * blockHeight) / USE_QUEUE_NUM;
    }

    __aicore__ inline void L2CacheHintCfg(int splitmode)
    {
        if (splitmode == FAST_SPLIT_SINGLE) {
            // 多核QKV不共享，不开启QKVL2Cache
            qGt.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            kGt.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            vGt.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            attnOutputGt.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        } else if (splitmode == FAST_SPLIT) {
            // 多核Q不共享，不开启Q L2Cache
            qGt.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            attnOutputGt.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        } else {
            // 多核QKV共享，需要开启L2 Cache
            attnOutputGt.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
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

        if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
            attnOutputHalfGt.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(attnOutput));
        } else {
            attnOutputGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnOutput));
        }

        const uint32_t coreNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;

        int64_t oneBlockMidElem = TraitParams::blockM * TraitParams::blockN * COMPUTE_PIPE_NUM;
        int64_t oneCoreMidElem = coreNum * oneBlockMidElem;

        int64_t oneBlockMidTransElem = TraitParams::blockM * TraitParams::blockK * TRANS_PIPE_NUM;
        int64_t oneCoreTransMidElem = coreNum * oneBlockMidTransElem;
        int64_t kvOffset = oneCoreMidElem + oneCoreTransMidElem * 3; // svResultGt midkGt midvGt

        attnScoreGt.SetGlobalBuffer(reinterpret_cast<__gm__ scoreType*>(workspace) + GetBlockIdx() * oneBlockMidElem);
        // 申请一个attnScoreFp8Gt的workspace
        if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
            attnScoreFp8Gt.SetGlobalBuffer(
                reinterpret_cast<__gm__ fp8_e4m3fn_t*>(
                    reinterpret_cast<__gm__ int32_t*>(workspace) + oneCoreMidElem + coreNum * oneBlockMidTransElem +
                    coreNum * DATA_ALIGN_BYTES / sizeof(int32_t)) + GetBlockIdx() * oneBlockMidElem);
        }
        svResultGt.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace) + oneCoreMidElem + GetBlockIdx() * oneBlockMidTransElem,
            oneBlockMidTransElem);
        syncGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(workspace) + oneCoreMidElem + coreNum * oneBlockMidTransElem);

        pipe->InitBuffer(vecIn, USE_QUEUE_NUM, DATA_ALIGN_BYTES);

        pipe->InitBuffer(scm, USE_QUEUE_NUM, BLOCK_N * BLOCK_M * sizeof(qType));

        // Init pipe total 32K * 5 = 160K
        transUbBlockElem = vectorScoreUbBlockElem;
        pipe->InitBuffer(queIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(queOut, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(tmpBuff, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(biasIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(queMaskIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(vecOutFp8, USE_QUEUE_NUM,
            TraitParams::blockM * TraitParams::blockN * sizeof(fp8_e4m3fn_t));

        if constexpr (TraitParams::deterministic) {
            SyncAll<true>();
            if (GetBlockIdx() == 0) {
                uint32_t zeroNumber = coreNum * DATA_ALIGN_BYTES / sizeof(int32_t);
                auto zeroBuff = queOut.template AllocTensor<int32_t>();
                Duplicate<int32_t>(zeroBuff, 0, zeroNumber);
                queOut.EnQue(zeroBuff);
                auto overBuff = queOut.DeQue<int32_t>();
                pipe_barrier(PIPE_ALL);
                DataCopy(syncGm, overBuff, zeroNumber);
                queOut.template FreeTensor<int32_t>(overBuff);
            }
            SyncAll<true>();
        }
    }

    template <typename T>
    __aicore__ inline void CastQtype2Float(LocalTensor<float> distTensor, LocalTensor<T> srcTensor,
                                           LocalTensor<qType> midTensor, int64_t len)
    {
        if constexpr (!std::is_same<T, float>::value) {
            DataCopy<qType>(midTensor, srcTensor, len);
            Cast(distTensor, midTensor, RoundMode::CAST_NONE, len);
        }
    }

    __aicore__ inline void CastFloat2Qtype(LocalTensor<scoreType>& distTensor, LocalTensor<float>& srcTensor,
                                           LocalTensor<float>& midTensor, int64_t len)
    {
        if constexpr (!std::is_same<scoreType, float>::value) {
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
                CastQtype2Float<qType>(inMaskLtFp32, inMaskLt, tmpLt, thisLen);
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
            CastQtype2Float<qType>(newBiasLt, biasLt, tmpLt, thisLen);
            Add<float>(newInLt, newInLt, newBiasLt, thisLen);
            biasIn.FreeTensor(biasLt);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32NoRab(
        LocalTensor<scoreType>& inLt,
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
            SiluCompute<float>(biasLtFp32, tmpLtFp32, alpha, thisLen);
            DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, biasLtFp32, thisLen, needMask, scale);

            auto outLt = queOut.AllocTensor<qType>();
            Cast(outLt, biasLtFp32, RoundMode::CAST_RINT, thisLen);
            queOut.EnQue(outLt);
        } else {
            queIn.DeQue();

            auto outLt = queOut.AllocTensor<scoreType>();
            SiluCompute<scoreType>(outLt, inLt, alpha, thisLen);
            queIn.FreeTensor(inLt);

            DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, outLt, thisLen, needMask, scale);
            queOut.EnQue(outLt);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32(
        LocalTensor<scoreType>& inLt,
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
        CastQtype2Float<scoreType>(newInLt, inLt, tmpLt, thisLen);
        DoBiasOptional(newInLt, biasLt, tmpLt, thisLen);

        auto outLt = queOut.AllocTensor<scoreType>();
        auto newOutLt = outLt.template ReinterpretCast<float>();
        SiluCompute<float>(newOutLt, newInLt, alpha, thisLen);

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
        align = (maxSeqLenK % ElementOfBlock == 0) && (alignOfN == blockLen);

        uint16_t dstStride = (TraitParams::blockN - alignOfN) * sizeof(qType) / DATA_ALIGN_BYTES;

        if (align) {
            uint16_t copyLen = alignOfN * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t srcStride = (maxSeqLenK - blockLen) * sizeof(qType) / DATA_ALIGN_BYTES;

            DataCopyParams copyParms = { copyBlock, copyLen, srcStride, dstStride };
            DataCopy(lt, gt[offset], copyParms);
        } else {
            uint16_t copyLenBytes = blockLen * sizeof(qType);
            uint16_t srcStrideBytes = (maxSeqLenK - blockLen) * sizeof(qType);

            uint8_t padLens = alignOfN - blockLen;
            DataCopyParams copyParms = { copyBlock, copyLenBytes, srcStrideBytes, dstStride };
            DataCopyPadParams padParms = { true, 0, padLens, 0 };
            DataCopyPad(lt, gt[offset], copyParms, padParms);
        }
    }

    __aicore__ inline bool GenMask(
        LocalTensor<float>& inMaskLt, int causalMask, int64_t maskLen, int64_t maskOffset, float scale)
    {
        bool needMask = false;
        if (causalMask == 1) {
            DoCausalMask<float, CausalMaskT::MASK_TRIL>(inMaskLt, maskOffset, maskLen, TraitParams::blockN,
                maskLen / TraitParams::blockN, scale);
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
            if constexpr (std::is_same<MaskInfoType, uint32_t>::value) {
                // 处理 uint32_t 类型 dense格式使用
                needMask = GenMask(
                    inMaskLtFp32,
                    maskinfo,
                    thisLen,
                    ((maskinfo > 0) ? (blockOffset) : n),  // blockOffset为行号
                    scale);
            } else {
                // 处理 BlockMaskParams 类型
                BlockMaskGenerator blkMaskGen(maskinfo);
                needMask =
                    blkMaskGen.GenMask(inMaskLtFp32, blockOffset, thisLen / TraitParams::blockN, TraitParams::blockN);
            }

            queMaskIn.EnQue(inMaskLtFp32);
        } else if constexpr (TraitParams::maskType == CausalMaskT::MASK_CUSTOM) {
            int64_t thisMaskOffset = maskOffset + blockOffset * maxSeqLenK;
            inMaskLt = queMaskIn.AllocTensor<qType>();
            DataCopyMayPad(inMaskLt, attnMaskGt,
                           (uint16_t)(thisLen / TraitParams::blockN), n, thisMaskOffset);
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
            int64_t thisBiasOffset = biasOffset + blockOffset * maxSeqLenK;
            biasLt = biasIn.AllocTensor<qType>();
            DataCopyMayPad(biasLt, attnBiasGt,
                           (uint16_t)(thisLen / TraitParams::blockN), n, thisBiasOffset);
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
        int64_t total = m * TraitParams::blockN;
        int64_t offset = midResultIdx * TraitParams::blockM * TraitParams::blockN;

        auto tmpLt = tmpBuff.AllocTensor<qType>();
        auto tmpLtFp32 = tmpLt.template ReinterpretCast<float>();
        LocalTensor<qType> biasLt;
        LocalTensor<qType> inMaskLt;
        LocalTensor<float> inMaskLtFp32;
        LocalTensor<fp8_e4m3fn_t> tmpLtfp8;

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
            auto inLt = queIn.AllocTensor<scoreType>();
            DataCopy(inLt, attnScoreGt[thisOffset], thisLen);

            queIn.EnQue(inLt);

            int64_t blockOffset = (total - remain) / TraitParams::blockN;
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
            auto outLt = queOut.DeQue<scoreType>();
            if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
                tmpLtfp8 = vecOutFp8.AllocTensor<fp8_e4m3fn_t>();
                Cast(tmpLtfp8, outLt, RoundMode::CAST_RINT, thisLen);
                vecOutFp8.EnQue(tmpLtfp8);
                tmpLtfp8 = vecOutFp8.DeQue<fp8_e4m3fn_t>();
                DataCopy(attnScoreFp8Gt[thisOffset], tmpLtfp8, thisLen);
                vecOutFp8.FreeTensor(tmpLtfp8);
            } else {
                DataCopy(attnScoreGt[thisOffset], outLt, thisLen);
            }
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
        if constexpr (isFirst && UseL1Cache<qType, TraitParams::blockK>()) {
            this->scm.FreeTensor(this->scmQKTensor);
            int64_t dim = xDim3;
            int64_t headNum = xDim2;
            auto alignOfM = AlignUp(m, ALIGN_16);
            Nd2NzParams param = {
                1, static_cast<uint16_t>(m), static_cast<uint16_t>(k), 0,
                static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfM), 1, 0
            };
            LocalTensor<qType> scmLocal = scm.AllocTensor<qType>();
            DataCopy(scmLocal, qGt[qOffset], param);
            scm.EnQue(scmLocal);
            scmQKTensor = scm.DeQue<qType>();
        }

        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outOffset = midResultIdx * TraitParams::blockM * TraitParams::blockN;
        if constexpr (UseL1Cache<qType, TraitParams::blockK>()) {
            qkMatmul.SetTensorA(scmQKTensor);
        } else {
            qkMatmul.SetTensorA(qGt[qOffset]);
        }
        qkMatmul.SetTensorB(kGt[kOffset], true);
        qkMatmul.SetTail(m, n, k);
        qkMatmul.SetSelfDefineData(copyHeadNum); // 设置CopyQK的自定义headNum数据

        qkMatmul.template IterateAll<false>(attnScoreGt[outOffset], 0, false, true);
    }

    template<bool isFirst = true>
    __aicore__ inline void DoQkMatmulImpl(int64_t qOffset, int64_t kOffset, uint32_t taskId, uint32_t m, uint32_t n,
                                          uint32_t k, const GlobalTensor<qType>& midkGt)
    {
        if constexpr (isFirst && UseL1Cache<qType, TraitParams::blockK>()) {
            this->scm.FreeTensor(this->scmQKTensor);
            int64_t dim = xDim3;
            int64_t headNum = xDim2;
            auto alignOfM = AlignUp(m, ALIGN_16);
            Nd2NzParams param = {
                1, static_cast<uint16_t>(m), static_cast<uint16_t>(k), 0,
                static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfM), 1, 0
            };
            LocalTensor<qType> scmLocal = scm.AllocTensor<qType>();
            DataCopy(scmLocal, qGt[qOffset], param);
            scm.EnQue(scmLocal);
            scmQKTensor = scm.DeQue<qType>();
        }

        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outOffset = midResultIdx * TraitParams::blockM * TraitParams::blockN;

        if constexpr (UseL1Cache<qType, TraitParams::blockK>()) {
            qkMatmul.SetTensorA(scmQKTensor);
        } else {
            qkMatmul.SetTensorA(qGt[qOffset]);
        }
        qkMatmul.SetTensorB(midkGt[kOffset], true);
        qkMatmul.SetTail(m, n, k);
        qkMatmul.SetSelfDefineData(copyHeadNum); // 设置CopyQK的自定义headNum数据

        qkMatmul.template IterateAll<false>(attnScoreGt[outOffset], 0, false, true);
    }

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, int isAtomicAdd,
                                          uint32_t m, uint32_t n, uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * TraitParams::blockM * TraitParams::blockK;
        int64_t sOffset = midResultIdx * TraitParams::blockM * TraitParams::blockN;

        if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
            svMatmul.SetTensorA(attnScoreFp8Gt[sOffset]);
        } else {
            svMatmul.SetTensorA(attnScoreGt[sOffset]);
        }
        svMatmul.SetTensorB(vGt[vOffset]);
        svMatmul.SetTail(m, n, k);
        svMatmul.SetSelfDefineData(copyHeadNum); // 设置CopyQK的自定义headNum数据

        svMatmul.template IterateAll<false>(svResultGt[outOffset], static_cast<uint8_t>(isAtomicAdd), false, true);
    }

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, int isAtomicAdd,
                                          uint32_t m, uint32_t n, uint32_t k, const GlobalTensor<qType>& midvGt)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * TraitParams::blockM * TraitParams::blockK;
        int64_t sOffset = midResultIdx * TraitParams::blockM * TraitParams::blockN;

        if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
            svMatmul.SetTensorA(attnScoreFp8Gt[sOffset]);
        } else {
            svMatmul.SetTensorA(attnScoreGt[sOffset]);
        }
        svMatmul.SetTensorB(midvGt[vOffset]);
        svMatmul.SetTail(m, n, k);
        svMatmul.SetSelfDefineData(copyHeadNum); // 设置拷贝v矩阵的自定义headNum数据

        if (isAtomicAdd == 0) {
            // Override
            svMatmul.template IterateAll<false>(svResultGt[outOffset], 0, false, true);
        } else {
            // Automic Add
            svMatmul.template IterateAll<false>(svResultGt[outOffset], 1, false, true);
        }
    }

    template <bool needAtomic = false>
    __aicore__ inline void DoTransSvImpl(int64_t transTaskId, int64_t outStartOffset, uint32_t m)
    {
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t inOffset = outMidIndex * TraitParams::blockM * TraitParams::blockK;

        int64_t total = m * vDim;
        int64_t remain = total;

        DataCopyParams srcCopyParams;
        srcCopyParams.blockLen = vDim * sizeof(float) / DATA_ALIGN_BYTES;
        srcCopyParams.srcStride = (TraitParams::blockK - vDim) * sizeof(float) / DATA_ALIGN_BYTES;
        srcCopyParams.dstStride = 0;

        DataCopyParams dstCopyParams;
        dstCopyParams.blockLen = vDim * sizeof(qType) / DATA_ALIGN_BYTES;
        dstCopyParams.srcStride = 0;
        dstCopyParams.dstStride = (xDim2 * vDim - vDim) * sizeof(qType) / DATA_ALIGN_BYTES;

        DataCopyParams dstCopyParamsFp8;
        if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
            dstCopyParamsFp8.blockLen = vDim * sizeof(half) / DATA_ALIGN_BYTES;
            dstCopyParamsFp8.srcStride = 0;
            dstCopyParamsFp8.dstStride = (xDim2 * vDim - vDim) * sizeof(half) / DATA_ALIGN_BYTES;
        }

        int64_t copyLenEachLoopAlignHeadDim = transUbBlockElem / vDim * vDim;

        if constexpr (needAtomic) {
            AscendC::SetAtomicNone();
        }

        while (remain > 0) {
            int64_t thisLen = copyLenEachLoopAlignHeadDim;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t kThisOffset = inOffset + (total - remain) / vDim * TraitParams::blockK;
            srcCopyParams.blockCount = static_cast<uint16_t>(thisLen / vDim);
            LocalTensor<float> inLt = queIn.AllocTensor<float>();
            DataCopy(inLt, svResultGt[kThisOffset], srcCopyParams);

            queIn.EnQue(inLt);

            LocalTensor<float> newInLt = queIn.DeQue<float>();
            LocalTensor<qType> outLt;
            LocalTensor<half> outLtHalf;
            if constexpr (std::is_same<qType, float>::value) {
                outLt = queOut.AllocTensor<qType>();
                DataCopy(outLt.template ReinterpretCast<float>(), newInLt, thisLen);
                queOut.EnQue(outLt);
            } else if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
                outLtHalf = queOut.AllocTensor<half>();
                Cast(outLtHalf, newInLt, RoundMode::CAST_RINT, thisLen);
                queOut.EnQue(outLtHalf);
            } else {
                outLt = queOut.AllocTensor<qType>();
                Cast(outLt, newInLt, RoundMode::CAST_RINT, thisLen);
                queOut.EnQue(outLt);
            }
            queIn.FreeTensor(newInLt);

            dstCopyParams.blockCount = static_cast<uint16_t>(thisLen / vDim);
            if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
                dstCopyParamsFp8.blockCount = static_cast<uint16_t>(thisLen / vDim);
            }
            int64_t thisLineOffset = (total - remain) / vDim;
            int64_t outOffset = outStartOffset + thisLineOffset * xDim2 * vDim;
            if constexpr (std::is_same<qType, fp8_e4m3fn_t>::value) {
                if constexpr (needAtomic) {
                    LocalTensor<half> newOutLtHalf = queOut.DeQue<half>();
                    AscendC::SetAtomicAdd<half>();
                    AscendC::SetAtomicType<half>();
                    DataCopy(attnOutputHalfGt[outOffset], newOutLtHalf, dstCopyParamsFp8);
                    AscendC::SetAtomicNone();
                    queOut.FreeTensor(newOutLtHalf);
                } else {
                    LocalTensor<half> newOutLtHalf = queOut.DeQue<half>();
                    DataCopy(attnOutputHalfGt[outOffset], newOutLtHalf, dstCopyParamsFp8);
                    queOut.FreeTensor(newOutLtHalf);
                }
            } else {
                if constexpr (needAtomic) {
                    LocalTensor<qType> newOutLt = queOut.DeQue<qType>();
                    AscendC::SetAtomicAdd<qType>();
                    AscendC::SetAtomicType<qType>();
                    DataCopy(attnOutputGt[outOffset], newOutLt, dstCopyParams);
                    AscendC::SetAtomicNone();
                    queOut.FreeTensor(newOutLt);
                } else {
                    LocalTensor<qType> newOutLt = queOut.DeQue<qType>();
                    DataCopy(attnOutputGt[outOffset], newOutLt, dstCopyParams);
                    queOut.FreeTensor(newOutLt);
                }
            }

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
    int64_t xDim0;
    int64_t xDim1;
    int64_t xDim2;
    int64_t xDim3;
    int64_t vDim;
    
    int64_t maxSeqLenK;

    // Tiling
    int64_t seqBlockNumQk;

    // Tiling-QK
    int64_t qkTotalBlock;

    // Ub
    int64_t transUbBlockElem;

    // split
    int64_t blockSplitNum;

    // Attr
    float siluScale;
    float alpha;

    // copyQKV
    uint64_t copyHeadNum;

    int vectorScoreUbBlockElem;

    // Tpipe
    TPipe *pipe;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> biasIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queMaskIn;
    TQue<TPosition::VECCALC, USE_QUEUE_NUM> tmpBuff;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queOut;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> qkQueInA;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> qkQueInB;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> vecIn;
    TSCM<TPosition::GM, USE_QUEUE_NUM> scm;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> vecOutFp8;

    // Gt
    GlobalTensor<qType> qGt;
    GlobalTensor<qType> kGt;
    GlobalTensor<qType> vGt;
    GlobalTensor<qType> attnOutputGt;
    GlobalTensor<scoreType> attnScoreGt;
    GlobalTensor<qType> attnBiasGt;
    GlobalTensor<qType> attnMaskGt;
    GlobalTensor<float> svResultGt;
    GlobalTensor<int32_t> syncGm;
    GlobalTensor<qType> attnScoreFp8Gt;
    GlobalTensor<half> attnOutputHalfGt;

    LocalTensor<qType> scmQKTensor;

    using CopyFun = MatmulCopyFun<qType, TraitParams::blockM, TraitParams::blockN, TraitParams::blockK, TilingDataType>;

    // Matmul
    using QK_MM_A_T = std::conditional_t<
        UseL1Cache<qType, TraitParams::blockK>(),
        matmul::MatmulType<TPosition::TSCM, CubeFormat::NZ, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>
        >;
    using QK_MM_CB_T = std::conditional_t<
        UseL1Cache<qType, TraitParams::blockK>(),
        matmul::MatmulCallBackFunc<nullptr, nullptr, CopyFun::CopyQKB1>,
        matmul::MatmulCallBackFunc<nullptr, CopyFun::CopyQKA1, CopyFun::CopyQKB1>
        >;
    using QK_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>;
    using QK_MM_C_T = matmul::MatmulType<qkMMCPos, CubeFormat::ND, scoreType, false>;
    using QK_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    static constexpr auto staticQkTilingCfg = GetMatmulApiTiling<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T>(
        qkMMConfig, GetL1CacheSize<qType, TraitParams::blockM, TraitParams::blockN, TraitParams::blockK>());
    matmul::Matmul<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T, staticQkTilingCfg, QK_MM_CB_T> qkMatmul;

    using SV_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false, LayoutMode::NONE, false,
                                         TPosition::VECOUT>;
    using SV_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using SV_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using SV_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using SV_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, nullptr, CopyFun::CopySVB1>;
    static constexpr auto staticSvTilingCfg = GetMatmulApiTiling<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T>(
        svMMConfig, GetL1CacheSize<qType, TraitParams::blockM, TraitParams::blockN, TraitParams::blockK>());
    matmul::Matmul<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T, staticSvTilingCfg, SV_MM_CB_T> svMatmul;
};
}  // namespace HstuDenseForward
#endif
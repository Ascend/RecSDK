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

#ifndef HSTU_DENSE_CAUSAL_MASK_H
#define HSTU_DENSE_CAUSAL_MASK_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"

using namespace AscendC;

namespace HstuDenseBackward {

enum class CausalMaskT {
    MASK_TRIL = 0,  // 下三角
    MASK_TRIU,      // 上三角
    MASK_NONE,      // 不使能mask
    MASK_CUSTOM    // 用户自定义mask
};

struct BlockMaskParams {
    uint32_t qSeqId;          // 该基本块所属Query 输入的第几个seq block 一个block是256条seq
    uint32_t kSeqId;          // 该基本块所属Key 输入的第几个seq block 一个block是256条seq
    uint32_t seqlenQ;         // 序列总长度
    uint32_t seqlenK;         // 序列总长度
    int64_t blockHeight;      // 基本块高度
    int64_t numContext;       // context 掩码长度
    int64_t numTarget;        // target 掩码长度
    int64_t targetGroupSize;  // target 掩码group size
    int64_t historicalLen;    // 历史context长度
    float value;              // 掩码值

    __aicore__ inline BlockMaskParams() {}

    __aicore__ inline BlockMaskParams(uint32_t qSeq, uint32_t kSeq, uint32_t lenQ, uint32_t lenK, int64_t bHeight,
                                      int64_t nContext, int64_t nTarget, int64_t groupSize, float val)
        : qSeqId(qSeq),
          kSeqId(kSeq),
          seqlenQ(lenQ),
          seqlenK(lenK),
          blockHeight(bHeight),
          numContext(nContext),
          numTarget(nTarget),
          targetGroupSize(groupSize),
          value(val)

    {
    }

    __aicore__ inline bool NoComputation()
    {
        return !NeedCausalMask(false) && !NeedContextMask();
    }

    __aicore__ inline void GetFlippedBlk(int64_t* result)
    {
        // 以top到bottom为轴进行翻折，判断翻折后的block是否有计算量 用于填0逻辑
        const uint32_t top = GetTop(seqlenQ, seqlenK, blockHeight);
        result[0] = kSeqId - top + 1;  // flipped Q ID
        result[1] = top - 1 + qSeqId;  // flipped K ID
    }

    __aicore__ inline bool DiagonalNoComputation()
    {
        int64_t blk[2];
        GetFlippedBlk(blk);
        int64_t qSeqIdFlipped = blk[0];
        int64_t kSeqIdFlipped = blk[1];

        if (qSeqIdFlipped < 0 || kSeqIdFlipped < 0) {
            return false;
        }
        const uint32_t numBlockForContextMaskQ = CeilDiv(numContext, blockHeight);
        const uint32_t numBlockForContextMaskK = CeilDiv(seqlenK - numTarget, blockHeight);
        bool needCtxMask = (numContext > 0) && (qSeqIdFlipped < numBlockForContextMaskQ) &&
               (kSeqIdFlipped < numBlockForContextMaskK);

        const uint32_t top = GetTop(seqlenQ, seqlenK, blockHeight);
        bool needCasMask = (kSeqIdFlipped < qSeqIdFlipped + top);
        return !needCtxMask && !needCasMask;
    }

    __aicore__ inline bool NeedContextMask()
    {
        const uint32_t numBlockForContextMaskQ = CeilDiv(numContext, blockHeight);
        const uint32_t numBlockForContextMaskK = CeilDiv(seqlenK - numTarget, blockHeight);
        return (numContext > 0) && (qSeqId < numBlockForContextMaskQ) &&
               (kSeqId < numBlockForContextMaskK);
    }

    __aicore__ inline bool NeedCausalMask(bool diagonal = true)
    {
        const uint32_t deltaQK = seqlenK - seqlenQ;
        const int rcol = (qSeqId + 1) * blockHeight + deltaQK - 1;
        const int rblk = rcol / blockHeight;
        bool belowDig = (kSeqId <= rblk);
        // 非对角模式: 判断是否在下三角范围内
        if (!diagonal || !belowDig) {
            return belowDig;
        }
        // 对角模式: 判断是否需要创建对角线causal mask
        const int lcol = qSeqId * blockHeight + deltaQK;
        const int lblk = lcol / blockHeight;
        return (kSeqId >= lblk);
    }

    __aicore__ inline bool NeedTargetMask()
    {
        auto tbase = (seqlenK - numTarget) / blockHeight;
        return (numTarget > 0) && (targetGroupSize > 0) && (tbase <= kSeqId) && NeedCausalMask(false);
    }

    __aicore__ inline bool NeedMask()
    {
        bool triu = !NeedCausalMask(false);
        bool ctx = NeedContextMask();
        if (triu) {
            return ctx;
        }
        bool diagonal = NeedCausalMask();
        bool tar = NeedTargetMask();
        return diagonal || tar;
    }

    __aicore__ inline bool IsFirstBlockNeedOverride()
    {
        const uint32_t numBlockForContextMaskK = CeilDiv(seqlenK - numTarget, blockHeight);
        bool kIdNotInContext = (numContext <= 0) || (kSeqId >= numBlockForContextMaskK);
        const uint32_t top = GetTop(seqlenQ, seqlenK, blockHeight);
        bool isDiagonal = (kSeqId == qSeqId + top - 1);
        return qSeqId == 0 || (isDiagonal && kIdNotInContext);
    }

    static __aicore__ inline uint32_t GetTop(uint32_t lenQ, uint32_t lenK, uint32_t blkLen)
    {
        return (lenK - lenQ + 2 * blkLen - 1) / blkLen;
    }
};

class BlockMaskGenerator {
public:
    __aicore__ inline BlockMaskGenerator(BlockMaskParams* params)
    {
        qSeqId = params->qSeqId;
        kSeqId = params->kSeqId;
        seqlenQ = params->seqlenQ;
        seqlenK = params->seqlenK;
        blockHeight = params->blockHeight;
        numContext = params->numContext;
        numTarget = params->numTarget;
        targetGroupSize = params->targetGroupSize;
        value = params->value;

        needMask = params->NeedMask();
        contextMask = params->NeedContextMask();
        causalMask = params->NeedCausalMask();
        targetMask = params->NeedTargetMask();
    }

    /**
     * (Hblock x Hblock)中[line, line+height]行mask生成
     * @param inMaskLt mask写入的local tensor
     * @param line (Hblock x Hblock)中的第几行
     * @param height
     * @param width
     */
    __aicore__ inline bool GenMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int64_t total = height * width;
        if (!needMask) {
            return false;
        }
        Duplicate<float>(inMaskLt, 0, total);
        if (contextMask) {
            GenContextMask(inMaskLt, line, height, width);
        }
        if (causalMask) {
            GenCausalMask(inMaskLt, line, height, width);
        }
        if (targetMask) {
            if (!causalMask) {
                Duplicate<float>(inMaskLt, value, total);
            }
            GenTargetMask(inMaskLt, line, height, width);
        }
        return needMask;
    }

    __aicore__ inline bool NeedMask()
    {
        return needMask;
    }
    
private:
    uint32_t qSeqId;
    uint32_t kSeqId;
    uint32_t seqlenQ;
    uint32_t seqlenK;
    int64_t blockHeight;
    int64_t numContext;
    int64_t numTarget;
    int64_t targetGroupSize;
    float value;

    bool needMask;
    bool contextMask;
    bool causalMask;
    bool targetMask;

    __aicore__ inline void GenContextMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int cmaskWidth = (seqlenK - numTarget);
        int validWidth = cmaskWidth - kSeqId * blockHeight;
        if (validWidth > blockHeight) {
            validWidth = blockHeight;
        }
        uint32_t globalRowLine = qSeqId * blockHeight + line;
        for (int i = 0; i < height; i++) {
            if ((globalRowLine + i) >= numContext) {
                break;
            }
            Duplicate<float>(inMaskLt[i * width], value, validWidth);
        }
    }

    __aicore__ inline void GenCausalMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int deltaQK = seqlenK - seqlenQ;
        int startMaskWidth = deltaQK - (kSeqId - qSeqId) * blockHeight + line;
        int startMaskRow = 0;
        if (startMaskWidth < 0) {
            startMaskRow = -startMaskWidth;
            startMaskWidth = 0;
        }
        int64_t thisIndexMask = startMaskWidth + 1;
        for (int i = startMaskRow; i < height; i++) {
            thisIndexMask = thisIndexMask > blockHeight ? blockHeight : thisIndexMask;
            Duplicate<float>(inMaskLt[i * width], value, thisIndexMask);
            thisIndexMask++;
        }
    }

    __aicore__ inline void GenTargetMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int tbaseQ = seqlenQ - numTarget;
        int tbaseK = seqlenK - numTarget;
        int blkLeft = kSeqId * blockHeight;
        int blkRight = (kSeqId + 1) * blockHeight;
        int blkTop = qSeqId * blockHeight;
        int blkBottom = (qSeqId + 1) * blockHeight;

        for (int i = 0; i < height; i++) {
            // 1.找最近的小三角形
            int64_t lineInScore = line + i + blkTop;
            int64_t triNum = (lineInScore - tbaseQ) / targetGroupSize;  // 该行上方有多少个三角形
            // 2.计算三角形下方挖空边界
            int64_t triRight = tbaseK + triNum * targetGroupSize;
            if (triNum < 1 || triRight < blkLeft) {
                continue;
            }
            // 3.计算valid_rbound = min(triRight, blkRight)
            int64_t validRbound = (triRight >= blkRight) ? blkRight : triRight;
            int64_t validRboundInBlk = validRbound - blkLeft;
            // 4.计算valid_lbound = max(tbase, blkLeft)
            int64_t validLbound = (tbaseK >= blkLeft) ? tbaseK : blkLeft;
            int64_t validLboundInBlk = validLbound - blkLeft;
            // 5.计算挖空宽度
            int alignStart = validLboundInBlk * sizeof(float) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES / sizeof(float);
            int64_t validWidth = validRboundInBlk - alignStart;
            // 6.挖空
            Duplicate<float>(inMaskLt[i * width + alignStart], 0, validWidth);
            if (alignStart != validLboundInBlk) {
                int unalignlen = validLboundInBlk - alignStart;
                Duplicate<float>(inMaskLt[i * width + alignStart], value, unalignlen);
            }
        }
    }
};

template <typename qType, CausalMaskT maskType>
__aicore__ inline void DoCausalMask(LocalTensor<qType>& inMaskLt, int64_t maskOffset, int64_t maskLens,
                                    int64_t maskStride, int64_t repeatTimes, qType value)
{
    if constexpr (maskType == CausalMaskT::MASK_TRIL) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            int64_t thisIndexMask = maskOffset + i + 1;
            Duplicate<qType>(inMaskLt[i * maskStride], value, thisIndexMask);
        }
    } else if constexpr (maskType == CausalMaskT::MASK_TRIU) {
        ASCENDC_ASSERT((false), "DoCausalMask triu is unreadlized");
    } else if constexpr (maskType == CausalMaskT::MASK_NONE) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            Duplicate<qType>(inMaskLt[i * maskStride], value, maskOffset);
        }
    } else {
        ASCENDC_ASSERT((false), "DoCausalMask custom is unreadlized");
    }
}
}  // namespace HstuDenseBackward
#endif
/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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
    MASK_CUSTOME    // 用户自定义mask
};

struct BlockMaskParams {
    uint32_t qSeqId;          // 该基本块所属Query 输入的第几个seq block 一个block是256条seq
    uint32_t kSeqId;          // 该基本块所属Key 输入的第几个seq block 一个block是256条seq
    uint32_t seqlen;          // 序列总长度
    int64_t blockHeight;      // 基本块高度
    int64_t numContext;       // context 掩码长度
    int64_t numTarget;        // target 掩码长度
    int64_t targetGroupSize;  // target 掩码group size
    float value;              // 掩码值

    __aicore__ inline BlockMaskParams() {}

    __aicore__ inline BlockMaskParams(uint32_t qSeq, uint32_t kSeq, uint32_t len, int64_t bHeight, int64_t nContext,
                                      int64_t nTarget, int64_t groupSize, float val)
        : qSeqId(qSeq),
          kSeqId(kSeq),
          seqlen(len),
          blockHeight(bHeight),
          numContext(nContext),
          numTarget(nTarget),
          targetGroupSize(groupSize),
          value(val)
    {
    }

    __aicore__ inline bool NoComputation()
    {
        bool noCausal = (kSeqId > qSeqId);
        bool noContext =
            (numContext <= 0) || (qSeqId > numContext / blockHeight) || (kSeqId > (seqlen - numTarget) / blockHeight);
        return noCausal && noContext;
    }

    __aicore__ inline bool DiagonalNoComputation()
    {
        int64_t kSeqId_diagonal = qSeqId;
        int64_t qSeqId_diagonal = kSeqId;
        bool noCausal = (kSeqId_diagonal > qSeqId_diagonal);
        bool noContext = (numContext <= 0) || (qSeqId_diagonal > numContext / blockHeight) ||
                         (kSeqId_diagonal > (seqlen - numTarget) / blockHeight);
        return noCausal && noContext;
    }

    __aicore__ inline bool NeedContextMask()
    {
        return (numContext > 0) && (qSeqId <= numContext / blockHeight) &&
               (kSeqId <= (seqlen - numTarget) / blockHeight);
    }

    __aicore__ inline bool NeedCausalMask()
    {
        return (qSeqId == kSeqId);
    }

    __aicore__ inline bool NeedTargetMask()
    {
        auto tbase = (seqlen - numTarget) / blockHeight;
        return (numTarget > 0) && (targetGroupSize > 0) && (tbase <= kSeqId) && (kSeqId <= qSeqId);
    }

    __aicore__ inline bool NeedMask()
    {
        return NeedContextMask() || NeedCausalMask() || NeedTargetMask();
    }

    __aicore__ inline bool IsFirstBlockNeedOverride()
    {
        bool kIdNotInContext = (numContext <= 0) || (kSeqId > (seqlen - numTarget) / blockHeight);
        return qSeqId == 0 || (kSeqId == qSeqId && kIdNotInContext);
    }
};

class BlockMaskGenerator {
public:
    __aicore__ inline BlockMaskGenerator(BlockMaskParams* params)
    {
        qSeqId = params->qSeqId;
        kSeqId = params->kSeqId;
        seqlen = params->seqlen;
        blockHeight = params->blockHeight;
        numContext = params->numContext;
        numTarget = params->numTarget;
        targetGroupSize = params->targetGroupSize;
        value = params->value;
        contextMask = NeedContextMask();
        causalMask = NeedCausalMask();
        targetMask = NeedTargetMask();
    }

    __aicore__ inline bool NeedContextMask()
    {
        return (numContext > 0) && (qSeqId <= numContext / blockHeight) &&
               (kSeqId <= (seqlen - numTarget) / blockHeight);
    }

    __aicore__ inline bool NeedCausalMask()
    {
        return (qSeqId == kSeqId);
    }

    __aicore__ inline bool NeedTargetMask()
    {
        auto tbase = (seqlen - numTarget) / blockHeight;
        return (numTarget > 0) && (targetGroupSize > 0) && (tbase <= kSeqId) && (kSeqId <= qSeqId);
    }

    /**
     * (Hblcok x Hblock)中[line, line+height]行mask生成
     * @param inMaskLt mask写入的local tensor
     * @param line (Hblcok x Hblock)中的第几行
     * @param height
     * @param width
     */
    __aicore__ inline bool GenMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int64_t total = height * width;
        bool needMask = (contextMask || causalMask || targetMask);
        if (!needMask) {
            return needMask;
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

private:
    uint32_t qSeqId;
    uint32_t kSeqId;
    uint32_t seqlen;
    int64_t blockHeight;
    int64_t numContext;
    int64_t numTarget;
    int64_t targetGroupSize;
    float value;

    bool contextMask;
    bool causalMask;
    bool targetMask;

    __aicore__ inline void GenContextMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int cmaskWidth = (seqlen - numTarget);
        int validWidth = cmaskWidth - kSeqId * blockHeight;
        if (validWidth > blockHeight) {
            validWidth = blockHeight;
        }
        for (int i = 0; i < height; i++) {
            if ((line + i) >= numContext) {
                break;
            }
            Duplicate<float>(inMaskLt[i * width], value, validWidth);
        }
    }

    __aicore__ inline void GenCausalMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        for (int i = 0; i < height; i++) {
            int64_t thisIndexMask = line + i + 1;
            Duplicate<float>(inMaskLt[i * width], value, thisIndexMask);
        }
    }

    __aicore__ inline void GenTargetMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int tbase = seqlen - numTarget;
        int blkLeft = kSeqId * blockHeight;
        int blkRight = (kSeqId + 1) * blockHeight;
        int blkTop = qSeqId * blockHeight;
        int blkBottom = (qSeqId + 1) * blockHeight;

        for (int i = 0; i < height; i++) {
            // 1.找最近的小三角形
            int64_t lineInScore = line + i + blkTop;
            int64_t triNum = (lineInScore - tbase) / targetGroupSize;  // 该行上方有多少个三角形
            // 2.计算三角形下方挖空边界
            int64_t triBottom = tbase + triNum * targetGroupSize;
            if (triNum < 1 || triBottom < blkLeft) {
                continue;
            }
            // 3.计算valid_rbound = min(triBottom, blkRight)
            int64_t validRbound = (triBottom >= blkRight) ? blkRight : triBottom;
            int64_t validRboundInBlk = validRbound - blkLeft;
            // 4.计算valid_lbound = max(tbase, blkLeft)
            int64_t validLbound = (tbase >= blkLeft) ? tbase : blkLeft;
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
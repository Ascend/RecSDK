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
#include <tuple>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"

using namespace AscendC;

namespace HstuForward {

struct BlockMaskParams {
    int64_t qSeqId;           // 该基本块所属Query 输入的第几个seq block 一个block是256条seq
    int64_t kSeqId;           // 该基本块所属Key 输入的第几个seq block 一个block是256条seq
    int64_t qSeqLen;          // Q序列总长度
    int64_t kSeqLen;          // K序列总长度
    int64_t blockM;           // 基本块高度
    int64_t blockN;           // 基本块宽度
    int64_t numContext;       // context 掩码长度
    int64_t numTarget;        // target 掩码长度
    int64_t targetGroupSize;  // target 掩码group size
    float value;              // 掩码值

    __aicore__ inline BlockMaskParams() {}

    __aicore__ inline BlockMaskParams(int64_t qId,
                                      int64_t kId,
                                      int64_t qLen,
                                      int64_t kLen,
                                      int64_t blockM,
                                      int64_t blockN,
                                      int64_t nContext,
                                      int64_t nTarget,
                                      int64_t groupSize,
                                      float val)
        : qSeqId(qId),
          kSeqId(kId),
          qSeqLen(qLen),
          kSeqLen(kLen),
          blockM(blockM),
          blockN(blockN),
          numContext(nContext),
          numTarget(nTarget),
          targetGroupSize(groupSize),
          value(val) {}

    __aicore__ inline bool NoComputation(CausalMaskT maskType)
    {
        if (maskType != CausalMaskT::MASK_TRIL) {
            return false;
        }
        return !NeedCausalMask(false) && !NeedContextMask();
    }

    __aicore__ inline bool NeedContextMask()
    {
        const uint32_t numBlockForContextMaskQ = CeilDiv(numContext, blockM);
        const uint32_t numBlockForContextMaskK = CeilDiv(kSeqLen - numTarget, blockN);
        return (numContext > 0) && (qSeqId < numBlockForContextMaskQ) &&
               (kSeqId < numBlockForContextMaskK);
    }

    __aicore__ inline bool NeedCausalMask(bool diagonal = true)
    {
        const int deltaQK = kSeqLen - qSeqLen;
        const int64_t pointLeftBottom[2] = {(qSeqId + 1) * blockM, kSeqId * blockN};
        bool noCompute = AboveDiag(pointLeftBottom);
        // 非对角模式: 判断是否在下三角范围内
        if (!diagonal || noCompute) {
            return !noCompute;
        }
        // 对角模式: 判断是否需要创建对角线causal mask
        const int64_t pointRightTop[2] = {qSeqId * blockM, (kSeqId + 1) * blockN};
        bool nonFullMask = AboveDiag(pointRightTop);
        return nonFullMask;
    }

    __aicore__ inline bool NeedTargetMask()
    {
        const int tbase = (kSeqLen - numTarget) / blockN;
        return (numTarget > 0) && (targetGroupSize > 0) && (kSeqId >= tbase) && NeedCausalMask(false);
    }

    template<typename T>
    __aicore__ inline bool AboveDiag(T* point)
    {
        const int deltaQK = kSeqLen - qSeqLen;
        return point[1] > point[0] + deltaQK;
    }
};

class BlockMaskGenerator {
public:
    __aicore__ inline BlockMaskGenerator(BlockMaskParams& params)
    {
        qSeqId = params.qSeqId;
        kSeqId = params.kSeqId;
        qSeqLen = params.qSeqLen;
        kSeqLen = params.kSeqLen;
        blockM = params.blockM;
        blockN = params.blockN;
        numContext = params.numContext;
        numTarget = params.numTarget;
        targetGroupSize = params.targetGroupSize;
        value = params.value;
        contextMask = params.NeedContextMask();
        causalMask = params.NeedCausalMask();
        targetMask = params.NeedTargetMask();

        bool tril = params.NeedCausalMask(false);
        fullMask = tril && !causalMask && !targetMask;
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
        bool needMask = (contextMask || causalMask || targetMask);
        if (!needMask || fullMask) {
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

private:
    uint32_t qSeqId;
    uint32_t kSeqId;
    uint32_t qSeqLen;
    uint32_t kSeqLen;
    int64_t blockM;
    int64_t blockN;
    int64_t numContext;
    int64_t numTarget;
    int64_t targetGroupSize;
    float value;

    bool contextMask;
    bool causalMask;
    bool targetMask;
    bool fullMask;

    __aicore__ inline void GenContextMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        auto lineInScore = line + qSeqId * blockM;
        if (lineInScore >= numContext) {
            return;
        }
        int cmaskWidth = (kSeqLen - numTarget);
        int validWidth = cmaskWidth - kSeqId * blockN;
        if (validWidth > blockN) {
            validWidth = blockN;
        }
        int heightLeft = numContext - lineInScore;
        int validHeight = (height > heightLeft) ? heightLeft : height;
        for (int i = 0; i < validHeight; i++) {
            Duplicate<float>(inMaskLt[i * width], value, validWidth);
        }
    }

    __aicore__ inline void GenCausalMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int deltaQK = kSeqLen - qSeqLen;
        int w0 = deltaQK + qSeqId * blockM - kSeqId * blockN;

        int startLine;
        int startWidth;
        if ((w0 + line) >= 0) {
            startLine = 0;
            startWidth = w0 + line;
        } else {
            startLine = -(w0 + line);
            startWidth = 0;
        }
        int thisIndexMask = startWidth + 1;
        for (int i = startLine; i < height; i++) {
            thisIndexMask = thisIndexMask > width ? width : thisIndexMask;
            Duplicate<float>(inMaskLt[i * width], value, thisIndexMask);
            thisIndexMask++;
        }
    }

    __aicore__ inline void GenTargetMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width)
    {
        int tbaseQ = qSeqLen - numTarget;
        int tbaseK = kSeqLen - numTarget;

        int blkLeft = kSeqId * blockN;
        int blkRight = (kSeqId + 1) * blockN;
        int blkTop = qSeqId * blockM;

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
            // 4.计算valid_lbound = max(tbaseK, blkLeft)
            int64_t validLbound = (tbaseK >= blkLeft) ? tbaseK : blkLeft;
            int64_t validLboundInBlk = validLbound - blkLeft;
            // 5.计算挖空宽度
            int64_t alignStart = validLboundInBlk * sizeof(float) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES / sizeof(float);
            int64_t validWidth = validRboundInBlk - alignStart;
            // 6.挖空
            Duplicate<float>(inMaskLt[i * width + alignStart], 0, validWidth);
            if (alignStart != validLboundInBlk) {
                int64_t unalignlen = validLboundInBlk - alignStart;
                Duplicate<float>(inMaskLt[i * width + alignStart], value, unalignlen);
            }
        }
    }
};

template<typename qType, CausalMaskT maskType>
__aicore__ inline void DoCausalMask(
    LocalTensor<qType>& inMaskLt,
    int64_t maskOffset,
    int64_t maskLens,
    int64_t maskStride,
    int64_t repeatTimes,
    qType value
)
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
}  // namespace HstuForward
#endif

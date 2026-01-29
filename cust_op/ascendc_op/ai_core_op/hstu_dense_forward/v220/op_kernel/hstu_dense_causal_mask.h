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

namespace HstuDenseForward {

enum class CausalMaskT {
    MASK_TRIL = 0,  // 下三角
    MASK_TRIU,      // 上三角
    MASK_NONE,      // 不使能mask
    MASK_CUSTOM,   // 用户自定义mask
};

struct BlockMaskParams {
    int64_t qSeqId;           // 该基本块所属Query 输入的第几个seq block 一个block是256条seq
    int64_t kSeqId;           // 该基本块所属Key 输入的第几个seq block 一个block是256条seq
    int64_t qSeqLen;          // Q序列总长度
    int64_t kSeqLen;          // K序列总长度
    int64_t blockM;          // 基本块高度
    int64_t blockN;      // 基本块宽度
    int64_t numContext;       // context 掩码长度
    int64_t numTarget;        // target 掩码长度
    int64_t targetGroupSize;  // target 掩码group size
    float value;              // 掩码值
    int64_t maskOffset1;      // causal mask第一个块内偏移
    int64_t maskOffset2;      // causal mask第二个块内偏移
    int64_t nblk;             // DeltaQK/blockHeight
    bool isDeltaQK;           // 是否存在块内偏移

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
                                      float val,
                                      int64_t maskOffset1,
                                      int64_t maskOffset2,
                                      int64_t nblk,
                                      bool isDeltaQK)
        : qSeqId(qId),
          kSeqId(kId),
          qSeqLen(qLen),
          kSeqLen(kLen),
          blockM(blockM),
          blockN(blockN),
          numContext(nContext),
          numTarget(nTarget),
          targetGroupSize(groupSize),
          value(val),
          maskOffset1(maskOffset1),
          maskOffset2(maskOffset2),
          nblk(nblk),
          isDeltaQK(isDeltaQK) {}

    __aicore__ inline bool NoComputation(CausalMaskT maskType)
    {
        if (maskType != CausalMaskT::MASK_TRIL) {
            return false;
        }
        int64_t DeltaQK = kSeqLen - qSeqLen;
        int64_t qTileEnd = CeilDiv(qSeqId * blockM + DeltaQK + blockM, blockN);
        bool noCausal = (kSeqId >= qTileEnd);
        int64_t qBlks = CeilDiv(numContext, blockM);
        int64_t kBlks = CeilDiv((kSeqLen - numTarget), blockN);
        bool noContext = (qSeqId >= qBlks) || (kSeqId >= kBlks);
        return noCausal && noContext;
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
        nblk = params.nblk;
        maskOffset1 = params.maskOffset1;
        maskOffset2 = params.maskOffset2;
        contextMask = NeedContextMask();
        causalMask = NeedCausalMask();
        targetMask = NeedTargetMask();
    }

    __aicore__ inline bool NeedContextMask()
    {
        if (numContext <= 0) {
            return false;
        }
        int64_t deltaBlock = qSeqId * blockM - kSeqId * blockN;
        if (deltaBlock > blockN) {
            return false;
        }
        int64_t qBlks = CeilDiv(numContext, blockM);
        int64_t kBlks = CeilDiv((kSeqLen - numTarget), blockN);
        return (qSeqId < qBlks) && (kSeqId < kBlks);
    }

    __aicore__ inline int NeedCausalMask()
    {
        const int64_t edgeStart = qSeqId * blockM + kSeqLen - qSeqLen;
        const int64_t edgeEnd = edgeStart + blockM;
        const int64_t TileStart = kSeqId * blockN;
        const int64_t TileEnd = kSeqId * blockN + blockN;

        if (TileEnd <= edgeStart || edgeEnd <= TileStart) {
            return NO_CAUSAL_MASK;
        }
        if (edgeStart < TileStart && TileStart < edgeEnd) {
            return DIAGONAL_MASK_RIGHT;
        }
        return DIAGONAL_MASK_LEFT;
    }

    __aicore__ inline bool NeedTargetMask()
    {
        auto tbaseQ = (qSeqLen - numTarget) / blockM;
        auto tbaseK = (kSeqLen - numTarget) / blockN;
        return (numTarget > 0) && (targetGroupSize > 0) && (tbaseQ <= qSeqId) && (tbaseK <= kSeqId);
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
        if (!needMask) {
            return needMask;
        }
        Duplicate<float>(inMaskLt, 0, total);
        if (contextMask) {
            GenContextMask(inMaskLt, line, height, width);
        }
        
        // 对角线左侧梯形因果掩码
        if (causalMask == DIAGONAL_MASK_LEFT) {
            GenCausalMask(inMaskLt, line + maskOffset1, height, width);
        }
        // 对角线右侧三角因果掩码
        if (causalMask == DIAGONAL_MASK_RIGHT) {
            GenCausalMask(inMaskLt, line + maskOffset2, height, width);
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
    uint32_t nblk;
    int64_t blockM;
    int64_t blockN;
    int64_t numContext;
    int64_t numTarget;
    int64_t targetGroupSize;
    int64_t maskOffset1;
    int64_t maskOffset2;
    int64_t causalMask;
    float value;

    bool contextMask;
    bool targetMask;

    static constexpr int NO_CAUSAL_MASK = 0;      // 不需要因果掩码
    static constexpr int DIAGONAL_MASK_LEFT = 1;  // 对角线左侧梯形因果掩码
    static constexpr int DIAGONAL_MASK_RIGHT = 2; // 对角线右侧三角因果掩码

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

    __aicore__ inline void GenCausalMask(LocalTensor<float>& inMaskLt, int64_t line, int64_t height, int64_t width) //?
    {
        auto startLine = line > 0 ? 0 : -line;
        for (int i = startLine; i < height; i++) {
            int64_t thisIndexMask = line + i + 1;
            if (thisIndexMask > width) {
                Duplicate<float>(inMaskLt[i * width], value, width);
            } else {
                Duplicate<float>(inMaskLt[i * width], value, thisIndexMask);
            }
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
}  // namespace HstuDenseForward
#endif

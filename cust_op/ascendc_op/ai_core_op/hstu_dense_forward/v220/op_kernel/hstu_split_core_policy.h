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


#ifndef HSTU_SPLIT_CORE_POLICY_H
#define HSTU_SPLIT_CORE_POLICY_H

#include <unistd.h>
#include <cstdint>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"

using namespace AscendC;

namespace HstuForward {

template <typename oType, CausalMaskT maskType>
class BlockTaskAssign {
public:
    __aicore__ inline BlockTaskAssign(uint32_t coreNum,
                                      int64_t batchSize,
                                      int64_t headNum,
                                      int64_t tgsize,
                                      int64_t blockM,
                                      int64_t blockN,
                                      GlobalTensor<oType>& seqOffsetsQGt,
                                      GlobalTensor<oType>& seqOffsetsKGt,
                                      GlobalTensor<oType>& numContextGt,
                                      GlobalTensor<oType>& numTargetGt,
                                      int splitMode)
    {
        this->coreNum = coreNum;
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->tgsize = tgsize;
        this->blockM = blockM;
        this->blockN = blockN;
        this->seqOffsetsQGt = seqOffsetsQGt;
        this->seqOffsetsKGt = seqOffsetsKGt;
        this->numContextGt = numContextGt;
        this->numTargetGt = numTargetGt;
        this->bxn = batchSize * headNum;
        this->splitMode = splitMode;
    }
    
    __aicore__ inline void SplitCoreFast(int (&result)[4], int coreId)
    {
        uint32_t totalTaskNum = 0;
        if (this->splitMode == FAST_SPLIT_SINGLE) {
            totalTaskNum = this->bxn;
        } else {
            for (auto batchId = 0; batchId < batchSize; batchId++) {
                int64_t seqlenQ = seqOffsetsQGt.GetValue(batchId + 1) - seqOffsetsQGt.GetValue(batchId);
                int64_t numBlkQ = CeilDiv(seqlenQ, blockM);
                totalTaskNum += headNum * numBlkQ;
            }
        }
        uint32_t usedCoreNum = (this->coreNum > totalTaskNum) ? totalTaskNum : this->coreNum;

        uint32_t splitNextCoreProcNum = totalTaskNum / usedCoreNum;
        uint32_t splitPrevCoreProcNum = splitNextCoreProcNum + 1;
        uint32_t splitCoreIdx = totalTaskNum % usedCoreNum;
        if (coreId < splitCoreIdx) {
            result[0] = 0;
            result[1] = 0;
            result[2] = coreId * splitPrevCoreProcNum;
            result[3] = result[2] + splitPrevCoreProcNum;
        } else if (coreId < usedCoreNum) {
            result[0] = 0;
            result[1] = 0;
            result[2] = splitCoreIdx * splitPrevCoreProcNum + (coreId - splitCoreIdx) * splitNextCoreProcNum;
            result[3] = result[2] + splitNextCoreProcNum;
        } else {
            result[0] = 0;
            result[1] = 0;
            result[2] = 0;
            result[3] = 0;
        }
    }

    __aicore__ inline void SplitCoreStreamK(int (&result)[4], int coreId)
    {
        // 计算总任务量
        uint32_t totalTaskNum = 0;
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            int64_t seqLenQ = seqOffsetsQGt.GetValue(batchId + 1) - seqOffsetsQGt.GetValue(batchId);
            int64_t numBlkQ = CeilDiv(seqLenQ, blockM);
            int64_t seqLenK = seqOffsetsKGt.GetValue(batchId + 1) - seqOffsetsKGt.GetValue(batchId);
            int64_t numBlkK = CeilDiv(seqLenK, blockN);
            int64_t numCtx = numContextGt.GetValue(batchId);
            int64_t numTg = numTargetGt.GetValue(batchId);
            int64_t DeltaQK = seqLenK - seqLenQ;
            bool isDeltaQK = DeltaQK != 0;

            uint32_t seqTaskNum = ComputeSeqTaskNum(isDeltaQK, seqLenQ, seqLenK, numBlkQ, numBlkK, numCtx, numTg);
            totalTaskNum += headNum * seqTaskNum;
        }
        uint32_t usedCoreNum = (this->coreNum > totalTaskNum)? totalTaskNum : this->coreNum;
        uint32_t splitNextCoreProcNum = totalTaskNum / usedCoreNum;
        uint32_t splitPrevCoreProcNum = splitNextCoreProcNum + 1;
        uint32_t splitCoreIdx = totalTaskNum % usedCoreNum;
        
        if (coreId < splitCoreIdx) {
            result[0] = coreId * splitPrevCoreProcNum;
            result[1] = result[0] + splitPrevCoreProcNum;
        } else if (coreId < usedCoreNum) {
            result[0] = splitCoreIdx * splitPrevCoreProcNum + (coreId - splitCoreIdx) * splitNextCoreProcNum;
            result[1] = result[0] + splitNextCoreProcNum;
        } else {
            result[0] = 0;
            result[1] = 0;
            result[2] = 0;
            result[3] = 0;
        }

        int64_t eachCoreTaskNumLimit = CeilDiv(totalTaskNum, this->coreNum);
        totalTaskNum = 0;
        auto totalTaskBlock = 0;
        bool signSBlk = true;
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            int64_t seqLenQ = seqOffsetsQGt.GetValue(batchId + 1) - seqOffsetsQGt.GetValue(batchId);
            int64_t numBlkQ = CeilDiv(seqLenQ, blockM);
            int64_t seqLenK = seqOffsetsKGt.GetValue(batchId + 1) - seqOffsetsKGt.GetValue(batchId);
            int64_t numBlkK = CeilDiv(seqLenK, blockN);
            int64_t numCtx = numContextGt.GetValue(batchId);
            int64_t numTg = numTargetGt.GetValue(batchId);
            int64_t DeltaQK = seqLenK - seqLenQ;
            bool isDeltaQK = DeltaQK != 0;
            uint32_t innerSeqTaskNum = ComputeSeqTaskNum(isDeltaQK, seqLenQ, seqLenK, numBlkQ, numBlkK, numCtx, numTg);
            if (signSBlk && totalTaskNum + innerSeqTaskNum * headNum >= result[0]) {
                LocateQKseqId(result[2], result[0], isDeltaQK, seqLenQ, seqLenK,
                    numBlkQ, numBlkK, numCtx, numTg, totalTaskNum, innerSeqTaskNum);
                result[2] += totalTaskBlock;
                signSBlk = false;
            }
            if (totalTaskNum + innerSeqTaskNum * headNum >= result[1]) {
                LocateQKseqId(result[3], result[1], isDeltaQK, seqLenQ, seqLenK,
                    numBlkQ, numBlkK, numCtx, numTg, totalTaskNum, innerSeqTaskNum);
                result[3] += totalTaskBlock;
                break;
            }
            totalTaskNum += headNum * innerSeqTaskNum;
            totalTaskBlock += headNum * numBlkQ;
        }
    }

    __aicore__ inline void Compute(int (&result)[4], int coreId)
    {
        if (this->splitMode == STREAM_K) {
            SplitCoreStreamK(result, coreId);
        } else {
            SplitCoreFast(result, coreId);
        }
    }

private:
    __aicore__ inline uint32_t ComputeSeqTaskBlockNum(bool isDeltaQK, int64_t seqLenQ, int64_t seqLenK,
                             int64_t numBlkQ, int64_t numBlkK, int64_t numCtx, int64_t numTg, uint32_t seqTaskNum)
    {
        if constexpr (maskType != CausalMaskT::MASK_TRIL) {
            return seqTaskNum / numBlkK;
        }
        
        int64_t sum = 0;
        int64_t numBlkidx = 0;
        int64_t numCtxCol = min(numBlkQ, CeilDiv(numCtx, blockM));
        int64_t numCtxRow = CeilDiv(seqLenK - numTg, blockN);

        for (; numBlkidx < numCtxCol; numBlkidx++) {
            sum += numCtxRow;
            if (seqTaskNum < sum) {
                return numBlkidx;
            }
        }
        numBlkidx++;
        for (; numBlkidx < numBlkQ; numBlkidx++) {
            sum += CeilDiv(seqLenK - seqLenQ + numBlkidx * blockM, blockN);
            if (seqTaskNum < sum) {
                return numBlkidx - 1;
            }
        }
        return numBlkQ - 1;
    }

    __aicore__ inline uint32_t ComputeSeqTaskNum(bool isDeltaQK, int64_t seqLenQ, int64_t seqLenK,
                                                int64_t numBlkQ, int64_t numBlkK, int64_t numCtx, int64_t numTg)
    {
        if constexpr (maskType != CausalMaskT::MASK_TRIL) {
            return numBlkQ * numBlkK;
        }
        
        int64_t numCtxCol = min(numBlkQ, CeilDiv(numCtx, blockM));
        int64_t numCtxRow = CeilDiv(seqLenK - numTg, blockN);
        uint32_t seqTaskNum = numCtxRow * numCtxCol;
        int64_t numBlkidx = numCtxCol;

        for (; numBlkidx < numBlkQ - 1; numBlkidx++) {
            seqTaskNum += CeilDiv(seqLenK - seqLenQ + numBlkidx * blockM + blockM, blockN);
        }

        if (numBlkidx == numBlkQ - 1) {
            seqTaskNum += CeilDiv(seqLenK, blockN); // 处理最后一行，可能存在缺角
        }
        
        return seqTaskNum;
    }

    __aicore__ inline void LocateQKseqId(int32_t &QseqId, int32_t &KseqId, bool isDeltaQK, int64_t seqLenQ,
        int64_t seqLenK, int64_t numBlkQ, int64_t numBlkK, int64_t numCtx,
        int64_t numTg, uint32_t totalTaskNum, uint32_t innerSeqTaskNum)
    {
        QseqId = ComputeSeqTaskBlockNum(isDeltaQK, seqLenQ, seqLenK, numBlkQ,
            numBlkK, numCtx, numTg, (KseqId - totalTaskNum) % innerSeqTaskNum) +
            (KseqId - totalTaskNum) / innerSeqTaskNum * numBlkQ;
        KseqId = (KseqId - totalTaskNum) % innerSeqTaskNum;
        if constexpr (maskType != CausalMaskT::MASK_TRIL) {
            KseqId %= numBlkK;
            return;
        }
        int64_t step = 1;
        int64_t numCtxRow = CeilDiv(seqLenK - numTg, blockN);
        int64_t task = step > CeilDiv(numCtx, blockM) ? CeilDiv(seqLenK - seqLenQ + step * blockM, blockN) : numCtxRow;
        while (KseqId >= task) {
            KseqId -= task;
            step++;
            task = step > CeilDiv(numCtx, blockM) ? CeilDiv(seqLenK - seqLenQ + step * blockM, blockN) : numCtxRow;
        }
    }
    uint32_t coreNum;
    int64_t blockLen;
    int64_t batchSize;
    int64_t headNum;
    int64_t tgsize;
    uint32_t bxn;  // 总序列数
    uint32_t splitMode;
    int64_t blockM;
    int64_t blockN;

    GlobalTensor<oType> seqOffsetsQGt;
    GlobalTensor<oType> seqOffsetsKGt;
    GlobalTensor<oType> numContextGt;
    GlobalTensor<oType> numTargetGt;
};
}  // namespace HstuDenseForward
#endif

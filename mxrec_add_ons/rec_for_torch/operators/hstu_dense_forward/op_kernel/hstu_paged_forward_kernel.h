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
#ifndef HSTU_DENSE_FORWARD_PAGED_KERNEL_FUN_H
#define HSTU_DENSE_FORWARD_PAGED_KERNEL_FUN_H


#include "hstu_dense_forward_kernel_patten_bsnd.h"
#include "hstu_dense_forward_jagged_kernel.h"

using namespace AscendC;

namespace HstuDenseForward {
constexpr int KVUBSIZE = 16384; // 176KB-160KB
constexpr int64_t CONST_2 = 2;
template <typename qType, typename oType>
class HstuDenseForwardPagedKernel : public HstuDenseForwardJaggedKernel<qType, oType> {
public:
    __aicore__ inline HstuDenseForwardPagedKernel() {}
    __aicore__ inline void Compute(const HstuDenseForwardTilingData* __restrict tilingDataPtr);
    __aicore__ inline void Init(const Args& args,
                                const HstuDenseForwardTilingData* __restrict tilingDataPtr,
                                TPipe* pipePtr);
    __aicore__ inline void ComputeAllBlock();
    __aicore__ inline void ComputeTailBlock(uint32_t taskId,
                                            uint32_t currentTaskId,
                                            uint32_t preTaskId,
                                            uint32_t transtaskId);
    __aicore__ inline void ComputeSvMatmul(uint32_t taskId);
    __aicore__ inline void ComputeQkMatmul(uint32_t taskId);
private:
    __aicore__ inline void InitPagedArgs(const Args& args,
                                         const HstuDenseForwardTilingData* __restrict tilingDataPtr,
                                         TPipe* pipePtr);

    __aicore__ inline void GetTaskInfo(uint32_t sBlkId);

    __aicore__ inline void UpdateTaskInfo(uint32_t taskId);

    __aicore__ inline void FillTaskInfoPaged(uint32_t batchId, uint32_t taskId);

    __aicore__ inline void FetchKvMayFromCache(uint32_t taskId);

    __aicore__ inline void CopyFromKvCache(uint32_t pageSid, uint32_t pageNum, uint32_t taskId);

    __aicore__ inline void CopySeqFromGT(const GlobalTensor<qType>& dstGt,
                                         const GlobalTensor<qType>& srcGt,
                                         uint32_t seqLen);
    __aicore__ inline void CopyFromKvInputCache(uint32_t taskId, uint32_t pageSid, uint32_t pageNum,
                                                uint32_t cacheLen, uint32_t candLen);

    // GM_ADDR
    GM_ADDR kvCache;
    GM_ADDR pageOffset;
    GM_ADDR pageIds;
    GM_ADDR lastPageLen;

    int32_t pageSize;
    GlobalTensor<oType> pageOffsetGt;
    GlobalTensor<oType> pageIdsGt;
    GlobalTensor<oType> lastPageLenGt;
    GlobalTensor<qType> kvCacheGt;
    GlobalTensor<qType> midkGt;
    GlobalTensor<qType> midvGt;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, USE_QUEUE_NUM> queKv;
    uint32_t kvLtUbSize = KVUBSIZE;
};

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::Compute(
    const HstuDenseForwardTilingData* __restrict tilingDataPtr)
{
    int ret = this->PreInit(tilingDataPtr);
    if (ret == -1) {
        return; // no task
    }
    ComputeAllBlock();
}


template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::Init(
    const Args& args,
    const HstuDenseForwardTilingData* __restrict tilingDataPtr,
    TPipe* pipePtr)
{
    this->InitArgs(args, tilingDataPtr);
    this->InitPipe(pipePtr);
    InitPagedArgs(args, tilingDataPtr, pipePtr);
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::InitPagedArgs(
    const Args& args,
    const HstuDenseForwardTilingData* __restrict tilingDataPtr,
    TPipe* pipePtr)
{
    kvCache = args.kvCache;
    pageOffset = args.pageOffsets;
    pageIds = args.pageIds;
    lastPageLen = args.lastPageLen;
    pageSize = tilingDataPtr->pageSize;

    int64_t oneBlockMidElem = this->blockHeight * this->blockHeight * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = this->blockHeight * this->xDim3 * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

    int64_t kOffset = (oneCoreMidElem + oneCoreTransMidElem) * sizeof(float) / sizeof(qType);
    int64_t vOffset = kOffset + oneCoreTransMidElem;

    midkGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(this->workspace) + kOffset + \
                           GetBlockIdx() * oneBlockMidTransElem, oneBlockMidTransElem);
    midvGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(this->workspace) + vOffset + \
                           GetBlockIdx() * oneBlockMidTransElem, oneBlockMidTransElem);
    pageOffsetGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(pageOffset), this->xDim0 + 1);
    pageIdsGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(pageIds)); // todo 初始化大小
    lastPageLenGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(lastPageLen), this->xDim0 + 1);
    kvCacheGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(kvCache));
    // 176 - 160 -> 16KB
    pipePtr->InitBuffer(queKv, USE_QUEUE_NUM, kvLtUbSize);

    // copy kv
    this->copyHeadNum = 1;
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::CopyFromKvCache(uint32_t pageSid,
                                                                                  uint32_t pageNum,
                                                                                  uint32_t taskId)
{
    // datacopy pageSid
    int64_t headId = static_cast<int64_t>(this->computeTaskInfo[taskId % COMPUTE_PIPE_NUM].headId);
    uint32_t taskOffset = taskId * this->blockHeight * this->xDim3;

    uint32_t batchId = this->computeTaskInfo[taskId].batchId;
    int64_t totalPageNum = this->computeTaskInfo[taskId].pageNum;
    int64_t lastPageIdx = pageOffsetGt.GetValue(batchId + 1) - 1;
    int64_t lastPageLen = lastPageLenGt.GetValue(batchId);

    for (uint32_t i = 0; i < pageNum; i++) {
        int64_t pageIdx = pageIdsGt.GetValue(pageSid + i); // [page_num, 2, pageSize, num_head, head_dim]
        // [pageIdx, 0, pageSize, headId, head_dim]
        int64_t offsetK = pageIdx * CONST_2 * pageSize * this->xDim2 * this->xDim3 + \
                          headId * this->xDim3;
        
        // [pageIdx, 1, pageSize, headId, head_dim]
        int64_t offsetV = pageIdx * CONST_2 * pageSize * this->xDim2 * this->xDim3 + \
                          pageSize * this->xDim2 * this->xDim3 + \
                          headId * this->xDim3;
        int64_t dstOffset = taskOffset + pageSize * i * this->xDim3;

        if (lastPageLen > 0 && (i + pageSid) == lastPageIdx) {
            CopySeqFromGT(midkGt[dstOffset], kvCacheGt[offsetK], lastPageLen);
            CopySeqFromGT(midvGt[dstOffset], kvCacheGt[offsetV], lastPageLen);
        } else {
            CopySeqFromGT(midkGt[dstOffset], kvCacheGt[offsetK], pageSize);
            CopySeqFromGT(midvGt[dstOffset], kvCacheGt[offsetV], pageSize);
        }
    }
    this->computeTaskInfo[taskId].kvOffset = taskOffset;
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::CopySeqFromGT(
    const GlobalTensor<qType>& dstGt,
    const GlobalTensor<qType>& srcGt,
    uint32_t seqLen)
{
    uint16_t blockLen = this->xDim3 * sizeof(qType) / DATA_ALIGN_BYTES;
    // 前一个数据尾和后一个数据头的间隔 (num_head - 1) * head_dim
    uint16_t srcGap = (this->xDim2 - 1) * this->xDim3 * sizeof(qType) / DATA_ALIGN_BYTES;
    uint16_t totalBlockCnt = seqLen;
    uint16_t maxBlockCnt = kvLtUbSize / this->xDim3 / sizeof(qType);
    uint16_t dstGap = (this->blockHeight - pageSize) * sizeof(qType) / DATA_ALIGN_BYTES;
    uint32_t copyedCnt = 0;
    while (totalBlockCnt > 0) {
        // Copy K from kvCache->Lt
        auto kvLt = queKv.template AllocTensor<qType>();
        uint16_t blockCnt = totalBlockCnt > maxBlockCnt ? maxBlockCnt : totalBlockCnt;
        DataCopyParams copyInK(blockCnt, blockLen, srcGap, 0);

        DataCopy(kvLt, srcGt[copyedCnt * this->xDim2 * this->xDim3], copyInK);
        queKv.EnQue(kvLt);
        auto newKvLt = queKv.DeQue<qType>();

        // Copy Lt-> workspace
        DataCopyParams copyOut(blockCnt, blockLen, 0, 0);
        DataCopy(dstGt[copyedCnt * this->xDim3], newKvLt, blockCnt * this->xDim3);
        copyedCnt += blockCnt;
        totalBlockCnt -= blockCnt;
        queKv.template FreeTensor(kvLt);
    }
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::CopyFromKvInputCache(uint32_t taskId,
                                                                                       uint32_t pageSid,
                                                                                       uint32_t pageNum,
                                                                                       uint32_t cacheLen,
                                                                                       uint32_t candLen)
{
    // copy kv from kv cache
    int64_t taskOffset = taskId * this->blockHeight * this->xDim3;
    CopyFromKvCache(pageSid, pageNum, taskId);
    // copy kv from input 偏移newhistorylen
    int64_t offset = this->computeTaskInfo[taskId].batchOffset * this->headDim * this->headNum + \
                     this->computeTaskInfo[taskId].actualNewHistLen * this->headNum * this->headDim + \
                     this->computeTaskInfo[taskId].headId * this->headDim;
    CopySeqFromGT(midkGt[taskOffset + cacheLen * this->xDim3], this->kGt[offset], candLen);
    CopySeqFromGT(midvGt[taskOffset + cacheLen * this->xDim3], this->vGt[offset], candLen);

    this->computeTaskInfo[taskId].kvOffset = taskOffset;
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::FetchKvMayFromCache(uint32_t taskId)
{
    auto batchId = this->computeTaskInfo[taskId].batchId; // 当前seqlen计算的长度
    auto computeLen = this->computeTaskInfo[taskId].computeBSeqLen;
    auto seqLenStart = this->computeTaskInfo[taskId].kSeqId * this->blockHeight;
    auto seqLenEnd = seqLenStart + computeLen;
    uint32_t pageNum = this->blockHeight / pageSize;
    if (seqLenEnd <= this->computeTaskInfo[taskId].actualHistLen) { // 当前计算结尾小于历史序列
        int32_t pageSid = seqLenStart / pageSize + pageOffsetGt.GetValue(batchId);
        CopyFromKvCache(pageSid, this->blockHeight / pageSize, taskId);
    } else if (seqLenStart >= this->computeTaskInfo[taskId].actualHistLen) {
        // 从inputkv里拷贝
        uint32_t taskOffset = taskId * this->blockHeight * this->headDim;
        auto diffHistLen = this->computeTaskInfo[taskId].actualHistLen - this->computeTaskInfo[taskId].actualNewHistLen;
        auto inputkvStart = seqLenStart - diffHistLen;
        int64_t offset = this->computeTaskInfo[taskId].batchOffset * this->headDim * this->headNum + \
                          inputkvStart * this->headNum * this->headDim + \
                          this->computeTaskInfo[taskId].headId * this->headDim;
        CopySeqFromGT(midkGt[taskOffset], this->kGt[offset], computeLen);
        CopySeqFromGT(midvGt[taskOffset], this->vGt[offset], computeLen);
        this->computeTaskInfo[taskId].kvOffset = taskOffset;
    } else {
        // 部分从inputkv拷贝部分从kvcache拷贝
        // kvCache拷贝起始pageid
        int32_t pageSid = seqLenStart / pageSize + pageOffsetGt.GetValue(batchId);
        int32_t kvpageNum = (this->computeTaskInfo[taskId].actualHistLen - seqLenStart + pageSize - 1) / pageSize;
        int32_t candLen = seqLenStart + computeLen - this->computeTaskInfo[taskId].actualHistLen;
        CopyFromKvInputCache(taskId,
                             pageSid,
                             kvpageNum,
                             this->computeTaskInfo[taskId].actualHistLen - seqLenStart,
                             candLen);
    }
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::ComputeSvMatmul(uint32_t taskId)
{
    int isAtomic = 1;
    if (this->computeTaskInfo[taskId].kSeqId == 0) {
        isAtomic = 0;
    }

    this->DoSvMatmulImpl(this->computeTaskInfo[taskId].kvOffset, taskId, this->computeTaskInfo[taskId].transTaskId,
                         isAtomic, this->computeTaskInfo[taskId].computeASeqLen, this->headDim,
                         this->computeTaskInfo[taskId].computeBSeqLen, this->midvGt);
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::ComputeQkMatmul(uint32_t taskId)
{
    this->DoQkMatmulImpl(this->computeTaskInfo[taskId].ioOffset, this->computeTaskInfo[taskId].kvOffset, taskId,
                         this->computeTaskInfo[taskId].computeASeqLen, this->computeTaskInfo[taskId].computeBSeqLen,
                         this->headDim, this->midkGt);
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::ComputeAllBlock()
{
    GetTaskInfo(this->sBlkId);

    uint32_t taskId = 0;
    uint32_t transtaskId = 0;

    uint32_t currentTaskId = 0;
    uint32_t preTaskId = 0;
    uint32_t prePreTaskId = 0;
    uint32_t nextTaskId = 0;
    uint32_t nblk = 0;

    for (auto blkId = this->sBlkId; blkId < this->eBlkId; blkId++) {
        auto kSeqNum = this->computeTaskInfo[taskId % COMPUTE_PIPE_NUM].kSeqNum;
        auto deltaQK = this->computeTaskInfo[taskId % COMPUTE_PIPE_NUM].deltaQK;
        nblk = deltaQK / this->blockHeight;
        bool isDeltaQK = deltaQK % this->blockHeight != 0;
        int64_t maskOffset1 = deltaQK % this -> blockHeight;
        int64_t maskOffset2 = deltaQK % this -> blockHeight - this -> blockHeight;

        for (auto kSeqId = 0; kSeqId < kSeqNum; kSeqId++) {
            auto taskinfo = this->computeTaskInfo[taskId % COMPUTE_PIPE_NUM];
            BlockMaskParams maskinfo = {
                taskinfo.qSeqId,
                static_cast<uint32_t> (kSeqId),
                taskinfo.actualSeqLen,
                this->blockHeight,
                taskinfo.numContext,
                taskinfo.numTarget,
                this->targetGroupSize,
                taskinfo.scale,
                maskOffset1,
                maskOffset2,
                nblk,
                isDeltaQK
            };
            // 在下三角下跳过运算
            if (maskinfo.NoComputation(this->maskType)) {
                break;
            }
            currentTaskId = taskId % COMPUTE_PIPE_NUM;
            preTaskId = (taskId - 1) % COMPUTE_PIPE_NUM;
            prePreTaskId = (taskId - 2) % COMPUTE_PIPE_NUM;
            nextTaskId = (taskId + 1) % COMPUTE_PIPE_NUM;

            this->maskTaskInfo[currentTaskId] = maskinfo;
            this->computeTaskInfo[currentTaskId].transTaskId = transtaskId % TRANS_PIPE_NUM;
            this->computeTaskInfo[currentTaskId].kSeqId = kSeqId;
            this->computeTaskInfo[currentTaskId].computeBSeqLen =
                   (kSeqId != (kSeqNum - 1)) ? (this->blockHeight) :
                   (this->computeTaskInfo[currentTaskId].actualSeqLenK - kSeqId * this->blockHeight);
            
            // fetch kv data
            FetchKvMayFromCache(currentTaskId);
            pipe_barrier(PIPE_ALL);

            // matmul qk
            this->ComputeQkMatmul(currentTaskId);

            // matmul sv
            if (taskId > 1) {
                this->ComputeSvMatmul(prePreTaskId);
            }

            // VecScore
            if (taskId > 0) {
                this->ComputeVecScore(preTaskId);
            }

            // wait qk
            this->WaitQkMatmul();

            // wait sv
            if (taskId > 1) {
                this->WaitSvMatmul();
            }

            this->computeTaskInfo[nextTaskId] = this->computeTaskInfo[currentTaskId];
            this->maskTaskInfo[nextTaskId] = this->maskTaskInfo[currentTaskId];
            taskId++;
        }

        this->transTaskInfo[transtaskId % TRANS_PIPE_NUM] = this->computeTaskInfo[currentTaskId];
        if (transtaskId > 1) {
            this->TransResult((transtaskId - 2) % TRANS_PIPE_NUM);
        }
        transtaskId++;

        this->UpdateTaskInfo(taskId % COMPUTE_PIPE_NUM);
    }
    ComputeTailBlock(taskId, currentTaskId, preTaskId, transtaskId);
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::ComputeTailBlock(uint32_t taskId,
                                                                                   uint32_t currentTaskId,
                                                                                   uint32_t preTaskId,
                                                                                   uint32_t transtaskId)
{
    if (taskId == 0) {
        return;
    }

    if (taskId == 1) {
        this->ComputeVecScore(currentTaskId);
        pipe_barrier(PIPE_ALL);

        this->ComputeSvMatmul(currentTaskId);
        this->WaitSvMatmul();

        this->TransResult((transtaskId - 1) % TRANS_PIPE_NUM);
        return;
    }

    if (transtaskId == 1) {
        this->ComputeSvMatmul(preTaskId);
        this->WaitSvMatmul();

        this->ComputeVecScore(currentTaskId);
        pipe_barrier(PIPE_ALL);

        this->ComputeSvMatmul(currentTaskId);
        this->WaitSvMatmul();
        this->TransResult((transtaskId - 1) % TRANS_PIPE_NUM);
        return;
    }

    this->ComputeSvMatmul(preTaskId);
    this->WaitSvMatmul();

    this->ComputeVecScore(currentTaskId);
    pipe_barrier(PIPE_ALL);

    this->ComputeSvMatmul(currentTaskId);
    this->WaitSvMatmul();

    this->TransResult((transtaskId - 2) % TRANS_PIPE_NUM);
    this->TransResult((transtaskId - 1) % TRANS_PIPE_NUM);
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::FillTaskInfoPaged(uint32_t batchId,
                                                                                    uint32_t taskId)
{
    if (batchId >= this->batchSize) {
        return;
    }

    taskId = taskId % COMPUTE_PIPE_NUM;
    this->computeTaskInfo[taskId].actualHistLen = this->computeTaskInfo[taskId].actualSeqLenK -
                                                  this->numTargetGt.GetValue(batchId);
    this->computeTaskInfo[taskId].actualNewHistLen = this->computeTaskInfo[taskId].actualSeqLen -
                                                     this->numTargetGt.GetValue(batchId);
    this->computeTaskInfo[taskId].pageNum = pageOffsetGt.GetValue(batchId + 1) - pageOffsetGt.GetValue(batchId);
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::UpdateTaskInfo(uint32_t taskId)
{
    auto batchId = this->computeTaskInfo[taskId].batchId;
    auto headId = this->computeTaskInfo[taskId].headId;

    int64_t seqGlobalOffset = this->computeTaskInfo[taskId].seqGlobalOffset;
    int64_t gap = this->computeTaskInfo[taskId].headSeqLimit - seqGlobalOffset;

    if (gap <= this->blockHeight) {
        headId++;
        if (headId >= this->headNum) {
            batchId++;
        }

        if (batchId >= this->batchSize) {
            return;
        }

        seqGlobalOffset = seqGlobalOffset + gap;
        headId = headId % this->headNum;
        this->FillTaskInfo(batchId, headId, seqGlobalOffset, taskId);
        FillTaskInfoPaged(batchId, taskId);
    } else {
        this->computeTaskInfo[taskId].seqGlobalOffset = seqGlobalOffset + this->blockHeight;

        uint32_t computeASeqLen = this->blockHeight;
        uint32_t computeHeadSeq = this->computeTaskInfo[taskId].seqGlobalOffset + this->blockHeight;
        if (computeHeadSeq > this->computeTaskInfo[taskId].headSeqLimit) {
            computeASeqLen = this->computeTaskInfo[taskId].headSeqLimit -
                             this->computeTaskInfo[taskId].seqGlobalOffset;
        }

        auto batchInnerOffset =
            this->computeTaskInfo[taskId].seqGlobalOffset - (this->computeTaskInfo[taskId].batchOffset * this->headNum);
        this->computeTaskInfo[taskId].qSeqId =
            (batchInnerOffset - this->computeTaskInfo[taskId].headId * this->computeTaskInfo[taskId].actualSeqLen) /
            this->blockHeight;
        this->computeTaskInfo[taskId].ioOffset =
            this->computeTaskInfo[taskId].batchOffset * this->headDim * this->headNum + \
            this->computeTaskInfo[taskId].qSeqId * this->blockHeight * this->headNum * this->headDim + \
            this->computeTaskInfo[taskId].headId * this->headDim;
        this->computeTaskInfo[taskId].computeASeqLen = computeASeqLen;
    }
}

template <typename qType, typename oType>
__aicore__ inline void HstuDenseForwardPagedKernel<qType, oType>::GetTaskInfo(uint32_t sBlkId)
{
    uint32_t offsetOfBlk = 0;
    int64_t offsetOfSeq = 0;
    int64_t seqGlobalOffset = 0;

    for (auto index = 0; index < this->batchSize * this->headNum; index++) {
        uint32_t batchId = index / this->headNum;
        uint32_t headId = index % this->headNum;

        uint32_t batchSeqSize = this->seqOffsetsQGt.GetValue(batchId + 1) - this->seqOffsetsQGt.GetValue(batchId);
        uint32_t batchBlkSize = (batchSeqSize + this->blockHeight - 1) / this->blockHeight;

        if (sBlkId < (offsetOfBlk + batchBlkSize)) {
            uint32_t innerBlkId = sBlkId - offsetOfBlk;
            seqGlobalOffset = seqGlobalOffset + innerBlkId * this->blockHeight;
            this->FillTaskInfo(batchId, headId, seqGlobalOffset, 0);
            FillTaskInfoPaged(batchId, 0);
            return;
        }

        offsetOfBlk += batchBlkSize;
        seqGlobalOffset += batchSeqSize;
    }
}
}

#endif
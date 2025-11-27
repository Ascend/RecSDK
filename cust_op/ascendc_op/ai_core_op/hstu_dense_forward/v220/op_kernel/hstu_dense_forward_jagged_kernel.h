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
#ifndef HSTU_DENSE_FORWARD_JAGGED_KERNEL_FUN_H
#define HSTU_DENSE_FORWARD_JAGGED_KERNEL_FUN_H

#include "hstu_dense_forward_kernel_patten_bsnd.h"
#include "hstu_split_core_policy.h"
#include "hstu_dense_causal_mask.h"

using namespace AscendC;

namespace HstuDenseForward {

template <typename oType>
__aicore__ inline int64_t GetBatchSizeFromJaggedOffset(GlobalTensor<oType>& seqOffsetData, int32_t seqOffsetLens)
{
    if (seqOffsetLens <= 0) {
        return 0;
    }

    // 二分法找出有效batch
    int64_t maxValue = seqOffsetData.GetValue(seqOffsetLens - 1);
    int32_t left = 0;
    int32_t right = seqOffsetLens - 1;
    int32_t firstMaxIdx = seqOffsetLens - 1;
    while (left <= right) {
        int32_t mid = left + (right - left) / 2;  // 二分法除以2找到剩余中间位置
        if (seqOffsetData.GetValue(mid) == maxValue) {
            firstMaxIdx = mid;
            right = mid - 1;
        } else if (seqOffsetData.GetValue(mid) < maxValue) {
            left = mid + 1;
        }
    }

    int64_t batchSize = static_cast<int64_t>(firstMaxIdx);
    return batchSize;
}

struct JaggedTaskArgs {
    uint32_t batchId = 0;         // 该基本块所属的batch
    uint32_t headId = 0;          // 该基本块所属的head
    uint32_t qSeqId = 0;          // 该基本块所属Query 输入的第几个seq block 一个block是256条seq
    uint32_t kSeqId = 0;          // 该基本块所属Key 输入的第几个seq block 一个block是256条seq
    uint32_t actualSeqLen = 0;    // Q序列的基本块实际的序列长度
    uint32_t actualSeqLenK = 0;    // K序列的基本块实际的序列长度
    uint32_t actualHistLen = 0;    // KV序列的历史序列长度
    uint32_t actualNewHistLen = 0; // Q序列的历史序列长度
    uint32_t qSeqNum = 0;         // 该Batch下Qblock数
    uint32_t kSeqNum = 0;         // 该基本块在K轴需要乘多少次
    uint32_t transTaskId = 0;     // 该基本块转置任务的id
    uint32_t computeASeqLen = 0;  // 该基本块matmul计算左矩阵的序列长度
    uint32_t computeBSeqLen = 0;  // 该基本块matmul计算右矩阵的序列长度
    float scale = 0.0f;           // 该基本块的siluScale
    int64_t numContext = 0;       // 该基本块所属序列的numContext
    int64_t numTarget = 0;        // 该基本块所属序列的numTarget
    int64_t seqGlobalOffset = 0;  // 该基本块的全局序列偏移
    int64_t batchOffset = 0;      // 该基本块的batch偏移
    int64_t batchOffsetK = 0;      // K序列的基本块的batch偏移
    int64_t headSeqLimit = 0;     // 该基本块的head offset最大长度, 超过则需要考虑切换head_id
    int64_t kvOffset = 0;         // 该基本块的key value计算偏移
    int64_t ioOffset = 0;         // 该基本块的query attnOutput计算偏移
    int32_t deltaQK = 0;         // QK序列长度差
    int64_t pageNum = 0;          // 该基本块存在kvcache中的page个数
};

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
class HstuDenseForwardJaggedKernel : public HstuDenseForwardKernelPattenBsnd<qType, enableBias, isQkUseUb, maskType> {
public:
    __aicore__ inline HstuDenseForwardJaggedKernel() {}

    __aicore__ inline void Compute(const HstuDenseForwardTilingData* __restrict tilingDataPtr);

    __aicore__ inline void ComputeAllBlock();

    __aicore__ inline int PreInit(const HstuDenseForwardTilingData* __restrict tilingDataPtr);

    __aicore__ inline void GetTaskInfo(uint32_t sBlkId);

    __aicore__ inline void UpdateTaskInfo(uint32_t taskId);

    __aicore__ inline void FillTaskInfo(uint32_t batchId, uint32_t head_id, int64_t seqGlobalOffset, uint32_t taskId);

    __aicore__ inline void ComputeQkMatmul(uint32_t taskId);

    __aicore__ inline void ComputeVecScore(uint32_t taskId);

    __aicore__ inline void ComputeSvMatmul(uint32_t taskId);

    __aicore__ inline void TransResult(uint32_t transtaskId);

    uint32_t sBlkId{0};
    uint32_t eBlkId{0};

    uint32_t batchSize{0};
    uint32_t seqLen{0};
    uint32_t headNum{0};
    uint32_t headDim{0};

    BlockMaskParams maskTaskInfo[COMPUTE_PIPE_NUM];
    JaggedTaskArgs computeTaskInfo[COMPUTE_PIPE_NUM];
    JaggedTaskArgs transTaskInfo[TRANS_PIPE_NUM];
    GlobalTensor<oType> seqOffsetsQGt;
    GlobalTensor<oType> seqOffsetsKGt;
    GlobalTensor<oType> numContextGt;
    GlobalTensor<oType> numTargetGt;
};

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::Compute(
    const HstuDenseForwardTilingData* __restrict tilingDataPtr)
{
    int ret = PreInit(tilingDataPtr);
    if (ret == -1) {
        return; // no task
    }
    ComputeAllBlock();
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::ComputeSvMatmul(
    uint32_t taskId)
{
    int isAtomic = 1;
    if (computeTaskInfo[taskId].kSeqId == 0) {
        isAtomic = 0;
    }

    this->DoSvMatmulImpl(computeTaskInfo[taskId].kvOffset, taskId, computeTaskInfo[taskId].transTaskId, isAtomic,
                         computeTaskInfo[taskId].computeASeqLen, this->headDim, computeTaskInfo[taskId].computeBSeqLen);
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::ComputeQkMatmul(
    uint32_t taskId)
{
    this->DoQkMatmulImpl(computeTaskInfo[taskId].ioOffset, computeTaskInfo[taskId].kvOffset, taskId,
                         computeTaskInfo[taskId].computeASeqLen, computeTaskInfo[taskId].computeBSeqLen, this->headDim);
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::ComputeVecScore(
    uint32_t taskId)
{
    int64_t biasOffset = computeTaskInfo[taskId].batchId * this->headNum * this->maxSeqLenQ * this->maxSeqLenK +
                         computeTaskInfo[taskId].headId * this->maxSeqLenQ * this->maxSeqLenK +
                         computeTaskInfo[taskId].qSeqId * this->maxSeqLenK * this->blockHeight +
                         computeTaskInfo[taskId].kSeqId * this->blockHeight;

    int64_t maskOffset = biasOffset;

    this->template VecScoreImpl<BlockMaskParams>(taskId, biasOffset, maskOffset, computeTaskInfo[taskId].scale,
                                                 maskTaskInfo[taskId], computeTaskInfo[taskId].computeASeqLen,
                                                 computeTaskInfo[taskId].computeBSeqLen);
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::TransResult(
    uint32_t transtaskId)
{
    this->DoTransSvImpl(transtaskId, transTaskInfo[transtaskId].ioOffset, transTaskInfo[transtaskId].computeASeqLen);
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::ComputeAllBlock()
{
    GetTaskInfo(this->sBlkId);

    uint32_t taskId = 0;
    uint32_t transtaskId = 0;

    uint32_t currentTaskId = 0;
    uint32_t preTaskId = 0;
    uint32_t prePreTaskId = 0;
    uint32_t nextTaskId = 0;
    uint32_t nblk = 0;

    this->AllocQkUbTensor();

    for (auto blkId = sBlkId; blkId < eBlkId; blkId++) {
        auto kSeqNum = computeTaskInfo[taskId % COMPUTE_PIPE_NUM].kSeqNum;
        auto deltaQK = computeTaskInfo[taskId % COMPUTE_PIPE_NUM].deltaQK;
        nblk = deltaQK / this->blockHeight;
        bool isDeltaQK = deltaQK % this->blockHeight != 0;
        int64_t maskOffset1 = deltaQK % this -> blockHeight;
        int64_t maskOffset2 = deltaQK % this -> blockHeight - this -> blockHeight;

        for (auto kSeqId = 0; kSeqId < kSeqNum; kSeqId++) {
            auto taskinfo = this->computeTaskInfo[taskId % COMPUTE_PIPE_NUM];
            BlockMaskParams maskinfo = {
                taskinfo.qSeqId,
                kSeqId,
                taskinfo.actualSeqLen,
                taskinfo.actualSeqLenK,
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
            if (maskinfo.NoComputation(maskType)) {
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
            uint64_t kvHeadId = this->computeTaskInfo[currentTaskId].headId / this->headRatio;
            this->computeTaskInfo[currentTaskId].kvOffset =
                this->computeTaskInfo[currentTaskId].batchOffsetK * this->headDim * this->headNumK +
                this->computeTaskInfo[currentTaskId].kSeqId * this->blockHeight * this->headNumK * this->headDim +
                kvHeadId * this->headDim;

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

            computeTaskInfo[nextTaskId] = computeTaskInfo[currentTaskId];
            maskTaskInfo[nextTaskId] = maskTaskInfo[currentTaskId];
            taskId++;
        }

        this->transTaskInfo[transtaskId % TRANS_PIPE_NUM] = this->computeTaskInfo[currentTaskId];
        if (transtaskId > 1) {
            this->TransResult((transtaskId - 2) % TRANS_PIPE_NUM);
        }
        transtaskId++;

        this->UpdateTaskInfo(taskId % COMPUTE_PIPE_NUM);
    }

    if (taskId == 0) {
        this->FreeQkUbTensor();
        return;
    }

    if (taskId == 1) {
        this->ComputeVecScore(currentTaskId);
        pipe_barrier(PIPE_ALL);

        this->ComputeSvMatmul(currentTaskId);
        this->WaitSvMatmul();

        this->TransResult((transtaskId - 1) % TRANS_PIPE_NUM);
        this->FreeQkUbTensor();
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
        this->FreeQkUbTensor();
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
    this->FreeQkUbTensor();
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::FillTaskInfo(
    uint32_t batchId, uint32_t headId, int64_t seqGlobalOffset, uint32_t taskId)
{
    if (batchId >= this->batchSize) {
        return;
    }

    taskId = taskId % COMPUTE_PIPE_NUM;

    auto nextBatchSeqOffset = this->seqOffsetsQGt.GetValue(batchId + 1);
    auto currentBatchSeqOffset = this->seqOffsetsQGt.GetValue(batchId);

    auto nextBatchSeqOffsetK = this->seqOffsetsKGt.GetValue(batchId + 1);
    auto currentBatchSeqOffsetK = this->seqOffsetsKGt.GetValue(batchId);

    auto numContext = this->numContextGt.GetValue(batchId);
    auto numTarget = this->numTargetGt.GetValue(batchId);

    computeTaskInfo[taskId].seqGlobalOffset = seqGlobalOffset;
    computeTaskInfo[taskId].batchId = batchId;
    computeTaskInfo[taskId].actualSeqLen = nextBatchSeqOffset - currentBatchSeqOffset;
    computeTaskInfo[taskId].actualSeqLenK = nextBatchSeqOffsetK - currentBatchSeqOffsetK;
    computeTaskInfo[taskId].deltaQK = computeTaskInfo[taskId].actualSeqLenK - computeTaskInfo[taskId].actualSeqLen;

    computeTaskInfo[taskId].scale = this->siluScale;
    computeTaskInfo[taskId].numTarget = numTarget;
    computeTaskInfo[taskId].numContext = numContext;
    computeTaskInfo[taskId].batchOffset = currentBatchSeqOffset;
    computeTaskInfo[taskId].batchOffsetK = currentBatchSeqOffsetK;
    // 每个注意力头在序列维度上的处理边界
    computeTaskInfo[taskId].headSeqLimit =
        computeTaskInfo[taskId].batchOffset * this->headNum + computeTaskInfo[taskId].actualSeqLen * (headId + 1);
    auto batchInnerOffset = seqGlobalOffset - (computeTaskInfo[taskId].batchOffset * this->headNum);
    computeTaskInfo[taskId].headId = headId;
    computeTaskInfo[taskId].qSeqId =
        (batchInnerOffset - computeTaskInfo[taskId].headId * computeTaskInfo[taskId].actualSeqLen) / this->blockHeight;
    computeTaskInfo[taskId].kSeqNum = CeilDiv(computeTaskInfo[taskId].actualSeqLenK,
                                              static_cast<uint32_t> (this->blockHeight));
    computeTaskInfo[taskId].qSeqNum = CeilDiv(computeTaskInfo[taskId].actualSeqLen,
                                              static_cast<uint32_t> (this->blockHeight));

    computeTaskInfo[taskId].ioOffset =
        computeTaskInfo[taskId].batchOffset * this->headDim * this->headNum +
        computeTaskInfo[taskId].qSeqId * this->blockHeight * this->headNum * this->headDim +
        computeTaskInfo[taskId].headId * this->headDim;

    if ((computeTaskInfo[taskId].headSeqLimit - seqGlobalOffset) >= this->blockHeight) {
        computeTaskInfo[taskId].computeASeqLen = this->blockHeight;
    } else {
        computeTaskInfo[taskId].computeASeqLen = computeTaskInfo[taskId].headSeqLimit - seqGlobalOffset;
    }
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::UpdateTaskInfo(
    uint32_t taskId)
{
    auto batchId = computeTaskInfo[taskId].batchId;
    auto headId = computeTaskInfo[taskId].headId;

    int64_t seqGlobalOffset = computeTaskInfo[taskId].seqGlobalOffset;
    int64_t gap = computeTaskInfo[taskId].headSeqLimit - seqGlobalOffset;

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
    } else {
        computeTaskInfo[taskId].seqGlobalOffset = seqGlobalOffset + this->blockHeight;

        uint32_t computeASeqLen = this->blockHeight;
        if ((computeTaskInfo[taskId].seqGlobalOffset + this->blockHeight) > computeTaskInfo[taskId].headSeqLimit) {
            computeASeqLen = computeTaskInfo[taskId].headSeqLimit - computeTaskInfo[taskId].seqGlobalOffset;
        }

        auto batchInnerOffset =
            computeTaskInfo[taskId].seqGlobalOffset - (computeTaskInfo[taskId].batchOffset * this->headNum);
        computeTaskInfo[taskId].qSeqId =
            (batchInnerOffset - computeTaskInfo[taskId].headId * computeTaskInfo[taskId].actualSeqLen) /
            this->blockHeight;
        computeTaskInfo[taskId].ioOffset =
            computeTaskInfo[taskId].batchOffset * this->headDim * this->headNum +
            computeTaskInfo[taskId].qSeqId * this->blockHeight * this->headNum * this->headDim +
            computeTaskInfo[taskId].headId * this->headDim;
        computeTaskInfo[taskId].computeASeqLen = computeASeqLen;
    }
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline void HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::GetTaskInfo(
    uint32_t sBlkId)
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
            return;
        }

        offsetOfBlk += batchBlkSize;
        seqGlobalOffset += batchSeqSize;
    }
}

template <typename qType, typename oType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
__aicore__ inline int HstuDenseForwardJaggedKernel<qType, oType, enableBias, isQkUseUb, maskType>::PreInit(
    const HstuDenseForwardTilingData* __restrict tilingDataPtr)
{
    seqOffsetsQGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(this->seqOffsetQ), this->xDim0 + 1);
    seqOffsetsKGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(this->seqOffsetK), this->xDim0 + 1);
    auto validBatchSize = GetBatchSizeFromJaggedOffset(seqOffsetsQGt, this->xDim0 + 1);
    ASCENDC_ASSERT((validBatchSize > 0 && validBatchSize <= MAX_BATCH_SIZE), "batchSize exceed limit of (0, 20480]\n");

    const int blockId = GetBlockIdx();
    const uint32_t coreNum = GetBlockNum() * GetTaskRation();
    this->batchSize = validBatchSize;
    this->xDim0 = validBatchSize;
    this->seqLen = this->xDim1;
    this->headNum = this->xDim2;
    this->headDim = this->xDim3;

    numContextGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(this->numContext), this->batchSize);
    numTargetGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(this->numTarget), this->batchSize);

    bool fastMode = false;
    if (this->maxSeqLenK <= BLOCK_M && this->maxSeqLenQ * this->maxSeqLenK <= BLOCK_MN) {
        fastMode = true;
    }
    int blocks[2] = {0}; // start block id, end block id
    auto taskAssigner =
        BlockTaskAssign(maskType, coreNum, this->blockHeight, this->batchSize, this->headNum,
                        this->targetGroupSize, seqOffsetsQGt, seqOffsetsKGt, numContextGt, numTargetGt, fastMode);
    taskAssigner.Compute(blocks, blockId);

    this->sBlkId = blocks[0];
    this->eBlkId = blocks[1];
    if (this->sBlkId == this->eBlkId && this->eBlkId == 0) {
        return -1;
    }
    return 0;
}
}  // namespace HstuDenseForward

#endif
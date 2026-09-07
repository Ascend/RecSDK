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
#ifndef HSTU_DENSE_FORWARD_JAGGED_KERNEL_FUN_H
#define HSTU_DENSE_FORWARD_JAGGED_KERNEL_FUN_H

#include "hstu_dense_forward_kernel_patten_bsnd.h"
#include "hstu_split_core_policy.h"
#include "hstu_dense_causal_mask.h"

using namespace AscendC;

namespace HstuDenseForward {

template <typename TraitParams, typename TilingDataType>
class HstuDenseForwardJaggedKernel : public HstuDenseForwardKernelPattenBsnd<TraitParams, TilingDataType> {
public:
    using qType = typename TraitParams::qType;
    using oType = typename TraitParams::oType;
#ifdef SUPPORT_950
    __aicore__ inline HstuDenseForwardJaggedKernel(int vecPerProcess = (std::is_same_v<qType, fp8_e4m3fn_t> ? 32 : 48))
        : HstuDenseForwardKernelPattenBsnd<TraitParams, TilingDataType>(vecPerProcess)
    {
    }
#else
    __aicore__ inline HstuDenseForwardJaggedKernel() {}
#endif

    __aicore__ inline void Compute();

    __aicore__ inline void Init(const Args& args, const HstuJaggedForwardTilingData* __restrict tilingDataPtr,
                                TPipe* pipePtr);

    __aicore__ inline void InitArgs(const Args& args, const HstuJaggedForwardTilingData* __restrict tilingDataPtr);

    __aicore__ inline void ComputeAllBlock();

    __aicore__ inline int PreInit();

    __aicore__ inline void GetTaskInfo(uint32_t sBlkId);

    __aicore__ inline void UpdateTaskInfo(uint32_t taskId);

    __aicore__ inline void FillTaskInfo(uint32_t batchId, uint32_t head_id, int64_t seqGlobalOffset, uint32_t taskId);

    __aicore__ inline void ComputeQkMatmul(uint32_t taskId);

    __aicore__ inline void ComputeVecScore(uint32_t taskId);

    __aicore__ inline void ComputeSvMatmul(uint32_t taskId);

    __aicore__ inline void TransResult(uint32_t transtaskId);

    __aicore__ inline void NotifypreBlock();

    __aicore__ inline void WaitNextBlock(uint32_t transtaskId);

    GM_ADDR seqOffsetQ;
    GM_ADDR seqOffsetK;
    GM_ADDR numContext;
    GM_ADDR numTarget;

    int64_t maxSeqLenQ;
    int64_t targetGroupSize;

    // GQA
    uint64_t headNumK;
    uint64_t headRatio;

    uint32_t sBlkId{0};
    uint32_t eBlkId{0};
    uint32_t skSeqBlkId{0};
    uint32_t ekSeqBlkId{0};
    uint32_t batchSize{0};
    uint32_t seqLen{0};
    uint32_t headNum{0};
    uint32_t headDim{0};
    uint32_t headDimV{0};
    int32_t splitMode{DEFAULT_SPLIT};

    BlockMaskParams maskTaskInfo[COMPUTE_PIPE_NUM];
    JaggedTaskArgs computeTaskInfo[COMPUTE_PIPE_NUM];
    JaggedTaskArgs transTaskInfo[TRANS_PIPE_NUM];
    GlobalTensor<oType> seqOffsetsQGt;
    GlobalTensor<oType> seqOffsetsKGt;
    GlobalTensor<oType> numContextGt;
    GlobalTensor<oType> numTargetGt;
};

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::Compute()
{
    int ret = PreInit();
    if (ret == -1) {
        return;  // no task
    }
    ComputeAllBlock();
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::Init(
    const Args& args, const HstuJaggedForwardTilingData* __restrict tilingDataPtr, TPipe* pipePtr)
{
    InitArgs(args, tilingDataPtr);
    this->InitPipe(pipePtr);
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::InitArgs(
    const Args& args, const HstuJaggedForwardTilingData* __restrict tilingDataPtr)
{
    this->q = args.q;
    this->k = args.k;
    this->v = args.v;
    this->attnBias = args.attnBias;
    this->mask = args.mask;

    this->attnOutput = args.attnOutput;
    this->workspace = args.workspace;

    // Batch Size
    this->xDim0 = tilingDataPtr->batchSize;
    // Seq Len
    this->xDim1 = tilingDataPtr->seqLen;
    // Head Num
    this->xDim2 = tilingDataPtr->headNum;
    // Embedding Dim
    this->xDim3 = tilingDataPtr->dim;
    this->vDim = tilingDataPtr->vDim;

    this->maxSeqLenK = tilingDataPtr->maxSeqLenk;

    // attr
    this->siluScale = tilingDataPtr->siluScale;
    this->alpha = tilingDataPtr->alpha;

    // copyKV
    this->copyHeadNum = tilingDataPtr->headNumK;

    seqOffsetQ = args.seqOffsetQ;
    seqOffsetK = args.seqOffsetK;
    numContext = args.numContext;
    numTarget = args.numTarget;

    maxSeqLenQ = tilingDataPtr->maxSeqLenq;

    targetGroupSize = tilingDataPtr->targetGroupSize;

    // GQA
    headNumK = tilingDataPtr->headNumK;
    headRatio = tilingDataPtr->headRatio;
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::ComputeSvMatmul(uint32_t taskId)
{
    int isAtomic = 1;
    if (computeTaskInfo[taskId].isFirstSeqBlk) {
        isAtomic = 0;
    }

    this->DoSvMatmulImpl(computeTaskInfo[taskId].vOffset, taskId, computeTaskInfo[taskId].transTaskId, isAtomic,
                         computeTaskInfo[taskId].computeASeqLen, this->headDimV,
                         computeTaskInfo[taskId].computeBSeqLen);
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::ComputeQkMatmul(uint32_t taskId)
{
    if (computeTaskInfo[taskId].isFirstSeqBlk) {
        this->template DoQkMatmulImpl<true>(computeTaskInfo[taskId].iOffset, computeTaskInfo[taskId].kOffset, taskId,
                                            computeTaskInfo[taskId].computeASeqLen,
                                            computeTaskInfo[taskId].computeBSeqLen, this->headDim,
                                            computeTaskInfo[taskId].bufferIdx);
    } else {
        this->template DoQkMatmulImpl<false>(computeTaskInfo[taskId].iOffset, computeTaskInfo[taskId].kOffset, taskId,
                                             computeTaskInfo[taskId].computeASeqLen,
                                             computeTaskInfo[taskId].computeBSeqLen, this->headDim,
                                             computeTaskInfo[taskId].bufferIdx);
    }
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::ComputeVecScore(uint32_t taskId)
{
    int64_t biasOffset = computeTaskInfo[taskId].batchId * this->headNum * this->maxSeqLenQ * this->maxSeqLenK +
                         computeTaskInfo[taskId].headId * this->maxSeqLenQ * this->maxSeqLenK +
                         computeTaskInfo[taskId].qSeqId * this->maxSeqLenK * TraitParams::blockM +
                         computeTaskInfo[taskId].kSeqId * TraitParams::blockN;

    int64_t maskOffset = biasOffset;

    this->template VecScoreImpl<BlockMaskParams>(taskId, biasOffset, maskOffset, computeTaskInfo[taskId].scale,
                                                 maskTaskInfo[taskId], computeTaskInfo[taskId].computeASeqLen,
                                                 computeTaskInfo[taskId].computeBSeqLen,
                                                 computeTaskInfo[taskId].bufferIdx);
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::TransResult(uint32_t transtaskId)
{
    uint32_t transtaskIdModed = transtaskId % TRANS_PIPE_NUM;
    if constexpr (TraitParams::deterministic) {
        if (transTaskInfo[transtaskIdModed].isEndToTail) {
            this->template DoTransSvImpl<false>(transtaskId, transTaskInfo[transtaskIdModed].oOffset,
                                                transTaskInfo[transtaskIdModed].computeASeqLen);
        } else {
            this->template DoTransSvImpl<true>(transtaskId, transTaskInfo[transtaskIdModed].oOffset,
                                               transTaskInfo[transtaskIdModed].computeASeqLen);
        }
    } else {
        this->template DoTransSvImpl<true>(transtaskId, transTaskInfo[transtaskIdModed].oOffset,
                                           transTaskInfo[transtaskIdModed].computeASeqLen);
    }

    if (transtaskId == 0) {
        NotifypreBlock();
    }
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::NotifypreBlock()
{
    if constexpr (TraitParams::deterministic) {
        if (GetBlockIdx() > 0 && transTaskInfo[0].isStartFromZero == 0) {
            auto syncBuf = this->vecIn.template AllocTensor<int32_t>();
            AscendC::IBSet<false>(this->syncGm, syncBuf, GetBlockIdx(), 0);
            this->vecIn.FreeTensor(syncBuf);
        }
    }
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::WaitNextBlock(uint32_t transtaskId)
{
    if constexpr (TraitParams::deterministic) {
        if (GetBlockIdx() + 1 < GetBlockNum() * GetTaskRation() &&
            transTaskInfo[transtaskId % TRANS_PIPE_NUM].isEndToTail == 0) {
            auto syncBuf = this->vecIn.template AllocTensor<int32_t>();
            AscendC::IBWait<false>(this->syncGm, syncBuf, GetBlockIdx() + 1, 0);
            this->vecIn.FreeTensor(syncBuf);
        }
    }
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::ComputeAllBlock()
{
    GetTaskInfo(this->sBlkId);

    uint32_t transtaskId = 0;
    uint32_t taskId = 0;

    uint32_t currentTaskId = 0;
    uint32_t preTaskId = 0;
    uint32_t prePreTaskId = 0;
    uint32_t nextTaskId = 0;
    uint32_t kSeqId = this->skSeqBlkId;
    uint32_t kSeqNum = 0;

    this->scmQKTensor = this->qkL1In.template AllocTensor<qType>();
    for (auto blkId = this->sBlkId; blkId <= this->eBlkId; blkId++) {
        kSeqNum = computeTaskInfo[taskId % COMPUTE_PIPE_NUM].kSeqNum;
        auto limit = (blkId == this->eBlkId) ? this->ekSeqBlkId : kSeqNum;
        uint32_t isStartFromZero = (kSeqId == 0);
        uint32_t isEndToTail = false;
        for (; kSeqId < limit; kSeqId++) {
            auto taskinfo = this->computeTaskInfo[taskId % COMPUTE_PIPE_NUM];
            BlockMaskParams maskinfo = {taskinfo.qSeqId,       kSeqId,
                                        taskinfo.actualSeqLen, taskinfo.actualSeqLenK,
                                        TraitParams::blockM,   TraitParams::blockN,
                                        taskinfo.numContext,   taskinfo.numTarget,
                                        this->targetGroupSize, taskinfo.scale};

            const int64_t qBlockBegin = static_cast<int64_t>(taskinfo.qSeqId) * TraitParams::blockM;
            const int64_t kBlockBegin = static_cast<int64_t>(kSeqId) * TraitParams::blockN;
            const int64_t kBlockEnd = kBlockBegin + TraitParams::blockN;
            const int64_t targetQBegin = taskinfo.actualSeqLen - taskinfo.numTarget;
            const int64_t targetKBegin = taskinfo.actualSeqLenK - taskinfo.numTarget;

            // 跳过 target mask 三角形下方完全无效的 K block；边界 block 仍交给 mask 处理。
            // target 白区仅由 MASK_TRIL 的内核 mask 保证恒为 0；CUSTOM mask 的值由外部输入决定，不能跳过。
            if (TraitParams::maskType == CausalMaskT::MASK_TRIL) {
                if (taskinfo.numTarget > 0 && this->targetGroupSize > 0 && qBlockBegin >= targetQBegin &&
                    kBlockBegin >= targetKBegin) {
                    const int64_t targetGroupIndex = (qBlockBegin - targetQBegin) / this->targetGroupSize;
                    if (targetGroupIndex > 0) {
                        const int64_t targetGroupLimit = targetKBegin + targetGroupIndex * this->targetGroupSize;
                        if (kBlockEnd <= targetGroupLimit) {
                            continue;
                        }
                    }
                }
            }

            // 在下三角下跳过运算
            if (maskinfo.NoComputation(TraitParams::maskType)) {
                isEndToTail = true;
                break;
            }

            currentTaskId = taskId % COMPUTE_PIPE_NUM;
            preTaskId = (taskId + COMPUTE_PIPE_NUM - 1) % COMPUTE_PIPE_NUM;
            prePreTaskId = (taskId + COMPUTE_PIPE_NUM - 2) % COMPUTE_PIPE_NUM;
            nextTaskId = (taskId + 1) % COMPUTE_PIPE_NUM;

            this->maskTaskInfo[currentTaskId] = maskinfo;
            this->computeTaskInfo[currentTaskId].transTaskId = transtaskId % TRANS_PIPE_NUM;
            this->computeTaskInfo[currentTaskId].kSeqId = kSeqId;
            this->computeTaskInfo[currentTaskId].computeBSeqLen =
                (kSeqId != (kSeqNum - 1))
                    ? (TraitParams::blockN)
                    : (this->computeTaskInfo[currentTaskId].actualSeqLenK - kSeqId * TraitParams::blockN);
            uint64_t kvHeadId = this->computeTaskInfo[currentTaskId].headId / this->headRatio;
            this->computeTaskInfo[currentTaskId].kOffset =
                this->computeTaskInfo[currentTaskId].batchOffsetK * this->headDim * this->headNumK +
                this->computeTaskInfo[currentTaskId].kSeqId * TraitParams::blockN * this->headNumK * this->headDim +
                kvHeadId * this->headDim;
            this->computeTaskInfo[currentTaskId].vOffset =
                this->computeTaskInfo[currentTaskId].batchOffsetK * this->headDimV * this->headNumK +
                this->computeTaskInfo[currentTaskId].kSeqId * TraitParams::blockN * this->headNumK * this->headDimV +
                kvHeadId * this->headDimV;
            this->computeTaskInfo[currentTaskId].bufferIdx = taskId % TraitParams::GetInTQueNumber();

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
            this->computeTaskInfo[nextTaskId].isFirstSeqBlk = 0;
            maskTaskInfo[nextTaskId] = maskTaskInfo[currentTaskId];
            taskId++;
        }
        if (blkId == this->eBlkId && ekSeqBlkId == 0) {
            break;
        }

        this->transTaskInfo[transtaskId % TRANS_PIPE_NUM] = this->computeTaskInfo[currentTaskId];
        this->transTaskInfo[transtaskId % TRANS_PIPE_NUM].isStartFromZero = isStartFromZero;
        this->transTaskInfo[transtaskId % TRANS_PIPE_NUM].isEndToTail = isEndToTail | (kSeqId == kSeqNum);
        if (transtaskId > 1) {
            this->TransResult(transtaskId - 2);
        }
        transtaskId++;

        this->UpdateTaskInfo(taskId % COMPUTE_PIPE_NUM);
        kSeqId = 0;
    }

    this->qkL1In.FreeTensor(this->scmQKTensor);
    if (taskId == 0) {
        return;
    }

    if (taskId == 1) {
        this->ComputeVecScore(currentTaskId);
        pipe_barrier(PIPE_ALL);

        this->ComputeSvMatmul(currentTaskId);
        this->WaitSvMatmul();
        WaitNextBlock(transtaskId - 1);
        this->TransResult(transtaskId - 1);
        return;
    }

    if (transtaskId == 1) {
        this->ComputeSvMatmul(preTaskId);
        this->WaitSvMatmul();

        this->ComputeVecScore(currentTaskId);
        pipe_barrier(PIPE_ALL);

        this->ComputeSvMatmul(currentTaskId);
        this->WaitSvMatmul();
        WaitNextBlock(transtaskId - 1);
        this->TransResult(transtaskId - 1);
        return;
    }

    this->ComputeSvMatmul(preTaskId);
    this->WaitSvMatmul();

    this->ComputeVecScore(currentTaskId);
    pipe_barrier(PIPE_ALL);

    this->ComputeSvMatmul(currentTaskId);
    this->WaitSvMatmul();

    this->TransResult(transtaskId - 2);
    WaitNextBlock(transtaskId - 1);
    this->TransResult(transtaskId - 1);
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::FillTaskInfo(uint32_t batchId,
                                                                                               uint32_t headId,
                                                                                               int64_t seqGlobalOffset,
                                                                                               uint32_t taskId)
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
        (batchInnerOffset - computeTaskInfo[taskId].headId * computeTaskInfo[taskId].actualSeqLen) /
        TraitParams::blockM;
    computeTaskInfo[taskId].kSeqNum =
        CeilDiv(computeTaskInfo[taskId].actualSeqLenK, static_cast<uint32_t>(TraitParams::blockN));
    computeTaskInfo[taskId].qSeqNum =
        CeilDiv(computeTaskInfo[taskId].actualSeqLen, static_cast<uint32_t>(TraitParams::blockM));

    computeTaskInfo[taskId].iOffset =
        computeTaskInfo[taskId].batchOffset * this->headDim * this->headNum +
        computeTaskInfo[taskId].qSeqId * TraitParams::blockM * this->headNum * this->headDim +
        computeTaskInfo[taskId].headId * this->headDim;
    computeTaskInfo[taskId].oOffset =
        computeTaskInfo[taskId].batchOffset * this->headDimV * this->headNum +
        computeTaskInfo[taskId].qSeqId * TraitParams::blockM * this->headNum * this->headDimV +
        computeTaskInfo[taskId].headId * this->headDimV;

    if ((computeTaskInfo[taskId].headSeqLimit - seqGlobalOffset) >= TraitParams::blockM) {
        computeTaskInfo[taskId].computeASeqLen = TraitParams::blockM;
    } else {
        computeTaskInfo[taskId].computeASeqLen = computeTaskInfo[taskId].headSeqLimit - seqGlobalOffset;
    }
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::UpdateTaskInfo(uint32_t taskId)
{
    auto batchId = computeTaskInfo[taskId].batchId;
    auto headId = computeTaskInfo[taskId].headId;

    int64_t seqGlobalOffset = computeTaskInfo[taskId].seqGlobalOffset;
    int64_t gap = computeTaskInfo[taskId].headSeqLimit - seqGlobalOffset;

    if (gap <= TraitParams::blockM) {
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
        computeTaskInfo[taskId].seqGlobalOffset = seqGlobalOffset + TraitParams::blockM;

        uint32_t computeASeqLen = TraitParams::blockM;
        if ((computeTaskInfo[taskId].seqGlobalOffset + TraitParams::blockM) > computeTaskInfo[taskId].headSeqLimit) {
            computeASeqLen = computeTaskInfo[taskId].headSeqLimit - computeTaskInfo[taskId].seqGlobalOffset;
        }

        auto batchInnerOffset =
            computeTaskInfo[taskId].seqGlobalOffset - (computeTaskInfo[taskId].batchOffset * this->headNum);
        computeTaskInfo[taskId].qSeqId =
            (batchInnerOffset - computeTaskInfo[taskId].headId * computeTaskInfo[taskId].actualSeqLen) /
            TraitParams::blockM;
        computeTaskInfo[taskId].iOffset =
            computeTaskInfo[taskId].batchOffset * this->headDim * this->headNum +
            computeTaskInfo[taskId].qSeqId * TraitParams::blockM * this->headNum * this->headDim +
            computeTaskInfo[taskId].headId * this->headDim;
        computeTaskInfo[taskId].oOffset =
            computeTaskInfo[taskId].batchOffset * this->headDimV * this->headNum +
            computeTaskInfo[taskId].qSeqId * TraitParams::blockM * this->headNum * this->headDimV +
            computeTaskInfo[taskId].headId * this->headDimV;
        computeTaskInfo[taskId].computeASeqLen = computeASeqLen;
    }
    computeTaskInfo[taskId].isFirstSeqBlk = 1;
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline void HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::GetTaskInfo(uint32_t sBlkId)
{
    uint32_t offsetOfBlk = 0;
    int64_t offsetOfSeq = 0;
    int64_t seqGlobalOffset = 0;
    for (auto index = 0; index < this->batchSize * this->headNum; index++) {
        uint32_t batchId = index / this->headNum;
        uint32_t headId = index % this->headNum;

        uint32_t batchSeqSize = this->seqOffsetsQGt.GetValue(batchId + 1) - this->seqOffsetsQGt.GetValue(batchId);
        uint32_t batchBlkSize = (batchSeqSize + TraitParams::blockM - 1) / TraitParams::blockM;
        if (this->sBlkId < (offsetOfBlk + batchBlkSize)) {
            uint32_t innerBlkId = sBlkId - offsetOfBlk;
            seqGlobalOffset = seqGlobalOffset + innerBlkId * TraitParams::blockM;
            this->FillTaskInfo(batchId, headId, seqGlobalOffset, 0);
            return;
        }

        offsetOfBlk += batchBlkSize;
        seqGlobalOffset += batchSeqSize;
    }
}

template <typename TraitParams, typename TilingDataType>
__aicore__ inline int HstuDenseForwardJaggedKernel<TraitParams, TilingDataType>::PreInit()
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
    this->headDimV = this->vDim;

    numContextGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(this->numContext), this->batchSize);
    numTargetGt.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(this->numTarget), this->batchSize);

    this->splitMode = STREAM_K;
    if (this->maxSeqLenQ <= TraitParams::blockM && this->maxSeqLenK <= TraitParams::blockN) {
        this->splitMode = FAST_SPLIT_SINGLE;
    } else if (this->maxSeqLenK <= TraitParams::blockN) {
        this->splitMode = FAST_SPLIT;
    }

    int blocks[4] = {0};  // start block id, end block id
    if constexpr (TraitParams::maskType == CausalMaskT::MASK_TRIL) {
        auto taskAssigner = BlockTaskAssign<oType, CausalMaskT::MASK_TRIL>(
            coreNum, this->batchSize, this->headNum, this->targetGroupSize, TraitParams::blockM, TraitParams::blockN,
            seqOffsetsQGt, seqOffsetsKGt, numContextGt, numTargetGt, this->splitMode);
        taskAssigner.Compute(blocks, blockId);
    } else if constexpr (TraitParams::maskType == CausalMaskT::MASK_CUSTOM) {
        auto taskAssigner = BlockTaskAssign<oType, CausalMaskT::MASK_CUSTOM>(
            coreNum, this->batchSize, this->headNum, this->targetGroupSize, TraitParams::blockM, TraitParams::blockN,
            seqOffsetsQGt, seqOffsetsKGt, numContextGt, numTargetGt, this->splitMode);
        taskAssigner.Compute(blocks, blockId);
    } else {
        auto taskAssigner = BlockTaskAssign<oType, CausalMaskT::MASK_NONE>(
            coreNum, this->batchSize, this->headNum, this->targetGroupSize, TraitParams::blockM, TraitParams::blockN,
            seqOffsetsQGt, seqOffsetsKGt, numContextGt, numTargetGt, this->splitMode);
        taskAssigner.Compute(blocks, blockId);
    }

    this->L2CacheHintCfg(this->splitMode);

    this->skSeqBlkId = blocks[0];
    this->ekSeqBlkId = blocks[1];
    this->sBlkId = blocks[2];
    this->eBlkId = blocks[3];
    if (this->skSeqBlkId == this->ekSeqBlkId && this->sBlkId == this->eBlkId && this->sBlkId == 0 &&
        this->skSeqBlkId == 0) {
        return -1;
    }
    return 0;
}
}  // namespace HstuDenseForward

#endif

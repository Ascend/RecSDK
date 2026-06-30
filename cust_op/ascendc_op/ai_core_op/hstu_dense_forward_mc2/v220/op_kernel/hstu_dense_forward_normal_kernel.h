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

#ifndef HSTU_DENSE_FORWARD_KERNEL_FUN_H
#define HSTU_DENSE_FORWARD_KERNEL_FUN_H
#include "hstu_dense_kernel_patten_bsnd.h"
#include "collectives.h"
#include "sync_collectives.h"

// Fused mode: 1=AllGather executed inside kernel (computation-communication overlap), 0=separate
#define HSTU_FUSED_MODE 1
// K/V dimension is 8x Q dimension after AllGather in MC2 scenario
constexpr int64_t MC2_KV_DIM_FACTOR = 8;
// Total core count per AI CPU, used for inter-core work distribution
constexpr int64_t MC2_COMPUTE_CORE_NUM = 48;
// AllGather completion flag value written to windowOut for batch-level sync
constexpr int64_t AG_COMPLETE_FLAG = 2025;
// Maximum number of HCCL handles per kernel instance
constexpr int64_t MAX_HANDLE_ID = 256;
namespace HstuDenseForward {

// Arguments for Q*K Matmul tiling iteration
struct QkMatmulArgs {
    int64_t taskId = INVALID_TASK_ID;  // Matmul task slot (0..COMPUTE_PIPE_NUM-1)
    int64_t qkBlockId;                 // Global block index = batch*head*qSeqBlocks + headSeqBlock
    int64_t batchId;                   // Batch index
    int64_t headId;                    // Head index
    int64_t qSeqId;                    // Q sequence block index
    int64_t kSeqId;                    // K sequence block index
};

// Arguments for Score (SiLU+Mask) vector computation
struct ScoreVectorArgs {
    int64_t taskId = INVALID_TASK_ID;  // Matmul task slot
    int64_t scoreBlockId;              // Score block index
    int64_t batchId;                   // Batch index
    int64_t headId;                    // Head index
    int64_t qSeqId;                    // Q sequence block index
    int64_t kSeqId;                    // K sequence block index
};

// Arguments for Score*V Matmul tiling iteration
struct SvMatmulArgs {
    int64_t transTaskId = INVALID_TASK_ID;  // Transpose task slot
    int64_t taskId = INVALID_TASK_ID;       // Matmul task slot
    int64_t scoreBlockId;                   // Score block index
    int64_t batchId;                        // Batch index
    int64_t headId;                         // Head index
    int64_t qSeqId;                         // Q sequence block index
    int64_t kSeqId;                         // K sequence block index
    int64_t vSeqId;                         // V sequence block index
};

// Arguments for SV transpose/datacopy to output
struct SVTransArgs {
    int64_t transTaskId = INVALID_TASK_ID;  // Transpose task slot
    int64_t scoreBlockId;                   // Score block index
    int64_t batchId;                        // Batch index
    int64_t headId;                         // Head index
    int64_t qSeqId;                         // Q sequence block index
};

// Flag value written to windowOut for batch-level synchronization between AICPU and AI Core
constexpr int64_t FLAG_OFFSET = 2000 * 1024 * 1024;
// Number of batches processed per split batch iteration
constexpr int64_t SPLIT_BATCH_SIZE = 2;

template <typename qType>
class HstuDenseForwardKernel : public HstuDenseForwardKernelPattenBsnd<qType> {
public:
    __aicore__ inline HstuDenseForwardKernel() {}

    __aicore__ inline void PreInit(const HstuDenseForwardTilingData* __restrict tilingDataPtr)
    {
        seqBlockNumQk = DivCeil(this->xDim1, this->blockHeight);  // Round up to at least one block
        seqBlockNumKV = DivCeil(this->xDim1 * MC2_KV_DIM_FACTOR, this->blockHeight);
        qkTotalBlock = this->xDim0 * this->xDim2 * seqBlockNumQk;
    }

    __aicore__ inline void VecScore(ScoreVectorArgs& scoreArgs)
    {
        if (scoreArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t attnBiasOffset =
            scoreArgs.batchId * this->xDim2 * this->xDim1 * this->xDim1 + scoreArgs.headId * this->xDim1 * this->xDim1 +
            scoreArgs.qSeqId * this->blockHeight * this->xDim1 + scoreArgs.kSeqId * this->blockHeight;

        int64_t maskOffset = attnBiasOffset;

        int causalMask = ((scoreArgs.qSeqId == scoreArgs.kSeqId) && (this->maskType == CausalMaskT::MASK_TRIL)) ? 1 : 0;

        int64_t m = (scoreArgs.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight
                                                              : (this->xDim1 - scoreArgs.qSeqId * this->blockHeight);
        int64_t n = (scoreArgs.kSeqId != (seqBlockNumKV - 1))
                        ? this->blockHeight
                        : (this->xDim1 * MC2_KV_DIM_FACTOR - scoreArgs.kSeqId * this->blockHeight);

        this->VecScoreImpl(scoreArgs.taskId, attnBiasOffset, maskOffset, this->siluScale, causalMask, m, n);
    }

    __aicore__ inline void DoQkMatmul(QkMatmulArgs& qkPosArgs)
    {
        if (qkPosArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t qOffset = qkPosArgs.batchId * this->xDim1 * this->xDim2 * this->xDim3 +
                          qkPosArgs.qSeqId * this->blockHeight * this->xDim2 * this->xDim3 +
                          qkPosArgs.headId * this->xDim3;
        int64_t kOffset = qkPosArgs.batchId * this->xDim1 * this->xDim2 * this->xDim3 +
                          qkPosArgs.kSeqId * this->blockHeight * this->xDim2 * this->xDim3 +
                          qkPosArgs.headId * this->xDim3;

        int64_t m = (qkPosArgs.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight
                                                              : (this->xDim1 - qkPosArgs.qSeqId * this->blockHeight);
        int64_t n = (qkPosArgs.kSeqId != (seqBlockNumKV - 1))
                        ? this->blockHeight
                        : (this->xDim1 * MC2_KV_DIM_FACTOR - qkPosArgs.kSeqId * this->blockHeight);

        this->DoQkMatmulImpl(qOffset, kOffset, qkPosArgs.taskId, m, n, this->xDim3);
    }

    __aicore__ inline void DoSvMatmul(SvMatmulArgs& svArgs)
    {
        if (svArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t vOffset = svArgs.batchId * this->xDim1 * this->xDim2 * this->xDim3 +
                          svArgs.vSeqId * this->blockHeight * this->xDim2 * this->xDim3 + svArgs.headId * this->xDim3;

        int64_t m = (svArgs.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight
                                                           : (this->xDim1 - svArgs.qSeqId * this->blockHeight);
        int64_t n = (svArgs.vSeqId != (seqBlockNumKV - 1))
                        ? this->blockHeight
                        : (this->xDim1 * MC2_KV_DIM_FACTOR - svArgs.vSeqId * this->blockHeight);

        if (svArgs.vSeqId == 0) {
            // Override
            this->DoSvMatmulImpl(vOffset, svArgs.taskId, svArgs.transTaskId, 0, m, this->xDim3, n);
        } else {
            // Automic Add
            this->DoSvMatmulImpl(vOffset, svArgs.taskId, svArgs.transTaskId, 1, m, this->xDim3, n);
        }
    }

    __aicore__ inline void DoTransSv(SVTransArgs& args)
    {
        if (args.transTaskId == INVALID_TASK_ID) {
            return;
        }

        int64_t outStartOffset = args.batchId * this->xDim1 * this->xDim2 * this->xDim3 +
                                 args.qSeqId * this->blockHeight * this->xDim2 * this->xDim3 +
                                 args.headId * this->xDim3;

        int64_t m =
            (args.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight : (this->xDim1 - args.qSeqId * this->blockHeight);

        this->DoTransSvImpl(args.transTaskId, outStartOffset, m);
    }

    __aicore__ inline void SplitData(const int64_t totalLen, const int64_t useCoreNum, const int64_t useCoreIdx,
                                     int64_t& dataOffset, int64_t& dataLen)
    {
        // Evenly divide data across cores, round up
        dataLen = totalLen / useCoreNum;
        int64_t dataRemain = totalLen % useCoreNum;

        // First dataRemain cores get one extra element
        if (useCoreIdx < dataRemain) {
            dataLen = dataLen + 1;
            dataOffset = useCoreIdx * dataLen;
        } else {
            dataOffset = (dataLen + 1) * dataRemain + dataLen * (useCoreIdx - dataRemain);
        }
    }

    __aicore__ inline void AICPUAllGather(const HstuDenseForwardTilingData* __restrict tilingDataPtr)
    {
        this->pipe->InitBuffer(coll.tBuf, 256);

        blockIdx = GetBlockIdx();
        pipe_barrier(PIPE_ALL);
        SyncAll<true>();

        uint64_t batchCount = this->xDim1 * this->xDim2 * this->xDim3;
        uint64_t totalCount = this->xDim0 * batchCount;
        uint64_t copyCount = BATCH_SIZE_ONCE * batchCount;

        windowOutGM_ = this->hccl_.GetWindowsOutAddr(this->rankId);
        windowInGM_ = this->hccl_.GetWindowsInAddr(this->rankId);

        windowsOutGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(windowOutGM_));
        windowsInGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(windowInGM_));
        pipe_barrier(PIPE_ALL);
        SyncAll<true>();

        if (blockIdx == 0) {
            auto inLt = this->queIn.template AllocTensor<int16_t>();
            auto newinLt = inLt.template ReinterpretCast<qType>();
            inLt.SetValue(0, 0);

            for (int batchId = 0; batchId < this->xDim0; batchId++) {
                DataCopy(windowsOutGt[FLAG_OFFSET / sizeof(qType) + batchId * 8], newinLt, 8);
                pipe_barrier(PIPE_ALL);
            }
            this->queIn.template FreeTensor(inLt);
        }

        if (blockIdx == 0) {
            HcclHandle handleId1 = this->hccl_.template AllGather<true>(windowOutGM_, windowInGM_, 1,
                                                                        HcclDataType::HCCL_DATA_TYPE_FP32, 0);
            this->hccl_.Wait(handleId1);
            pipe_barrier(PIPE_ALL);

            coll.CpGM2GMPingPong(sizeof(qType) * totalCount, this->vGtOri, windowsOutGt[totalCount], COPYONLY);
            pipe_barrier(PIPE_ALL);
            coll.CpGM2GMPingPong(sizeof(qType) * totalCount, this->kGtOri, windowsOutGt, COPYONLY);
            pipe_barrier(PIPE_ALL);

            for (int batchId = 0; batchId < this->xDim0; batchId++) {
                handleId[batchId] =
                    this->hccl_.template AllGather<true>(windowOutGM_ + sizeof(qType) * batchId * batchCount,
                                                         windowInGM_ + sizeof(qType) * batchCount * this->rankSize +
                                                             sizeof(qType) * batchId * batchCount * this->rankSize,
                                                         batchCount, HcclDataType::HCCL_DATA_TYPE_FP32, 0);

                handleId[batchId + this->xDim0] = this->hccl_.template AllGather<true>(
                    windowOutGM_ + sizeof(qType) * totalCount + sizeof(qType) * batchId * batchCount,
                    windowInGM_ + sizeof(qType) * batchCount * this->rankSize +
                        sizeof(qType) * totalCount * this->rankSize +
                        sizeof(qType) * batchId * batchCount * this->rankSize,
                    batchCount, HcclDataType::HCCL_DATA_TYPE_FP32, 0);
            }
        }

        pipe_barrier(PIPE_ALL);
        SyncAll<true>();

        if (blockIdx == 0) {
            auto inLt = this->queIn.template AllocTensor<int16_t>();
            auto newinLt = inLt.template ReinterpretCast<qType>();
            inLt.SetValue(0, AG_COMPLETE_FLAG);

            for (int batchId = 0; batchId < this->xDim0; batchId++) {
                this->hccl_.Wait(handleId[batchId]);
                this->hccl_.Wait(handleId[batchId + this->xDim0]);
                DataCopy(windowsOutGt[FLAG_OFFSET / sizeof(qType) + batchId * 8], newinLt, 8);
                pipe_barrier(PIPE_ALL);
            }
            this->queIn.template FreeTensor(inLt);
        }
    }

    __aicore__ inline void Compute(const HstuDenseForwardTilingData* __restrict tilingDataPtr)
    {
        if (blockIdx == 0) {
            return;
        }
        PreInit(tilingDataPtr);

        int64_t lenOfThisSplitBatch = SPLIT_BATCH_SIZE * this->xDim2 * seqBlockNumQk;
        int64_t offsetOfThisSplitBatch;

        for (int64_t i = 0; i < DivCeil(this->xDim0, SPLIT_BATCH_SIZE); i++) {
            offsetOfThisSplitBatch = i * lenOfThisSplitBatch;
            if (i == DivCeil(this->xDim0, SPLIT_BATCH_SIZE) - 1) {
                lenOfThisSplitBatch = this->qkTotalBlock - offsetOfThisSplitBatch;
            }
            int64_t taskId = 0;
            int64_t transTaskId = 0;

            int64_t blockNumOfOneBatch = this->xDim2 * this->seqBlockNumQk;
            int64_t blockNumOfOneHead = this->seqBlockNumQk;

            int64_t lenOfThisCore;
            int64_t offsetOfThisCore;

#if HSTU_FUSED_MODE
            SplitData(lenOfThisSplitBatch, MC2_COMPUTE_CORE_NUM - 1, blockIdx - 1, offsetOfThisCore, lenOfThisCore);
#else
            SplitData(lenOfThisSplitBatch, MC2_COMPUTE_CORE_NUM, blockIdx - 1, offsetOfThisCore, lenOfThisCore);
#endif

            SVTransArgs lastSvTrans;
            SVTransArgs lastLastSvTrans;

            ScoreVectorArgs lastVectorScore;
            SvMatmulArgs lastSvMatmulArgs;
            SvMatmulArgs lastLastSvMatmulArgs;

            for (int64_t qBlockId = offsetOfThisSplitBatch + offsetOfThisCore;
                 qBlockId < offsetOfThisSplitBatch + offsetOfThisCore + lenOfThisCore; qBlockId++) {
                int64_t batchId = qBlockId / blockNumOfOneBatch;

                int64_t batchRemain = qBlockId % blockNumOfOneBatch;

                int64_t headId = batchRemain / blockNumOfOneHead;
                int64_t headReamin = batchRemain % blockNumOfOneHead;

                int64_t qSeqId = headReamin;

#if HSTU_FUSED_MODE
                {
                    auto inLt = this->queIn.template AllocTensor<qType>();
                    auto newinLt = inLt.template ReinterpretCast<int16_t>();
                    DataCopy(inLt, windowsOutGt[FLAG_OFFSET / sizeof(qType) + batchId * 8], 8);
                    pipe_barrier(PIPE_ALL);
                    while (newinLt.GetValue(0) != AG_COMPLETE_FLAG) {
                        DataCopy(inLt, windowsOutGt[FLAG_OFFSET / sizeof(qType) + batchId * 8], 8);
                        pipe_barrier(PIPE_ALL);
                    }
                    this->queIn.template FreeTensor(inLt);
                }
#endif

                for (int64_t kSeqId = 0; kSeqId < this->seqBlockNumKV; kSeqId++) {
                    if ((this->maskType == CausalMaskT::MASK_TRIL) and (kSeqId > qSeqId)) {
                        continue;
                    }

                    int qkBlockId = qBlockId * this->seqBlockNumKV + kSeqId;
                    QkMatmulArgs qkArgs = {taskId, qkBlockId, batchId, headId, qSeqId, kSeqId};
                    this->DoQkMatmul(qkArgs);
                    if (taskId > 1) {
                        this->DoSvMatmul(lastLastSvMatmulArgs);
                    }
                    ScoreVectorArgs scoreArgs = {taskId, qkBlockId, batchId, headId, qSeqId, kSeqId};
                    if (taskId > 0) {
                        this->VecScore(lastVectorScore);
                    }
                    lastVectorScore = scoreArgs;

                    int64_t vSeqId = kSeqId;
                    SvMatmulArgs svMatmulArgs = {transTaskId, taskId, qkBlockId, batchId,
                                                 headId,      qSeqId, kSeqId,    vSeqId};
                    lastLastSvMatmulArgs = lastSvMatmulArgs;
                    lastSvMatmulArgs = svMatmulArgs;

                    this->WaitQkMatmul();
                    if (taskId > 1) {
                        this->WaitSvMatmul();
                    }

                    taskId += 1;
                }

                SVTransArgs svTransArgs = {transTaskId, qBlockId * this->seqBlockNumQk, batchId, headId, qSeqId};
                if (transTaskId > 1) {
                    this->DoTransSv(lastLastSvTrans);
                }
                lastLastSvTrans = lastSvTrans;
                lastSvTrans = svTransArgs;
                transTaskId += 1;
            }

            if (taskId == 0) {
                return;
            }

            if (taskId == 1) {
                this->VecScore(lastVectorScore);
                pipe_barrier(PIPE_ALL);
                this->DoSvMatmul(lastSvMatmulArgs);
                this->WaitSvMatmul();
                this->DoTransSv(lastSvTrans);
                return;
            }

            this->DoSvMatmul(lastLastSvMatmulArgs);
            this->VecScore(lastVectorScore);
            pipe_barrier(PIPE_ALL);
            this->DoSvMatmul(lastSvMatmulArgs);
            this->WaitSvMatmul();
            this->DoTransSv(lastLastSvTrans);
            this->WaitSvMatmul();
            this->DoTransSv(lastSvTrans);
        }
    }

private:
    int64_t seqBlockNumQk;
    int64_t seqBlockNumKV;
    int64_t qkTotalBlock;
    int64_t blockIdx;
    Collectives coll{0, 0, 0, this->pipe};
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    HcclHandle handleId[MAX_HANDLE_ID];
    GlobalTensor<qType> windowsOutGt;
    GlobalTensor<qType> windowsInGt;
};

}  // namespace HstuDenseForward

template <typename qType>
__aicore__ inline void InvokeHstuNormalOpImpl(HstuDenseForward::Args& args)
{
    TPipe tPipe;
    HstuDenseForward::HstuDenseForwardKernel<qType> op;
    GET_TILING_DATA_WITH_STRUCT(HstuDenseForwardTilingData, tilingData, args.tiling);
    const HstuDenseForwardTilingData* __restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, &tilingDataPtr->qkMatmul, op.svMatmul,
                      &tilingDataPtr->svMatmul);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    op.qkMatmul.SetUserDefInfo(tilingPtr);
    op.svMatmul.SetUserDefInfo(tilingPtr);
    op.Init(args, tilingDataPtr, &tPipe);
#if HSTU_FUSED_MODE
    op.AICPUAllGather(tilingDataPtr);
#endif
    op.Compute(tilingDataPtr);
    op.hccl_.Finalize();
}

#endif

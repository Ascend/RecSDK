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
#ifndef HSTU_JAGGED_FORWARD_KERNEL_H
#define HSTU_JAGGED_FORWARD_KERNEL_H

#include "hstu_split_core_policy.h"
#include "matmul_mgmt.h"
#include "trans.h"
#include "vector_score_inter.h"

using namespace AscendC;
using namespace HstuForward;

namespace HstuJaggedForward {

struct JaggedArgs {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR attnBias;
    // jagged
    GM_ADDR seqOffsetQ;
    GM_ADDR seqOffsetK;
    // mask_
    GM_ADDR numContext;
    GM_ADDR numTarget;

    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;

    const HstuJaggedForwardTilingData* __restrict tilingDataPtr {nullptr};
};

template <typename TraitParams, typename TilingDataType, typename MatmulMgmtType, typename VectorScoreType>
class HstuJaggedForwardKernel {
public:
    using qType = typename TraitParams::qType;
    using oType = typename TraitParams::oType;

    using vecScoreInter = VectorScoreInter<qType, TraitParams::maskType, HstuForward::BlockMaskParams, VectorScoreType>;

    __aicore__ inline HstuJaggedForwardKernel() {};

    __aicore__ inline void Compute(const JaggedArgs& args, MatmulMgmtType* mmMgmt, vecScoreInter* vecScore)
    {
        if (!Init(args, mmMgmt, vecScore)) {
            return;  // no task
        }
        ComputeAllBlock();
    }

    __aicore__ inline bool Init(const JaggedArgs& args, MatmulMgmtType* mmMgmt, vecScoreInter* vecScore)
    {
        InitArgs(args);
        InitGtInfo();
        InitPipe();
        InitSubModules(mmMgmt, vecScore);

        return InitSplitCore();
    }

    __aicore__ inline void InitArgs(const JaggedArgs& args)
    {
        q_ = args.q;
        k_ = args.k;
        v_ = args.v;
        attnBias_ = args.attnBias;
        mask_ = args.mask;
        seqOffsetQ_ = args.seqOffsetQ;
        seqOffsetK_ = args.seqOffsetK;

        attnOutput_ = args.attnOutput;
        workspace_ = args.workspace;

        numContext_ = args.numContext;
        numTarget_ = args.numTarget;
        // Batch Size
        batchSize_ = args.tilingDataPtr->batchSize;
        // Seq Len
        seqLen_ = args.tilingDataPtr->seqLen;
        // Head Num
        headNum_ = args.tilingDataPtr->headNum;
        // Embedding Dim
        headDim_ = args.tilingDataPtr->dim;
        headDimV_ = args.tilingDataPtr->vDim;

        maxSeqLenQ_ = args.tilingDataPtr->maxSeqLenq;
        maxSeqLenK_ = args.tilingDataPtr->maxSeqLenk;

        // attr
        siluScale_ = args.tilingDataPtr->siluScale;
        alpha_ = args.tilingDataPtr->alpha;
        targetGroupSize_ = args.tilingDataPtr->targetGroupSize;
        enableNumContext_ = args.tilingDataPtr->enableNumContext;
        enableNumTarget_ = args.tilingDataPtr->enableNumTarget;

        // copyKV
        copyHeadNum_ = args.tilingDataPtr->headNumK;

        // GQA
        headNumK_ = args.tilingDataPtr->headNumK;
        headRatio_ = args.tilingDataPtr->headRatio;
    }

    __aicore__ inline void InitGtInfo()
    {
        // Gt
        qGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(q_));
        kGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(k_));
        vGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(v_));

        if constexpr (TraitParams::enableBias) {
            attnBiasGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnBias_));
        }

        if constexpr (TraitParams::maskType == CausalMaskT::MASK_CUSTOM) {
            attnMaskGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(mask_));
        }

        attnOutputGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnOutput_));

        const uint32_t coreNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;

        int64_t oneBlockMidElem = BLOCK_HEIGHT_256 * BLOCK_HEIGHT_256 * COMPUTE_PIPE_NUM;
        int64_t oneCoreMidElem = coreNum * oneBlockMidElem;

        int64_t oneBlockMidTransElem = BLOCK_HEIGHT_256 * MAX_BLOCK_DIM * TRANS_PIPE_NUM;
        int64_t oneCoreTransMidElem = coreNum * oneBlockMidTransElem;
        int64_t kvOffset = oneCoreMidElem + oneCoreTransMidElem * 3; // svResultGt_ midkGt midvGt

        attnScoreGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(workspace_) + GetBlockIdx() * oneBlockMidElem);
        svResultGt_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace_) + oneCoreMidElem + GetBlockIdx() * oneBlockMidTransElem,
            oneBlockMidTransElem);
        syncGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(workspace_) + oneCoreMidElem + coreNum * oneBlockMidTransElem);
    }

    __aicore__ inline void InitPipe()
    {
        pipe_.InitBuffer(vecIn_, 1, 8 * sizeof(int32_t));

        pipe_.InitBuffer(scm_, 1, TraitParams::blockM * TraitParams::blockN * sizeof(qType));
    }

    __aicore__ inline void InitSubModules(MatmulMgmtType* mmMgmt, vecScoreInter* vecScore)
    {
        mmMgmt_ = mmMgmt;
        vecScore_ = vecScore;

        VecScoreRabGtInfo<qType> gtInfo = { attnBiasGt_, attnMaskGt_ };
        vecScore_->Init(&pipe_, gtInfo, siluScale_, alpha_, TraitParams::blockM, TraitParams::blockN, maxSeqLenK_);

        if constexpr (TraitParams::deterministic) {
            vecScore_->InitSyncGm(syncGm_);
        }

        transSVResult_.Init(&pipe_, headNum_, headDimV_, TraitParams::blockK);
    }

    __aicore__ inline bool InitSplitCore()
    {
        seqOffsetsQGt_.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(seqOffsetQ_), batchSize_ + 1);
        seqOffsetsKGt_.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(seqOffsetK_), batchSize_ + 1);
        auto validBatchSize = GetBatchSizeFromJaggedOffset(seqOffsetsQGt_, batchSize_ + 1);
        ASCENDC_ASSERT((validBatchSize > 0 && validBatchSize <= MAX_BATCH_SIZE),
            "batchSize_ exceed limit of (0, 2048]\n");

        const int blockId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum() * GetTaskRation();
        batchSize_ = validBatchSize;

        numContextGt_.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(numContext_), batchSize_);
        numTargetGt_.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(numTarget_), batchSize_);

        splitMode_ = STREAM_K;
        if (maxSeqLenQ_ <= TraitParams::blockM && maxSeqLenK_ <= TraitParams::blockN) {
            splitMode_ = FAST_SPLIT_SINGLE;
        }

        int blocks[4] = {0};  // start block id, end block id
        if constexpr (TraitParams::maskType == CausalMaskT::MASK_TRIL) {
            auto taskAssigner = BlockTaskAssign<oType, CausalMaskT::MASK_TRIL>(
                coreNum, batchSize_, headNum_, targetGroupSize_, TraitParams::blockM,
                TraitParams::blockN, seqOffsetsQGt_, seqOffsetsKGt_, numContextGt_, numTargetGt_, splitMode_);
            taskAssigner.Compute(blocks, blockId);
        } else if constexpr (TraitParams::maskType == CausalMaskT::MASK_CUSTOM) {
            auto taskAssigner = BlockTaskAssign<oType, CausalMaskT::MASK_CUSTOM>(
                coreNum, batchSize_, headNum_, targetGroupSize_, TraitParams::blockM,
                TraitParams::blockN, seqOffsetsQGt_, seqOffsetsKGt_, numContextGt_, numTargetGt_, splitMode_);
            taskAssigner.Compute(blocks, blockId);
        } else {
            auto taskAssigner = BlockTaskAssign<oType, CausalMaskT::MASK_NONE>(
                coreNum, batchSize_, headNum_, targetGroupSize_, TraitParams::blockM,
                TraitParams::blockN, seqOffsetsQGt_, seqOffsetsKGt_, numContextGt_, numTargetGt_, splitMode_);
            taskAssigner.Compute(blocks, blockId);
        }

        skSeqBlkId_ = blocks[0];
        ekSeqBlkId_ = blocks[1];
        sBlkId_ = blocks[2];
        eBlkId_ = blocks[3];
        if (skSeqBlkId_ == ekSeqBlkId_ && sBlkId_ == eBlkId_ && sBlkId_ == 0 && skSeqBlkId_ == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void ComputeAllBlock()
    {
        GetTaskInfo();

        uint32_t transtaskId = 0;
        uint32_t taskId = 0;

        uint32_t currentTaskId = 0;
        uint32_t preTaskId = 0;
        uint32_t prePreTaskId = 0;
        uint32_t nextTaskId = 0;
        uint32_t kSeqId = skSeqBlkId_;
        uint32_t kSeqNum = 0;

        scmQKTensor_ =  scm_.template AllocTensor<qType>();
        for (auto blkId = sBlkId_; blkId <= eBlkId_; blkId++) {
            kSeqNum = computeTaskInfo_[taskId % COMPUTE_PIPE_NUM].kSeqNum;
            auto deltaQK = computeTaskInfo_[taskId % COMPUTE_PIPE_NUM].deltaQK;
            auto nblk = deltaQK / TraitParams::blockN;
            bool isDeltaQK = deltaQK % TraitParams::blockN != 0;
            auto limit = (blkId == eBlkId_) ? ekSeqBlkId_ : kSeqNum;
            uint32_t isStartFromZero = (kSeqId == 0);
            uint32_t isEndToTail = false;
            for (; kSeqId < limit; kSeqId++) {
                auto taskinfo = computeTaskInfo_[taskId % COMPUTE_PIPE_NUM];
                int64_t maskOffset1 = (deltaQK +  (int64_t)taskinfo.qSeqId * TraitParams::blockM) % TraitParams::blockN;
                int64_t maskOffset2 = (deltaQK + (int64_t)taskinfo.qSeqId * TraitParams::blockM + TraitParams::blockM) %
                                    TraitParams::blockN - TraitParams::blockM;
                HstuForward::BlockMaskParams maskinfo = {
                    taskinfo.qSeqId,
                    kSeqId,
                    taskinfo.actualSeqLen,
                    taskinfo.actualSeqLenK,
                    TraitParams::blockM,
                    TraitParams::blockN,
                    taskinfo.numContext,
                    taskinfo.numTarget,
                    targetGroupSize_,
                    taskinfo.scale,
                    maskOffset1,
                    maskOffset2,
                    nblk,
                    isDeltaQK
                };
                // 在下三角下跳过运算
                if (maskinfo.NoComputation(TraitParams::maskType)) {
                    isEndToTail = true;
                    break;
                }
                
                currentTaskId = taskId % COMPUTE_PIPE_NUM;
                preTaskId = (taskId + COMPUTE_PIPE_NUM - 1) % COMPUTE_PIPE_NUM;
                prePreTaskId = (taskId + COMPUTE_PIPE_NUM - 2) % COMPUTE_PIPE_NUM;
                nextTaskId = (taskId + 1) % COMPUTE_PIPE_NUM;

                maskTaskInfo_[currentTaskId] = maskinfo;
                computeTaskInfo_[currentTaskId].transTaskId = transtaskId % TRANS_PIPE_NUM;
                computeTaskInfo_[currentTaskId].kSeqId = kSeqId;
                computeTaskInfo_[currentTaskId].computeBSeqLen =
                    (kSeqId != (kSeqNum - 1))
                        ? (TraitParams::blockN)
                        : (computeTaskInfo_[currentTaskId].actualSeqLenK - kSeqId * TraitParams::blockN);
                uint64_t kvHeadId = computeTaskInfo_[currentTaskId].headId / headRatio_;
                computeTaskInfo_[currentTaskId].kOffset =
                    computeTaskInfo_[currentTaskId].batchOffsetK * headDim_ * headNumK_ +
                    computeTaskInfo_[currentTaskId].kSeqId * TraitParams::blockN * headNumK_ * headDim_ +
                    kvHeadId * headDim_;
                computeTaskInfo_[currentTaskId].vOffset =
                    computeTaskInfo_[currentTaskId].batchOffsetK * headDimV_ * headNumK_ +
                    computeTaskInfo_[currentTaskId].kSeqId * TraitParams::blockN * headNumK_ * headDimV_ +
                    kvHeadId * headDimV_;

                // matmul qk
                ComputeQkMatmul(currentTaskId);

                // matmul sv
                if (taskId > 1) {
                    ComputeSvMatmul(prePreTaskId);
                }

                // VecScore
                if (taskId > 0) {
                    ComputeVecScore(preTaskId);
                }

                // wait qk
                WaitQkMatmul();

                // wait sv
                if (taskId > 1) {
                    WaitSvMatmul();
                }

                computeTaskInfo_[nextTaskId] = computeTaskInfo_[currentTaskId];
                computeTaskInfo_[nextTaskId].isFirstSeqBlk = 0;
                maskTaskInfo_[nextTaskId] = maskTaskInfo_[currentTaskId];
                taskId++;
            }
            if (blkId == eBlkId_ && ekSeqBlkId_ == 0) {
                break;
            }

            transTaskInfo_[transtaskId % TRANS_PIPE_NUM] = computeTaskInfo_[currentTaskId];
            transTaskInfo_[transtaskId % TRANS_PIPE_NUM].isStartFromZero = isStartFromZero;
            transTaskInfo_[transtaskId % TRANS_PIPE_NUM].isEndToTail = isEndToTail | (kSeqId == kSeqNum);
            if (transtaskId > 1) {
                TransResult(transtaskId - 2);
            }
            transtaskId++;

            UpdateTaskInfo(taskId % COMPUTE_PIPE_NUM);
            kSeqId = 0;
        }

        scm_.FreeTensor(scmQKTensor_);
        ComputeTailBlock(taskId, currentTaskId, preTaskId, transtaskId);
    }

    __aicore__ inline void ComputeTailBlock(uint32_t taskId,
                                            uint32_t currentTaskId,
                                            uint32_t preTaskId,
                                            uint32_t transtaskId)
    {
        if (taskId == 0) {
            return;
        }

        if (taskId == 1) {
            ComputeVecScore(currentTaskId);
            pipe_barrier(PIPE_ALL);

            ComputeSvMatmul(currentTaskId);
            WaitSvMatmul();
            WaitNextBlock(transtaskId - 1);
            TransResult(transtaskId - 1);
            return;
        }

        if (transtaskId == 1) {
            ComputeSvMatmul(preTaskId);
            WaitSvMatmul();

            ComputeVecScore(currentTaskId);
            pipe_barrier(PIPE_ALL);

            ComputeSvMatmul(currentTaskId);
            WaitSvMatmul();
            WaitNextBlock(transtaskId - 1);
            TransResult(transtaskId - 1);
            return;
        }

        ComputeSvMatmul(preTaskId);
        WaitSvMatmul();

        ComputeVecScore(currentTaskId);
        pipe_barrier(PIPE_ALL);

        ComputeSvMatmul(currentTaskId);
        WaitSvMatmul();

        TransResult(transtaskId - 2);
        WaitNextBlock(transtaskId - 1);
        TransResult(transtaskId - 1);
    }

    __aicore__ inline void GetTaskInfo()
    {
        uint32_t offsetOfBlk = 0;
        int64_t offsetOfSeq = 0;
        int64_t seqGlobalOffset = 0;
        for (auto index = 0; index < batchSize_ * headNum_; index++) {
            uint32_t batchId = index / headNum_;
            uint32_t headId = index % headNum_;

            uint32_t batchSeqSize = seqOffsetsQGt_.GetValue(batchId + 1) - seqOffsetsQGt_.GetValue(batchId);
            uint32_t batchBlkSize = (batchSeqSize + TraitParams::blockM - 1) / TraitParams::blockM;
            if (sBlkId_ < (offsetOfBlk + batchBlkSize)) {
                uint32_t innerBlkId = sBlkId_ - offsetOfBlk;
                seqGlobalOffset = seqGlobalOffset + innerBlkId * TraitParams::blockM;
                FillTaskInfo(batchId, headId, seqGlobalOffset, 0);
                return;
            }

            offsetOfBlk += batchBlkSize;
            seqGlobalOffset += batchSeqSize;
        }
    }

    __aicore__ inline void FillTaskInfo(uint32_t batchId, uint32_t headId, int64_t seqGlobalOffset, uint32_t taskId)
    {
        if (batchId >= batchSize_) {
            return;
        }
    
        taskId = taskId % COMPUTE_PIPE_NUM;
    
        auto nextBatchSeqOffset = seqOffsetsQGt_.GetValue(batchId + 1);
        auto currentBatchSeqOffset = seqOffsetsQGt_.GetValue(batchId);
    
        auto nextBatchSeqOffsetK = seqOffsetsKGt_.GetValue(batchId + 1);
        auto currentBatchSeqOffsetK = seqOffsetsKGt_.GetValue(batchId);
    
        auto numContext = numContextGt_.GetValue(batchId);
        auto numTarget = numTargetGt_.GetValue(batchId);
    
        computeTaskInfo_[taskId].seqGlobalOffset = seqGlobalOffset;
        computeTaskInfo_[taskId].batchId = batchId;
        computeTaskInfo_[taskId].actualSeqLen = nextBatchSeqOffset - currentBatchSeqOffset;
        computeTaskInfo_[taskId].actualSeqLenK = nextBatchSeqOffsetK - currentBatchSeqOffsetK;
        computeTaskInfo_[taskId].deltaQK =
            computeTaskInfo_[taskId].actualSeqLenK - computeTaskInfo_[taskId].actualSeqLen;
    
        computeTaskInfo_[taskId].scale = siluScale_;
        computeTaskInfo_[taskId].numTarget = numTarget;
        computeTaskInfo_[taskId].numContext = numContext;
        computeTaskInfo_[taskId].batchOffset = currentBatchSeqOffset;
        computeTaskInfo_[taskId].batchOffsetK = currentBatchSeqOffsetK;
        // 每个注意力头在序列维度上的处理边界
        computeTaskInfo_[taskId].headSeqLimit =
            computeTaskInfo_[taskId].batchOffset * headNum_ + computeTaskInfo_[taskId].actualSeqLen * (headId + 1);
        auto batchInnerOffset = seqGlobalOffset - (computeTaskInfo_[taskId].batchOffset * headNum_);
        computeTaskInfo_[taskId].headId = headId;
        computeTaskInfo_[taskId].qSeqId =
            (batchInnerOffset - computeTaskInfo_[taskId].headId *
             computeTaskInfo_[taskId].actualSeqLen) / TraitParams::blockM;
        computeTaskInfo_[taskId].kSeqNum =
            CeilDiv(computeTaskInfo_[taskId].actualSeqLenK, static_cast<uint32_t>(TraitParams::blockN));
        computeTaskInfo_[taskId].qSeqNum =
            CeilDiv(computeTaskInfo_[taskId].actualSeqLen, static_cast<uint32_t>(TraitParams::blockM));
    
        computeTaskInfo_[taskId].iOffset =
            computeTaskInfo_[taskId].batchOffset * headDim_ * headNum_ +
            computeTaskInfo_[taskId].qSeqId * TraitParams::blockM * headNum_ * headDim_ +
            computeTaskInfo_[taskId].headId * headDim_;
        computeTaskInfo_[taskId].oOffset =
            computeTaskInfo_[taskId].batchOffset * headDimV_ * headNum_ +
            computeTaskInfo_[taskId].qSeqId * TraitParams::blockM * headNum_ * headDimV_ +
            computeTaskInfo_[taskId].headId * headDimV_;
    
        if ((computeTaskInfo_[taskId].headSeqLimit - seqGlobalOffset) >= TraitParams::blockM) {
            computeTaskInfo_[taskId].computeASeqLen = TraitParams::blockM;
        } else {
            computeTaskInfo_[taskId].computeASeqLen = computeTaskInfo_[taskId].headSeqLimit - seqGlobalOffset;
        }
    }

    __aicore__ inline void ComputeQkMatmul(uint32_t taskId)
    {
        MatmulArgs args = {
            .leftOffset = computeTaskInfo_[taskId].iOffset,
            .rightOffset = computeTaskInfo_[taskId].kOffset,
            .outOffset = (taskId % COMPUTE_PIPE_NUM) * TraitParams::blockM * TraitParams::blockM,
            .m = computeTaskInfo_[taskId].computeASeqLen,
            .n = computeTaskInfo_[taskId].computeBSeqLen,
            .k = static_cast<uint32_t>(headDim_),
            .headNum = copyHeadNum_
        };
        if (computeTaskInfo_[taskId].isFirstSeqBlk) {
            mmMgmt_->template DoQKMatmul<true>(args, qGt_, kGt_, attnScoreGt_);
        } else {
            mmMgmt_->template DoQKMatmul<false>(args, qGt_, kGt_, attnScoreGt_);
        }
    }

    __aicore__ inline void ComputeSvMatmul(uint32_t taskId)
    {
        uint8_t isAtomic = (computeTaskInfo_[taskId].isFirstSeqBlk) ? 0 : 1;
        MatmulArgs args = {
            .leftOffset = (taskId % COMPUTE_PIPE_NUM) * TraitParams::blockM * TraitParams::blockN,
            .rightOffset = computeTaskInfo_[taskId].vOffset,
            .outOffset =
                (computeTaskInfo_[taskId].transTaskId % TRANS_PIPE_NUM) * TraitParams::blockM * TraitParams::blockK,
            .m = computeTaskInfo_[taskId].computeASeqLen,
            .n = static_cast<uint32_t>(headDimV_),
            .k = computeTaskInfo_[taskId].computeBSeqLen,
            .headNum = copyHeadNum_,
            .isAtomicAdd = isAtomic
        };

        mmMgmt_->DoSVMatmul(args, attnScoreGt_, vGt_, svResultGt_);
    }

    __aicore__ inline void ComputeVecScore(uint32_t taskId)
    {
        int64_t srcOffset = (taskId % COMPUTE_PIPE_NUM) * TraitParams::blockM * TraitParams::blockM;
        int64_t biasOffset = computeTaskInfo_[taskId].batchId * headNum_ * maxSeqLenQ_ * maxSeqLenK_ +
                         computeTaskInfo_[taskId].headId * maxSeqLenQ_ * maxSeqLenK_ +
                         computeTaskInfo_[taskId].qSeqId * maxSeqLenK_ * TraitParams::blockM +
                         computeTaskInfo_[taskId].kSeqId * TraitParams::blockN;
                         
        int64_t maskOffset = biasOffset;

        VecScoreRabParam<HstuForward::BlockMaskParams> vecScoreParam = {
            .srcOffset = srcOffset,
            .biasOffset = biasOffset,
            .maskOffset = maskOffset,
            .m = computeTaskInfo_[taskId].computeASeqLen,
            .n = computeTaskInfo_[taskId].computeBSeqLen,
            .maskinfo = maskTaskInfo_[taskId]
        };

        vecScore_->VecScoreImpl(vecScoreParam, attnScoreGt_);
    }

    __aicore__ inline void WaitQkMatmul()
    {
        mmMgmt_->WaitQKMatmul();
    }

    __aicore__ inline void WaitSvMatmul()
    {
        mmMgmt_->WaitSVMatmul();
    }

    __aicore__ inline void TransResult(uint32_t transTaskId)
    {
        uint32_t transtaskIdModed = transTaskId % TRANS_PIPE_NUM;
        int64_t fromOffset = (transTaskId % TRANS_PIPE_NUM) * TraitParams::blockM * TraitParams::blockK;
        int64_t toOffset = transTaskInfo_[transtaskIdModed].oOffset;
        int64_t total = transTaskInfo_[transtaskIdModed].computeASeqLen * headDimV_;

        if constexpr (TraitParams::deterministic) {
            if (transTaskInfo_[transtaskIdModed].isEndToTail) {
                transSVResult_.template TransResult<false>(svResultGt_, attnOutputGt_, fromOffset, toOffset, total);
            } else {
                transSVResult_.template TransResult<true>(svResultGt_, attnOutputGt_, fromOffset, toOffset, total);
            }
        } else {
            transSVResult_.template TransResult<true>(svResultGt_, attnOutputGt_, fromOffset, toOffset, total);
        }
        
        if (transTaskId == 0) {
            NotifypreBlock();
        }
    }

    __aicore__ inline void NotifypreBlock()
    {
        if constexpr (TraitParams::deterministic) {
            if (GetBlockIdx() > 0 && transTaskInfo_[0].isStartFromZero == 0) {
                auto syncBuf = vecIn_.template AllocTensor<int32_t>();
                AscendC::IBSet<false>(syncGm_, syncBuf, GetBlockIdx(), 0);
                vecIn_.FreeTensor(syncBuf);
            }
        }
    }

    __aicore__ inline void UpdateTaskInfo(uint32_t taskId)
    {
        auto batchId = computeTaskInfo_[taskId].batchId;
        auto headId = computeTaskInfo_[taskId].headId;

        int64_t seqGlobalOffset = computeTaskInfo_[taskId].seqGlobalOffset;
        int64_t gap = computeTaskInfo_[taskId].headSeqLimit - seqGlobalOffset;

        if (gap <= TraitParams::blockM) {
            headId++;
            if (headId >= headNum_) {
                batchId++;
            }

            if (batchId >= batchSize_) {
                return;
            }

            seqGlobalOffset = seqGlobalOffset + gap;
            headId = headId % headNum_;
            FillTaskInfo(batchId, headId, seqGlobalOffset, taskId);
        } else {
            computeTaskInfo_[taskId].seqGlobalOffset = seqGlobalOffset + TraitParams::blockM;

            uint32_t computeASeqLen = TraitParams::blockM;
            if ((computeTaskInfo_[taskId].seqGlobalOffset + TraitParams::blockM) >
                computeTaskInfo_[taskId].headSeqLimit) {
                computeASeqLen = computeTaskInfo_[taskId].headSeqLimit - computeTaskInfo_[taskId].seqGlobalOffset;
            }

            auto batchInnerOffset =
                computeTaskInfo_[taskId].seqGlobalOffset - (computeTaskInfo_[taskId].batchOffset * headNum_);
            computeTaskInfo_[taskId].qSeqId =
                (batchInnerOffset - computeTaskInfo_[taskId].headId * computeTaskInfo_[taskId].actualSeqLen) /
                TraitParams::blockM;
            computeTaskInfo_[taskId].iOffset =
                computeTaskInfo_[taskId].batchOffset * headDim_ * headNum_ +
                computeTaskInfo_[taskId].qSeqId * TraitParams::blockM * headNum_ * headDim_ +
                computeTaskInfo_[taskId].headId * headDim_;
            computeTaskInfo_[taskId].oOffset =
                computeTaskInfo_[taskId].batchOffset * headDimV_ * headNum_ +
                computeTaskInfo_[taskId].qSeqId * TraitParams::blockM * headNum_ * headDimV_ +
                computeTaskInfo_[taskId].headId * headDimV_;
            computeTaskInfo_[taskId].computeASeqLen = computeASeqLen;
        }
        computeTaskInfo_[taskId].isFirstSeqBlk = 1;
    }

    __aicore__ inline void WaitNextBlock(uint32_t transtaskId)
    {
        if constexpr (TraitParams::deterministic) {
            if (GetBlockIdx() + 1 < GetBlockNum() * GetTaskRation() &&
                transTaskInfo_[transtaskId % TRANS_PIPE_NUM].isEndToTail == 0) {
                auto syncBuf = vecIn_.template AllocTensor<int32_t>();
                AscendC::IBWait<false>(syncGm_, syncBuf, GetBlockIdx() + 1, 0);
                vecIn_.FreeTensor(syncBuf);
            }
        }
    }

    MatmulMgmtType* mmMgmt_;
    vecScoreInter* vecScore_;
    Trans<float, qType> transSVResult_;

    // GM_ADDR
    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR attnBias_;
    GM_ADDR mask_;
    GM_ADDR seqOffsetQ_;
    GM_ADDR seqOffsetK_;

    GM_ADDR numContext_;
    GM_ADDR numTarget_;

    GM_ADDR attnOutput_;
    GM_ADDR workspace_;
    GM_ADDR tiling_;

    // Shape
    int64_t batchSize_;
    int64_t seqLen_;
    int64_t headNum_;
    int64_t headDim_;
    int64_t headDimV_;
    int64_t maxSeqLenQ_;
    int64_t maxSeqLenK_;
    bool enableNumContext_;
    bool enableNumTarget_;

    // Attr
    float siluScale_;
    float alpha_;
    int64_t targetGroupSize_;

    // copyQKV
    uint64_t copyHeadNum_;

    // GQA
    uint64_t headNumK_;
    uint64_t headRatio_;

    // Tpipe
    TPipe pipe_;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queOut_;
    TQue<TPosition::VECIN, 1> vecIn_;
    TSCM<TPosition::GM, 1> scm_;

    // Gt
    GlobalTensor<qType> qGt_;
    GlobalTensor<qType> kGt_;
    GlobalTensor<qType> vGt_;
    GlobalTensor<qType> attnOutputGt_;
    GlobalTensor<qType> attnScoreGt_;
    GlobalTensor<qType> attnBiasGt_;
    GlobalTensor<qType> attnMaskGt_;
    GlobalTensor<float> svResultGt_;
    GlobalTensor<int32_t> syncGm_;

    GlobalTensor<oType> seqOffsetsQGt_;
    GlobalTensor<oType> seqOffsetsKGt_;
    GlobalTensor<oType> numContextGt_;
    GlobalTensor<oType> numTargetGt_;

    LocalTensor<qType> scmQKTensor_;

    uint32_t sBlkId_ {0};
    uint32_t eBlkId_ {0};
    uint32_t skSeqBlkId_ {0};
    uint32_t ekSeqBlkId_ {0};
    int32_t splitMode_ {DEFAULT_SPLIT};

    HstuForward::BlockMaskParams maskTaskInfo_[COMPUTE_PIPE_NUM];
    JaggedTaskArgs computeTaskInfo_[COMPUTE_PIPE_NUM];
    JaggedTaskArgs transTaskInfo_[TRANS_PIPE_NUM];
};
}  // namespace HstuJaggedForward

#endif

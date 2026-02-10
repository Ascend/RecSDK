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

#ifndef LN_LINEAR_SILU_BACKWARD_H
#define LN_LINEAR_SILU_BACKWARD_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "split_core.h"

using namespace AscendC;

namespace InLinearSiluBackward {

constexpr int VEC_PER_PROCESS = 32;
constexpr int UB_SIZE = 170 * 1024;  // 170KB
constexpr int QUEUE_IN_NUM = 2;
constexpr int SPLIT_CORE = 2;
constexpr int ALIGN_16 = 16;
constexpr int COMPUTE_PIPE_NUM = 2;
constexpr int INT_ALIGN_NUM = 8;
constexpr int BLOCK_HEIGHT_128 = 128;
constexpr int BLOCK_HEIGHT_256 = 256;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int USE_QUEUE_NUM = 1;
constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int VECTOR_SCORE_UB_SIZE = 12 * 1024;
constexpr int UVQK_NUM = 4;
constexpr int UVQK_INDEX_U = 0;
constexpr int UVQK_INDEX_V = 1;
constexpr int UVQK_INDEX_Q = 2;
constexpr int UVQK_INDEX_K = 3;
constexpr float ONE_FLOAT = 1.0f;

struct Args {
    // input
    GM_ADDR x;
    GM_ADDR weight;
    GM_ADDR bias;
    GM_ADDR user_grad;
    GM_ADDR value_grad;
    GM_ADDR query_grad;
    GM_ADDR key_grad;
    GM_ADDR linear_output;
    // output
    GM_ADDR x_grad;
    GM_ADDR weight_grad;
    GM_ADDR bias_grad;
    // workspace
    GM_ADDR workspace;
    GM_ADDR tiling;
};

struct TaskInfo {
    int32_t rowId = 0;
    int32_t blockId = 0;
    int32_t eId = 0;
    int32_t taskId = 0;
    int32_t computeMSeqLen = 0;
    int32_t computeNSeqLen = 0;
    int32_t computeKSeqLen = 0;
    int32_t kOffset = 0; // LinearOut k轴总的偏移
    int32_t uvqkSeqOffset = 0; // uvqk 单个偏移
    int32_t uvqkId = 0;
    int32_t isFirst = 0;
};


template <typename xType, typename wType, bool enableBias, bool isTrans>
class InLinearSiluBackward {
public:
    static constexpr int ElementOfBlock = DATA_ALIGN_BYTES / sizeof(xType);
    static constexpr int blockHeight = BLOCK_HEIGHT_256;
    static constexpr int vectorScoreUbBlockElem = VECTOR_SCORE_UB_SIZE / QUEUE_IN_NUM;
    __aicore__ inline InLinearSiluBackward() {}
    __aicore__ inline void Init(const Args& args, const InLinearSiluBackwardTilingData* __restrict tilingDataPtr,
                             TPipe* pipePtr)
    {
        InitArgs(args, tilingDataPtr);
        InitPipe(pipePtr);
    }

    __aicore__ inline void InitArgs(const Args& args, const InLinearSiluBackwardTilingData* __restrict tilingDataPtr)
    {
        x = args.x;
        weight = args.weight;
        bias = args.bias;
        user_grad = args.user_grad;
        value_grad = args.value_grad;
        query_grad = args.query_grad;
        key_grad = args.key_grad;
        linear_output = args.linear_output;
        x_grad = args.x_grad;
        weight_grad = args.weight_grad;
        bias_grad = args.bias_grad;
        workspace = args.workspace;
        seqLen = tilingDataPtr->seqLen;
        blockM = BLOCK_HEIGHT_256;
        blockK = tilingDataPtr->blockK;
        hiddenSize = tilingDataPtr->hiddenSize;
        dim = tilingDataPtr->embedDim;
        uDim = tilingDataPtr->uDim;
        vDim = tilingDataPtr->vDim;
        qDim = tilingDataPtr->qDim;
        kDim = tilingDataPtr->kDim;

        totalDim = uDim + vDim + qDim + kDim;
        splitList[UVQK_INDEX_U] = uDim;
        splitList[UVQK_INDEX_V] = vDim;
        splitList[UVQK_INDEX_Q] = qDim;
        splitList[UVQK_INDEX_K] = kDim;
    }

    __aicore__ inline void InitPipe(TPipe* pipePtr)
    {
        pipe = pipePtr;

        // input
        xGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x));
        weightGt.SetGlobalBuffer(reinterpret_cast<__gm__ wType*>(weight));
        uGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(user_grad));
        vGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(value_grad));
        qGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(query_grad));
        kGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(key_grad));
        linearOutGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(linear_output));

        // output
        xGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_grad));
        weightGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight_grad));

        if constexpr (enableBias) {
            biasGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
            biasGradGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias_grad));
        }

        coreNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;

        int64_t oneBlockMidElem = blockK * blockM * COMPUTE_PIPE_NUM;
        int64_t weightStart = coreNum * oneBlockMidElem;
        int64_t weightOffset = blockK * dim;
        int64_t xStart = weightStart + weightOffset * coreNum;
        int64_t xOffset = blockM * dim;

        SiluGradGt.SetGlobalBuffer(
            reinterpret_cast<__gm__ xType*>(workspace) + oneBlockMidElem * GetBlockIdx(),
            oneBlockMidElem);
        pipe->InitBuffer(queIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(queOut, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(tmpBuff, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
    }

    __aicore__ inline void Compute()
    {
        // splitcore
        int blocks[2] = {0};
        auto taskAssigner = BlockTaskAssign(coreNum, blockM, blockK, seqLen, hiddenSize, dim);
        taskAssigner.SplitCoreFast(blocks, GetBlockIdx());
        if (blocks[0] == blocks[1] && blocks[0] == 0) {
            return;
        }
        this->sRow = blocks[0];
        this->eRow = blocks[1];
        ComputeAllBlock();
    }

    __aicore__ inline void Wait_V_MTE2()
    {
        AscendC::TQueSync<PIPE_V, PIPE_MTE2> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
    }
    __aicore__ inline void Wait_V_MTE3()
    {
        AscendC::TQueSync<PIPE_V, PIPE_MTE3> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
    }
    __aicore__ inline void Wait_MTE2_V()
    {
        AscendC::TQueSync<PIPE_MTE2, PIPE_V> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
    }

    __aicore__ inline void SiluGrad(TaskInfo &taskInfo, int32_t taskId)
    {
        GlobalTensor<xType>uvqkGt[UVQK_NUM] = {
            uGradGt, vGradGt, qGradGt, kGradGt,
        };
        // datacopyIn linearOut
        int32_t totalLen = taskInfo.computeMSeqLen * taskInfo.computeKSeqLen;
        int32_t siluOffset = blockM * blockK * (taskId % COMPUTE_PIPE_NUM);

        int32_t calcLen = 0;
        int32_t lines = 0;
        int32_t offset = 0;
        while (totalLen > 0) {
            if (totalLen > (vectorScoreUbBlockElem)) {
                lines = vectorScoreUbBlockElem / taskInfo.computeKSeqLen;
            } else {
                lines = totalLen / taskInfo.computeKSeqLen;
            }
            calcLen = lines * taskInfo.computeKSeqLen;

            LocalTensor<xType> inputLt = queIn.template AllocTensor<xType>();
            LocalTensor<float> tmpLt = tmpBuff.template AllocTensor<float>();
            LocalTensor<float> dstLt = queOut.template AllocTensor<float>();

            uint16_t blockCount = lines;
            uint16_t blockLen = taskInfo.computeKSeqLen * sizeof(xType) / DATA_ALIGN_BYTES;
            uint16_t dstGap = 0;
            uint16_t srcGap = (totalDim - taskInfo.computeKSeqLen) * sizeof(xType) / DATA_ALIGN_BYTES;
            DataCopyParams lnOutparams = {blockCount, blockLen, srcGap, dstGap};

            uint16_t uvqkSrcGap = (splitList[taskInfo.uvqkId] - taskInfo.computeKSeqLen) * \
                sizeof(xType) / DATA_ALIGN_BYTES;
            DataCopyParams uvqkParams = {blockCount, blockLen, uvqkSrcGap, dstGap};
            
            uint32_t srcOffset = (taskInfo.rowId * blockM + offset) * totalDim + taskInfo.kOffset;

            DataCopy(inputLt, linearOutGt[srcOffset], lnOutparams);
            queIn.template EnQue(inputLt);
            inputLt = queIn.template DeQue<xType>();
            
            if constexpr(!std::is_same<xType, float>::value) {
                Cast(tmpLt, inputLt, RoundMode::CAST_NONE, calcLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                auto srcLt = inputLt.template ReinterpretCast<float>();
                LinearOutGradCalcFp32(tmpLt, srcLt, dstLt, calcLen);
                // datacopyin uvqk
                srcOffset = (taskInfo.rowId * this->blockM + offset) * splitList[taskInfo.uvqkId] + \
                            taskInfo.uvqkSeqOffset;
                Wait_V_MTE2();
                DataCopy(inputLt, uvqkGt[taskInfo.uvqkId][srcOffset], uvqkParams);

                // x * d(sigmoid(x))/dx
                queIn.template EnQue(inputLt);
                inputLt = queIn.template DeQue<xType>();
                
                Cast(tmpLt, inputLt, RoundMode::CAST_NONE, calcLen);
                AscendC::PipeBarrier<PIPE_ALL>();

                Mul(srcLt, dstLt, tmpLt, calcLen);

                GetBiasGrad(srcLt, dstLt, lines, taskInfo.computeKSeqLen, taskInfo.kOffset);
                AscendC::PipeBarrier<PIPE_ALL>();

                // cast-> xType
                auto outLt = dstLt.template ReinterpretCast<xType>();
                Cast(outLt, srcLt, RoundMode::CAST_RINT, calcLen);
                queOut.template EnQue(outLt);
                outLt = queOut.template DeQue<xType>();
                DataCopy(SiluGradGt[siluOffset + offset * taskInfo.computeKSeqLen], outLt, calcLen);
            } else {
                LinearOutGradCalcFp32(inputLt, tmpLt, dstLt, calcLen);
                srcOffset = (taskInfo.rowId * this->blockM + offset) * splitList[taskInfo.uvqkId] +
                    taskInfo.uvqkSeqOffset;
                Wait_V_MTE2();
                DataCopy(inputLt, uvqkGt[taskInfo.uvqkId][srcOffset], uvqkParams);

                // x * d(sigmoid(x))/dx
                Wait_MTE2_V();

                Mul(dstLt, dstLt, inputLt, calcLen);
                Wait_V_MTE3();

                DataCopy(SiluGradGt[siluOffset + offset * taskInfo.computeKSeqLen], dstLt, calcLen);
                GetBiasGrad(dstLt, inputLt, lines, taskInfo.computeKSeqLen, taskInfo.kOffset);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            offset += lines;
            totalLen -= calcLen;
            
            queIn.template FreeTensor(inputLt);
            queOut.template FreeTensor(dstLt);
            tmpBuff.template FreeTensor(tmpLt);
        }
    }

    __aicore__ inline void GetBiasGrad(const LocalTensor<float>& srcLt,
                                       const LocalTensor<float>& dstLt,
                                       uint32_t row,
                                       uint32_t col,
                                       uint32_t kOffset)
    {
        if (!enableBias) {
            return;
        }
        constexpr bool isReUse = false; // 不允许对srcLt进行复用
        constexpr bool srcInnerPad = false; // k轴切分时已对齐
        uint32_t srcShape[2] = {row, col}; // 对第一维进行reduce RA
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, isReUse>(dstLt, srcLt, srcShape, srcInnerPad);
        Wait_V_MTE3();
        AscendC::SetAtomicAdd<float>();
        DataCopy(biasGradGt[kOffset], dstLt, col);
        AscendC::SetAtomicNone();
    }

    __aicore__ inline void LinearOutGradCalcFp32(const LocalTensor<float>& srcLt,
                                                 const LocalTensor<float>& tmpLt,
                                                 const LocalTensor<float>& dstLt,
                                                 uint32_t calcLen)
    {
        // silu_grad(x) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        // = sigmoid(x) * (1 + x*(1-sigmoid(x)))
        
        // Step 1: 计算 sigmoid(x) 并存储到 tmpLt
        Sigmoid<float>(tmpLt, srcLt, calcLen);
        
        // Step 2: 计算 1 - sigmoid(x) 并存储到 dstLt
        Duplicate<float>(dstLt, 1.0, calcLen);
        Sub<float>(dstLt, dstLt, tmpLt, calcLen);
        
        // Step 3: 计算 x*(1-sigmoid(x)) 并存储到 dstLt
        Mul<float>(dstLt, srcLt, dstLt, calcLen);
        
        // Step 4: 计算 1 + x*(1-sigmoid(x)) 并存储到 dstLt
        Adds<float>(dstLt, dstLt, 1.0, calcLen);
        
        // Step 5: 计算 sigmoid(x) * (1 + x*(1-sigmoid(x))) 并存储到 dstLt
        Mul<float>(dstLt, tmpLt, dstLt, calcLen);
    }

    __aicore__ inline void FillTaskInfo(TaskInfo &taskInfo, uint32_t rowId)
    {
        // 按行分核
        // 按照blockK进行计算，uvqk核小dim可能存在较小尾块，记录computeSeqLenOffset, blockM, blockK
        // 计算 uvqk索引下标，uvqk拷贝起始地址，linear_output拷贝起始地址
        taskInfo.rowId = rowId;
        taskInfo.computeMSeqLen = (taskInfo.rowId + 1) * blockM < this->seqLen ? blockM :
                             this->seqLen - taskInfo.rowId * blockM;
        taskInfo.computeKSeqLen = ((taskInfo.uvqkSeqOffset + blockK) < splitList[taskInfo.uvqkId]) ? blockK :
                                  splitList[taskInfo.uvqkId] - taskInfo.uvqkSeqOffset;
    }

    __aicore__ inline void UpdateTaskInfo(uint32_t taskId)
    {
        int32_t nextTaskId = (taskId + 1) % COMPUTE_PIPE_NUM;
        int32_t currTaskId = taskId % COMPUTE_PIPE_NUM;
        int32_t lastTaskOffset = computeTaskInfo[currTaskId].computeKSeqLen + computeTaskInfo[currTaskId].uvqkSeqOffset;
        computeTaskInfo[nextTaskId] = computeTaskInfo[currTaskId];
        if (lastTaskOffset >= splitList[computeTaskInfo[currTaskId].uvqkId]) {
            if (computeTaskInfo[currTaskId].uvqkId < UVQK_INDEX_K) { // 换uvqk
                computeTaskInfo[nextTaskId].rowId = computeTaskInfo[currTaskId].rowId;
                computeTaskInfo[nextTaskId].uvqkId = computeTaskInfo[currTaskId].uvqkId + 1;
                computeTaskInfo[nextTaskId].kOffset = computeTaskInfo[currTaskId].computeKSeqLen +
                computeTaskInfo[currTaskId].kOffset;
            } else { // 换行 todo待测试
                computeTaskInfo[nextTaskId].rowId = computeTaskInfo[currTaskId].rowId + 1;
                computeTaskInfo[nextTaskId].uvqkId = UVQK_INDEX_U;
                computeTaskInfo[nextTaskId].kOffset = 0;
                computeTaskInfo[nextTaskId].computeMSeqLen = (computeTaskInfo[nextTaskId].rowId + 1) *
                    blockM < this->seqLen ? blockM :
                    this->seqLen - computeTaskInfo[nextTaskId].rowId * blockM;
            }
            computeTaskInfo[nextTaskId].uvqkSeqOffset = 0;
        } else {
            computeTaskInfo[nextTaskId].uvqkSeqOffset = computeTaskInfo[currTaskId].uvqkSeqOffset +
                computeTaskInfo[currTaskId].computeKSeqLen;
            computeTaskInfo[nextTaskId].kOffset = computeTaskInfo[currTaskId].kOffset +
                computeTaskInfo[currTaskId].computeKSeqLen;
        }
        int32_t uvqkLen = splitList[computeTaskInfo[nextTaskId].uvqkId];
        computeTaskInfo[nextTaskId].computeKSeqLen = ((computeTaskInfo[nextTaskId].uvqkSeqOffset + blockK) < uvqkLen) ?
            blockK : uvqkLen - computeTaskInfo[nextTaskId].uvqkSeqOffset;
    }

    __aicore__ inline void DolwMatmul(TaskInfo &taskInfo, uint32_t taskId)
    {
        // 一次性拷完weight
        // [sq, H] [H, d]
        int32_t siluOffset = blockM * blockK * taskId;
        // 右矩阵偏移
        int32_t startOffsetB = taskInfo.kOffset * dim;
        int32_t outOffset = taskInfo.rowId * blockM * dim;
        lwMatmul.SetTensorA(SiluGradGt[siluOffset]);
        lwMatmul.SetTensorB(weightGt[startOffsetB], false);
        lwMatmul.SetTail(taskInfo.computeMSeqLen, dim, taskInfo.computeKSeqLen);

        lwMatmul.template IterateAll<false>(xGradGt[outOffset], 1, false, true);
    }

    __aicore__ inline void DolxMatmul(TaskInfo &taskInfo, uint32_t taskId)
    {
        // 一次性拷完weight
        // [sq, H] [sq, d]
        int32_t siluOffset = blockM * blockK * taskId;
        // 右矩阵偏移
        int32_t startOffsetB = taskInfo.rowId * blockM * dim;
        int32_t outOffset = taskInfo.kOffset * dim;
        lxMatmul.SetTensorA(SiluGradGt[siluOffset], true);
        lxMatmul.SetTensorB(xGt[startOffsetB]);
        lxMatmul.SetTail(taskInfo.computeKSeqLen, dim, taskInfo.computeMSeqLen);
        lxMatmul.template IterateAll<false>(weightGradGt[outOffset], 1, false, true);
    }

    __aicore__ inline void WaitLxMatmul()
    {
        lxMatmul.WaitIterateAll();
        lxMatmul.End();
    }
    
    __aicore__ inline void WaitLwMatmul()
    {
        lwMatmul.WaitIterateAll();
        lwMatmul.End();
    }
    
    __aicore__ inline void ComputeAllBlock()
    {
        uint32_t taskId = 0;
        uint32_t currentTaskId = taskId;
        uint32_t prevTaskId = taskId - 1;
        FillTaskInfo(computeTaskInfo[taskId], this->sRow);
        while (computeTaskInfo[currentTaskId].rowId < this->eRow) {
                if (taskId > 0) {
                    DolxMatmul(computeTaskInfo[prevTaskId], prevTaskId);
                    DolwMatmul(computeTaskInfo[prevTaskId], prevTaskId);
                }

                SiluGrad(computeTaskInfo[currentTaskId], currentTaskId);
                if (taskId > 0) {
                    WaitLxMatmul();
                    WaitLwMatmul();
                }
                UpdateTaskInfo(taskId);
                taskId++;
                currentTaskId = taskId % COMPUTE_PIPE_NUM;
                prevTaskId = (taskId - 1) % COMPUTE_PIPE_NUM;
        }
           
        if (taskId >= 1) {
            pipe_barrier(PIPE_ALL);
            DolxMatmul(computeTaskInfo[prevTaskId], prevTaskId);
            DolwMatmul(computeTaskInfo[prevTaskId], prevTaskId);
            WaitLxMatmul();
            WaitLwMatmul();
        }
    }

    // GM_ADDR
    GM_ADDR x;
    GM_ADDR weight;
    GM_ADDR bias;
    GM_ADDR user_grad;
    GM_ADDR value_grad;
    GM_ADDR query_grad;
    GM_ADDR key_grad;
    GM_ADDR linear_output;

    GM_ADDR weight_grad;
    GM_ADDR bias_grad;
    GM_ADDR x_grad;
    GM_ADDR workspace;
    GM_ADDR tiling;
    // Ub
    int32_t transUbBlockElem;
    int32_t coreNum;
    int32_t blockM;
    int32_t blockK;
    int32_t seqLen;
    int32_t hiddenSize;
    int32_t dim;
    int32_t sRow;
    int32_t eRow;
    int32_t uDim;
    int32_t vDim;
    int32_t qDim;
    int32_t kDim;
    int32_t totalDim;
    int32_t splitList[UVQK_NUM];
    TaskInfo computeTaskInfo[COMPUTE_PIPE_NUM];

    // Tpipe
    TPipe *pipe;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn;
    TQue<TPosition::VECCALC, USE_QUEUE_NUM> tmpBuff;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queOut;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> transOut;

    // Gt
    GlobalTensor<xType> xGt;
    GlobalTensor<wType> weightGt;
    GlobalTensor<float> biasGt;
    GlobalTensor<xType> uGradGt;
    GlobalTensor<xType> vGradGt;
    GlobalTensor<xType> qGradGt;
    GlobalTensor<xType> kGradGt;
    GlobalTensor<xType> linearOutGt;
   
    GlobalTensor<float> xGradGt;
    GlobalTensor<float> weightGradGt;
    GlobalTensor<float> biasGradGt;

    GlobalTensor<xType> SiluGradGt;

    matmul::Matmul<
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, xType, true>,
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, xType, false>,
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, CFG_NORM,
    matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> lxMatmul;

    matmul::Matmul<
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, xType, false>,
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, xType, false>,
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
    matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, CFG_NORM,
    matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>> lwMatmul;
};
}  // namespace InLinearSiluBackward
#endif
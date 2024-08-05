/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef ATTENTION_FUSION_KERNEL_H
#define ATTENTION_FUSION_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "q_k_bmm_compute.h"
#include "k_v_bmm_compute.h"
#include "normalize_compute.h"

using namespace AscendC;

namespace Attention_Kernel {

struct AttentionFusionArgs {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR attnMask;
    GM_ADDR attenScore;
    GM_ADDR softmaxOut;
    
    uint8_t normalizeAttr;
    int32_t queryDim1;
    int32_t queryDim2;
    int32_t keyDim1;
    int32_t valueDim2;
    int32_t batchNum;
    int32_t normalizeLoop;
    int32_t normalizeRow;
    int32_t normalizeColumn;
    int32_t maskIsOn;
    float normalizeSqrt;
    uint64_t maxSharedTmpBuf;

    const TCubeTiling* qkMatmulTiling;
    const TCubeTiling* kvMatmulTiling;
    const SoftMaxTiling* softMaxTilingData;

    const ConfusionTransposeTiling* confusionTransposeTilingData;
    const ConfusionTransposeTiling* confusionTransposeTilingData1;
    const ConfusionTransposeTiling* confusionTransposeTilingData2;
    const ConfusionTransposeTiling* confusionTransposeTilingData3;
};

struct AttentionFusionPipe {
    TPipe* pipe;
};

template<typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    if (b == 0) {
        return 0;
    }
    return (a + b -1) / b;
}

template<typename qType, typename kType, typename vType>
class AttentionFusionKernel {
public:
    __aicore__ inline AttentionFusionKernel() {};

    __aicore__ inline void Compute(AttentionFusionArgs args)
    {
        // Args
        this->args = args;

        // Matmul Register
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), qKBmmCompute.mm, args.qkMatmulTiling, kvBmmCompute.mm,
            args.kvMatmulTiling);

        // batch offset
        GetBatchOffsetAndLen(args.batchNum, this->batchOffset, this->batchLen);

        // QKBmm Initialize
        QKBmmArgs qKBmmArgs {
            args.query, args.key, args.softmaxOut,
            args.queryDim1, args.keyDim1, args.queryDim2,
            batchOffset, batchLen
        };
        QKBmmPipeArgs qKBmmPipeArgs {&pipe};
        qKBmmCompute.Init(qKBmmArgs, qKBmmPipeArgs);

        NormalizeArgs normalArgs {
            &pipe, args.normalizeAttr, args.queryDim1, args.keyDim1, args.normalizeLoop, args.normalizeRow,
            args.normalizeColumn, args.maskIsOn, args.normalizeSqrt, args.maxSharedTmpBuf, args.softMaxTilingData,
            args.confusionTransposeTilingData, args.confusionTransposeTilingData1,
            args.confusionTransposeTilingData2, args.confusionTransposeTilingData3
        };
        normalizeCompute.Init(normalArgs);

        // KVBmm Initialize
        KVBmmArgs kvBmmArgs {
            args.softmaxOut, args.value, args.attenScore,
            args.queryDim1, args.valueDim2, args.keyDim1,
            batchOffset
        };
        KVBmmPipeArgs kvBmmPipeArgs {&pipe};
        kvBmmCompute.Init(kvBmmArgs, kvBmmPipeArgs);

        // Start compute
        Process();
    }

    __aicore__ inline void Process()
    {
        QKBmmComputePart();
        NormalizeMatmulFusion();
    }

    __aicore__ inline void QKBmmComputePart()
    {
        qKBmmCompute.Process();
    }

    __aicore__ inline void NormalizeMatmulFusion()
    {
        GlobalTensor<kType> softmaxOutGbTensorThisCore;
        softmaxOutGbTensorThisCore.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.softmaxOut),
                                                    batchLen * args.queryDim1 * args.keyDim1);
        GlobalTensor<kType> softmaxGbMaskThisCore;
        softmaxGbMaskThisCore.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.attnMask),
                                                    batchLen * args.queryDim1 * args.keyDim1);
        for (int i = 0; i < batchLen + 1; i++) {
            if (i != batchLen) {
                GlobalTensor<kType> softmaxOutGbTensor =
                    softmaxOutGbTensorThisCore[(batchOffset + i) * args.queryDim1 * args.keyDim1];
                GlobalTensor<kType> softmaxGbMaskTensor =
                    softmaxGbMaskThisCore[(batchOffset + i) * args.queryDim1 * args.keyDim1];
                /* normallize */
                normalizeCompute.Process(softmaxOutGbTensor, softmaxGbMaskTensor);
            }
            
            if (i != 0) {
                /* matmul */
                kvBmmCompute.ComputeOneBatch(i - 1);
            }
        }
    }

    __aicore__ inline void GetBatchOffsetAndLen(int batchNum, int& batchOffset, int& batchLen)
    {
        // batch offset
        int blockLenPerCore = CeilDiv(batchNum, (GetBlockNum() * 2));
        batchOffset = blockLenPerCore * GetBlockIdx();
        batchLen = blockLenPerCore;
        if (batchOffset + batchLen > batchNum) {
            batchLen = batchNum - batchOffset;
        }
    }

private:
    TPipe pipe;
    AttentionFusionArgs args;
    QKBmmCompute<qType, kType> qKBmmCompute;
    KVBmmCompute<qType, vType> kvBmmCompute;
    NormalizeCompute<qType> normalizeCompute;

    int batchOffset;
    int batchLen;
};
}
#endif
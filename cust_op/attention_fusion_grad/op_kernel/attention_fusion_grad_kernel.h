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

#ifndef ATTENTION_FUSION_GRAD_KERNEL_H
#define ATTENTION_FUSION_GRAD_KERNEL_H

#include <cstdint>

#include "args.h"
#include "kernel_operator.h"
#include "normalize_grad.h"
#include "q_k_mm_grad.h"
#include "small_k_matmul.h"
#include "utils.h"
#include "v_s_mm_grad.h"

using namespace AscendC;

namespace AscendFusionGrad {

constexpr int SPECAIL_CASE[] = {1, 1000, 80};

struct AttentionFusionGradPipe {
    TPipe* pipe;
};

template <typename tType>
class AttentionFusionGradKernel {
public:
    __aicore__ inline AttentionFusionGradKernel(){};

    __aicore__ inline void Compute(AttentionFusionGradArgs args)
    {
        this->args = args;
        // Matmul Register
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), vSmm.mmGradV, args.tilingArgs.gardVMatmulTiling, vSmm.mmGradS,
                          args.tilingArgs.gardSMatmulTiling, qKmm.mmGradQ, args.tilingArgs.gardQMatmulTiling,
                          qKmm.mmGradK, args.tilingArgs.gardKMatmulTiling);
        vSmm.Init(args);
        qKmm.Init(args);
        Process();
    }

    __aicore__ inline void Process()
    {
        TQue<QuePosition::VECIN, 1> vecInQueue;
        TQue<QuePosition::VECIN, 1> vecInGradQueue;
        TQue<QuePosition::VECOUT, 1> vecOutQueue;
        TBuf<TPosition::VECCALC> tmpBuff;

        int64_t oneRowSize =
            args.shapeTilingArgs.numRowOfNormalizeOne * args.shapeTilingArgs.paddingKeyDim1 * sizeof(tType);

        pipe.InitBuffer(vecInQueue, 1, oneRowSize);
        pipe.InitBuffer(vecInGradQueue, 1, oneRowSize);
        pipe.InitBuffer(vecOutQueue, 1, oneRowSize);
        pipe.InitBuffer(tmpBuff, oneRowSize);

        NormGradPipeArgs normGradPipe{&pipe, &vecInQueue, &vecInGradQueue, &vecOutQueue, &tmpBuff};

        NormalizeGrad<tType> normalCompute;
        normalCompute.Init(args, normGradPipe);

        SmallKMatmul<tType> smallKMatmul;
        smallKMatmul.Init(args, normGradPipe);

        // 80对齐，且UB能放下
        bool specialCase = args.shapeArgs.queryDim1 == SPECAIL_CASE[0] && args.shapeArgs.keyDim1 == SPECAIL_CASE[1] &&
                           args.shapeArgs.valueDim2 == SPECAIL_CASE[2];
        for (int thisBatch = 0; thisBatch < args.shapeTilingArgs.batchLen; thisBatch++) {
            vSmm.ProcessDS(thisBatch);
            if (specialCase == true) {
                smallKMatmul.ProcessDV(thisBatch);
            }
        }

        for (int thisBatch = 0; thisBatch < args.shapeTilingArgs.batchLen; thisBatch++) {
            if (specialCase == false) {
                vSmm.ProcessDV(thisBatch);
            }

            normalCompute.ProcessOneBatch(thisBatch);
        }
        for (int thisBatch = 0; thisBatch < args.shapeTilingArgs.batchLen; thisBatch++) {
            qKmm.ProcessDQ(thisBatch);
            qKmm.ProcessDK(thisBatch);
        }
    }
    AttentionFusionGradArgs args;
    VSMmGrad<tType> vSmm;
    QKMmGrad<tType> qKmm;
    TPipe pipe;
};
}  // namespace AscendFusionGrad

#endif
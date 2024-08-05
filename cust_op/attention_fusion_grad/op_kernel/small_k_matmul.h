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

#ifndef SMALL_K_MATMUL_H
#define SMALL_K_MATMUL_H
#include <cstdint>

#include "args.h"
#include "kernel_operator.h"
#include "utils.h"
using namespace AscendC;


namespace AscendFusionGrad {

constexpr int DIM_NUMS = 2;
template <typename tType>
class SmallKMatmul {
public:
    __aicore__ inline SmallKMatmul() {}

    __aicore__ inline void Init(AttentionFusionGradArgs args, NormGradPipeArgs pipeArgs)
    {
        this->args = args;

        doutShapeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.valueDim2;
        gradValueShapeOfOneBatch = args.shapeArgs.valueDim1 * args.shapeArgs.valueDim2;
        softmaxShpeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.keyDim1;

        softmaxOut.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.softmaxOut),
                                   args.shapeArgs.batchNum * softmaxShpeOfOneBatch);
        dout.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.dout),
                             args.shapeArgs.batchNum * doutShapeOfOneBatch);
        gradValue.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.outputArgs.gradValue),
                                  args.shapeArgs.batchNum * gradValueShapeOfOneBatch);

        vecInQueue = pipeArgs.vecInQueue;
        vecInGradQueue = pipeArgs.vecInGradQueue;
        vecOutQueue = pipeArgs.vecOutQueue;
        tmpBuff = pipeArgs.tmpBuff;
    }

    __aicore__ inline void ProcessDV(uint32_t batchI)
    {
        batchI += args.shapeTilingArgs.batchOffset;
        GlobalTensor<tType> thisBatchSoftmaxGb = softmaxOut[batchI * softmaxShpeOfOneBatch];
        GlobalTensor<tType> thisBatchDoutGb = dout[batchI * doutShapeOfOneBatch];
        GlobalTensor<tType> gradValueGb = gradValue[batchI * gradValueShapeOfOneBatch];

        int valueDim2 = args.shapeArgs.valueDim2;
        int total = softmaxShpeOfOneBatch;
        int remain = total;

        int numOfOneMul = args.shapeTilingArgs.numRowOfNormalizeOne * args.shapeArgs.keyDim1 / valueDim2;

        LocalTensor<tType> inLocalTensor = vecInQueue->AllocTensor<tType>();
        DataCopy(inLocalTensor, thisBatchDoutGb, valueDim2);
        vecInQueue->EnQue(inLocalTensor);
        LocalTensor<tType> inLocalTensorCompute = vecInQueue->DeQue<tType>();
        LocalTensor<tType> inGradLocalTensor = vecInGradQueue->AllocTensor<tType>();

        const uint32_t dstShape_[]{(uint32_t)numOfOneMul, (uint32_t)valueDim2};
        const uint32_t srcShape_[]{1, (uint32_t)valueDim2};
        
        BroadCast<float, DIM_NUMS, 0>(inGradLocalTensor, inLocalTensorCompute, dstShape_, srcShape_);

        DataCopy(inLocalTensorCompute, inGradLocalTensor, numOfOneMul * valueDim2);

        while (remain > 0) {
            // caculate basic
            int thisLen = numOfOneMul;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int offset = total - remain;

            LocalTensor<tType> outLocalTensor = vecOutQueue->AllocTensor<tType>();
            for (int j = 0; j < thisLen; j++) {
                float v = *(thisBatchSoftmaxGb.GetPhyAddr() + offset + j);
                Duplicate<float>(inGradLocalTensor[j * valueDim2], v, valueDim2);
            }

            Mul(outLocalTensor, inGradLocalTensor, inLocalTensorCompute, thisLen * valueDim2);
            vecOutQueue->EnQue(outLocalTensor);
            outLocalTensor = vecOutQueue->DeQue<tType>();
            DataCopy(gradValueGb[offset * valueDim2], outLocalTensor, thisLen * valueDim2);
            vecOutQueue->FreeTensor(outLocalTensor);
            remain = remain - thisLen;
        }
        vecInQueue->FreeTensor(inLocalTensorCompute);
        vecInGradQueue->FreeTensor(inGradLocalTensor);
    }

private:
    int doutShapeOfOneBatch;
    int gradValueShapeOfOneBatch;
    int softmaxShpeOfOneBatch;

    AttentionFusionGradArgs args;

    TQue<QuePosition::VECIN, 1>* vecInQueue;
    TQue<QuePosition::VECIN, 1>* vecInGradQueue;
    TQue<QuePosition::VECOUT, 1>* vecOutQueue;
    TBuf<TPosition::VECCALC>* tmpBuff;

    GlobalTensor<tType> gradValue;
    GlobalTensor<tType> dout;
    GlobalTensor<tType> softmaxOut;
};
}

#endif
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

#ifndef VS_MM_GRAD_H
#define VS_MM_GRAD_H
#include <cstdint>

#include "args.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "utils.h"
using namespace AscendC;

namespace AscendFusionGrad {
template <typename tType>
class VSMmGrad {
public:
    __aicore__ inline VSMmGrad() {}

    __aicore__ inline void Init(AttentionFusionGradArgs& args)
    {
        this->args = args;
        softmaxOutShapeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.keyDim1;
        valueShapeOfOneBatch = args.shapeArgs.valueDim1 * args.shapeArgs.valueDim2;
        doutShapeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.valueDim2;
        gradSShapeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.keyDim1;
        gradValueShapeOfOneBatch = args.shapeArgs.valueDim1 * args.shapeArgs.valueDim2;

        softmaxOut.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.softmaxOut),
                                   args.shapeArgs.batchNum * softmaxOutShapeOfOneBatch);

        value.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.value),
                              args.shapeArgs.batchNum * valueShapeOfOneBatch);

        dout.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.dout),
                             args.shapeArgs.batchNum * doutShapeOfOneBatch);

        gradS.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.workspace),
                              args.shapeArgs.batchNum * gradSShapeOfOneBatch);

        gradValue.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.outputArgs.gradValue),
                                  args.shapeArgs.batchNum * gradValueShapeOfOneBatch);
    }

    __aicore__ inline void ProcessDV(int batchI)
    {
        if (batchI != 0) {
            mmGradV.WaitIterateAll();
            mmGradV.End();
        }
        batchI += args.shapeTilingArgs.batchOffset;
        mmGradV.SetTensorA(softmaxOut[batchI * softmaxOutShapeOfOneBatch], true);
        mmGradV.SetTensorB(dout[batchI * doutShapeOfOneBatch]);

        mmGradV.template IterateAll<false>(gradValue[batchI * gradValueShapeOfOneBatch], 0, false, true);
    }

    __aicore__ inline void ProcessDS(int batchI)
    {
        batchI += args.shapeTilingArgs.batchOffset;
        mmGradS.SetTensorA(dout[batchI * doutShapeOfOneBatch]);
        mmGradS.SetTensorB(value[batchI * valueShapeOfOneBatch], true);

        mmGradS.template IterateAll<false>(gradS[batchI * gradSShapeOfOneBatch], 0, false, true);
        mmGradS.WaitIterateAll();
    }

    matmul::Matmul<matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, true>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>>
        mmGradV;

    matmul::Matmul<matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, true>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>>
        mmGradS;

private:
    int softmaxOutShapeOfOneBatch;
    int valueShapeOfOneBatch;
    int doutShapeOfOneBatch;
    int gradSShapeOfOneBatch;
    int gradValueShapeOfOneBatch;
    AttentionFusionGradArgs args;
    GlobalTensor<tType> softmaxOut;
    GlobalTensor<tType> value;
    GlobalTensor<tType> dout;
    GlobalTensor<tType> gradS;
    GlobalTensor<tType> gradValue;
};
}
#endif
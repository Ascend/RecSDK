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

#ifndef QK_MM_GRAD_H
#define QK_MM_GRAD_H
#include "args.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "utils.h"
using namespace AscendC;

namespace AscendFusionGrad {

template <typename tType>
class QKMmGrad {
public:
    __aicore__ inline QKMmGrad() {}

    __aicore__ inline void Init(AttentionFusionGradArgs args)
    {
        this->args = args;
        queryShapeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.queryDim2;
        keyShapeOfOneBatch = args.shapeArgs.keyDim1 * args.shapeArgs.keyDim2;
        gradSoftmaxShapeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.keyDim1;
        gradQueryShapeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.queryDim2;
        gradKeyShapeOfOneBatch = args.shapeArgs.keyDim1 * args.shapeArgs.keyDim2;

        query.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.query),
                              args.shapeArgs.batchNum * queryShapeOfOneBatch);

        key.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.key),
                            args.shapeArgs.batchNum * keyShapeOfOneBatch);

        gradSoftmax.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.workspace),
                                    args.shapeArgs.batchNum * gradSoftmaxShapeOfOneBatch);

        gradQuery.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.outputArgs.gradQuery),
                                  args.shapeArgs.batchNum * gradQueryShapeOfOneBatch);

        gradKey.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.outputArgs.gradKey),
                                args.shapeArgs.batchNum * gradKeyShapeOfOneBatch);
    }

    __aicore__ inline void ProcessDQ(int batchI)
    {
        if (batchI != 0) {
            mmGradQ.WaitIterateAll();
            mmGradQ.End();
        }
        batchI += args.shapeTilingArgs.batchOffset;
        mmGradQ.SetTensorA(gradSoftmax[batchI * gradSoftmaxShapeOfOneBatch]);
        mmGradQ.SetTensorB(key[batchI * keyShapeOfOneBatch]);

        mmGradQ.template IterateAll<false>(gradQuery[batchI * gradQueryShapeOfOneBatch], 0, false, true);
    }

    __aicore__ inline void ProcessDK(int batchI)
    {
        if (batchI != 0) {
            mmGradK.WaitIterateAll();
            mmGradK.End();
        }
        batchI += args.shapeTilingArgs.batchOffset;
        mmGradK.SetTensorA(gradSoftmax[batchI * gradSoftmaxShapeOfOneBatch], true);
        mmGradK.SetTensorB(query[batchI * queryShapeOfOneBatch]);

        mmGradK.template IterateAll<false>(gradKey[batchI * gradKeyShapeOfOneBatch], 0, false, true);
    }

    matmul::Matmul<matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>>
        mmGradQ;

    matmul::Matmul<matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, true>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
                   matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>>
        mmGradK;

private:
    int queryShapeOfOneBatch;
    int keyShapeOfOneBatch;
    int gradSoftmaxShapeOfOneBatch;
    int gradQueryShapeOfOneBatch;
    int gradKeyShapeOfOneBatch;
    AttentionFusionGradArgs args;

    GlobalTensor<tType> query;
    GlobalTensor<tType> key;
    GlobalTensor<tType> gradSoftmax;
    GlobalTensor<tType> gradQuery;
    GlobalTensor<tType> gradKey;
};
}  // namespace AscendFusionGrad

#endif
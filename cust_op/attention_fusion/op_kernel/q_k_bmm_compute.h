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

#ifndef QK_BMMM_COMPUTE__H
#define QK_BMMM_COMPUTE__H
#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace Attention_Kernel {
struct QKBmmArgs {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR out;
    
    int dimM;
    int dimN;
    int dimK;

    int batchOffset;
    int batchLen;
};

struct QKBmmPipeArgs {
    TPipe* pipe;
};

template<typename qType, typename kType>
class QKBmmCompute {
public:
    __aicore__ inline QKBmmCompute() {}

    __aicore__ inline void Init(QKBmmArgs qKBmmArgs, QKBmmPipeArgs pipeArgs)
    {
        this->qKBmmArgs = qKBmmArgs;

        // kernel batch offset
        qGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(qKBmmArgs.query),
                                qKBmmArgs.batchLen * qKBmmArgs.dimM * qKBmmArgs.dimK);
        qGlobal = qGlobal[qKBmmArgs.batchOffset * qKBmmArgs.dimM * qKBmmArgs.dimK];

        kGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.key),
                                qKBmmArgs.batchLen * qKBmmArgs.dimN * qKBmmArgs.dimK);
        kGlobal = kGlobal[qKBmmArgs.batchOffset * qKBmmArgs.dimN * qKBmmArgs.dimK];

        outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.out),
                                qKBmmArgs.batchLen * qKBmmArgs.dimM * qKBmmArgs.dimN);
        outGlobal = outGlobal[qKBmmArgs.batchOffset * qKBmmArgs.dimM * qKBmmArgs.dimN];
    }

    __aicore__ inline void Process()
    {
        for (int thisBatch = 0 ; thisBatch < qKBmmArgs.batchLen; thisBatch++) {
            mm.SetTensorA(qGlobal[thisBatch * qKBmmArgs.dimM * qKBmmArgs.dimK]);
            mm.SetTensorB(kGlobal[thisBatch * qKBmmArgs.dimN * qKBmmArgs.dimK], true);

            mm.IterateAll(outGlobal[thisBatch * qKBmmArgs.dimM * qKBmmArgs.dimN], 0, false);
        }
        mm.End();
    }
    
    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType <matmul::TPosition::GM, CubeFormat::ND, kType, true>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType>
        > mm;
private:
    QKBmmArgs qKBmmArgs;
    GlobalTensor<qType> qGlobal;
    GlobalTensor<kType> kGlobal;
    GlobalTensor<kType> outGlobal;
};
}
#endif
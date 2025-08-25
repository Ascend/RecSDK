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

#ifndef KV_BMMM_COMPUTE__H
#define KV_BMMM_COMPUTE__H

#include "kernel_operator.h"
using namespace AscendC;

namespace Attention_Kernel {
struct KVBmmArgs {
    GM_ADDR softmaxOut;
    GM_ADDR value;
    GM_ADDR out;
    
    int dimM;
    int dimN;
    int dimK;

    int batchOffset;
};

struct KVBmmPipeArgs {
    TPipe* pipe;
};

template<typename sType, typename vType>
class KVBmmCompute {
public:
    __aicore__ inline KVBmmCompute() {}

    __aicore__ inline void Init(KVBmmArgs kvBmmArgs, KVBmmPipeArgs pipeArgs)
    {
        this->kvBmmArgs = kvBmmArgs;
        this->pipeArgs = pipeArgs;

        // kernel batch offset
        sGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ sType*>(kvBmmArgs.softmaxOut), kvBmmArgs.dimM * kvBmmArgs.dimK);
        sGlobal = sGlobal[kvBmmArgs.batchOffset * kvBmmArgs.dimM * kvBmmArgs.dimK];
        vGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(kvBmmArgs.value),  kvBmmArgs.dimN * kvBmmArgs.dimK);
        vGlobal = vGlobal[kvBmmArgs.batchOffset * kvBmmArgs.dimN * kvBmmArgs.dimK];
        outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(kvBmmArgs.out), kvBmmArgs.dimM * kvBmmArgs.dimN);
        outGlobal = outGlobal[kvBmmArgs.batchOffset * kvBmmArgs.dimM * kvBmmArgs.dimN];
    }

    __aicore__ inline void ComputeOneBatch(int batchI)
    {
        if (batchI != 0) {
            mm.WaitIterateAll();
            mm.End();
        }

        mm.SetTensorA(sGlobal[batchI * kvBmmArgs.dimM * kvBmmArgs.dimK]);
        mm.SetTensorB(vGlobal[batchI * kvBmmArgs.dimN * kvBmmArgs.dimK]);

        mm.template IterateAll<false>(outGlobal[batchI * kvBmmArgs.dimM * kvBmmArgs.dimN], 0, false, true);
    }

    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, sType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, vType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, vType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, vType>
        > mm;
private:
    KVBmmArgs kvBmmArgs;
    KVBmmPipeArgs pipeArgs;
    GlobalTensor<sType> sGlobal;
    GlobalTensor<vType> vGlobal;
    GlobalTensor<vType> outGlobal;
};
}
#endif
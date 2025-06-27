/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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
#ifndef HSTU_DENSE_BACKWARD_KERNEL_MATMUL_H
#define HSTU_DENSE_BACKWARD_KERNEL_MATMUL_H

#include "hstu_dense_backward_kernel.h"

namespace HstuDenseBackwardFuxi {

template <typename qType> class HstuDenseBackwardKernelMatmulFuxi: public HstuDenseBackwardKernelFuxi<qType> {
public:
    __aicore__ inline HstuDenseBackwardKernelMatmulFuxi() {}

    __aicore__ inline void DoQKMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(this->q[left]);
        qkMatmul.SetTensorB(this->k[right], true);

        qkMatmul.template IterateAll<false>(this->qkTemp[out], 0, false, true);
    }

    __aicore__ inline void DoGVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(this->grad[left]);
        qkMatmul.SetTensorB(this->v[right], true);

        qkMatmul.template IterateAll<false>(this->gvTemp[out], 0, false, true);
    }

    __aicore__ inline void DoGpVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(this->gradPosition[left]);
        qkMatmul.SetTensorB(this->v[right], true);

        qkMatmul.template IterateAll<false>(this->tempGposVT[out], 0, false, true);
    }

    __aicore__ inline void DoGtVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(this->gradTimestamp[left]);
        qkMatmul.SetTensorB(this->v[right], true);

        qkMatmul.template IterateAll<false>(this->tempGtsVT[out], 0, false, true);
    }

    __aicore__ inline void DoQGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        qGradMatmul.SetTensorA(this->attnBiasGrad[left]);
        qGradMatmul.SetTensorB(this->k[right]);
        if (isNew) {
            qGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 0, false, true);
        } else {
            qGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoKGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        kGradMatmul.SetTensorA(this->attnBiasGrad[left], true);
        kGradMatmul.SetTensorB(this->q[right]);
        if (isNew) {
            kGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 0, false, true);
        } else {
            kGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoVGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        vGradMatmul.SetTensorA(this->scoreTemp[left], true);
        vGradMatmul.SetTensorB(this->grad[right]);
        if (isNew) {
            vGradMatmul.template IterateAll<false>(this->vGradAccumTemp[out], 0, false, true);
        } else {
            vGradMatmul.template IterateAll<false>(this->vGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoBtGtMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        vGradMatmul.SetTensorA(this->tempBtsM[left], true);
        vGradMatmul.SetTensorB(this->gradTimestamp[right]);
        if (isNew) {
            vGradMatmul.template IterateAll<false>(this->tempBtsGtsAccum[out], 0, false, true);
        } else {
            vGradMatmul.template IterateAll<false>(this->tempBtsGtsAccum[out], 1, false, true);
        }
    }

    __aicore__ inline void DoBpGpMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        vGradMatmul.SetTensorA(this->tempBposM[left], true);
        vGradMatmul.SetTensorB(this->gradPosition[right]);
        if (isNew) {
            vGradMatmul.template IterateAll<false>(this->tempBposGposAccum[out], 0, false, true);
        } else {
            vGradMatmul.template IterateAll<false>(this->tempBposGposAccum[out], 1, false, true);
        }
    }

    // Matmul
    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
        matmul::MatmulCallBackFunc<nullptr, CopyQKA1<qType>, CopyQKB1<qType>>>
        qkMatmul;

    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
        matmul::MatmulCallBackFunc<nullptr, CopyQGradA1<qType>, CopyVGradB1<qType>>>
        qGradMatmul;

    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
        matmul::MatmulCallBackFunc<nullptr, CopyKGradA1<qType>, CopyVGradB1<qType>>>
        kGradMatmul;

    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
        matmul::MatmulCallBackFunc<nullptr, nullptr, CopyVGradB1<qType>>>
        vGradMatmul;

    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
        matmul::MatmulCallBackFunc<nullptr, nullptr, CopyVGradB1<qType>>>
        biasMaskMatmul;
};
}

#endif // HSTU_DENSE_BACKWARD_KERNEL_MATMUL_H
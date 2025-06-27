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
        qkMatmul.SetTensorA(q[left]);
        qkMatmul.SetTensorB(k[right], true);

        qkMatmul.template IterateAll<false>(qkTemp[out], 0, false, true);
    }

    __aicore__ inline void DoGVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(grad[left]);
        qkMatmul.SetTensorB(v[right], true);

        qkMatmul.template IterateAll<false>(gvTemp[out], 0, false, true);
    }

    __aicore__ inline void DoGpVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(gradPosition[left]);
        qkMatmul.SetTensorB(v[right], true);

        qkMatmul.template IterateAll<false>(tempGposVT[out], 0, false, true);
    }

    __aicore__ inline void DoGtVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(gradTimestamp[left]);
        qkMatmul.SetTensorB(v[right], true);

        qkMatmul.template IterateAll<false>(tempGtsVT[out], 0, false, true);
    }

    // __aicore__ inline void DoQGradMatmul(int64_t taskId)
    // {
    //     int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
    //     int64_t midAccumIdx = taskInfo[curTaskId].accumId % MID_USE_TIMES;
    //     int64_t outOffset = midAccumIdx * blockHeight * headDim;

    //     bool isNew = taskInfo[curTaskId].colId == 0;

    //     qGradMatmul.SetTail(taskInfo[curTaskId].rowLine, headDim, taskInfo[curTaskId].colLine);
    //     DoQGradMatmulImpl(taskInfo[curTaskId].kGradLeftOffset,
    //                       taskInfo[curTaskId].vGradRightOffset,
    //                       outOffset, isNew);
    // }

    __aicore__ inline void DoQGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        qGradMatmul.SetTensorA(attnBiasGrad[left]);
        qGradMatmul.SetTensorB(k[right]);
        if (isNew) {
            qGradMatmul.template IterateAll<false>(kGradAccumTemp[out], 0, false, true);
        } else {
            qGradMatmul.template IterateAll<false>(kGradAccumTemp[out], 1, false, true);
        }
    }

    // __aicore__ inline void DoKGradMatmul(int64_t taskId)
    // {
    //     int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
    //     int64_t midAccumIdx = taskInfo[curTaskId].accumId % MID_USE_TIMES;
    //     int64_t outOffset = midAccumIdx * blockHeight * headDim;

    //     bool isNew = false;
    //     if (IfMask(maskType, MaskType::MASK_TRIL)) {
    //         isNew = taskInfo[curTaskId].rowId == taskInfo[curTaskId].colId;
    //     } else {
    //         isNew = taskInfo[curTaskId].rowId == 0;
    //     }

    //     kGradMatmul.SetTail(taskInfo[curTaskId].colLine, headDim, taskInfo[curTaskId].rowLine);
    //     DoKGradMatmulImpl(taskInfo[curTaskId].kGradLeftOffset,
    //                       taskInfo[curTaskId].vGradRightOffset,
    //                       outOffset, isNew);
    // }

    __aicore__ inline void DoKGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        kGradMatmul.SetTensorA(attnBiasGrad[left], true);
        kGradMatmul.SetTensorB(q[right]);
        if (isNew) {
            kGradMatmul.template IterateAll<false>(kGradAccumTemp[out], 0, false, true);
        } else {
            kGradMatmul.template IterateAll<false>(kGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoVGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        vGradMatmul.SetTensorA(scoreTemp[left], true);
        vGradMatmul.SetTensorB(grad[right]);
        if (isNew) {
            vGradMatmul.template IterateAll<false>(vGradAccumTemp[out], 0, false, true);
        } else {
            vGradMatmul.template IterateAll<false>(vGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoBtGtMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        vGradMatmul.SetTensorA(tempBtsM[left], true);
        vGradMatmul.SetTensorB(gradTimestamp[right]);
        if (isNew) {
            vGradMatmul.template IterateAll<false>(tempBtsGtsAccum[out], 0, false, true);
        } else {
            vGradMatmul.template IterateAll<false>(tempBtsGtsAccum[out], 1, false, true);
        }
    }

    __aicore__ inline void DoBpGpMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        vGradMatmul.SetTensorA(tempBposM[left], true);
        vGradMatmul.SetTensorB(gradPosition[right]);
        if (isNew) {
            vGradMatmul.template IterateAll<false>(tempBposGposAccum[out], 0, false, true);
        } else {
            vGradMatmul.template IterateAll<false>(tempBposGposAccum[out], 1, false, true);
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
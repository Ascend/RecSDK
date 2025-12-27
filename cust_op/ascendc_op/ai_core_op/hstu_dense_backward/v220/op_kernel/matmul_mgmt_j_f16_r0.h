/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#ifndef GT_JAGGED_FP16_R0_KERNEL_H
#define GT_JAGGED_FP16_R0_KERNEL_H
#include <cstdint>
#include "hstu_dense_backward_kernel_common.h"
#include "matmul_j_f16_r0_const.h"

struct MatmulArgs {
    const int64_t scoreMidPipeId;
    const int64_t accumPipeId;
    const int64_t QOrGOffset;  // Q和G的位置是一样的，所以可以复用
    const int64_t KOrVOffset;  // K和V的位置是一样的，所以可以复用
    const int64_t ActureQLen;
    const int64_t ActureKLen;
};
namespace HstuDenseBackward {

template <typename qType, int64_t blockHeightQ, int64_t blockHeightK, int64_t headNum, int64_t headDim>
class MmMgmtFp16R0Jagged {
public:
    __aicore__ inline MmMgmtFp16R0Jagged() {}

    __aicore__ inline void Init(const AddrArgs* addrArgs, const BaseShapeArgs* baseShape)
    {
        this->addrArgs_ = addrArgs;
        this->baseShape_ = baseShape;
        GM_ADDR workspace = addrArgs_->workspace;
        GM_ADDR curAICWorkspace;

        const int64_t batchSize = baseShape_->batchSize;
        const int64_t maxSeqLen = baseShape_->maxSeqLen;

        const uint32_t aivNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;

        grad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs_->grad));
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs_->q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs_->k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs_->v));

        qGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs_->qGrad));
        kGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs_->kGrad));
        vGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs_->vGrad));

        const int64_t qkMatmulTempSpace = blockHeightQ * blockHeightK;
        const int64_t gvMatmulTempSpace = blockHeightQ * blockHeightK;
        const int64_t vGradAccumTempSpace = blockHeightK * headDim;
        const int64_t kGradAccumTempSpace = blockHeightK * headDim;
        const int64_t scoreTempSpace = blockHeightQ * blockHeightK;
        const int64_t maskTempSpace = blockHeightQ * blockHeightK;
        const int64_t biasOrSGradTempSpace = blockHeightQ * blockHeightK;
        const int64_t qGradAccumTempSpace = batchSize * headNum * maxSeqLen * headDim;

        int64_t totalTempSpaceForOneVec = MID_USE_TIMES * (vGradAccumTempSpace + kGradAccumTempSpace) * sizeof(float) +
                                          (qkMatmulTempSpace + gvMatmulTempSpace) * sizeof(qType) * COMPUTE_PIPE_NUM +
                                          maskTempSpace * sizeof(qType);

        curAICWorkspace = reinterpret_cast<__gm__ uint8_t*>(workspace) + GetBlockIdx() * totalTempSpaceForOneVec;

        qkTemp_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(curAICWorkspace), qkMatmulTempSpace * COMPUTE_PIPE_NUM);
        curAICWorkspace += qkMatmulTempSpace * sizeof(qType) * COMPUTE_PIPE_NUM;

        gvTemp_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(curAICWorkspace), gvMatmulTempSpace * COMPUTE_PIPE_NUM);
        curAICWorkspace += gvMatmulTempSpace * sizeof(qType) * COMPUTE_PIPE_NUM;

        vGradAccumTemp_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(curAICWorkspace),
                                        vGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += vGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        kGradAccumTemp_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(curAICWorkspace),
                                        kGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += kGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        qGradAccumTemp_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(workspace) +
                                                                        aivNum * totalTempSpaceForOneVec),
                                        qGradAccumTempSpace);

        // 这些GM的L2利用率非常低，不能使用
        qGradAccumTemp_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        qGrad_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        kGrad_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        vGrad_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        // 所有核共享一片globalMemory，且存在累加操作，每次执行需要清理内存防止上次执行结果残留数据影响本次结果
        // 多核执行后需要调用SyncAll保证多核间同步正常
        int64_t unitClear = qGradAccumTempSpace / aivNum;
        int64_t leftClear = qGradAccumTempSpace % aivNum;
        uint64_t globalOffset = GetBlockIdx() * unitClear;
        uint64_t clearLen = unitClear;
        if (GetBlockIdx() == aivNum - 1) {
            clearLen += leftClear;
        }
        GlobalTensor<float> thisBlockQGrad;
        thisBlockQGrad.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(workspace) +
                                            aivNum * totalTempSpaceForOneVec + globalOffset * sizeof(float)),
            clearLen);
        InitGlobalMemory(thisBlockQGrad, clearLen, static_cast<float>(0));
        SyncAll();
    }

    __aicore__ inline void DoQkMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        qkOrGvMatmul_.SetTail(actureQLen, actureKLen, headDim);
        qkOrGvMatmul_.SetTensorA(q_[qOrGOffset]);
        qkOrGvMatmul_.SetTensorB(k_[kOrVOffset], true);
        qkOrGvMatmul_.template IterateAll<false>(qkTemp_[scoreMidOffset], 0, false, true);
    }

    __aicore__ inline void DoGvMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        qkOrGvMatmul_.SetTail(actureQLen, actureKLen, headDim);
        qkOrGvMatmul_.SetTensorA(grad_[qOrGOffset]);
        qkOrGvMatmul_.SetTensorB(v_[kOrVOffset], true);
        qkOrGvMatmul_.template IterateAll<false>(gvTemp_[scoreMidOffset], 0, false, true);
    }

    __aicore__ inline void DoKGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        kGradMatmul_.SetTail(actureKLen, headDim, actureQLen);
        kGradMatmul_.SetTensorA(gvTemp_[scoreMidOffset], true);
        kGradMatmul_.SetTensorB(q_[qOrGOffset]);
        kGradMatmul_.template IterateAll<false>(kGradAccumTemp_[accumOffset], isFirstBlock ? 0 : 1, false, true);
    }

    __aicore__ inline void DoVGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        vGradMatmul_.SetTail(actureKLen, headDim, actureQLen);
        vGradMatmul_.SetTensorA(qkTemp_[scoreMidOffset], true);
        vGradMatmul_.SetTensorB(grad_[qOrGOffset]);
        vGradMatmul_.template IterateAll<false>(vGradAccumTemp_[accumOffset], isFirstBlock ? 0 : 1, false, true);
    }

    __aicore__ inline void DoQGradMatmul(const int64_t qGradOutoffset, const int64_t scoreMidOffset,
                                         const int64_t kOrVOffset, const int64_t actureQLen, const int64_t actureKLen)
    {
        qGradMatmul_.SetTail(actureQLen, headDim, actureKLen);
        qGradMatmul_.SetTensorA(gvTemp_[scoreMidOffset]);
        qGradMatmul_.SetTensorB(k_[kOrVOffset]);
        qGradMatmul_.template IterateAll<false>(qGradAccumTemp_[qGradOutoffset], 1, false, true);
    }

    __aicore__ inline void QkOrGvMatmulWait()
    {
        qkOrGvMatmul_.WaitIterateAll();
        qkOrGvMatmul_.End();
    }

    __aicore__ inline void kGradMatmulWait()
    {
        kGradMatmul_.WaitIterateAll();
        kGradMatmul_.End();
    }

    __aicore__ inline void vGradMatmulWait()
    {
        vGradMatmul_.WaitIterateAll();
        vGradMatmul_.End();
    }

    __aicore__ inline void qGradMatmulWait()
    {
        qGradMatmul_.WaitIterateAll();
        qGradMatmul_.End();
    }

    const AddrArgs* __restrict addrArgs_;
    const BaseShapeArgs* __restrict baseShape_;

    // Gt
    GlobalTensor<qType> grad_;
    GlobalTensor<qType> q_;
    GlobalTensor<qType> k_;
    GlobalTensor<qType> v_;

    GlobalTensor<qType> qGrad_;
    GlobalTensor<qType> kGrad_;
    GlobalTensor<qType> vGrad_;
    GlobalTensor<qType> attnBiasGrad_;

    GlobalTensor<qType> qkTemp_;
    GlobalTensor<qType> gvTemp_;
    GlobalTensor<float> kGradAccumTemp_;  // qGrad share temp space with kGrad
    GlobalTensor<float> vGradAccumTemp_;
    GlobalTensor<float> qGradAccumTemp_;

    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headNum, headDim>::QK_OR_GV_MATMUL qkOrGvMatmul_;
    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headNum, headDim>::K_GRAD_MATMUL kGradMatmul_;
    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headNum, headDim>::V_GRAD_MATMUL vGradMatmul_;
    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headNum, headDim>::Q_GRAD_MATMUL qGradMatmul_;
};
}  // namespace HstuDenseBackward

#endif
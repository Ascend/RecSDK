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
#include "matmul_const.h"

struct MatmulArgs {
    const int64_t scoreMidPipeId;
    const int64_t accumPipeId;
    const int64_t QOrGOffset;  // Q和G的位置是一样的，所以可以复用
    const int64_t KOrVOffset;  // K和V的位置是一样的，所以可以复用
    const int64_t ActureQLen;
    const int64_t ActureKLen;
};

namespace HstuDenseBackward {

// 静态接口实现多态的办法
template <typename qType, uint32_t blockHeightQ, uint32_t blockHeightK, class TilingDataType,class HstuMatmulMgmtStrategy>
class HstuMatmulMgmtInterface {
public:
    __aicore__ inline HstuMatmulMgmtInterface() {}

    __aicore__ inline void RegisterMatmul(TPipe* pipe, const TilingDataType* tilingPtr, GM_ADDR tiling)
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->RegisterMatmul(pipe, tilingPtr, tiling);
    }

    __aicore__ inline void InitGt(const AddrArgs* addrArgs)
    {
        int64_t qSize = totalBatchSize_ * headNum_ * headDim_;
        grad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs->grad), qSize);
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs->q), qSize);
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs->k), qSize);
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs->v), qSize);

        qGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs->qGrad), qSize);
        kGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs->kGrad), qSize);
        vGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(addrArgs->vGrad), qSize);
    }

    __aicore__ inline void InitTempSpace(GM_ADDR workspace, int64_t& totalTempSpaceForOneVec)
    {
        GM_ADDR curAICWorkspace;
        const int64_t qkMatmulTempSpace = blockHeightQ * blockHeightK;
        const int64_t gvMatmulTempSpace = blockHeightQ * blockHeightK;
        const int64_t vGradAccumTempSpace = blockHeightK * headDim_;
        const int64_t kGradAccumTempSpace = blockHeightK * headDim_;
        const int64_t scoreTempSpace = blockHeightQ * blockHeightK;
        const int64_t maskTempSpace = blockHeightQ * blockHeightK;
        const int64_t biasOrSGradTempSpace = blockHeightQ * blockHeightK;
        const int64_t qGradAccumTempSpace = batchSize_ * headNum_ * maxSeqLen_ * headDim_;

        totalTempSpaceForOneVec = MID_USE_TIMES * (vGradAccumTempSpace + kGradAccumTempSpace) * sizeof(float) +
                                          (qkMatmulTempSpace + gvMatmulTempSpace) * sizeof(qType) * COMPUTE_PIPE_NUM;

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
                                                                        aivNum_ * totalTempSpaceForOneVec),
                                        qGradAccumTempSpace);
    }

    __aicore__ inline void DisableL2Cache()
    {
        qGradAccumTemp_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        qGrad_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        kGrad_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        vGrad_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    __aicore__ inline void ClearQGradAccumTemp(GM_ADDR workspace, int64_t totalTempSpaceForOneVec)
    {
        int64_t qGradAccumTempSpace = batchSize_ * headNum_ * maxSeqLen_ * headDim_;

        // 所有核共享一片globalMemory，且存在累加操作，每次执行需要清理内存防止上次执行结果残留数据影响本次结果
        // 多核执行后需要调用SyncAll保证多核间同步正常
        int64_t unitClear = qGradAccumTempSpace / aivNum_;
        int64_t leftClear = qGradAccumTempSpace % aivNum_;
        uint64_t globalOffset = GetBlockIdx() * unitClear;
        uint64_t clearLen = unitClear;
        if (GetBlockIdx() == aivNum_ - 1) {
            clearLen += leftClear;
        }
        GlobalTensor<float> thisBlockQGrad;
        thisBlockQGrad.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(workspace) +
                                            aivNum_ * totalTempSpaceForOneVec + globalOffset * sizeof(float)),
            clearLen);
        InitGlobalMemory(thisBlockQGrad, clearLen, static_cast<float>(0));
        SyncAll();
    }

    __aicore__ inline void Init(const AddrArgs* addrArgs, const BaseShapeArgs* baseShape)
    {
        // Shape Init
        GM_ADDR workspace = addrArgs->workspace;
        totalBatchSize_ = baseShape->totalBatchSize;
        batchSize_ = baseShape->batchSize;
        maxSeqLen_ = baseShape->maxSeqLen;
        headDim_ = baseShape->headDim;
        headNum_ = baseShape->headNum;
        aivNum_ = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;

        InitGt(addrArgs);
        int64_t totalTempSpaceForOneVec;
        // 计算TempSpace大小
        InitTempSpace(workspace, totalTempSpaceForOneVec);
        // 这些GM的L2利用率非常低，不能使用
        DisableL2Cache();
        // 所有核共享一片globalMemory，且存在累加操作，每次执行需要清理内存防止上次执行结果残留数据影响本次结果
        // 多核执行后需要调用SyncAll保证多核间同步正常
        ClearQGradAccumTemp(workspace, totalTempSpaceForOneVec);
    }

    __aicore__ inline void DoQkMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->DoQkMatmul(scoreMidOffset, qOrGOffset, kOrVOffset, actureQLen, actureKLen);
    }

    __aicore__ inline void DoGvMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->DoGvMatmul(scoreMidOffset, qOrGOffset, kOrVOffset, actureQLen, actureKLen);
    }

    __aicore__ inline void DoKGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->DoKGradMatmul(accumOffset, scoreMidOffset, qOrGOffset, actureQLen, actureKLen, isFirstBlock);
    }

    __aicore__ inline void DoVGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->DoVGradMatmul(accumOffset, scoreMidOffset, qOrGOffset, actureQLen, actureKLen, isFirstBlock);
    }

    __aicore__ inline void DoQGradMatmul(const int64_t qGradOutoffset, const int64_t scoreMidOffset,
                                         const int64_t kOrVOffset, const int64_t actureQLen, const int64_t actureKLen)
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->DoQGradMatmul(qGradOutoffset, scoreMidOffset, kOrVOffset, actureQLen, actureKLen);
    }

    __aicore__ inline void QkOrGvMatmulWait()
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->QkOrGvMatmulWait();
    }

    __aicore__ inline void kGradMatmulWait()
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->kGradMatmulWait();
    }

    __aicore__ inline void vGradMatmulWait()
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->vGradMatmulWait();
    }

    __aicore__ inline void qGradMatmulWait()
    {
        static_cast<HstuMatmulMgmtStrategy*>(this)->qGradMatmulWait();
    }

    // Shape
    uint32_t totalBatchSize_;
    uint32_t batchSize_;
    uint32_t maxSeqLen_;
    uint32_t headDim_;
    uint32_t headNum_;

    // AIC
    uint32_t aivNum_;

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
};


template <typename qType, int64_t blockHeightQ, int64_t blockHeightK, int64_t headDimPadding, class TilingDataType>
class MmMgmtFp16R0Jagged : public HstuMatmulMgmtInterface<qType, blockHeightQ, blockHeightK, TilingDataType,
        MmMgmtFp16R0Jagged<qType, blockHeightQ, blockHeightK, headDimPadding, TilingDataType>> {
public:
    __aicore__ inline MmMgmtFp16R0Jagged() {}

    __aicore__ inline void DoQkMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        qkOrGvMatmul_.SetTail(actureQLen, actureKLen, this->headDim_);
        qkOrGvMatmul_.SetTensorA(this->q_[qOrGOffset]);
        qkOrGvMatmul_.SetTensorB(this->k_[kOrVOffset], true);
        qkOrGvMatmul_.template IterateAll<false>(this->qkTemp_[scoreMidOffset], 0, false, true);
    }

    __aicore__ inline void DoGvMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        qkOrGvMatmul_.SetTail(actureQLen, actureKLen, this->headDim_);
        qkOrGvMatmul_.SetTensorA(this->grad_[qOrGOffset]);
        qkOrGvMatmul_.SetTensorB(this->v_[kOrVOffset], true);
        qkOrGvMatmul_.template IterateAll<false>(this->gvTemp_[scoreMidOffset], 0, false, true);
    }

    __aicore__ inline void DoKGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        kGradMatmul_.SetTail(actureKLen, this->headDim_, actureQLen);
        kGradMatmul_.SetTensorA(this->gvTemp_[scoreMidOffset], true);
        kGradMatmul_.SetTensorB(this->q_[qOrGOffset]);
        kGradMatmul_.template IterateAll<false>(this->kGradAccumTemp_[accumOffset], isFirstBlock ? 0 : 1, false, true);
    }

    __aicore__ inline void DoVGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        vGradMatmul_.SetTail(actureKLen, this->headDim_, actureQLen);
        vGradMatmul_.SetTensorA(this->qkTemp_[scoreMidOffset], true);
        vGradMatmul_.SetTensorB(this->grad_[qOrGOffset]);
        vGradMatmul_.template IterateAll<false>(this->vGradAccumTemp_[accumOffset], isFirstBlock ? 0 : 1, false, true);
    }

    __aicore__ inline void DoQGradMatmul(const int64_t qGradOutoffset, const int64_t scoreMidOffset,
                                         const int64_t kOrVOffset, const int64_t actureQLen, const int64_t actureKLen)
    {
        qGradMatmul_.SetTail(actureQLen, this->headDim_, actureKLen);
        qGradMatmul_.SetTensorA(this->gvTemp_[scoreMidOffset]);
        qGradMatmul_.SetTensorB(this->k_[kOrVOffset]);
        qGradMatmul_.template IterateAll<false>(this->qGradAccumTemp_[qGradOutoffset], 1, false, true);
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

    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headDimPadding, TilingDataType>::QK_OR_GV_MATMUL
        qkOrGvMatmul_;
    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headDimPadding, TilingDataType>::K_GRAD_MATMUL
        kGradMatmul_;
    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headDimPadding, TilingDataType>::V_GRAD_MATMUL
        vGradMatmul_;
    typename MatmulJF16R0Const<qType, blockHeightQ, blockHeightK, headDimPadding, TilingDataType>::Q_GRAD_MATMUL
        qGradMatmul_;
};

template <typename qType, int64_t blockHeightQ, int64_t blockHeightK, int64_t headDimPadding, class TilingDataType>
class MmMgmtCommon : public HstuMatmulMgmtInterface<qType, blockHeightQ, blockHeightK, TilingDataType,
        MmMgmtCommon<qType, blockHeightQ, blockHeightK, headDimPadding, TilingDataType>> {
public:
    __aicore__ inline MmMgmtCommon() {}

    __aicore__ inline void DoQkMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        qkOrGvMatmul_.SetTail(actureQLen, actureKLen, this->headDim_);
        qkOrGvMatmul_.SetTensorA(this->q_[qOrGOffset]);
        qkOrGvMatmul_.SetTensorB(this->k_[kOrVOffset], true);
        qkOrGvMatmul_.template IterateAll<false>(this->qkTemp_[scoreMidOffset], 0, false, true);
    }

    __aicore__ inline void DoGvMatmul(const int64_t scoreMidOffset, const int64_t qOrGOffset, const int64_t kOrVOffset,
                                      const int64_t actureQLen, const int64_t actureKLen)
    {
        qkOrGvMatmul_.SetTail(actureQLen, actureKLen, this->headDim_);
        qkOrGvMatmul_.SetTensorA(this->grad_[qOrGOffset]);
        qkOrGvMatmul_.SetTensorB(this->v_[kOrVOffset], true);
        qkOrGvMatmul_.template IterateAll<false>(this->gvTemp_[scoreMidOffset], 0, false, true);
    }

    __aicore__ inline void DoKGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        kGradMatmul_.SetTail(actureKLen, this->headDim_, actureQLen);
        kGradMatmul_.SetTensorA(this->gvTemp_[scoreMidOffset], true);
        kGradMatmul_.SetTensorB(this->q_[qOrGOffset]);
        kGradMatmul_.template IterateAll<false>(this->kGradAccumTemp_[accumOffset], isFirstBlock ? 0 : 1, false, true);
    }

    __aicore__ inline void DoVGradMatmul(const int64_t accumOffset, const int64_t scoreMidOffset,
                                         const int64_t qOrGOffset, const int64_t actureQLen, const int64_t actureKLen,
                                         const bool isFirstBlock)
    {
        vGradMatmul_.SetTail(actureKLen, this->headDim_, actureQLen);
        vGradMatmul_.SetTensorA(this->qkTemp_[scoreMidOffset], true);
        vGradMatmul_.SetTensorB(this->grad_[qOrGOffset]);
        vGradMatmul_.template IterateAll<false>(this->vGradAccumTemp_[accumOffset], isFirstBlock ? 0 : 1, false, true);
    }

    __aicore__ inline void DoQGradMatmul(const int64_t qGradOutoffset, const int64_t scoreMidOffset,
                                         const int64_t kOrVOffset, const int64_t actureQLen, const int64_t actureKLen)
    {
        qGradMatmul_.SetTail(actureQLen, this->headDim_, actureKLen);
        qGradMatmul_.SetTensorA(this->gvTemp_[scoreMidOffset]);
        qGradMatmul_.SetTensorB(this->k_[kOrVOffset]);
        qGradMatmul_.template IterateAll<false>(this->qGradAccumTemp_[qGradOutoffset], 1, false, true);
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

    using CopyFun = MatmulStriCopyFun<qType, TilingDataType>;
    // Matmul
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, CopyFun::CopyQKA1_Strd, CopyFun::CopyQKB1_Strd_Trans>>
        qkOrGvMatmul_;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, nullptr, CopyFun::CopyVGradB1_Strd>>
        qGradMatmul_;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, nullptr, CopyFun::CopyVGradB1_Strd>>
        kGradMatmul_;

    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true, LayoutMode::NONE, false, TPosition::VECOUT>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
        matmul::MatmulCallBackFunc<nullptr, nullptr, CopyFun::CopyVGradB1_Strd>>
        vGradMatmul_;
};

}  // namespace HstuDenseBackward

#endif
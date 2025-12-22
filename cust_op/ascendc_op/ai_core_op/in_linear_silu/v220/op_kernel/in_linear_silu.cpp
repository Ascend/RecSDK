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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"

using namespace AscendC;

namespace kernels {
template <typename T, typename CT>
class InLinearSilu {
public:
    __aicore__ inline InLinearSilu(TPipe* pipeIn)
    {
        pipe_ = pipeIn;
    };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR user, GM_ADDR value, GM_ADDR query,
                                GM_ADDR key, GM_ADDR usrWorkspace, const InLinearSiluTilingData* tilingData)
    {
        coreId_ = GetBlockIdx();
        mShape_ = tilingData->mShape;
        nShape_ = tilingData->nShape;
        kShape_ = tilingData->kShape;
        uLength_ = tilingData->uLength;
        vLength_ = tilingData->vLength;
        qLength_ = tilingData->qLength;
        kLength_ = tilingData->kLength;
        formerNum_ = tilingData->formerNum;
        tailNum_ = tilingData->tailNum;
        formerSingleLen_ = tilingData->formerSingleLen;
        tailSingleLen_ = tilingData->tailSingleLen;
        formerLoop_ = tilingData->formerLoop;
        tailLoop_ = tilingData->tailLoop;
        formerPerLoopLen_ = tilingData->formerPerLoopLen;
        tailPerLoopLen_ = tilingData->tailPerLoopLen;
        formerRemain_ = tilingData->formerRemain;
        tailRemain_ = tilingData->tailRemain;
        buffseSize_ = tilingData->bufferSize;

        uint32_t xOffset = 0;
        uint32_t fomerLen = formerSingleLen_ * kShape_;
        uint32_t tailLen = tailSingleLen_ * kShape_;
        uint32_t localLen = 0;
        uint32_t uOffset = 0;
        uint32_t vOffset = 0;
        uint32_t qOffset = 0;
        uint32_t kOffset = 0;
        uint32_t uLen = 0;
        uint32_t vLen = 0;
        uint32_t qLen = 0;
        uint32_t kLen = 0;
        if (coreId_ < formerNum_) {
            xOffset = coreId_ * fomerLen;
            tileLength_ = fomerLen;
            localLen = formerPerLoopLen_ * nShape_ * sizeof(CT);
            uOffset = coreId_ * formerSingleLen_ * uLength_;
            vOffset = coreId_ * formerSingleLen_ * vLength_;
            qOffset = coreId_ * formerSingleLen_ * qLength_;
            kOffset = coreId_ * formerSingleLen_ * kLength_;
            uLen = formerSingleLen_ * uLength_;
            vLen = formerSingleLen_ * vLength_;
            qLen = formerSingleLen_ * qLength_;
            kLen = formerSingleLen_ * kLength_;
        } else {
            xOffset = formerNum_ * fomerLen + (coreId_ - formerNum_) * tailLen;
            tileLength_ = tailLen;
            localLen = tailPerLoopLen_ * nShape_ * sizeof(CT);
            uOffset = formerNum_ * formerSingleLen_ * uLength_ + (coreId_ - formerNum_) * tailSingleLen_ * uLength_;
            vOffset = formerNum_ * formerSingleLen_ * vLength_ + (coreId_ - formerNum_) * tailSingleLen_ * vLength_;
            qOffset = formerNum_ * formerSingleLen_ * qLength_ + (coreId_ - formerNum_) * tailSingleLen_ * qLength_;
            kOffset = formerNum_ * formerSingleLen_ * kLength_ + (coreId_ - formerNum_) * tailSingleLen_ * kLength_;
            uLen = tailSingleLen_ * uLength_;
            vLen = tailSingleLen_ * vLength_;
            qLen = tailSingleLen_ * qLength_;
            kLen = tailSingleLen_ * kLength_;
        }

        xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x + xOffset * sizeof(T)), tileLength_);
        weightGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight), nShape_ * kShape_);
        biasGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), nShape_);
        mmOutGM.SetGlobalBuffer(reinterpret_cast<__gm__ CT*>(usrWorkspace + buffseSize_ * 2 * coreId_));
        // gm 出参大小初始化
        userGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(user + uOffset * sizeof(T)), uLen);
        valueGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value + vOffset * sizeof(T)), vLen);
        queryGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query + qOffset * sizeof(T)), qLen);
        keyGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key + kOffset * sizeof(T)), kLen);
        pipe_->InitBuffer(mmOutQue, 1, localLen);
        pipe_->InitBuffer(siluOutQue, 1, localLen);
    }

    __aicore__ inline void Process()
    {
        // 统一变量名
        size_t loopCount = (coreId_ < formerNum_) ? formerLoop_ : tailLoop_;
        size_t perLoopLen = (coreId_ < formerNum_) ? formerPerLoopLen_ : tailPerLoopLen_;
        size_t remain = (coreId_ < formerNum_) ? formerRemain_ : tailRemain_;

        for (uint32_t i = 0; i <= loopCount; ++i) {
            uint32_t currentIdx = i % 2;         // 当前的idx对应的mmout缓冲块
            uint32_t previousIdx = (i - 1) % 2;  // 上一个循环对应的缓冲块
            // 等待并处理前一个矩阵乘法的结果
            if (i > 0) {
                WaitMatmul();
                SiluCompute(perLoopLen, previousIdx);
                CopyOut(i - 1, perLoopLen, perLoopLen);
            }

            // 启动新的矩阵乘法
            if (i < loopCount) {
                MatmulCompute(i, perLoopLen, perLoopLen, currentIdx);
            }
        }

        // 处理剩余部分
        if (remain > 0) {
            MatmulCompute(loopCount, perLoopLen, remain, 0);
            WaitMatmul();
            SiluCompute(remain, 0);
            CopyOut(loopCount, perLoopLen, remain);
        }
    }

    TPipe* pipe_;
    using XType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using WeightType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using OutType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, CT>;
    using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using MatmulObj = matmul::Matmul<XType, WeightType, OutType, BiasType>;
    MatmulObj mm;
private:
    __aicore__ inline void MatmulCompute(uint32_t idx, uint32_t offset, uint32_t perLoopLen, uint32_t currentIdx)
    {
        mm.SetTail(perLoopLen, nShape_, kShape_);
        mm.SetTensorA(xGM[idx * offset * kShape_]);
        mm.SetTensorB(weightGM, true);
        mm.SetBias(biasGM);
        mm.template IterateAll<false>(mmOutGM[currentIdx * perLoopLen * nShape_], 0, false, true);
    }

    __aicore__ inline void WaitMatmul()
    {
        mm.WaitIterateAll();
        mm.End();
    }

    __aicore__ inline void SiluCompute(uint32_t perLoopLen, uint32_t previousIdx)
    {
        mmOutLocal = mmOutQue.AllocTensor<CT>();
        DataCopy(mmOutLocal, mmOutGM[previousIdx * perLoopLen * nShape_], perLoopLen * nShape_);
        mmOutQue.EnQue(mmOutLocal);
        mmOutLocal = mmOutQue.DeQue<CT>();
        siluOutLocal = siluOutQue.AllocTensor<CT>();
        Silu(siluOutLocal, mmOutLocal, perLoopLen * nShape_);
        siluOutQue.EnQue(siluOutLocal);
        mmOutQue.FreeTensor(mmOutLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t offset, uint32_t loop)
    {
        uint32_t uIdx = idx * offset * uLength_;
        uint32_t vIdx = idx * offset * vLength_;
        uint32_t qIdx = idx * offset * qLength_;
        uint32_t kIdx = idx * offset * kLength_;
        siluOutLocal = siluOutQue.DeQue<CT>();
        if constexpr (IsSameType<T, CT>::value) {
            for (uint32_t l = 0; l < loop; ++l) {
                uint32_t localIdx = l * nShape_;
                DataCopy(userGM[uIdx + l * uLength_], siluOutLocal[localIdx], uLength_);
                DataCopy(valueGM[vIdx + l * vLength_], siluOutLocal[localIdx + uLength_], vLength_);
                DataCopy(queryGM[qIdx + l * qLength_], siluOutLocal[localIdx + uLength_ + vLength_], qLength_);
                DataCopy(keyGM[kIdx + l * kLength_], siluOutLocal[localIdx + uLength_ + vLength_ + qLength_], kLength_);
            }
        } else {
            LocalTensor<T> castOut = siluOutLocal.template ReinterpretCast<T>();
            Cast(castOut, siluOutLocal, RoundMode::CAST_RINT, loop * nShape_);
            TEventID eventIdVToMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            for (uint32_t l = 0; l < loop; ++l) {
                uint32_t localIdx = l * nShape_;
                DataCopy(userGM[uIdx + l * uLength_], castOut[localIdx], uLength_);
                DataCopy(valueGM[vIdx + l * vLength_], castOut[localIdx + uLength_], vLength_);
                DataCopy(queryGM[qIdx + l * qLength_], castOut[localIdx + uLength_ + vLength_], qLength_);
                DataCopy(keyGM[kIdx + l * kLength_], castOut[localIdx + uLength_ + vLength_ + qLength_], kLength_);
            }
        }
        siluOutQue.FreeTensor(siluOutLocal);
    }

    TQue<TPosition::VECIN, 1> mmOutQue;
    TQue<TPosition::VECOUT, 1> siluOutQue;
    GlobalTensor<T> xGM, weightGM, userGM, valueGM, queryGM, keyGM;
    GlobalTensor<CT> mmOutGM;
    GlobalTensor<float> biasGM;
    LocalTensor<CT> mmOutLocal;
    LocalTensor<CT> siluOutLocal;

    uint32_t coreId_;
    uint32_t mShape_;
    uint32_t kShape_;
    uint32_t nShape_;
    uint32_t uLength_;
    uint32_t vLength_;
    uint32_t qLength_;
    uint32_t kLength_;
    uint32_t formerNum_;
    uint32_t tailNum_;
    uint32_t formerSingleLen_;
    uint32_t tailSingleLen_;
    uint32_t formerLoop_;
    uint32_t tailLoop_;
    uint32_t formerPerLoopLen_;
    uint32_t tailPerLoopLen_;
    uint32_t formerRemain_;
    uint32_t tailRemain_;
    uint32_t blockDim_;
    uint32_t tileLength_;
    uint32_t buffseSize_;
};
}  // namespace kernels

extern "C" __global__ __aicore__ void in_linear_silu(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR user,
                                                     GM_ADDR value, GM_ADDR query, GM_ADDR key, GM_ADDR workspace,
                                                     GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        kernels::InLinearSilu<float16_t, float16_t> op(&pipe);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
        REGIST_MATMUL_OBJ(op.pipe_, GetSysWorkSpacePtr(), op.mm, &tiling_data.cubeTiling);
        if (GetBlockIdx() % 2 != 1 || GetBlockIdx() != tiling_data.blockDim) {
            op.Init(x, weight, bias, user, value, query, key, usrWorkspace, &tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        kernels::InLinearSilu<float, float> op(&pipe);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
        REGIST_MATMUL_OBJ(op.pipe_, GetSysWorkSpacePtr(), op.mm, &tiling_data.cubeTiling);
        if (GetBlockIdx() % 2 != 1 || GetBlockIdx() != tiling_data.blockDim) {
            op.Init(x, weight, bias, user, value, query, key, usrWorkspace, &tiling_data);
            op.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        kernels::InLinearSilu<bfloat16_t, float> op(&pipe);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
        REGIST_MATMUL_OBJ(op.pipe_, GetSysWorkSpacePtr(), op.mm, &tiling_data.cubeTiling);
        if (GetBlockIdx() % 2 != 1 || GetBlockIdx() != tiling_data.blockDim) {
            op.Init(x, weight, bias, user, value, query, key, usrWorkspace, &tiling_data);
            op.Process();
        }
    }
}

/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
============================================================================== */

/**
 • @file fast_rab_grad.hpp

 • @brief RAB (Relative Attention Bias) 梯度计算实现

 • @description 提供相对位置注意力偏置的反向传播梯度计算，

 •              将 SiLU score 的梯度传递到 RAB 参数

 */
#pragma once

#include "../../../catlass_hstu/epilogue/regbase/common.h"

namespace catlass::Epilogue::RegBase {

/**
 • @brief RAB 梯度向量化快速计算函数

 • @tparam Type 输入/输出数据类型

 • @tparam AccType 累加器数据类型

 • @tparam GrabType 梯度数据类型

 • @param ubGSPtr SiLU Score 梯度的 UBuf 指针

 • @param ubGRabPartPtr RAB 部分梯度的 UBuf 指针

 • @param ubGRabPtr 输出 RAB 总梯度的 UBuf 指针

 • @param count 有效元素数量

 • @param repeatTimes 重复次数 (向量化的循环次数)

 • @description 计算 RAB 的梯度: grad_rab = grad_score * 1 (直接传递)

 •              由于 d(silu(x))/dx 对 x 求导时 x 就是 RAB，所以梯度直接传递

 */
template <typename Type, typename AccType, typename GrabType>
__simd_vf__ inline void FastRabGradVf(__ubuf__ AccType *ubGSPtr, __ubuf__ GrabType *ubGRabPartPtr,
                                      __ubuf__ Type *ubGRabPtr, uint32_t count, uint32_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(AccType));

    AscendC::MicroAPI::RegTensor<AccType> vregA;
    AscendC::MicroAPI::RegTensor<AccType> vregT;

    AscendC::MicroAPI::MaskReg maskReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        maskReg = AscendC::MicroAPI::UpdateMask<AccType>(count);

        AscendC::MicroAPI::LoadAlign(vregA, ubGSPtr + i * oneRepElm);

        if constexpr (!std::is_same<GrabType, AccType>::value) {
            CastUpLoad<AccType, GrabType>(vregT, ubGRabPartPtr + i * oneRepElm, maskReg);
        } else {
            AscendC::MicroAPI::LoadAlign(vregT, ubGRabPartPtr + i * oneRepElm);
        }

        AscendC::MicroAPI::Mul(vregA, vregA, vregT, maskReg);

        if constexpr (!std::is_same<Type, AccType>::value) {
            CastDownStore<Type, AccType>(ubGRabPtr + i * oneRepElm, vregA, maskReg);
        } else {
            AscendC::MicroAPI::StoreAlign(ubGRabPtr + i * oneRepElm, vregA, maskReg);
        }
    }
}

}

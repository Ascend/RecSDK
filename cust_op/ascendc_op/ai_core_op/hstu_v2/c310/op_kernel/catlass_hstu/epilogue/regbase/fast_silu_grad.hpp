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
 • @file fast_silu_grad.hpp

 • @brief SiLU 激活函数梯度计算实现

 • @description 提供 SiLU 激活函数的反向传播梯度计算，用于 HSTU 注意力机制的 backward 过程

 •              计算公式: d(silu(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

 */
#pragma once

#include "../../../catlass_hstu/epilogue/regbase/fast_silu_score.hpp"

namespace catlass::Epilogue::RegBase {

/**
 • @brief SiLU 梯度向量化快速计算函数

 • @tparam Type 输入/输出数据类型

 • @tparam AccType 累加器数据类型

 • @tparam GrabType 梯度数据类型

 • @tparam HAS_RAB 是否有相对位置偏置

 • @param ubSPtr SiLU 输入 (QK^T) 的 UBuf 指针

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param ubSiluScorePtr SiLU 分数的 UBuf 指针

 • @param ubGradPartPtr 输出梯度的 UBuf 指针

 • @param alpha 注意力分数的缩放系数

 • @param scale SiLU 输出缩放因子

 • @param count 有效元素数量

 • @param repeatTimes 重复次数 (向量化的循环次数)

 • @description 实现 SiLU 梯度的向量化计算:

 •              1. 重新计算 SiLU score (调用 SiluScore)

 •              2. 计算 sigmoid 的导数: Z = 1 - sigmoid

 •              3. 计算梯度: T = T + S * (1 - Z) = sigmoid * (1 + x * (1 - sigmoid))

 •              4. 乘以 alpha 系数

 •              5. 存储梯度结果

 */
template <typename Type, typename AccType, typename GrabType, bool HAS_RAB>
__simd_vf__ inline void FastSiluGradVf(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr, __ubuf__ Type* ubMaskPtr,
                                       __ubuf__ Type* ubSiluScorePtr, __ubuf__ GrabType* ubGradPartPtr, AccType alpha,
                                       AccType scale, uint32_t count, uint32_t repeatTimes, bool needMask)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(AccType));

    AscendC::MicroAPI::RegTensor<AccType> vregA;
    AscendC::MicroAPI::RegTensor<AccType> vregZ;
    AscendC::MicroAPI::RegTensor<AccType> vregT;
    AscendC::MicroAPI::RegTensor<AccType> vregS;
    AscendC::MicroAPI::RegTensor<AccType> vregOnes;
    AscendC::MicroAPI::RegTensor<AccType> vregZeros;
    AscendC::MicroAPI::MaskReg maskReg;

    AscendC::MicroAPI::Duplicate(vregOnes, 1.0f);
    AscendC::MicroAPI::Duplicate(vregZeros, 0.0f);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        maskReg = AscendC::MicroAPI::UpdateMask<AccType>(count);

        SiluScore<Type, AccType, HAS_RAB>(ubSPtr + i * oneRepElm, ubRabPtr + i * oneRepElm, ubMaskPtr + i * oneRepElm,
                                          ubSiluScorePtr + i * oneRepElm, vregA, vregZ, vregT, vregS, vregOnes,
                                          vregZeros, alpha, maskReg, needMask);

        AscendC::MicroAPI::Sub(vregT, vregOnes, vregZ, maskReg);     // T = 1 - Z
        AscendC::MicroAPI::MulAddDst(vregZ, vregS, vregT, maskReg);  // Z = Z + S * T

        if constexpr (!std::is_same<GrabType, AccType>::value) {
            CastDownStore<GrabType, AccType>(ubGradPartPtr + i * oneRepElm, vregZ, maskReg);
        } else {
            AscendC::MicroAPI::StoreAlign(ubGradPartPtr + i * oneRepElm, vregZ, maskReg);
        }
    }
}

}  // namespace catlass::Epilogue::RegBase

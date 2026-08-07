/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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
 • @file fast_silu_score.hpp

 • @brief SiLU (Swish) 激活函数得分计算实现

 • @description 提供基于寄存器的 SiLU 激活函数前向计算、掩码乘法、RAB 相加等操作，

 •              用于 HSTU 注意力机制中 score = silu(QK^T + RAB) * scale * mask 的计算

 */
#pragma once

#include "../../../catlass_hstu/epilogue/regbase/common.h"

namespace catlass::Epilogue::RegBase {

/**
 • @brief Sigmoid 激活函数

 • @tparam RegType 寄存器类型

 • @param dst 目标寄存器，存储计算结果

 • @param src 源寄存器，输入值

 • @param ones 常量寄存器，值为 1.0

 • @param maskReg 掩码寄存器

 • @description 实现 sigmoid(x) = 1 / (1 + exp(-x)) 函数

 •              数学公式: y = 1 / (1 + exp(-x)) = sigmoid(x)

 */
// y = 1 / (1 + exp(-x))
template <typename RegType>
__simd_callee__ inline void Sigmoid(RegType& dst, RegType& src, RegType& ones, RegType& zeros,
                                    AscendC::MicroAPI::MaskReg& maskReg)
{
    AscendC::MicroAPI::ExpSub(dst, zeros, src, maskReg);  // dst = exp(0 - dst)
    AscendC::MicroAPI::Add(dst, ones, dst, maskReg);      // dst = 1 + dst
    AscendC::MicroAPI::Div(dst, ones, dst, maskReg);      // dst = 1 / dst
}

/**
 • @brief 加上相对位置注意力偏置 (RAB)

 • @tparam Type 输入数据类型

 • @tparam AccType 累加器数据类型

 • @tparam RegType 寄存器类型

 • @param dst 目标寄存器，存储计算结果

 • @param src 源寄存器，输入的注意力分数

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param maskReg 掩码寄存器

 • @description 将 RAB (Relative Attention Bias) 加到注意力分数上: dst = src + RAB

 */
template <typename Type, typename AccType, typename RegType>
__simd_callee__ inline void AddRab(RegType& dst, __ubuf__ Type* ubRabPtr, AccType alpha,
                                   AscendC::MicroAPI::MaskReg& maskReg)
{
    RegType vregRab;

    if constexpr (!std::is_same<Type, AccType>::value) {
        CastUpLoad<AccType, Type>(vregRab, ubRabPtr, maskReg);
    } else {
        AscendC::MicroAPI::LoadAlign(vregRab, ubRabPtr);
    }

    AscendC::MicroAPI::Axpy(dst, vregRab, alpha, maskReg);  // A = A + Rab * alpha
}

/**
 • @brief 掩码乘法操作

 • @tparam Type 输入数据类型

 • @tparam AccType 累加器数据类型

 • @tparam RegType 寄存器类型

 • @param dst 目标寄存器，存储计算结果

 • @param src 源寄存器，输入的注意力分数

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param maskReg 掩码寄存器

 • @description 将掩码应用到注意力分数上: dst = src * mask，用于屏蔽无效位置

 */
template <typename Type, typename AccType, typename RegType>
__simd_callee__ inline void MulMask(RegType& dst, RegType& src, __ubuf__ Type* ubMaskPtr,
                                    AscendC::MicroAPI::MaskReg& maskReg)
{
    RegType vregMask;

    if constexpr (!std::is_same<Type, AccType>::value) {
        CastUpLoad<AccType, Type>(vregMask, ubMaskPtr, maskReg);
    } else {
        AscendC::MicroAPI::LoadAlign(vregMask, ubMaskPtr);
    }
    AscendC::MicroAPI::Mul(dst, src, vregMask, maskReg);  // Z = Z * mask
}

/**
 • @brief SiLU Score 计算核心函数

 • @tparam Type 输入/输出数据类型

 • @tparam AccType 累加器数据类型

 • @tparam HAS_RAB 是否有相对位置偏置

 • @tparam HAS_MASK 是否有掩码

 • @tparam RegType 寄存器类型

 • @param ubSPtr SiLU 输入 (QK^T) 的 UBuf 指针

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param ubSiluScorePtr 输出 SiLU 分数的 UBuf 指针

 • @param vregA 工作寄存器 A

 • @param vregZ 工作寄存器 Z (sigmoid 结果)

 • @param vregT 工作寄存器 T (中间值)

 • @param vregS 工作寄存器 S (最终结果)

 • @param vregOnes 常量寄存器，值为 1.0

 • @param vregAlpha Alpha 系数寄存器

 • @param vregScale Scale 系数寄存器

 • @param maskReg 掩码寄存器

 • @description 实现完整的 SiLU Score 计算流程:

 •              1. 加载 QK^T 分数

 •              2. 加上 RAB (如果 HAS_RAB)

 •              3. 乘以 alpha

 •              4. 计算 sigmoid

 •              5. 乘以掩码 (如果 HAS_MASK)

 •              6. 乘以 scale

 •              7. 计算 silu(x) = x * sigmoid(x) 并存储

 */
template <typename Type, typename AccType, bool HAS_RAB, bool HAS_MASK, typename RegType>
__simd_callee__ inline void SiluScore(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr, __ubuf__ Type* ubMaskPtr,
                                      __ubuf__ Type* ubSiluScorePtr, RegType& vregA, RegType& vregZ, RegType& vregS,
                                      RegType& vregOnes, RegType& vregZeros, RegType& vregAlpha, RegType& vregScale,
                                      AscendC::MicroAPI::MaskReg& maskReg)
{
    AscendC::MicroAPI::LoadAlign(vregA, ubSPtr);

    if constexpr (HAS_RAB) {
        RegType vregRab;
        if constexpr (!std::is_same<Type, AccType>::value) {
            CastUpLoad<AccType, Type>(vregRab, ubRabPtr, maskReg);
        } else {
            AscendC::MicroAPI::LoadAlign(vregRab, ubRabPtr);
        }
        AscendC::MicroAPI::Add(vregA, vregA, vregRab, maskReg);  // A = S + Rab
    }

    AscendC::MicroAPI::Mul(vregA, vregA, vregAlpha, maskReg);  // A = A * alpha

    Sigmoid(vregZ, vregA, vregOnes, vregZeros, maskReg);  // Z = Sigmoid(A)

    if constexpr (HAS_MASK) {
        MulMask<Type, AccType>(vregZ, vregZ, ubMaskPtr, maskReg);  // Z = Z * mask
    }

    AscendC::MicroAPI::Mul(vregS, vregZ, vregA, maskReg);      // S = A * Z
    AscendC::MicroAPI::Mul(vregS, vregS, vregScale, maskReg);  // S = S * scale

    if constexpr (!std::is_same<Type, AccType>::value) {
        CastDownStore<Type, AccType>(ubSiluScorePtr, vregS, maskReg);
    } else {
        AscendC::MicroAPI::StoreAlign(ubSiluScorePtr, vregS, maskReg);
    }
}

/**
 • @brief SiLU Score 向量化快速计算函数

 • @tparam Type 输入/输出数据类型

 • @tparam AccType 累加器数据类型

 • @tparam HAS_RAB 是否有相对位置偏置

 • @tparam HAS_MASK 是否有掩码

 • @param ubSPtr SiLU 输入 (QK^T) 的 UBuf 指针

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param ubSiluScorePtr 输出 SiLU 分数的 UBuf 指针

 • @param alpha 注意力分数的缩放系数

 • @param scale SiLU 输出缩放因子

 • @param count 有效元素数量

 • @param repeatTimes 重复次数 (向量化的循环次数)

 • @description 向量化版本的 SiLU Score 计算，通过循环处理多次向量运算，

 •              每次处理一个向量长度 (GetVecLen/sizeof(AccType)) 的数据

 */
template <typename Type, typename AccType, bool HAS_RAB, bool HAS_MASK>
__simd_vf__ inline void FastSiluScoreVf(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr, __ubuf__ Type* ubMaskPtr,
                                        __ubuf__ Type* ubSiluScorePtr, AccType alpha, AccType scale, uint32_t count,
                                        uint32_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(AccType));

    AscendC::MicroAPI::RegTensor<AccType> vregA;
    AscendC::MicroAPI::RegTensor<AccType> vregZ;
    AscendC::MicroAPI::RegTensor<AccType> vregS;
    AscendC::MicroAPI::RegTensor<AccType> vregOnes;
    AscendC::MicroAPI::RegTensor<AccType> vregZeros;
    AscendC::MicroAPI::RegTensor<AccType> vregAlpha;
    AscendC::MicroAPI::RegTensor<AccType> vregScale;
    AscendC::MicroAPI::MaskReg maskReg;

    AscendC::MicroAPI::Duplicate(vregOnes, 1.0f);
    AscendC::MicroAPI::Duplicate(vregZeros, 0.0f);
    AscendC::MicroAPI::Duplicate(vregAlpha, alpha);
    AscendC::MicroAPI::Duplicate(vregScale, scale);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        // maskReg 必须在每次迭代前重算：Sigmoid 内 Div 等指令会消费/改写 mask 寄存器，
        // 循环外只算一次会导致后续迭代 mask 错误（3dba5595 精度回归根因）
        maskReg = AscendC::MicroAPI::UpdateMask<AccType>(count);
        SiluScore<Type, AccType, HAS_RAB, HAS_MASK>(ubSPtr + i * oneRepElm, ubRabPtr + i * oneRepElm,
                                                    ubMaskPtr + i * oneRepElm, ubSiluScorePtr + i * oneRepElm, vregA,
                                                    vregZ, vregS, vregOnes, vregZeros, vregAlpha, vregScale, maskReg);
    }
}

}  // namespace catlass::Epilogue::RegBase

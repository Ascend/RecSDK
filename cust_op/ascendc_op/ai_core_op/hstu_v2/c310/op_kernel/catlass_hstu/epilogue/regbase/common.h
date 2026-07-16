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
 • @file common.h

 • @brief HSTU CATLASS Epilogue 基础函数定义

 • @description 提供数据类型转换的加载和存储函数，用于不同精度数据类型之间的转换

 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace catlass::Epilogue::RegBase {

/**
 • @brief 将高精度数据转换为低精度数据并存储到 UBuf

 • @tparam dstType 目标数据类型（如 bfloat16, float16）

 • @tparam srcType 源数据类型（如 float32）

 • @param dst 目标 UBuf 指针

 • @param src 源寄存器张量

 • @param maskReg 掩码寄存器，用于处理非对齐数据

 • @description 将 float32 类型数据转换为 float16/bfloat16 并存储，

 •              常用于 GEMM 计算结果从累加器类型输出到目标类型

 */
template <typename dstType, typename srcType>
__simd_callee__ inline void CastDownStore(__ubuf__ dstType* dst, AscendC::MicroAPI::RegTensor<srcType>& src,
                                          AscendC::MicroAPI::MaskReg& maskReg)
{
    // float->bfloat16/float16
    // SAT: 溢出时饱和到 dstType 范围,避免回绕
    static constexpr AscendC::MicroAPI::CastTrait castNativeTrait = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    AscendC::MicroAPI::RegTensor<dstType> vregTmp;
    AscendC::MicroAPI::Cast<dstType, srcType, castNativeTrait>(vregTmp, src, maskReg);
    AscendC::MicroAPI::StoreAlign<dstType, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst, vregTmp, maskReg);
}

/**
 • @brief 从 UBuf 加载低精度数据并转换为高精度数据

 • @tparam dstType 目标数据类型（如 float32）

 • @tparam srcType 源数据类型（如 bfloat16, float16）

 • @param dst 目标寄存器张量

 • @param src 源 UBuf 指针

 • @param maskReg 掩码寄存器，用于处理非对齐数据

 • @description 从 UBuf 加载 float16/bfloat16 类型数据并转换为 float32，

 •              常用于输入数据从目标类型提升到计算类型进行高精度运算

 */
template <typename dstType, typename srcType>
__simd_callee__ inline void CastUpLoad(AscendC::MicroAPI::RegTensor<dstType>& dst, __ubuf__ srcType* src,
                                       AscendC::MicroAPI::MaskReg& maskReg)
{
    // bfloat16/float16->float
    static constexpr AscendC::MicroAPI::CastTrait castTrait = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    AscendC::MicroAPI::RegTensor<srcType> vregTmp;
    AscendC::MicroAPI::LoadAlign<srcType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregTmp, src);
    AscendC::MicroAPI::Cast<dstType, srcType, castTrait>(dst, vregTmp, maskReg);
}

}  // namespace catlass::Epilogue::RegBase

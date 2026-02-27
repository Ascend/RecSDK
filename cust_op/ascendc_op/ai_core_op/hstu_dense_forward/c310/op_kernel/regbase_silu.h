/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef VECTOR_SILU_VF_H
#define VECTOR_SILU_VF_H

#include "kernel_operator.h"

template<typename T>
__aicore__ inline void SiluComputeVF(__ubuf__ T* dst, __ubuf__ T* src,
    float alpha, uint32_t count, const uint16_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> tmpReg0;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(count);
        MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
        MicroAPI::Muls(srcVreg, srcVreg, alpha, mask); // x = x * alpha
        MicroAPI::Muls(tmpReg0, srcVreg, -1.0f, mask); // tmpReg0 = -x * alpha
        MicroAPI::Exp(tmpReg0, tmpReg0, mask);  // tmpReg0 = e^(-x * alpha)
        MicroAPI::Adds(tmpReg0, tmpReg0, 1.0f, mask); // tmpReg0 = 1 + e^(-x * alpha)
        MicroAPI::Div(dstVreg, srcVreg, tmpReg0, mask); // tmpReg0 = x * alpha / (1 + e^(-x * alpha))
        MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
    }
}

template<typename T>
__aicore__ inline void SiluCompute(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, float alpha,
    const uint32_t count)
{
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, half, float>(), "Silu only support half/float data type on current device!");
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, oneRepElm));

    VF_CALL<SiluComputeVF<T>>(
        (__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), alpha, count, repeatTimes);
}

#endif
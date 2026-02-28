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

#ifndef MATMUL_MGMT_H
#define MATMUL_MGMT_H

#include "matmul_constexpr.h"

template <typename qType, typename TilingDataType>
class MatmulMgmt {
public:
    static constexpr MatmulConfig qkMMConfig = std::is_same<qType, float>::value ?
        mmStaticConfigQKFp32 : mmStaticConfigQKFp16;
    static constexpr MatmulConfig svMMConfig = std::is_same<qType, float>::value ?
        mmStaticConfigSVFp32 : mmStaticConfigSVFp16;

    __aicore__ inline MatmulMgmt() {}
    
    template<bool isFirst = true>
    __aicore__ inline void DoQKMatmul(const MatmulArgs& args, const GlobalTensor<qType>& leftGt,
        const GlobalTensor<qType>& rightGt, GlobalTensor<qType>& outGt)
    {
        qkMatmul_.SetTensorA(leftGt[args.leftOffset]);
        qkMatmul_.SetTensorB(rightGt[args.rightOffset], true);
        qkMatmul_.SetTail(args.m, args.n, args.k);
        qkMatmul_.SetSelfDefineData(args.headNum); // 设置CopyQK的自定义headNum数据

        qkMatmul_.template IterateAll<false>(outGt[args.outOffset], 0, false, true);
    }

    __aicore__ inline void DoSVMatmul(const MatmulArgs& args, const GlobalTensor<qType>& leftGt,
        const GlobalTensor<qType>& rightGt, GlobalTensor<float>& outGt)
    {
        svMatmul_.SetTensorA(leftGt[args.leftOffset]);
        svMatmul_.SetTensorB(rightGt[args.rightOffset]);
        svMatmul_.SetTail(args.m, args.n, args.k);
        svMatmul_.SetSelfDefineData(args.headNum); // 设置CopyQK的自定义headNum数据

        svMatmul_.template IterateAll<false>(outGt[args.outOffset], args.isAtomicAdd, false, true);
    }

    __aicore__ inline void WaitQKMatmul()
    {
        qkMatmul_.WaitIterateAll();
        qkMatmul_.End();
    }

    __aicore__ inline void WaitSVMatmul()
    {
        svMatmul_.WaitIterateAll();
        svMatmul_.End();
    }

    using CopyFun = MatmulCopyFun<qType, TilingDataType>;

    using QK_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false, LayoutMode::NONE, false,
                                         TPosition::VECOUT>;
    using QK_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, CopyFun::CopyQKA1, CopyFun::CopyQKB1>;
    using QK_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>;
    using QK_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using QK_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    static constexpr auto staticQkTilingCfg = GetMatmulApiTiling<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T>(
        qkMMConfig, MATMUL_L1_SIZE);
    matmul::Matmul<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T, staticQkTilingCfg, QK_MM_CB_T> qkMatmul_;
    
    using SV_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false, LayoutMode::NONE, false,
                                        TPosition::VECOUT>;
    using SV_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using SV_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using SV_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using SV_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, nullptr, CopyFun::CopySVB1>;
    static constexpr auto staticSvTilingCfg = GetMatmulApiTiling<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T>(
        svMMConfig, MATMUL_L1_SIZE);
    matmul::Matmul<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T, staticSvTilingCfg, SV_MM_CB_T> svMatmul_;
};

#endif
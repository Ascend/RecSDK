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

#ifndef MATMUL_CONSTEXPR_H
#define MATMUL_CONSTEXPR_H

#include <cstdint>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"
/* fp16 rab 0的版本，qk和gv的shape是256x256x512,
 qgrad和kgrad的shape是256x512x128, vgrad的shape是256x512x128 */

using namespace AscendC;
using namespace matmul;

namespace HstuDenseBackward {
template <typename qType, class TilingDataType>
class MatmulStriCopyFun {
public:
    __aicore__ inline static void CopyQKA1_Strd(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row,
                                                int col, int useM, int useK, const uint64_t tilingPtr,
                                                const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useM * useK);

        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t headNum = tilingP->headNum;
        int64_t headDim = static_cast<int64_t>(dataPtr);

        int32_t baseM = tilingP->qkMatmul.baseM;
        int32_t baseN = tilingP->qkMatmul.baseN;
        int32_t baseK = tilingP->qkMatmul.baseK;

        uint16_t alignedUseM = AlignUp(useM, ALIGN_16);

        Nd2NzParams param{1, (uint16_t)useM, (uint16_t)useK, 0, (uint16_t)(headNum * headDim), alignedUseM, 1, 0};

        int64_t startIdx = row * baseM * headNum * headDim + col * baseK;
        DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    };

    __aicore__ inline static void CopyQKB1_Strd_Trans(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm,
                                                      int row, int col, int useK, int useN, const uint64_t tilingPtr,
                                                      const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t headNum = tilingP->headNum;
        int64_t headDim = static_cast<int64_t>(dataPtr);

        int32_t baseM = tilingP->qkMatmul.baseM;
        int32_t baseN = tilingP->qkMatmul.baseN;
        int32_t baseK = tilingP->qkMatmul.baseK;

        uint16_t alignedUseN = AlignUp(useN, ALIGN_16);

        Nd2NzParams param{1, (uint16_t)useN, (uint16_t)useK, 0, (uint16_t)(headNum * headDim), alignedUseN, 1, 0};

        int64_t startIdx = col * baseN * headNum * headDim + row * baseK;
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    };

    __aicore__ inline static void CopyQKGradB1_Strd(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row,
        int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t headNum = tilingP->headNum;
        int64_t headDimQK = tilingP->headDimQK;

        int32_t baseM = tilingP->qGradMatmul.baseM;
        int32_t baseN = tilingP->qGradMatmul.baseN;
        int32_t baseK = tilingP->qGradMatmul.baseK;

        uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

        Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useN, 0, (uint16_t)(headNum * headDimQK), alignedUseK, 1, 0};

        int64_t startIdx = row * baseK * headNum * headDimQK + col * baseN;
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    };

    __aicore__ inline static void CopyVGradB1_Strd(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row,
                                                   int col, int useK, int useN, const uint64_t tilingPtr,
                                                   const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t headNum = tilingP->headNum;
        int64_t headDimV = tilingP->headDimV;

        int32_t baseM = tilingP->vGradMatmul.baseM;
        int32_t baseN = tilingP->vGradMatmul.baseN;
        int32_t baseK = tilingP->vGradMatmul.baseK;

        uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

        Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useN, 0, (uint16_t)(headNum * headDimV), alignedUseK, 1, 0};

        int64_t startIdx = row * baseK * headNum * headDimV + col * baseN;
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    };
};
// HeadPadding是为了兼容不同headDim的场景，比如headDim为32、 80、156，需要将headDim填充为64或128 256
template <typename qType, int64_t blockQ, int64_t blockK, int64_t headDimPadding, class TilingDataType>
class MatmulJF16R0Const {
public:
    static constexpr int BLOCK_HEIGHT_256 = 256;
    static constexpr int BLOCK_HEIGHT_128 = 128;
    static constexpr int BASIC_K_32 = 32;
    static constexpr int BASIC_K_64 = 64;
    static constexpr int MAX_BLOCK_DIM = 512;
    static constexpr int MATMUL_L1_SIZE = 524288;  // 512KB

    static constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;

    // FTile Layout
    static constexpr MatmulShapeParams qkOrGvShapeFp16Params = {blockQ,           blockK,           headDimPadding,
                                                                BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_64};

    static constexpr MatmulShapeParams qGradShapeFp16Params = {blockQ,           headDimPadding,   blockK,
                                                               BLOCK_HEIGHT_256, BLOCK_HEIGHT_128, BASIC_K_64};

    static constexpr MatmulShapeParams kGradShapeFp16Params = {blockK,           headDimPadding,   blockQ,
                                                               BLOCK_HEIGHT_256, BLOCK_HEIGHT_128, BASIC_K_64};

    static constexpr MatmulShapeParams vGradShapeFp16Params = {blockK,           headDimPadding,   blockQ,
                                                               BLOCK_HEIGHT_256, BLOCK_HEIGHT_128, BASIC_K_64};

    // MM Config
    static constexpr MatmulQuantParams quantParams = {false, false};
    static constexpr MatmulBatchParams batchParams = {false, BatchMode::NONE};
    static constexpr MatmulFuncParams qkFuncParams = {false, false};
    static constexpr MatmulFuncParams svFuncParams = {false, false};

    static constexpr MatmulConfig mmStaticConfigQKOrGVFp16 =
        GetMMConfig<configMode>(qkOrGvShapeFp16Params, quantParams, batchParams, qkFuncParams);
    static constexpr MatmulConfig mmStaticConfigQGradFp16 =
        GetMMConfig<configMode>(qGradShapeFp16Params, quantParams, batchParams, qkFuncParams);
    static constexpr MatmulConfig mmStaticConfigKGradFp16 =
        GetMMConfig<configMode>(kGradShapeFp16Params, quantParams, batchParams, qkFuncParams);
    static constexpr MatmulConfig mmStaticConfigVGradFp16 =
        GetMMConfig<configMode>(vGradShapeFp16Params, quantParams, batchParams, qkFuncParams);

    __aicore__ static inline void CopyQKA1_Strd(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row,
                                                int col, int useM, int useK, const uint64_t tilingPtr,
                                                const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)));

        uint16_t alignedUseM = AlignUp(useM, ALIGN_16);
        const TilingDataType* tilingP = reinterpret_cast<const TilingDataType*>(tilingPtr);
        int64_t headDim = static_cast<int64_t>(dataPtr);
        const uint16_t headStride = tilingP->headNum * headDim;

        Nd2NzParams param{1, (uint16_t)useM, (uint16_t)useK, 0, headStride, alignedUseM, 1, 0};

        int64_t startIdx = row * qkOrGvShapeFp16Params.basicM * headStride + col * qkOrGvShapeFp16Params.basicK;
        DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    }

    __aicore__ static inline void CopyQKB1_Strd_Trans(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm,
                                                      int row, int col, int useK, int useN, const uint64_t tilingPtr,
                                                      const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)));
        uint16_t alignedUseN = AlignUp(useN, ALIGN_16);
        const TilingDataType* tilingP = reinterpret_cast<const TilingDataType*>(tilingPtr);
        int64_t headDim = static_cast<int64_t>(dataPtr);
        const uint16_t headStride = tilingP->headNum * headDim;
        Nd2NzParams param{1, (uint16_t)useN, (uint16_t)useK, 0, headStride, alignedUseN, 1, 0};

        int64_t startIdx = col * qkOrGvShapeFp16Params.basicN * headStride + row * qkOrGvShapeFp16Params.basicK;
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    };

    __aicore__ static inline void CopyQKGradB1_Strd(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row,
                                                   int col, int useK, int useN, const uint64_t tilingPtr,
                                                   const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)));
        const TilingDataType* tilingP = reinterpret_cast<const TilingDataType*>(tilingPtr);
        const uint16_t headStride = tilingP->headNum * tilingP->headDimQK;
        uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

        Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useN, 0, headStride, alignedUseK, 1, 0};

        int64_t startIdx = row * vGradShapeFp16Params.basicK * headStride + col * vGradShapeFp16Params.basicN;
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    }

    __aicore__ static inline void CopyVGradB1_Strd(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row,
                                                   int col, int useK, int useN, const uint64_t tilingPtr,
                                                   const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)));
        const TilingDataType* tilingP = reinterpret_cast<const TilingDataType*>(tilingPtr);
        const uint16_t headStride = tilingP->headNum * tilingP->headDimV;
        uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

        Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useN, 0, headStride, alignedUseK, 1, 0};

        int64_t startIdx = row * vGradShapeFp16Params.basicK * headStride + col * vGradShapeFp16Params.basicN;
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
    }

    // QkMatmul
    using QK_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using QK_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>;
    using QK_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using QK_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using QK_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, CopyQKA1_Strd, CopyQKB1_Strd_Trans>;

    static constexpr auto staticQkTilingCfg =
        GetMatmulApiTiling<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T>(mmStaticConfigQKOrGVFp16, MATMUL_L1_SIZE);
    using QK_OR_GV_MATMUL =
        matmul::Matmul<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T, staticQkTilingCfg, QK_MM_CB_T>;

    // KGradMatmul
    using KG_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>;
    using KG_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using KG_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using KG_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using KG_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, nullptr, CopyQKGradB1_Strd>;

    static constexpr auto staticKGradTilingCfg =
        GetMatmulApiTiling<KG_MM_A_T, KG_MM_B_T, KG_MM_C_T, KG_MM_BIAS_T>(mmStaticConfigKGradFp16, MATMUL_L1_SIZE);
    using K_GRAD_MATMUL =
        matmul::Matmul<KG_MM_A_T, KG_MM_B_T, KG_MM_C_T, KG_MM_BIAS_T, staticKGradTilingCfg, KG_MM_CB_T>;

    // VGradMatmul
    using VG_MM_A_T =
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true, LayoutMode::NONE, false, TPosition::VECOUT>;
    using VG_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using VG_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using VG_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using VG_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, nullptr, CopyVGradB1_Strd>;

    static constexpr auto staticVGradTilingCfg =
        GetMatmulApiTiling<VG_MM_A_T, VG_MM_B_T, VG_MM_C_T, VG_MM_BIAS_T>(mmStaticConfigVGradFp16, MATMUL_L1_SIZE);
    using V_GRAD_MATMUL =
        matmul::Matmul<VG_MM_A_T, VG_MM_B_T, VG_MM_C_T, VG_MM_BIAS_T, staticVGradTilingCfg, VG_MM_CB_T>;

    // QGradMatmul
    using QG_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using QG_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using QG_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using QG_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using QG_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, nullptr, CopyQKGradB1_Strd>;

    static constexpr auto staticQGradTilingCfg =
        GetMatmulApiTiling<QG_MM_A_T, QG_MM_B_T, QG_MM_C_T, QG_MM_BIAS_T>(mmStaticConfigQGradFp16, MATMUL_L1_SIZE);
    using Q_GRAD_MATMUL =
        matmul::Matmul<QG_MM_A_T, QG_MM_B_T, QG_MM_C_T, QG_MM_BIAS_T, staticQGradTilingCfg, QG_MM_CB_T>;
};
}  // namespace HstuDenseBackward
#endif

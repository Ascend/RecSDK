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

#include <array>
#include <tuple>
#include <cstdint>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"

using namespace HstuForward;

constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;

template <uint32_t m, uint32_t n, uint32_t k,
          uint32_t qkM, uint32_t qkN, uint32_t qkK,
          uint32_t svM, uint32_t svN, uint32_t svK>
struct TileCfgStorage {
    static inline constexpr MatmulQuantParams  quantParams  = {false, false};
    static inline constexpr MatmulBatchParams  batchParams  = {false, BatchMode::NONE};
    static inline constexpr MatmulFuncParams   qkFuncParams = {false, false};
    static inline constexpr MatmulFuncParams   svFuncParams = {false, false};

    static inline constexpr MatmulConfig mmStaticConfigQK =
        GetMMConfig<configMode>(
            MatmulShapeParams{m, n, k, qkM, qkN, qkK},
            quantParams, batchParams, qkFuncParams);

    static inline constexpr MatmulConfig mmStaticConfigSV =
        GetMMConfig<configMode>(
            MatmulShapeParams{m, k, n, svM, svN, svK},
            quantParams, batchParams, svFuncParams);
};

template <uint32_t wid, uint32_t m, uint32_t n, uint32_t k,
          uint32_t qkM, uint32_t qkN, uint32_t qkK,
          uint32_t svM, uint32_t svN, uint32_t svK>
struct TileCfgT {
    static constexpr uint32_t tileM = m;
    static constexpr uint32_t tileN = n;
    static constexpr uint32_t tileK = k;
    static constexpr uint32_t size = wid;

    static constexpr MatmulConfig getQKMatmulConfig()
    {
        return TileCfgStorage<m, n, k, qkM, qkN, qkK, svM, svN, svK>::mmStaticConfigQK;
    }
    static constexpr MatmulConfig getSVMatmulConfig()
    {
        return TileCfgStorage<m, n, k, qkM, qkN, qkK, svM, svN, svK>::mmStaticConfigSV;
    }

    __aicore__ constexpr bool match(uint32_t width, uint32_t blockM, uint32_t blockN, uint32_t blockK) const
    {
        return wid == width && m == blockM && n == blockN && k == blockK;
    }
};

using CfgList = std::tuple<
    TileCfgT<1, 256, 256, 512, 256, 256, 128, 256, 256, 32>,
    TileCfgT<2, 32, 32, 32, 32, 32, 32, 32, 32, 32>,
    TileCfgT<2, 32, 2048, 32, 32, 256, 32, 32, 32, 512>,
    TileCfgT<2, 64, 64, 32, 64, 64, 32, 64, 32, 64>,
    TileCfgT<2, 64, 1024, 32, 64, 256, 32, 64, 32, 256>,
    TileCfgT<2, 128, 128, 32, 128, 128, 32, 128, 32, 128>,
    TileCfgT<2, 128, 512, 32, 128, 256, 32, 128, 32, 128>,
    TileCfgT<2, 256, 256, 32, 256, 256, 32, 128, 32, 128>,
    TileCfgT<2, 64, 128, 32, 64, 128, 32, 64, 32, 128>,
    TileCfgT<2, 128, 256, 32, 128, 256, 32, 128, 32, 128>,
    TileCfgT<2, 32, 32, 64, 32, 32, 64, 32, 64, 32>,
    TileCfgT<2, 32, 2048, 64, 32, 256, 64, 32, 64, 256>,
    TileCfgT<2, 64, 64, 64, 64, 64, 64, 64, 64, 64>,
    TileCfgT<2, 64, 1024, 64, 64, 256, 64, 64, 64, 256>,
    TileCfgT<2, 128, 128, 64, 128, 128, 64, 128, 64, 128>,
    TileCfgT<2, 128, 512, 64, 128, 256, 64, 128, 64, 128>,
    TileCfgT<2, 256, 256, 64, 256, 256, 64, 256, 64, 64>,
    TileCfgT<2, 64, 128, 64, 64, 128, 64, 64, 64, 128>,
    TileCfgT<2, 128, 256, 64, 128, 256, 64, 128, 64, 128>,
    TileCfgT<2, 32, 32, 128, 32, 32, 128, 32, 128, 32>,
    TileCfgT<2, 32, 2048, 128, 32, 256, 64, 32, 128, 128>,
    TileCfgT<2, 64, 64, 128, 64, 64, 128, 64, 128, 64>,
    TileCfgT<2, 64, 1024, 128, 64, 256, 64, 64, 128, 128>,
    TileCfgT<2, 128, 128, 128, 128, 128, 128, 128, 128, 128>,
    TileCfgT<2, 128, 512, 128, 128, 256, 64, 128, 128, 128>,
    TileCfgT<2, 256, 256, 128, 256, 256, 64, 128, 128, 128>,
    TileCfgT<2, 64, 128, 128, 64, 128, 128, 64, 128, 128>,
    TileCfgT<2, 128, 256, 128, 128, 256, 64, 128, 128, 128>,
    TileCfgT<2, 32, 32, 256, 32, 32, 256, 32, 256, 32>,
    TileCfgT<2, 32, 2048, 256, 32, 256, 64, 32, 256, 64>,
    TileCfgT<2, 64, 64, 256, 64, 64, 256, 64, 256, 64>,
    TileCfgT<2, 64, 1024, 256, 64, 256, 64, 64, 256, 64>,
    TileCfgT<2, 128, 128, 256, 128, 128, 128, 128, 256, 64>,
    TileCfgT<2, 128, 512, 256, 128, 256, 64, 128, 256, 64>,
    TileCfgT<2, 256, 256, 256, 256, 256, 64, 256, 256, 64>,
    TileCfgT<2, 64, 128, 256, 64, 128, 128, 64, 256, 64>,
    TileCfgT<2, 128, 256, 256, 128, 256, 64, 128, 256, 64>,
    TileCfgT<2, 32, 32, 512, 32, 32, 512, 32, 256, 32>,
    TileCfgT<2, 32, 2048, 512, 32, 256, 64, 32, 256, 64>,
    TileCfgT<2, 64, 64, 512, 64, 64, 256, 64, 256, 64>,
    TileCfgT<2, 64, 1024, 512, 64, 256, 64, 64, 256, 64>,
    TileCfgT<2, 128, 128, 512, 128, 128, 128, 128, 256, 64>,
    TileCfgT<2, 128, 512, 512, 128, 256, 64, 128, 256, 64>,
    TileCfgT<2, 256, 256, 512, 256, 256, 64, 256, 256, 64>,
    TileCfgT<2, 64, 128, 512, 64, 128, 128, 64, 256, 64>,
    TileCfgT<2, 128, 256, 512, 128, 256, 64, 128, 256, 64>,
    TileCfgT<4, 32, 32, 32, 32, 32, 32, 32, 32, 32>,
    TileCfgT<4, 32, 2048, 32, 32, 256, 32, 32, 32, 256>,
    TileCfgT<4, 64, 64, 32, 64, 64, 32, 64, 32, 64>,
    TileCfgT<4, 64, 1024, 32, 64, 256, 32, 64, 32, 128>,
    TileCfgT<4, 128, 128, 32, 128, 128, 32, 128, 32, 64>,
    TileCfgT<4, 128, 512, 32, 128, 256, 32, 128, 32, 64>,
    TileCfgT<4, 256, 256, 32, 256, 256, 32, 128, 32, 64>,
    TileCfgT<4, 64, 128, 32, 64, 128, 32, 64, 32, 128>,
    TileCfgT<4, 128, 256, 32, 128, 256, 32, 128, 32, 64>,
    TileCfgT<4, 32, 32, 64, 32, 32, 64, 32, 64, 32>,
    TileCfgT<4, 32, 2048, 64, 32, 256, 32, 32, 64, 128>,
    TileCfgT<4, 64, 64, 64, 64, 64, 64, 64, 64, 64>,
    TileCfgT<4, 64, 1024, 64, 64, 256, 32, 64, 64, 128>,
    TileCfgT<4, 128, 128, 64, 128, 128, 64, 128, 64, 64>,
    TileCfgT<4, 128, 512, 64, 128, 256, 32, 128, 64, 64>,
    TileCfgT<4, 256, 256, 64, 256, 256, 32, 256, 64, 32>,
    TileCfgT<4, 64, 128, 64, 64, 128, 64, 64, 64, 128>,
    TileCfgT<4, 128, 256, 64, 128, 256, 32, 128, 64, 64>,
    TileCfgT<4, 32, 32, 128, 32, 32, 128, 32, 128, 32>,
    TileCfgT<4, 32, 2048, 128, 32, 256, 32, 32, 128, 64>,
    TileCfgT<4, 64, 64, 128, 64, 64, 128, 64, 128, 64>,
    TileCfgT<4, 64, 1024, 128, 64, 256, 32, 64, 128, 64>,
    TileCfgT<4, 128, 128, 128, 128, 128, 64, 128, 128, 64>,
    TileCfgT<4, 128, 512, 128, 128, 256, 32, 128, 128, 64>,
    TileCfgT<4, 256, 256, 128, 256, 256, 32, 256, 128, 32>,
    TileCfgT<4, 64, 128, 128, 64, 128, 64, 64, 128, 64>,
    TileCfgT<4, 128, 256, 128, 128, 256, 32, 128, 128, 64>,
    TileCfgT<4, 32, 32, 256, 32, 32, 256, 32, 256, 32>,
    TileCfgT<4, 32, 2048, 256, 32, 256, 32, 32, 256, 32>,
    TileCfgT<4, 64, 64, 256, 64, 64, 128, 64, 256, 32>,
    TileCfgT<4, 64, 1024, 256, 64, 256, 32, 64, 256, 32>,
    TileCfgT<4, 128, 128, 256, 128, 128, 64, 128, 256, 32>,
    TileCfgT<4, 128, 512, 256, 128, 256, 32, 128, 256, 32>,
    TileCfgT<4, 256, 256, 256, 256, 256, 32, 256, 256, 32>,
    TileCfgT<4, 64, 128, 256, 64, 128, 64, 64, 256, 32>,
    TileCfgT<4, 128, 256, 256, 128, 256, 32, 128, 256, 32>,
    TileCfgT<4, 32, 32, 512, 32, 32, 256, 32, 256, 32>,
    TileCfgT<4, 32, 2048, 512, 32, 256, 32, 32, 256, 32>,
    TileCfgT<4, 64, 64, 512, 64, 64, 128, 64, 256, 32>,
    TileCfgT<4, 64, 1024, 512, 64, 256, 32, 64, 256, 32>,
    TileCfgT<4, 128, 128, 512, 128, 128, 64, 128, 256, 32>,
    TileCfgT<4, 128, 512, 512, 128, 256, 32, 128, 256, 32>,
    TileCfgT<4, 256, 256, 512, 256, 256, 32, 256, 256, 32>,
    TileCfgT<4, 64, 128, 512, 64, 128, 64, 64, 256, 32>,
    TileCfgT<4, 128, 256, 512, 128, 256, 32, 128, 256, 32>
>;

template <std::size_t idx, uint32_t wid, uint32_t blockM, uint32_t blockN, uint32_t blockK>
__aicore__ constexpr auto FindShapeImpl() noexcept {
    using tilingTemp = std::tuple_element_t<idx, CfgList>;

    if constexpr (idx >= std::tuple_size_v<CfgList>) {
        return tilingTemp{};
    }
    else if constexpr (tilingTemp::size == wid && tilingTemp::tileM == blockM &&
                       tilingTemp::tileN == blockN && tilingTemp::tileK == blockK) {
        return tilingTemp{};
    }
    else if constexpr (idx + 1 < std::tuple_size_v<CfgList>) {
        return FindShapeImpl<idx + 1, wid, blockM, blockN, blockK>();
    }
    else {
        return tilingTemp{};
    }
}

template <uint32_t wid, uint32_t blockM, uint32_t blockN, uint32_t blockK>
__aicore__ constexpr auto FindShape() noexcept {
    return FindShapeImpl<0, wid, blockM, blockN, blockK>();
}

template <typename TraitParams, typename TilingDataType>
class MatmulCopyFun {
public:
    __aicore__ inline static void CopyQKA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col,
        int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<typename TraitParams::qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ typename TraitParams::qType*>
            (const_cast<__gm__ void*>(gm)), useM * useK);
        int blockLen = useM * useK;

        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t dim = tilingP->dim;
        int64_t headNum = tilingP->headNum;
        int32_t baseM = TraitParams::GetQKMatmulConfig().basicM;
        int32_t baseK = TraitParams::GetQKMatmulConfig().basicK;

        auto alignOfM = AlignUp(useM, ALIGN_16);
        Nd2NzParams param = {
            1, static_cast<uint16_t>(useM), static_cast<uint16_t>(useK), 0,
            static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfM), 1, 0
        };

        int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNum * static_cast<int64_t>(baseM) +
                            static_cast<int64_t>(col) * static_cast<int64_t>(baseK);
        DataCopy(aMatrix.ReinterpretCast<typename TraitParams::qType>(), globalGt[offsetOfGt], param);
    };

    __aicore__ inline static void CopyQKB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col,
        int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<typename TraitParams::qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ typename TraitParams::qType*>
            (const_cast<__gm__ void*>(gm)), useN * useK);

        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t dim = tilingP->dim;
        int32_t headNumK = static_cast<int32_t>(dataPtr);
        int32_t baseN = TraitParams::GetQKMatmulConfig().basicN;
        int32_t baseK = TraitParams::GetQKMatmulConfig().basicK;

        auto alignOfN = AlignUp(useN, ALIGN_16);
        Nd2NzParams param = {
            1, static_cast<uint16_t>(useN), static_cast<uint16_t>(useK), 0,
            static_cast<uint16_t>(dim * headNumK), static_cast<uint16_t>(alignOfN), 1, 0
        };

        int64_t offsetOfGt = static_cast<int64_t>(col) * dim * headNumK * static_cast<int64_t>(baseN) +
                            static_cast<int64_t>(row) * static_cast<int64_t>(baseK);
        DataCopy(bMatrix.ReinterpretCast<typename TraitParams::qType>(), globalGt[offsetOfGt], param);
    };

    __aicore__ inline static void CopySVB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col,
        int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<typename TraitParams::qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ typename TraitParams::qType*>
            (const_cast<__gm__ void*>(gm)), useN * useK);

        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t dim = tilingP->vDim;
        int32_t headNumK = static_cast<int32_t>(dataPtr);
        int32_t baseN = TraitParams::GetSVMatmulConfig().basicN;
        int32_t baseK = TraitParams::GetSVMatmulConfig().basicK;

        uint16_t alignOfK = 0;
        if constexpr (std::is_same<typename TraitParams::qType, fp8_e4m3fn_t>::value) {
            alignOfK = AlignUp(useK, ALIGN_32);
        } else {
            alignOfK = AlignUp(useK, ALIGN_16);
        }

        Nd2NzParams param = {
            1, static_cast<uint16_t>(useK), static_cast<uint16_t>(useN), 0,
            static_cast<uint16_t>(dim * headNumK), static_cast<uint16_t>(alignOfK), 1, 0
        };

        int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNumK * static_cast<int64_t>(baseK) +
                            static_cast<int64_t>(col) * static_cast<int64_t>(baseN);
        DataCopy(bMatrix.ReinterpretCast<typename TraitParams::qType>(), globalGt[offsetOfGt], param);
    };
};

#endif

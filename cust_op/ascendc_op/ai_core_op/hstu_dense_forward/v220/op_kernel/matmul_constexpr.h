/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"

using namespace HstuForward;

constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;

struct MatmulArgs {
    int64_t leftOffset {0};
    int64_t rightOffset {0};
    int64_t outOffset {0};
    uint32_t m {0};
    uint32_t n {0};
    uint32_t k {0};
    uint64_t headNum {0};
    uint8_t isAtomicAdd {0};
};

#ifdef SUPPORT_950
    // FP16/BFP16
    constexpr MatmulShapeParams qkShapeFp16Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_64
    };

    constexpr MatmulShapeParams svShapeFp16Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, BASIC_K_64
    };

    // FP32
    constexpr MatmulShapeParams qkShapeFp32Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, BASIC_K_32
    };

    constexpr MatmulShapeParams svShapeFp32Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_128, MAX_BLOCK_DIM, BASIC_K_32
    };

    // FP8
    constexpr MatmulShapeParams qkShapeFp8Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, 128
    };

    constexpr MatmulShapeParams svShapeFp8Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, 32
    };
#else
    // FP16/BFP16
    constexpr MatmulShapeParams qkShapeFp16Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_64
    };

    constexpr MatmulShapeParams svShapeFp16Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_64
    };

    // FP32
    constexpr MatmulShapeParams qkShapeFp32Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_32
    };

    constexpr MatmulShapeParams svShapeFp32Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_32
    };
    // FP8
    constexpr MatmulShapeParams qkShapeFp8Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, 128
    };

    constexpr MatmulShapeParams svShapeFp8Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, 32
    };
#endif

constexpr MatmulQuantParams quantParams = {false, false};
constexpr MatmulBatchParams batchParams = {false, BatchMode::NONE};
constexpr MatmulFuncParams qkFuncParams = {false, false};
constexpr MatmulFuncParams svFuncParams = {false, false};

constexpr MatmulConfig mmStaticConfigQKFp8 =
    GetMMConfig<configMode>(qkShapeFp8Params, quantParams, batchParams, qkFuncParams);
constexpr MatmulConfig mmStaticConfigSVFp8 =
    GetMMConfig<configMode>(svShapeFp8Params, quantParams, batchParams, svFuncParams);

constexpr MatmulConfig mmStaticConfigQKFp16 =
    GetMMConfig<configMode>(qkShapeFp16Params, quantParams, batchParams, qkFuncParams);
constexpr MatmulConfig mmStaticConfigSVFp16 =
    GetMMConfig<configMode>(svShapeFp16Params, quantParams, batchParams, svFuncParams);

constexpr MatmulConfig mmStaticConfigQKFp32 =
    GetMMConfig<configMode>(qkShapeFp32Params, quantParams, batchParams, qkFuncParams);
constexpr MatmulConfig mmStaticConfigSVFp32 =
    GetMMConfig<configMode>(svShapeFp32Params, quantParams, batchParams, svFuncParams);

template<typename qType, typename TilingDataType>
class MatmulCopyFun {
public:
    __aicore__ inline static void CopyQKA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm,
        int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useM * useK);
        int blockLen = useM * useK;
    
        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t dim = tilingP->dim;
        int64_t headNum = tilingP->headNum;
        int32_t baseM = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicM : mmStaticConfigQKFp16.basicM;
        int32_t baseK = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicK : mmStaticConfigQKFp16.basicK;
    
        auto alignOfM = AlignUp(useM, ALIGN_16);
        Nd2NzParams param = {
            1, static_cast<uint16_t>(useM), static_cast<uint16_t>(useK), 0,
            static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfM), 1, 0
        };
    
        int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNum * static_cast<int64_t>(baseM) +
                             static_cast<int64_t>(col) * static_cast<int64_t>(baseK);
        DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
    };
    
    __aicore__ inline static void CopyQKB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm,
        int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);
    
        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t dim = tilingP->dim;
        int32_t headNumK = static_cast<int32_t>(dataPtr);
        int32_t baseN = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicN : mmStaticConfigQKFp16.basicN;
        int32_t baseK = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicK : mmStaticConfigQKFp16.basicK;
    
        auto alignOfN = AlignUp(useN, ALIGN_16);
        Nd2NzParams param = {
            1, static_cast<uint16_t>(useN), static_cast<uint16_t>(useK), 0,
            static_cast<uint16_t>(dim * headNumK), static_cast<uint16_t>(alignOfN), 1, 0
        };
    
        int64_t offsetOfGt = static_cast<int64_t>(col) * dim * headNumK * static_cast<int64_t>(baseN) +
                             static_cast<int64_t>(row) * static_cast<int64_t>(baseK);
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
    };
    
    __aicore__ inline static void CopySVB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm,
        int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<qType> globalGt;
        globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);
    
        TilingDataType* tilingP = reinterpret_cast<TilingDataType*>(tilingPtr);
        int64_t dim = tilingP->vDim;
        int32_t headNumK = static_cast<int32_t>(dataPtr);
        int32_t baseN = std::is_same<qType, float>::value ? mmStaticConfigSVFp32.basicN : mmStaticConfigSVFp16.basicN;
        int32_t baseK = std::is_same<qType, float>::value ? mmStaticConfigSVFp32.basicK : mmStaticConfigSVFp16.basicK;
        auto alignOfK = AlignUp(useK, ALIGN_16);
    
        Nd2NzParams param = {
            1, static_cast<uint16_t>(useK), static_cast<uint16_t>(useN), 0,
            static_cast<uint16_t>(dim * headNumK), static_cast<uint16_t>(alignOfK), 1, 0
        };
    
        int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNumK * static_cast<int64_t>(baseK) +
                             static_cast<int64_t>(col) * static_cast<int64_t>(baseN);
        DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
    };
};

#endif  // MATMUL_CONSTEXPR_H
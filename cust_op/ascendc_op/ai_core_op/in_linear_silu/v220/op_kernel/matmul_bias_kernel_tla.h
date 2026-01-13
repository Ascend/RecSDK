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
==============================================================================*/

#ifndef MATMUL_BIAS_KERNEL_TLA_H
#define MATMUL_BIAS_KERNEL_TLA_H

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
namespace InLinearSilu_Kernel {
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MatmulBiasTlaKernel {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementL = typename BlockMmad::ElementA;
    using ElementBias = typename BlockMmad::ElementBias;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;
        EpilogueParams epilogueParams;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA const& layoutA_, GM_ADDR ptrB_,
               LayoutB const& layoutB_, GM_ADDR ptrC_, LayoutC const& layoutC_, GM_ADDR ptrBias_,
               EpilogueParams const& epilogueParams_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrBias(ptrBias_),
              epilogueParams(epilogueParams_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    MatmulBiasTlaKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);
        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
        // data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias);
        }

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(params.problemShape.m(), params.problemShape.n());
        layout::RowMajor layoutCEpilogue(params.problemShape.m(), params.problemShape.n());
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});
        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            aicoreIndex /= AscendC::GetSubBlockNum();
        }
        uint32_t loopStep = AscendC::GetBlockNum();

        GemmCoord blockShape(L1_TILE_M, L1_TILE_N, L1_TILE_K);
        // 统一循环
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += loopStep) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            if ASCEND_IS_AIC {
                auto tensorBlockA =
                    GetTile(tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB =
                    GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                // Compute block-scoped matrix multiply-add
                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                } else {
                    auto tensorBlockBias = GetTile(tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N),
                                                   tla::MakeShape(actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockBias);
                }
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            } else if ASCEND_IS_AIV {
                auto gmBlockC = gmC[layoutCEpilogue.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
                auto layoutBlockC = layoutCEpilogue.GetTileLayout(actualBlockShape.GetCoordMN());
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAicFinishStore);
                blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlockC, layoutBlockC);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};
}  // namespace InLinearSilu_Kernel

#endif
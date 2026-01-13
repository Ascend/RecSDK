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

#ifndef MATMUL_BIAS_KERNEL_H
#define MATMUL_BIAS_KERNEL_H

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

using namespace Catlass;
namespace InLinearSilu_Kernel {
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MatmulBiasKernel {
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
    using ElementBias = typename BlockMmad::ElementBias;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

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
        GM_ADDR ptrLinear;
        bool requiresGrad;
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
    MatmulBiasKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias);

        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
        int blockId = 0;
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            int64_t gmOffsetBias = blockCoord.n() * L1TileShape::N;

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], params.layoutC,
                      gmBias[gmOffsetBias], actualBlockShape);

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(crossCorePingPong[blockId % 2]);
            blockId++;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        // Loop through the epilogue calculations of each basic block
        GemmCoord blockShape = L1TileShape::ToCoord();
        int blockId = 0;
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            // Get the data and layout of C under the current basic block
            auto gmBlockC = gmC[layoutC.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());

            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(crossCorePingPong[blockId % 2]);
            blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlockC, layoutBlockC);
            AscendC::PipeBarrier<PIPE_ALL>();
            blockId++;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::CrossCoreFlagWithReverse<> crossCorePingPong0{2, 3};
    Arch::CrossCoreFlagWithReverse<> crossCorePingPong1{4, 5};
    Arch::CrossCoreFlagWithReverse<> crossCorePingPong[2] = {crossCorePingPong0, crossCorePingPong1};
    Arch::Resource<ArchTag> resource;
};
}  // namespace InLinearSilu_Kernel

#endif
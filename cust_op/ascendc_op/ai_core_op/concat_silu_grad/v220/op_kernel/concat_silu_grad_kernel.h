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
#ifndef CONCAT_SILU_GRAD_KERNEL
#define CONCAT_SILU_GRAD_KERNEL

#include <cstdio>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

using namespace Catlass;
namespace PROJECT_NAMESPACE {

// Template for matmul add kernel. Compute D = A * B + X
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class L1TileShape_>
class ConcatSiluGradKernel {
public:
    using L1TileShape = L1TileShape_;
    using BlockEpilogue = BlockEpilogue_;
    using ArchTag = typename BlockEpilogue::ArchTag;
    using ElementT = typename BlockEpilogue::ElementT;
    using LayoutT = typename BlockEpilogue::LayoutT;
    using EpilogueParams = typename BlockEpilogue::Params;
    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrSiluInput;
        GM_ADDR ptrGradSiluInput;
        LayoutT layoutT;
        EpilogueParams epilogueParams;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const& problemShape_, GM_ADDR ptrSiluInput_, GM_ADDR ptrGradSiluInput_,
               LayoutT const& layoutT_, EpilogueParams const& epilogueParams_)
            : problemShape(problemShape_),
              ptrSiluInput(ptrSiluInput_),
              ptrGradSiluInput(ptrGradSiluInput_),
              layoutT(layoutT_),
              epilogueParams(epilogueParams_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    ConcatSiluGradKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
    }
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        // Represent the full gm
        LayoutT layoutT = params.layoutT;
        AscendC::GlobalTensor<ElementT> gmSiluInput;
        gmSiluInput.SetGlobalBuffer((__gm__ ElementT*)params.ptrSiluInput);
        AscendC::GlobalTensor<ElementT> gmGradSiluInput;
        gmGradSiluInput.SetGlobalBuffer((__gm__ ElementT*)params.ptrGradSiluInput);

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        // Loop through the epilogue calculations of each basic block
        GemmCoord blockShape = L1TileShape::ToCoord();
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            // Get the data and layout of C under the current basic block
            auto blockOffset = layoutT.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN());
            auto gmBlockSiluInput = gmSiluInput[blockOffset];
            auto gmBlockGradSiluInput = gmGradSiluInput[blockOffset];
            auto layoutBlockSilu = layoutT.GetTileLayout(actualBlockShape.GetCoordMN());

            blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlockSiluInput, gmBlockGradSiluInput,
                          layoutBlockSilu);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::CrossCoreFlagWithReverse<> crossCorePingPong0{2, 3};
    Arch::CrossCoreFlagWithReverse<> crossCorePingPong1{4, 5};
    Arch::CrossCoreFlagWithReverse<> crossCorePingPong[2] = {crossCorePingPong0, crossCorePingPong1};
    Arch::Resource<ArchTag> resource;
};

}  // namespace PROJECT_NAMESPACE

#endif

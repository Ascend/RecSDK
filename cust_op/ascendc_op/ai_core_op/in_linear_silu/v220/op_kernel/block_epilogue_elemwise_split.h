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

#ifndef _BLOCK_EPILOGUE_ELEMWISE_SPLIT_H
#define _BLOCK_EPILOGUE_ELEMWISE_SPLIT_H

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace InLinearSilu_Kernel {
struct MyEpilogueElemWiseSplit {
    using ArchTag = Catlass::Arch::ARCH_CODE;
    static constexpr uint32_t OPERANDS_NUM = 2;
};
}  // namespace InLinearSilu_Kernel

namespace Catlass::Epilogue::Block {
template <class CType_, class DType_, class TileElemWiseEpilogue_, class TileCopy_>
class BlockEpilogue<InLinearSilu_Kernel::MyEpilogueElemWiseSplit, CType_, DType_, TileElemWiseEpilogue_, TileCopy_> {
public:
    // Type aliases
    using DispatchPolicy = InLinearSilu_Kernel::MyEpilogueElemWiseSplit;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogue = TileElemWiseEpilogue_;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using ElementE = typename TileElemWiseEpilogue::ElementCompute;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogue::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

    using ElementOut = ElementD;
    using LayoutComputeInUb = layout::RowMajor;

    static_assert(std::is_same_v<ElementC, ElementD>, "Element type of C must be equal than Element type of D");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutD, layout::RowMajor>,
                  "Layout type of C, D must be RowMajor");
    static_assert(std::is_same_v<typename TileElemWiseEpilogue::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(COMPUTE_LENGTH * (OPERANDS_NUM * sizeof(ElementC) + sizeof(ElementD)), "UB out of bounds");

    struct Params {
        GM_ADDR ptrLinear;
        GM_ADDR ptrUVQK[SPLIT_NUM];
        LayoutD layoutD;
        int64_t splitList[SPLIT_NUM];
        int64_t splitOffsets[SPLIT_NUM + 1];
        LayoutD layoutUVQK[SPLIT_NUM];
        bool requiresGrad;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrLinear_, GM_ADDR* ptrUVQK_, LayoutD const& layoutD_, const int64_t* splitList_,
               bool requiresGrad_)
            : ptrLinear(ptrLinear_),
              layoutD(layoutD_),
              requiresGrad(requiresGrad_)
        {
            splitOffsets[0] = 0;
            auto m = layoutD_.shape(0);
            for (int i = 0; i < 4; ++i) {
                ptrUVQK[i] = ptrUVQK_[i];
                auto splitN = splitList_[i];
                layoutUVQK[i] = LayoutD(m, splitN);
                splitList[i] = splitN;
                splitOffsets[i + 1] = splitOffsets[i] + splitN;
            }
        }
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params) : params(params)
    {
        uint32_t offset = 0;
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += COMPUTE_LENGTH * sizeof(ElementC);

        ubD = resource.ubBuf.template GetBufferByByte<ElementD>(offset);
        offset += COMPUTE_LENGTH * sizeof(ElementD);

        if constexpr (!std::is_same_v<ElementC, ElementE>) {
            ubEi = resource.ubBuf.template GetBufferByByte<ElementE>(offset);
            offset += COMPUTE_LENGTH * sizeof(ElementE);
        }
        if constexpr (!std::is_same_v<ElementE, ElementD>) {
            ubEo = resource.ubBuf.template GetBufferByByte<ElementE>(offset);
            offset += COMPUTE_LENGTH * sizeof(ElementE);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(GemmCoord const& blockShapeMNK, GemmCoord const& blockCoordMNK,
                    GemmCoord const& actualBlockShapeMNK, AscendC::GlobalTensor<ElementC> const& gmBlockC,
                    LayoutD const& layoutBlockC)
    {
        // Calculate the offset of the current block
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        // Calculate the offset and the shape of the current subblock
        MatrixCoord subblockShape{CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
                                  actualBlockShape.column()};

        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape =
            MatrixCoord::Min(subblockShape, actualBlockShape - subblockCoord * subblockShape);
        if (actualSubblockShape.row() == 0) {
            return;
        }
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        // Get the data and layout of C
        auto gmSubblockC = gmBlockC[layoutBlockC.GetOffset(subblockOffset)];
        auto layoutSubblockC = layoutBlockC.GetTileLayout(actualSubblockShape);

        // Get the layout on UB
        auto ubTileStride = MakeCoord(static_cast<int64_t>(actualSubblockShape.column()), 1L);
        LayoutComputeInUb layoutComputeInUb{actualSubblockShape, ubTileStride};

        // Copy the data of C
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        copyGmToUbC(ubC, gmSubblockC, layoutComputeInUb, layoutSubblockC);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // Perform epilogue calculation
        auto row = actualSubblockShape.row();
        auto column = actualBlockShape.column();

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // cast for input
        AscendC::LocalTensor<ElementE> ubEiTmp = ubEi;
        AscendC::LocalTensor<ElementE> ubEoTmp = ubEo;
        if constexpr (!std::is_same_v<ElementC, ElementE>) {
            AscendC::Cast(ubEi, ubC, AscendC::RoundMode::CAST_NONE, row * column);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            ubEiTmp = ubC;
        }

        if constexpr (std::is_same_v<ElementE, ElementD>) {
            ubEoTmp = ubD;
        }

        tileEpilogue(ubEoTmp, ubEiTmp, row * column);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        // cast for output
        if constexpr (!std::is_same_v<ElementE, ElementD>) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubD, ubEoTmp, AscendC::RoundMode::CAST_RINT, row * column);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // Copy the data of D
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // TO SPLIT UVQK
        auto gmSubblockCoordD = blockOffset + subblockOffset;
        auto blockColumnOffset = blockOffset.column();
        for (int i = 0; i < 4; ++i) {
            int64_t start = tla::max(params.splitOffsets[i], blockColumnOffset);
            int64_t end = tla::min(params.splitOffsets[i + 1], blockColumnOffset + column);
            if (end <= start) {
                // UVQK和本UB不重叠
                continue;
            }

            // 重叠的column数
            uint32_t columnUVQKOverLap = end - start;

            // 本UVQK对应的UB
            auto ubDOffset = start - gmSubblockCoordD.column();
            auto tileLayoutUbD =
                layoutComputeInUb.GetTileLayout(MatrixCoord(static_cast<uint32_t>(row), columnUVQKOverLap));

            // 本UVQK对应的GM
            AscendC::GlobalTensor<ElementD> gmD;
            gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params.ptrUVQK[i]));
            MatrixCoord gmSubblockUVQKCoord =
                MatrixCoord(static_cast<uint32_t>(gmSubblockCoordD.row()),
                            static_cast<uint32_t>(gmSubblockCoordD.column() - params.splitOffsets[i] + ubDOffset));
            auto layoutUVQK = params.layoutUVQK[i];
            auto tileLayoutGmUVQK =
                layoutUVQK.GetTileLayout(MatrixCoord(static_cast<uint32_t>(row), columnUVQKOverLap));
            auto gmSubblockUVQK = gmD[layoutUVQK.GetOffset(gmSubblockUVQKCoord)];

            copyUbToGmD(gmSubblockUVQK, ubD[ubDOffset], tileLayoutGmUVQK, tileLayoutUbD);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

private:
    Params params;
    AscendC::LocalTensor<ElementC> ubC;
    AscendC::LocalTensor<ElementD> ubD;
    AscendC::LocalTensor<ElementE> ubEi;  // 可选
    AscendC::LocalTensor<ElementE> ubEo;  // 可选

    TileElemWiseEpilogue tileEpilogue;
    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};
}  // namespace Catlass::Epilogue::Block

#endif
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
#ifndef CONCAT_SILU_GRAD_BLOCK_EPILOGUE
#define CONCAT_SILU_GRAD_BLOCK_EPILOGUE

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace PROJECT_NAMESPACE {

struct EpilogueConcatSiluGrad {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t OPERANDS_NUM = 2;
};
}  // namespace PROJECT_NAMESPACE

namespace Catlass::Epilogue::Block {
using namespace PROJECT_NAMESPACE;
template <class CType_, class DType_, class TileElemWiseEpilogue_, class TileCopy_>
class BlockEpilogue<PROJECT_NAMESPACE::EpilogueConcatSiluGrad, CType_, DType_, TileElemWiseEpilogue_, TileCopy_> {
public:
    // Type aliases
    using DispatchPolicy = PROJECT_NAMESPACE::EpilogueConcatSiluGrad;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementT = typename CType_::Element;
    using LayoutT = typename CType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogue = TileElemWiseEpilogue_;
    using ElementCompute = typename TileElemWiseEpilogue::ElementCompute;

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogue::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

    static constexpr int32_t eventSiluI = 1;
    static constexpr int32_t eventGradSiluI = 2;

    using LayoutComputeInUb = layout::RowMajor;

    // Check the element type of C and D
    static_assert(std::is_same_v<ElementC, ElementD>, "Element type of C, D must be same");

    static_assert(std::is_same_v<ElementCompute, float>, "Element type of ElementCompute must be float");

    // Check the layout type of C and D
    static_assert(std::is_same_v<LayoutT, layout::RowMajor>, "Layout type of C, D must be RowMajor");

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogue::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");

    // Check if compute length is valid
    static_assert(COMPUTE_LENGTH * (5 * sizeof(ElementCompute) + sizeof(ElementT)) <= ArchTag::UB_SIZE,
                  "UB out of bounds");

    // Epilogue params definition
    struct Params {
        LayoutT layoutSilu;
        GM_ADDR ptrUVQK[4];
        int64_t splitList[4];
        int64_t splitOffsets[SPLIT_NUM + 1];
        LayoutT layoutUVQK[SPLIT_NUM];

        CATLASS_HOST_DEVICE
        Params(GM_ADDR* inputGradAddr, LayoutT const& layoutSiluParam, const int64_t* splitListParam)
            : layoutSilu(layoutSiluParam)
        {
            splitOffsets[0] = 0;
            auto m = layoutSiluParam.shape(0);
            for (int i = 0; i < 4; i++) {
                ptrUVQK[i] = inputGradAddr[i];
                auto sliptN = splitListParam[i];
                layoutUVQK[i] = LayoutT(m, sliptN);
                splitList[i] = sliptN;
                splitOffsets[i + 1] = splitOffsets[i] + sliptN;
            }
        }
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params) : params(params)
    {
        // 真实UB分配
        uint32_t offset = 0;
        ubComI = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
        offset += COMPUTE_LENGTH * sizeof(ElementCompute);
        ubTmp1 = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
        offset += COMPUTE_LENGTH * sizeof(ElementCompute);
        ubTmp2 = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
        offset += COMPUTE_LENGTH * sizeof(ElementCompute);
        ubOnes = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
        offset += COMPUTE_LENGTH * sizeof(ElementCompute);
        ubGradComI = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
        offset += COMPUTE_LENGTH * sizeof(ElementCompute);
        if constexpr (!std::is_same_v<ElementCompute, ElementT>) {
            ubGradSiluO = resource.ubBuf.template GetBufferByByte<ElementT>(offset);
            offset += COMPUTE_LENGTH * sizeof(ElementT);
        }

        // 逻辑UB对应
        ubMulsO = ubAdd1O = ubExpO = ubSigmoid = ubTmp1;
        ubSubO = ubMul1O = ubAdd2O = ubMul2O = ubTmp2;
        ubComO = ubGradComI;

        if constexpr (std::is_same_v<ElementCompute, ElementT>) {
            ubGradSiluO = ubGradComI;
            ubGradSiluI = ubGradComI;
            ubSiluI = ubComI;
        } else {
            ubGradSiluI = ubGradComI.template ReinterpretCast<ElementT>();
            ubSiluI = ubTmp1.template ReinterpretCast<ElementT>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventSiluI);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventGradSiluI);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventGradSiluI);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventSiluI);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventGradSiluI);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventGradSiluI);
    }

    CATLASS_DEVICE
    void InitOnes(int32_t length)
    {
        Duplicate(ubOnes, static_cast<ElementCompute>(1.0), length);
    }

    CATLASS_DEVICE
    void operator()(GemmCoord const& blockShapeMNK, GemmCoord const& blockCoordMNK,
                    GemmCoord const& actualBlockShapeMNK, AscendC::GlobalTensor<ElementT> const& gmBlockSiluInput,
                    AscendC::GlobalTensor<ElementT> const& gmBlockGradSiluInput, LayoutT const& layoutBlockSilu)
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
        auto gmSublockSiluInput = gmBlockSiluInput[layoutBlockSilu.GetOffset(subblockOffset)];
        auto gmSublockGradSiluInput = gmBlockGradSiluInput[layoutBlockSilu.GetOffset(subblockOffset)];
        auto layoutSubblockSilu = layoutBlockSilu.GetTileLayout(actualSubblockShape);

        // Get the layout on UB
        auto ubTileStride = MakeCoord(static_cast<int64_t>(actualSubblockShape.column()), 1L);
        LayoutComputeInUb layoutComputeInUb{actualSubblockShape, ubTileStride};

        auto row = actualSubblockShape.row();
        auto column = actualBlockShape.column();
        auto length = row * column;

        // Copy the data of SiluInput
        {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventSiluI);
            copyGmToUbC(ubSiluI, gmSublockSiluInput, layoutComputeInUb, layoutSubblockSilu);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventSiluI);
            if constexpr (!std::is_same_v<ElementCompute, ElementT>) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventSiluI);
                AscendC::Cast(ubComI, ubSiluI, AscendC::RoundMode::CAST_NONE, length);
                AscendC::SetFlag<AscendC::HardEvent::V_V>(eventSiluI);
            }
        }

        // Copy the data of GradSiluInput
        {
            // TO SPLIK UVQK
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventGradSiluI);

            auto gmSubblockCoordD = blockOffset + subblockOffset;
            auto blockColumnOffset = blockOffset.column();
            for (int i = 0; i < 4; i++) {
                int64_t start = max(params.splitOffsets[i], blockColumnOffset);
                int64_t end = min(params.splitOffsets[i + 1], blockColumnOffset + column);
                if (end <= start) {
                    // UVQK和本UB不重叠
                    continue;
                }

                // 重叠的column数
                uint32_t columnUVQKOverLap = end - start;

                // 本UVQK对应的UB
                auto ubDOffset = start - gmSubblockCoordD.column();
                auto tileLayoutUbD = layoutComputeInUb.GetTileLayout(MatrixCoord((uint32_t)row, columnUVQKOverLap));

                // 本UVQK对应的GM
                AscendC::GlobalTensor<ElementD> gmD;
                gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params.ptrUVQK[i]));
                MatrixCoord gmSubblockUVQKCoord =
                    MatrixCoord((uint32_t)(gmSubblockCoordD.row()),
                                (uint32_t)(gmSubblockCoordD.column() - params.splitOffsets[i] + ubDOffset));
                auto layoutUVQK = params.layoutUVQK[i];
                auto tileLayoutGmUVQK = layoutUVQK.GetTileLayout(MatrixCoord((uint32_t)row, columnUVQKOverLap));
                auto gmSubblockUVQK = gmD[layoutUVQK.GetOffset(gmSubblockUVQKCoord)];
                copyGmToUbC(ubGradSiluO[ubDOffset], gmSubblockUVQK, tileLayoutUbD, tileLayoutGmUVQK);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventGradSiluI);
            if constexpr (!std::is_same_v<ElementCompute, ElementT>) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventGradSiluI);
                AscendC::Cast(ubGradComI, ubGradSiluO, AscendC::RoundMode::CAST_NONE, length);
                AscendC::SetFlag<AscendC::HardEvent::V_V>(eventGradSiluI);
            }
        }

        // 计算
        {
            if constexpr (!std::is_same_v<ElementCompute, ElementT>) {
                AscendC::WaitFlag<AscendC::HardEvent::V_V>(eventSiluI);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventSiluI);
            }
            // sigmoid
            Muls(ubMulsO, ubComI, (ElementCompute)-1, length);
            AscendC::PipeBarrier<PIPE_V>();

            Exp(ubExpO, ubMulsO, length);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(ubAdd1O, ubExpO, (ElementCompute)1, length);
            AscendC::PipeBarrier<PIPE_V>();

            InitOnes(length);
            AscendC::PipeBarrier<PIPE_V>();

            Div(ubSigmoid, ubOnes, ubAdd1O, length);
            AscendC::PipeBarrier<PIPE_V>();
            Sub(ubSubO, ubOnes, ubSigmoid, length);
            AscendC::PipeBarrier<PIPE_V>();

            Mul(ubMul1O, ubSubO, ubComI, length);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventSiluI);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(ubAdd2O, ubMul1O, (ElementCompute)1, length);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(ubMul2O, ubAdd2O, ubSigmoid, length);
            AscendC::PipeBarrier<PIPE_V>();

            if constexpr (!std::is_same_v<ElementCompute, ElementT>) {
                AscendC::WaitFlag<AscendC::HardEvent::V_V>(eventGradSiluI);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventGradSiluI);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventGradSiluI);
            Mul(ubComO, ubMul2O, ubGradComI, length);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventGradSiluI);
            // AscendC::DumpTensor(ubComO, 11, 16);
            if constexpr (!std::is_same_v<ElementCompute, ElementT>) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(ubGradSiluI, ubComO, AscendC::RoundMode::CAST_RINT, length);
                // AscendC::DumpTensor(ubGradSiluI, 12, 16);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }

        // copy out
        {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGmD(gmSublockGradSiluInput, ubGradSiluI, layoutSubblockSilu, layoutComputeInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventGradSiluI);
        }
    }

private:
    Params params;
    // 输入
    AscendC::LocalTensor<ElementT> ubSiluI;
    AscendC::LocalTensor<ElementT> ubGradSiluI;

    // Silu Grad Tensor
    AscendC::LocalTensor<ElementCompute> ubMulsO;
    AscendC::LocalTensor<ElementCompute> ubExpO;
    AscendC::LocalTensor<ElementCompute> ubAdd1O;
    AscendC::LocalTensor<ElementCompute> ubSigmoid;
    AscendC::LocalTensor<ElementCompute> ubSubO;
    AscendC::LocalTensor<ElementCompute> ubMul1O;
    AscendC::LocalTensor<ElementCompute> ubAdd2O;
    AscendC::LocalTensor<ElementCompute> ubMul2O;
    AscendC::LocalTensor<ElementCompute> ubComO;

    // 实际Tensor分配
    AscendC::LocalTensor<ElementCompute> ubComI;
    AscendC::LocalTensor<ElementCompute> ubTmp1;
    AscendC::LocalTensor<ElementCompute> ubTmp2;
    AscendC::LocalTensor<ElementCompute> ubOnes;
    AscendC::LocalTensor<ElementCompute> ubGradComI;
    AscendC::LocalTensor<ElementT> ubGradSiluO;  // 可选

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block

#endif

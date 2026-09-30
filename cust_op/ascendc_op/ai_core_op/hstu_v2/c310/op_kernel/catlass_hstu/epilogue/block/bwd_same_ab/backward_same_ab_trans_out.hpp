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
============================================================================== */

#pragma once

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"

namespace Catlass::Kernel::SameAB {

// Same AB dV/dK transpose output. Each instance is bound to its own FP32
// Fixpipe input and cross-core ready/free channel. The low-precision cast
// output is shared because vTrans/kTrans execute serially on each AIV.
template <class ArchTag_, class TileBuffer_, class Element_, class ElementAccumulator_>
class BackwardSameABTransOut {
public:
    using ArchTag = ArchTag_;
    using TileBuffer = TileBuffer_;
    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;

    static constexpr uint32_t ELEM_PER_BLOCK = Catlass::BYTE_PER_C0 / sizeof(Element);
    // Dedicated to the shared dV/dK output UB; IDs 0/1/2 belong to the
    // existing epilogue and k-block loop protocol. Seed/drain in PipeEventGuard.
    static constexpr uint32_t TRANS_EVENT_ID = EVENT_ID3;

    CATLASS_DEVICE BackwardSameABTransOut(uint32_t cubeFlag, uint32_t ubFreeFlag, int64_t stride,
                                          uint32_t transInOffset, Arch::Resource<ArchTag>& resource)
        : stride_(stride)
    {
        cubeReady_ = Arch::CrossCoreFlag(cubeFlag);
        ubFree_ = Arch::CrossCoreFlag(ubFreeFlag);
        transIn_ = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(transInOffset);
        transOut_ = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::TRANS_OUT);
    }

    template <class TensorDst>
    CATLASS_DEVICE void operator()(TensorDst& dstTensor)
    {
        // Cast is the first consumer of the FP32 Fixpipe result.
        AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(cubeReady_.id);

        uint32_t subBlock = AscendC::GetSubBlockIdx();
        uint32_t totalRows = tla::get<0>(dstTensor.shape());
        // The AIC pads the SPLIT_M M extent (the real K row count) to the
        // fractal boundary, so AIV0 owns RoundUp(totalRows, ELEM_PER_BLOCK) / 2
        // UB rows and AIV1 owns the remainder. Only rows that exist in
        // totalRows are written back to GM.
        uint32_t rowsCore0 = RoundUp<ELEM_PER_BLOCK>(totalRows) / 2;
        uint32_t rows = 0;
        if (subBlock == 0) {
            rows = totalRows < rowsCore0 ? totalRows : rowsCore0;
        } else if (totalRows > rowsCore0) {
            rows = totalRows - rowsCore0;
        }
        uint32_t cols = tla::get<1>(dstTensor.shape());
        if (rows == 0) {
            // This AIV does not consume transIn_. Return its ownership without
            // touching the local transOut_ event token.
            AscendC::CrossCoreSetFlag<0x2, PIPE_V>(ubFree_.id);
            return;
        }
        AscendC::DataCopyParams params;
        params.blockCount = rows;
        params.blockLen = cols / ELEM_PER_BLOCK;
        params.srcStride = 0;
        params.dstStride = (stride_ - cols) / ELEM_PER_BLOCK;
        auto dstOffset = dstTensor.layout()(dstTensor.coord()) + subBlock * rowsCore0 * stride_;
        uint32_t elements = rowsCore0 * cols;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(TRANS_EVENT_ID);
        AscendC::Cast(transOut_, transIn_, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(TRANS_EVENT_ID);

        // UBFREE protects the aliased FP32 input (transIn_/score), rather than
        // the low-precision transOut_. Once Cast has consumed transIn_, AIC may
        // reuse that region for the next score/Fixpipe result while MTE3 drains
        // the independent transOut_ buffer to GM. The V-pipe flag keeps the
        // ownership hand-off ordered after the Cast read.
        AscendC::CrossCoreSetFlag<0x2, PIPE_V>(ubFree_.id);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(TRANS_EVENT_ID);
        AscendC::DataCopy(dstTensor.data()[dstOffset], transOut_, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(TRANS_EVENT_ID);
    }

private:
    int64_t stride_{0};
    Arch::CrossCoreFlag cubeReady_;
    Arch::CrossCoreFlag ubFree_;
    AscendC::LocalTensor<ElementAccumulator> transIn_;
    AscendC::LocalTensor<Element> transOut_;
};

}  // namespace Catlass::Kernel::SameAB

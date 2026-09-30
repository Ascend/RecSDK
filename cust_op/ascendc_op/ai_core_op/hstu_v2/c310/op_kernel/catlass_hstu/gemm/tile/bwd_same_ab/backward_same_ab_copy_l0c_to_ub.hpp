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

#include "../copy_l0c_to_ub_a5.hpp"
#include "../../../kernel/bwd/same_ab/backward_same_ab_kernel_resource.hpp"

namespace Catlass::Kernel::SameAB {

/**
 * @brief Copy one complete Same AB result tile from L0C to the two AIV UBs.
 *
 * The AIC submits the complete logical M tile. Ascend950 Fixpipe SPLIT_M
 * routes the first and second M halves to the same UB address of AIV0 and
 * AIV1 respectively (dualDstCtl = 1). The destination tensor must therefore
 * describe the complete logical tile even though each AIV only owns
 * `L1_TILE_M / 2` rows of physical UB storage.
 *
 * This class only defines the data movement. The caller owns M_FIX/FIX_M and
 * cross-core ready/free event synchronization.
 */
template <class Resource_>
struct BackwardSameABCopyL0CToUB {
    using Resource = Resource_;
    using ArchTag = typename Resource::ArchTag;

    static constexpr uint32_t AIV_PER_AIC = Resource::AIV_PER_AIC;
    static constexpr uint32_t LOGICAL_M = Resource::L1_TILE_M;
    static constexpr uint32_t LOGICAL_N = Resource::L1_TILE_N;
    static constexpr uint32_t AIV_M = Resource::AIV_TILE_M;

    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Same AB L0C-to-UB copy currently supports Ascend950 only");
    static_assert(AIV_PER_AIC == 2, "Fixpipe SPLIT_M requires two AIVs per AIC");
    static_assert(LOGICAL_M % AIV_PER_AIC == 0, "Same AB logical M must split evenly across two AIVs");

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0) const
    {
        static_assert(TensorSrc::position == AscendC::TPosition::CO1, "Same AB Fixpipe source must be an L0C tensor");
        static_assert(TensorDst::position == AscendC::TPosition::VECCALC,
                      "Same AB Fixpipe destination must be a UB tensor");
        static_assert(tla::detail::isRowMajor<typename TensorDst::Layout>::value ||
                          tla::detail::isL0czN<typename TensorDst::Layout>::value,
                      "Same AB vector epilogue requires a RowMajor or zN UB tile");

        using CopySplitM =
            Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorSrc, TensorDst, Gemm::Tile::CopyL0CToUBMode::SPLIT_M,
                                       Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
        CopySplitM{}(dstTensor, srcTensor, unitFlag);
    }
};

}  // namespace Catlass::Kernel::SameAB

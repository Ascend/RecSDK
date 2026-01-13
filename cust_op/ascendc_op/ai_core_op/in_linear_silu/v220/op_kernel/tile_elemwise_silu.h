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

#ifndef _TILE_ELEMWISE_SILU_H
#define _TILE_ELEMWISE_SILU_H

#include "catlass/catlass.hpp"

namespace InLinearSilu_Kernel {
template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileElemWiseSilu {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileElemWiseSilu() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const& dstLocal,
                    AscendC::LocalTensor<ElementCompute> const& srcLocal, uint32_t length)
    {
        // split silu
        Muls(dstLocal, srcLocal, (ElementCompute)-1, length);
        AscendC::PipeBarrier<PIPE_V>();
        Exp(dstLocal, dstLocal, length);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(dstLocal, dstLocal, (ElementCompute)1, length);
        AscendC::PipeBarrier<PIPE_V>();
        Div(dstLocal, srcLocal, dstLocal, length);
    }
};
}  // namespace InLinearSilu_Kernel

#endif
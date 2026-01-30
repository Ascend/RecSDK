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
#ifndef CONCAT_SILU_GRAD_TILE
#define CONCAT_SILU_GRAD_TILE

#include "catlass/catlass.hpp"

namespace PROJECT_NAMESPACE {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    uint32_t COMPUTE_LENGTH_>
struct TileSiluGrad {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using LayoutCompute = typename ComputeType_::Layout;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileSiluGrad() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const& dstLocal,
                    AscendC::LocalTensor<ElementCompute> const& srcLocal, uint32_t length)
    {
    }
};

}  // namespace PROJECT_NAMESPACE

#endif

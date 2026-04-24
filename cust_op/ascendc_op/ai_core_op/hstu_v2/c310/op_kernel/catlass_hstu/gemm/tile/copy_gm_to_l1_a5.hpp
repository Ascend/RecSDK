/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

/**
 • @file copy_gm_to_l1_a5.hpp

 • @brief HSTU GM 到 L1 的数据拷贝 Tile 实现

 • @description 定义从 Global Memory 拷贝数据到 L1 缓存的 Tile 策略，

 •              支持 TLA (Tensor Layout Abstraction) 布局转换

 */
#pragma once

#include "../../../tla_hstu/layout.hpp"
#include "catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp"

namespace Catlass::Gemm::Tile {

/**
 • @brief TLA GM 到 L1 拷贝 Tile

 • @tparam ArchTag 架构标签

 • @tparam TensorSrc 源张量类型

 • @tparam TensorDst 目标张量类型

 • @description 定义从 Global Memory 拷贝数据到 L1 缓存的策略

 */
template <class ArchTag, class TensorSrc, class TensorDst>
struct TileCopyTNDTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTNDTla, can not find the specialization.");
};

/// Partial specialization for CopyGmToL1, AtlasA5, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTNDTla<Arch::Ascend950,
                      tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                      tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTNDTla(){};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, int64_t rows, int64_t cols,
                                   int64_t stride)
    {
        static_assert(
            TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN");

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = rows;
        intriParams.dValue = cols;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = stride;
        intriParams.dstNzC0Stride = RoundUp<Catlass::C0_NUM_PER_FRACTAL>(rows);
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data(), srcTensor.data()[srcOffset], intriParams);
    }
};

}

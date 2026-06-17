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
 * @file dual_tile_copy.hpp

 * @brief 双路 TileCopy 包装器

 * @description 将两个 PackedTileCopyTlaToUB 包装为一个类型，使 BlockMmad 可以同时使用两路
 *              CopyL0CToDst 输出策略（如 zN 格式输出到 Epilogue + RowMajor 格式输出到下一级 MMAD）

 *              继承 Primary，保证存量代码通过继承链仍可正常使用单路拷贝（Layout/CopyL1ToL0A 等）
 */

#pragma once

namespace Catlass::Gemm::Tile {

template <class PrimaryTileCopy_, class SecondaryTileCopy_ = PrimaryTileCopy_>
struct DualTileCopy : public PrimaryTileCopy_ {
    using Primary = PrimaryTileCopy_;
    using Secondary = SecondaryTileCopy_;
};

}  // namespace Catlass::Gemm::Tile

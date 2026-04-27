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
 • @file layout.hpp

 • @brief HSTU TLA 布局扩展定义

 • @description 定义 HSTU 算子特有的 Tensor Layout 扩展，包括 zN 布局和 L0C zN 布局的特化判断

 */

#pragma once

#include "tla/layout.hpp"

namespace tla {

namespace detail {

/**
 • @brief 判断是否为 L0C zN 布局

 • @tparam Layout 布局类型

 • @description 用于判断给定的布局是否是 L0C zN 格式

 */
template <class Layout, class Enable1 = void, class Enable2 = void>
struct isL0czN {
    static bool const value = false;
};

template <class Layout>
struct isL0czN<
    Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,    // 2 means depth and rank
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2>> {
    static bool const value =
        (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL &&
         stride<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && stride<1, 0>(Layout{}) == 1);
};

}

}

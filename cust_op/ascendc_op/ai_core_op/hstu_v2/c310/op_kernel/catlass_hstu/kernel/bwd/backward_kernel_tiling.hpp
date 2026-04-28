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
 • @file backward_kernel_tiling.hpp

 • @brief HSTU Backward 算子 Kernel Tiling 模板参数定义

 • @description 定义算子的模板参数，用于编译时生成不同配置的 Kernel:

 •              - HAS_RAB: 是否有相对位置偏置 (0: 无, 1: 有)

 •              - BLOCK_K: K 方向的块大小 (128 或 256)

 */

#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

/**
 • @brief 定义模板参数列表

 • @description HstuBackwardV2 算子支持两种模板参数:

 •              1. HAS_RAB: 0 表示无 RAB，1 表示有 RAB

 •              2. BLOCK_K: K 方向的块大小，支持 128 或 256

 */
ASCENDC_TPL_ARGS_DECL(HstuBackwardV2, ASCENDC_TPL_BOOL_DECL(HAS_RAB, 0, 1),
                      ASCENDC_TPL_UINT_DECL(BLOCK_K, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 128, 256));

/**
 • @brief 模板参数选择器

 • @description 组合 HAS_RAB 和 BLOCK_K 的所有可能取值，生成 4 种 Kernel 变体

 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(HAS_RAB, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(BLOCK_K, ASCENDC_TPL_UI_LIST, 128, 256)),);

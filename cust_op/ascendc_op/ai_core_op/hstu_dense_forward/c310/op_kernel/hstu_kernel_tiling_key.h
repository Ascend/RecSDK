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
#ifndef HSTU_KERNEL_TILING_KEY_H
#define HSTU_KERNEL_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// maskedType: { 0:mask_tril, 2:mask_none, 3:mask_custom }
// enableBias: { 0:no_bias, 1:bias }
// isQkUseUb: { 0:no_ub, 1:ub }
// typeTilingKey: { 0:normal, 1:jagged, 2:paged }
// deterministic: {0:False, 1:true}
ASCENDC_TPL_ARGS_DECL(HstuDenseForward, // 算子OpType
ASCENDC_TPL_UINT_DECL(maskedType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 2, 3),
ASCENDC_TPL_BOOL_DECL(enableBias, 0, 1),
ASCENDC_TPL_BOOL_DECL(isQkUseUb, 0, 1),
ASCENDC_TPL_UINT_DECL(typeTilingKey, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
ASCENDC_TPL_BOOL_DECL(deterministic, 0, 1),
ASCENDC_TPL_UINT_DECL(blockM, 9, ASCENDC_TPL_UI_LIST, 32, 64, 128, 256),
ASCENDC_TPL_UINT_DECL(blockN, 13, ASCENDC_TPL_UI_LIST, 32, 64, 128, 256, 512, 1024),
ASCENDC_TPL_UINT_DECL(blockK, 10, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512),
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
        ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
        ASCENDC_TPL_BOOL_SEL(isQkUseUb, 0, 1),
        ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
        ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
        ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
        ASCENDC_TPL_BOOL_SEL(isQkUseUb, 0, 1),
        ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
        ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 32),
        ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 32),
        ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
        ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
        ASCENDC_TPL_BOOL_SEL(isQkUseUb, 0, 1),
        ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
        ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 64),
        ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 64, 1024),
        ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
        ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
        ASCENDC_TPL_BOOL_SEL(isQkUseUb, 0, 1),
        ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
        ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 128, 256, 512),
        ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512),
    ),
    // dense格式和paged格式
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
        ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
        ASCENDC_TPL_BOOL_SEL(isQkUseUb, 0, 1),
        ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 0, 2),
        ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
        ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 256),
        ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 512),
    ),
);

#endif

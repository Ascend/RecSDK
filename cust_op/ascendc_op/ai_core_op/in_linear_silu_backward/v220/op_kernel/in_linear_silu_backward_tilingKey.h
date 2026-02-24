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
#ifndef IN_LINEAR_SILU_BACKWARD_TILINGKEY_H
#define IN_LINEAR_SILU_BACKWARD_TILINGKEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(InLinearSiluBackward, // 算子OpType
ASCENDC_TPL_BOOL_DECL(enableBias, 0, 1),
ASCENDC_TPL_BOOL_DECL(isTrans, 0, 1),
ASCENDC_TPL_BOOL_DECL(isVardim, 0, 1),
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_DECL(enableBias, 0, 1),
        ASCENDC_TPL_BOOL_DECL(isTrans, 0, 1),
        ASCENDC_TPL_BOOL_DECL(isVardim, 0, 1),
    ),
);

#endif

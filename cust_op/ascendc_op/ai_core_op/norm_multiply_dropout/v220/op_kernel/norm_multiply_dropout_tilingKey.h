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
#ifndef MXREC_NORM_MULTIPLY_DROPOUT_TILINGKEY_H
#define MXREC_NORM_MULTIPLY_DROPOUT_TILINGKEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(NormMultiplyDropout, ASCENDC_TPL_BOOL_DECL(isNeedDrop, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_DECL(isNeedDrop, 0, 1)));

#endif  // MXREC_NORM_MULTIPLY_DROPOUT_TILINGKEY_H

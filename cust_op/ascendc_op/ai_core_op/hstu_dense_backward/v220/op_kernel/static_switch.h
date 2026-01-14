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

#ifndef HSTU_STATIC_SWITCH_H
#define HSTU_STATIC_SWITCH_H

#define HEAD_DIM_SWITCH(PRECOND, COND, CONST_NAME, CONST_RESULT, ...) \
    if (PRECOND && COND == 64) {                                      \
        constexpr int64_t CONST_NAME = 64;                            \
        constexpr bool CONST_RESULT = true;                           \
        __VA_ARGS__;                                                  \
    } else if (PRECOND && COND == 128) {                              \
        constexpr int64_t CONST_NAME = 128;                           \
        constexpr bool CONST_RESULT = true;                           \
        __VA_ARGS__;                                                  \
    } else if (PRECOND && COND == 256) {                              \
        constexpr int64_t CONST_NAME = 256;                           \
        constexpr bool CONST_RESULT = true;                           \
        __VA_ARGS__;                                                  \
    } else {                                                          \
        constexpr int64_t CONST_NAME = 256;                           \
        constexpr bool CONST_RESULT = false;                          \
        __VA_ARGS__;                                                  \
    }
#endif
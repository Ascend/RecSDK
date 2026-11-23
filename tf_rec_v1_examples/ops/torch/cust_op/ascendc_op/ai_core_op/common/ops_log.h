/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
#ifndef COMMON_OPS_LOG_H
#define COMMON_OPS_LOG_H

#pragma once

#include "log_base.h"

/* 基础日志 */
#define OPS_LOG_D(OPS_DESC, ...) OPS_LOG_STUB_D(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_I(OPS_DESC, ...) OPS_LOG_STUB_I(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_W(OPS_DESC, ...) OPS_LOG_STUB_W(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_E(OPS_DESC, ...) OPS_INNER_ERR_STUB("EZ9999", OPS_DESC, __VA_ARGS__)
#define OPS_LOG_E_WITHOUT_REPORT(OPS_DESC, ...) OPS_LOG_STUB_E(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_EVENT(OPS_DESC, ...) OPS_LOG_STUB_EVENT(OPS_DESC, __VA_ARGS__)

/* 条件日志 */
#define OPS_LOG_D_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_D(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_I_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_I(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_W_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_W(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_E_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_E(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_EVENT_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_EVENT(OP_DESC, __VA_ARGS__), EXPR)

#define OPS_LOG_E_IF_NULL(OPS_DESC, PTR, EXPR)                                                                       \
    if (__builtin_expect((PTR) == nullptr, 0)) {                                                                      \
        OPS_LOG_STUB_E(OPS_DESC, "%s is nullptr!", #PTR);                                                             \
        OPS_CALL_ERR_STUB("EZ9999", OPS_DESC, "%s is nullptr!", #PTR);                                                \
        EXPR;                                                                                                         \
    }

#define OPS_CHECK_PTR_NULL(PTR, EXPR)  OPS_LOG_E_IF_NULL(#PTR, PTR, EXPR)

#define OPS_CHECK(COND, LOG_FUNC, EXPR)                                                                               \
    if (COND) {                                                                                                       \
        LOG_FUNC;                                                                                                     \
        EXPR;                                                                                                         \
    }
#endif
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
#ifndef COMMON_LOG_BASE_H
#define COMMON_LOG_BASE_H
#pragma once

#include <string>
#include <cstdint>
#include <sstream>
#include <unistd.h>
#include <sys/syscall.h>
#include "securec.h"
#include "base/err_msg.h"
#if __has_include("toolchain/slog.h")
    #include "toolchain/slog.h"
#elif __has_include("base/dlog_pub.h")
    #include "base/dlog_pub.h"
#else
    #error "Neither slog.h nor dlog_pub.h found!"
#endif

namespace ops {
namespace utils {

class LogBase {
public:
    static constexpr const int MAX_LOG_LEN = 16000;
    static constexpr const int MSG_HDR_LEN = 200;

    static inline uint64_t GetTid()
    {
        return static_cast<uint64_t>(syscall(__NR_gettid));
    }

    static inline const char *GetStr(const std::string &str)
    {
        return str.c_str();
    }

    static inline const char *GetStr(const char *str)
    {
        return str;
    }

    static inline const std::string &GetOpInfo(const std::string &str)
    {
        return str;
    }

    static inline const char *GetOpInfo(const char *str)
    {
        return str;
    }

    static inline std::string GetOpInfo(const gert::TilingContext *context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::TilingParseContext *context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::InferShapeContext *context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::InferDataTypeContext *context)
    {
        return GetOpInfoFromContext(context);
    }

private:
    template <class T> static inline std::string GetOpInfoFromContext(T context)
    {
        if (context == nullptr) {
            return "nil:nil";
        }
        std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
        opInfo += ":";
        opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
        return opInfo;
    }
};

} // namespace utils

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
        oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
}
oss << "]";
return oss.str();
}
} // namespace ops

// 使用本宏前需预定义标识子模块名称的 OPS_UTILS_LOG_SUB_MOD_NAME
// 如: #define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING" 或通过 CMake 传递预定义宏

#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING"
#define OPS_UTILS_LOG_PACKAGE_TYPE "MXREC"

#define OPS_LOG_STUB(MOD_ID, LOG_LEVEL, OPS_DESC, FMT, ...)                                                       \
    DlogSub(static_cast<int>(MOD_ID), (OPS_UTILS_LOG_SUB_MOD_NAME), (LOG_LEVEL), "%s[%s][%lu] OpName:[%s] " #FMT, \
            (OPS_UTILS_LOG_PACKAGE_TYPE), __FUNCTION__, ops::utils::LogBase::GetTid(),                            \
            ops::utils::LogBase::GetStr(ops::utils::LogBase::GetOpInfo(OPS_DESC)), ##__VA_ARGS__)

#define OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)                                                                   \
    do {                                                                                                        \
        static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool"); \
        if (__builtin_expect((COND), 0)) {                                                                      \
            LOG_FUNC;                                                                                           \
            EXPR;                                                                                               \
        }                                                                                                       \
    } while (0)

#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)        \
    do {                                                            \
        OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__); \
        REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);     \
    } while (0)

#define OPS_CALL_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)         \
    do {                                                            \
        OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__); \
        REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);     \
    } while (0)

#define OPS_LOG_STUB_D(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_DEBUG, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_I(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_INFO, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_W(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_WARN, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_E(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_EVENT(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_EVENT, OPS_DESC, FMT, ##__VA_ARGS__)

#endif
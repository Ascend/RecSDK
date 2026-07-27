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

/*!
 * \file aicpu_common.h
 * \brief
 */

#ifndef AICPU_COMMON_H
#define AICPU_COMMON_H

#include <cstdint>
#include <vector>
#include "cpu_context.h"

namespace aicpu {
template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, bool>::type GetAttrValue(CpuKernelContext& ctx,
                                                                               const std::string& name, T& value)
{
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = static_cast<T>(attr->GetInt());
    return true;
}

inline bool GetAttrValue(CpuKernelContext& ctx, const std::string& name, std::string& value)
{
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetString();
    return true;
}

inline bool GetAttrValue(CpuKernelContext& ctx, const std::string& name, bool& value)
{
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetBool();
    return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type GetAttrValueOpt(CpuKernelContext& ctx,
                                                                                  const std::string& name, T& value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = static_cast<T>(attr->GetInt());
    }
}

inline void GetAttrValueOpt(CpuKernelContext& ctx, const std::string& name, std::string& value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetString();
    }
}

inline void GetAttrValueOpt(CpuKernelContext& ctx, const std::string& name, bool& value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetBool();
    }
}

inline std::vector<int64_t> GetTensorDataAsInt64(Tensor* tensor, size_t size)
{
    std::vector<int64_t> result(size, 0);
    if (tensor == nullptr || tensor->GetData() == nullptr || size == 0) {
        return result;
    }
    void* data = tensor->GetData();
    switch (tensor->GetDataType()) {
        case DT_INT32:
            std::transform(static_cast<int32_t*>(data), static_cast<int32_t*>(data) + size, result.begin(),
                           [](int32_t v) { return static_cast<int64_t>(v); });
            break;
        case DT_INT64:
            std::copy(static_cast<int64_t*>(data), static_cast<int64_t*>(data) + size, result.begin());
            break;
        case DT_INT16:
            std::transform(static_cast<int16_t*>(data), static_cast<int16_t*>(data) + size, result.begin(),
                           [](int16_t v) { return static_cast<int64_t>(v); });
            break;
        case DT_UINT32:
            std::transform(static_cast<uint32_t*>(data), static_cast<uint32_t*>(data) + size, result.begin(),
                           [](uint32_t v) { return static_cast<int64_t>(v); });
            break;
        case DT_UINT64:
            std::transform(static_cast<uint64_t*>(data), static_cast<uint64_t*>(data) + size, result.begin(),
                           [](uint64_t v) { return static_cast<int64_t>(v); });
            break;
        case DT_UINT16:
            std::transform(static_cast<uint16_t*>(data), static_cast<uint16_t*>(data) + size, result.begin(),
                           [](uint16_t v) { return static_cast<int64_t>(v); });
            break;
        default:
            KERNEL_LOG_ERROR("unsupported tensor data type: %d", static_cast<int32_t>(tensor->GetDataType()));
            break;
    }
    return result;
}
}  // namespace aicpu

#endif

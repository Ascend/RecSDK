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
 * \file hstu_attn_metadata_backward_common.h
 * \brief AICPU 侧公共小工具：属性读取、tensor 数据读取、整除上取整。
 *
 * 本算子不带 deps/（不依赖 SectionStreamK），因此不复用 attention_common 的
 * aicpu_common.h，保持自包含，便于单独交叉编译与 host 侧单测。
 *
 * 注意：CMake 只把 *_aicpu.cpp 作为唯一翻译单元，本头里的函数必须 inline。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_COMMON_H
#define HSTU_ATTN_METADATA_BACKWARD_COMMON_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "cpu_context.h"
#include "cpu_tensor.h"
#include "log.h"

namespace aicpu {
namespace backward {
namespace detail {

/*! \brief 整除上取整；divisor 为 0 时返回 0（避免除零）。 */
inline uint32_t CeilDiv(uint64_t value, uint32_t divisor)
{
    return (divisor == 0U) ? 0U : static_cast<uint32_t>((value + divisor - 1U) / divisor);
}

/*! \brief 读 int 属性；属性缺失返回 false。 */
inline bool ReadAttr(CpuKernelContext& ctx, const std::string& name, int32_t& value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is missing: %s", name.c_str());
        return false;
    }
    value = static_cast<int32_t>(attr->GetInt());
    return true;
}

/*! \brief 读 string 属性；属性缺失返回 false。 */
inline bool ReadAttr(CpuKernelContext& ctx, const std::string& name, std::string& value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is missing: %s", name.c_str());
        return false;
    }
    value = attr->GetString();
    return true;
}

/*!
 * \brief 把一维的 int32/int64 tensor 读成 int64 序列（长度取第 0 维）。
 *
 * AICPU 侧不做搬运：GetData() 返回的就是可直接解引用的裸指针，
 * 这里只做「按运行期 dtype 分派 + 逐元素读取」，dtype 不在白名单内则失败。
 */
inline bool ReadTensorAsInt64(Tensor* tensor, std::vector<int64_t>& out)
{
    out.clear();
    if (tensor == nullptr || tensor->GetData() == nullptr) {
        KERNEL_LOG_ERROR("tensor data is null");
        return false;
    }
    auto shape = tensor->GetTensorShape();
    if (shape == nullptr) {
        KERNEL_LOG_ERROR("tensor shape is null");
        return false;
    }
    const int64_t len = shape->GetDimSize(0);
    if (len <= 0) {
        KERNEL_LOG_ERROR("tensor is empty, dim0 = %ld", len);
        return false;
    }

    out.resize(static_cast<size_t>(len));
    void* data = tensor->GetData();
    switch (tensor->GetDataType()) {
        case DT_INT32: {
            const int32_t* src = static_cast<const int32_t*>(data);
            std::transform(src, src + len, out.begin(), [](int32_t v) { return static_cast<int64_t>(v); });
            break;
        }
        case DT_INT64: {
            const int64_t* src = static_cast<const int64_t*>(data);
            std::copy(src, src + len, out.begin());
            break;
        }
        default:
            KERNEL_LOG_ERROR("unsupported dtype: %d", static_cast<int32_t>(tensor->GetDataType()));
            out.clear();
            return false;
    }
    return true;
}

}  // namespace detail
}  // namespace backward
}  // namespace aicpu

#endif  // HSTU_ATTN_METADATA_BACKWARD_COMMON_H

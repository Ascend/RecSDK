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

#ifndef RECSDK_REFACTORING_COMMON_FUNC_H
#define RECSDK_REFACTORING_COMMON_FUNC_H

#include <cstring>
#include <memory>
#include "securec.h"

namespace MxRec {
    extern const int GLOG_MAX_BUF_SIZE;
    extern const char* HUGE_TLB_ENABLE;

    template <typename... Args>
    std::string StringFormat(const std::string& format, Args... args)
    {
        auto size = static_cast<size_t>(GLOG_MAX_BUF_SIZE);
        auto buf = std::make_unique<char[]>(size); // LCOV_EXCL_BR_LINE
        memset_s(buf.get(), size, 0, size);
        int nChar = snprintf_s(buf.get(), size, size - 1, format.c_str(), args...);
        if (nChar == -1) { // LCOV_EXCL_BR_LINE
            throw std::invalid_argument("StringFormat failed");
        }
        return std::string(buf.get(), buf.get() + nChar);
    }
    uint32_t GetDeviceCount();
    std::string GetChipName(uint32_t devID);
}

#endif // RECSDK_REFACTORING_COMMON_FUNC_H

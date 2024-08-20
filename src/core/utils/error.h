/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef MXREC_CORE_UTILS_ERROR_H
#define MXREC_CORE_UTILS_ERROR_H

#include <cstdint>
#include <string>

#include "tl/expected.hpp"

namespace MxRec {

enum class ErrorType: uint8_t {
    NOT_FOUND,
    FILE_NOT_EXIST,
    NOT_SUPPORTED,
    INVALID_ARGUMENT,
    IO_ERROR,
    ACL_ERROR,
    MPI_ERROR,
    UNKNOWN,
};

enum class ModuleName: uint8_t {
    M_CHECK_POINT,
    M_EMB_TABLE,
    M_FILE_SYSTEM,
    M_HD_TRANSFER,
    M_HYBRID_MGMT,
    M_KEY_PROCESS,
    M_L3_STORAGE,
    M_SSD_ENGINE,
    M_UTILS,
};

class Error {
public:
    Error() = delete;
    Error(ModuleName mod, ErrorType e, const std::string& msg) : mod_(mod), err_(e), msg_(msg) {}

    std::string ToString() const;

private:
    std::string TypeAsString() const noexcept;
    std::string ModAsString() const noexcept;

    ModuleName mod_;
    ErrorType err_;
    std::string msg_;
};

template <typename T>
using Expected = tl::expected<T, Error>;

using UnExpected = tl::unexpected<Error>;

template <typename... Args, typename std::enable_if<std::is_constructible<Error, Args&&...>::value>::type* = nullptr>
UnExpected make_unexpected(Args&&... args)
{
    return tl::unexpected<Error>(std::forward<Args>(args)...);
}

}  // namespace MxRec

#endif

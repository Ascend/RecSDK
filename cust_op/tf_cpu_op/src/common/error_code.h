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

#ifndef RECBASE_ERROR_CODE_H
#define RECBASE_ERROR_CODE_H

namespace ock {
using RecBaseCode = enum : int {
    H_OK = 0,
    H_ERROR = 1,
    H_POINTER_NULL = 2,
    H_USING_EMPTY_KEY = 3,
    H_USING_DELETED_KEY = 4,
    H_INVALID_PARAMETER = 5,
    H_DIV_BY_ZERO = 6
};
}
#endif // RECBASE_ERROR_CODE_H
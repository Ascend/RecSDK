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

#include "select.h"

#include <arm_sve.h>

#include "common.h"
#include "error_code.h"
#include "external_logger.h"

namespace ock {
template <>
int Select<int64_t>(bool *cond, int64_t *thenBranch, int64_t *elseBranch, int64_t *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(cond == nullptr || thenBranch == nullptr || elseBranch == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Select Op.");
        return H_POINTER_NULL;
    }

    size_t i = 0;
    const size_t lanes = svlen(svint64_t());
    for (; i < length; i += lanes) {
        const svbool_t pg = svwhilelt_b64(i, length);
        // svld1sb_s64函数不支持bool类型作为输入，将bool类型强制转换为int8_t类型
        svint64_t cond_vec_int = svld1sb_s64(pg, reinterpret_cast<int8_t *>(cond + i));
        svint64_t then_vec = svld1_s64(pg, thenBranch + i);
        svint64_t else_vec = svld1_s64(pg, elseBranch + i);
        svbool_t cond_vec = svcmpeq_n_s64(pg, cond_vec_int, 1);
        // svsel_s64须作为svst1_s64的输入，不能将svsel_s64的返回值作为输入，会报错
        svst1_s64(pg, output + i, svsel_s64(cond_vec, then_vec, else_vec));
    }

    return H_OK;
}
} // namespace ock

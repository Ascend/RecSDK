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

#include "floor_mod.h"

#include <arm_sve.h>
#include <cmath>

#include <iostream>

#include "common.h"
#include "error_code.h"
#include "external_logger.h"

namespace ock {
inline int CheckSveHasZeroElement(const svbool_t pg_, svbool_t v_)
{
    if (svptest_any(pg_, v_)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Division by zero in the FloorMod Op.");
        return H_DIV_BY_ZERO;
    }
    return H_OK;
}

template <> int FloorMod<float>(float *input, float *mod, float *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input == nullptr || mod == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the FloorMod Op.");
        return H_POINTER_NULL;
    }

    size_t i = 0;
    const size_t lanes = svlen(svfloat32_t());
    const svfloat32_t zero_vec = svdup_f32(0.0f);
    for (; i < length; i += lanes) {
        const svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t input_vec = svld1_f32(pg, input + i);
        svfloat32_t mod_vec = svld1_f32(pg, mod + i);
        svbool_t zero_mask = svcmpeq_f32(pg, mod_vec, zero_vec);
        if (int err = CheckSveHasZeroElement(pg, zero_mask)) {
            return err;
        }
        svfloat32_t div_vec = svdiv_f32_z(pg, input_vec, mod_vec);
        svfloat32_t trunc_vec = svrintz_f32_z(pg, div_vec);
        svfloat32_t mult_vec = svmul_f32_z(pg, mod_vec, trunc_vec);
        svfloat32_t res_vec = svsub_f32_z(pg, input_vec, mult_vec);
        svst1_f32(pg, output + i, res_vec);
    }

    return H_OK;
}

template <> int FloorMod<double>(double *input, double *mod, double *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input == nullptr ||  output == nullptr || mod == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the FloorMod Op.");
        return H_POINTER_NULL;
    }

    size_t i = 0;
    const size_t lanes = svlen(svfloat64_t());
    const svfloat64_t zero_vec = svdup_f64(0.0);
    for (; i < length; i += lanes) {
        const svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t input_vec = svld1_f64(pg, input + i);
        svfloat64_t mod_vec = svld1_f64(pg, mod + i);
        svbool_t zero_mask = svcmpeq_f64(pg, mod_vec, zero_vec);
        if (int err = CheckSveHasZeroElement(pg, zero_mask)) {
            return err;
        }
        svfloat64_t div_vec = svdiv_f64_z(pg, input_vec, mod_vec);
        svfloat64_t trunc_vec = svrintz_f64_z(pg, div_vec);
        svfloat64_t mult_vec = svmul_f64_z(pg, mod_vec, trunc_vec);
        svfloat64_t res_vec = svsub_f64_z(pg, input_vec, mult_vec);
        svst1_f64(pg, output + i, res_vec);
    }

    return H_OK;
}

template <> int FloorMod<float>(float input, float *mod, float *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(output == nullptr || mod == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the FloorMod Op.");
        return H_POINTER_NULL;
    }

    size_t i = 0;
    const size_t lanes = svlen(svfloat32_t());
    const svfloat32_t zero_vec = svdup_f32(0.0f);
    svfloat32_t input_vec = svdup_f32(input);
    for (; i < length; i += lanes) {
        const svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t mod_vec = svld1_f32(pg, mod + i);
        svbool_t zero_mask = svcmpeq_f32(pg, mod_vec, zero_vec);
        if (int err = CheckSveHasZeroElement(pg, zero_mask)) {
            return err;
        }
        svfloat32_t div_vec = svdiv_f32_z(pg, input_vec, mod_vec);
        svfloat32_t trunc_vec = svrintz_f32_z(pg, div_vec);
        svfloat32_t mult_vec = svmul_f32_z(pg, mod_vec, trunc_vec);
        svfloat32_t res_vec = svsub_f32_z(pg, input_vec, mult_vec);
        svst1_f32(pg, output + i, res_vec);
    }

    return H_OK;
}

template <> int FloorMod<double>(double input, double *mod, double *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(output == nullptr || mod == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the FloorMod Op.");
        return H_POINTER_NULL;
    }

    size_t i = 0;
    const size_t lanes = svlen(svfloat64_t());
    const svfloat64_t zero_vec = svdup_f64(0.0);
    svfloat64_t input_vec = svdup_f64(input);
    for (; i < length; i += lanes) {
        const svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t mod_vec = svld1_f64(pg, mod + i);
        svbool_t zero_mask = svcmpeq_f64(pg, mod_vec, zero_vec);
        if (int err = CheckSveHasZeroElement(pg, zero_mask)) {
            return err;
        }
        svfloat64_t div_vec = svdiv_f64_z(pg, input_vec, mod_vec);
        svfloat64_t trunc_vec = svrintz_f64_z(pg, div_vec);
        svfloat64_t mult_vec = svmul_f64_z(pg, mod_vec, trunc_vec);
        svfloat64_t res_vec = svsub_f64_z(pg, input_vec, mult_vec);
        svst1_f64(pg, output + i, res_vec);
    }

    return H_OK;
}

template <> int FloorMod<float>(float *input, float mod, float *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(output == nullptr || input == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the FloorMod Op.");
        return H_POINTER_NULL;
    }

    size_t i = 0;
    const size_t lanes = svlen(svfloat32_t());
    const svfloat32_t zero_vec = svdup_f32(0.0f);
    svfloat32_t mod_vec = svdup_f32(mod);
    for (; i < length; i += lanes) {
        const svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t input_vec = svld1_f32(pg, input + i);
        svbool_t zero_mask = svcmpeq_f32(pg, mod_vec, zero_vec);
        if (int err = CheckSveHasZeroElement(pg, zero_mask)) {
            return err;
        }
        svfloat32_t div_vec = svdiv_f32_z(pg, input_vec, mod_vec);
        svfloat32_t trunc_vec = svrintz_f32_z(pg, div_vec);
        svfloat32_t mult_vec = svmul_f32_z(pg, mod_vec, trunc_vec);
        svfloat32_t res_vec = svsub_f32_z(pg, input_vec, mult_vec);
        svst1_f32(pg, output + i, res_vec);
    }

    return H_OK;
}

template <> int FloorMod<double>(double *input, double mod, double *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(output == nullptr || input == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the FloorMod Op.");
        return H_POINTER_NULL;
    }

    size_t i = 0;
    const size_t lanes = svlen(svfloat64_t());
    const svfloat64_t zero_vec = svdup_f64(0.0);
    svfloat64_t mod_vec = svdup_f64(mod);
    for (; i < length; i += lanes) {
        const svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t input_vec = svld1_f64(pg, input + i);
        svbool_t zero_mask = svcmpeq_f64(pg, mod_vec, zero_vec);
        if (int err = CheckSveHasZeroElement(pg, zero_mask)) {
            return err;
        }
        svfloat64_t div_vec = svdiv_f64_z(pg, input_vec, mod_vec);
        svfloat64_t trunc_vec = svrintz_f64_z(pg, div_vec);
        svfloat64_t mult_vec = svmul_f64_z(pg, mod_vec, trunc_vec);
        svfloat64_t res_vec = svsub_f64_z(pg, input_vec, mult_vec);
        svst1_f64(pg, output + i, res_vec);
    }

    return H_OK;
}
} // namespace ock
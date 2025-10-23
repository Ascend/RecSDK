/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#include "cmp.h"

#include <arm_sve.h>

#include "common.h"
#include "error_code.h"
#include "external_logger.h"

namespace ock {
namespace {
int ParamLengthCheck(const size_t length)
{
    if (OCK_PREDICT_FALSE(length == 0)) { // 检查 length 是否为 0，避免无效操作
        ExternalLogger::PrintLog(LogLevel::ERROR, "Length is 0.");
        return H_INVALID_PARAM;
    }

    // 检查 length 是否过大（防止整数溢出）
    constexpr size_t MAX_SAFE_LENGTH = SIZE_MAX / sizeof(int32_t);
    if (OCK_PREDICT_FALSE(length > MAX_SAFE_LENGTH)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Length exceeds maximum safe limit.");
        return H_INVALID_PARAM;
    }
    return H_OK;
}
}
// Less
template <> int Less<int64_t>(int64_t *input0, int64_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Less Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 2f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int64_t数据个数
        "cntd    x5\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.d, xzr, %[len]\n\t"
        "3:\n\t"
        // 将input0， input1，加载至sve寄存器，并做less运算
        "ld1d    z0.d, p0/z, [%[in1], x4, lsl 3]\n\t"
        "ld1d    z1.d, p0/z, [%[in0], x4, lsl 3]\n\t"
        "cmplt   p1.d, p2/z, z1.d, z0.d\n\t"
        "mov     z0.d, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.d, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.d, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    3b\n\t"
        "2:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

template <> int Less<int32_t>(int32_t *input0, int32_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Less Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 2f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int32_t数据个数
        "cntw    x5\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.s, xzr, %[len]\n\t"
        "3:\n\t"
        // 将input0， input1，加载至sve寄存器，并做less运算
        "ld1w    z0.s, p0/z, [%[in1], x4, lsl 2]\n\t"
        "ld1w    z1.s, p0/z, [%[in0], x4, lsl 2]\n\t"
        "cmplt   p1.s, p2/z, z1.s, z0.s\n\t"
        "mov     z0.s, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.s, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.s, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    3b\n\t"
        "2:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

// Less right
template <> int Less<int64_t>(int64_t *input0, int64_t input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Less Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 2f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int64_t数据个数
        "cntd    x5\n\t"
        "mov     z1.d, %[in1]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.d, xzr, %[len]\n\t"
        "3:\n\t"
        // 将input0， input1，加载至sve寄存器，并做less运算
        "ld1d    z0.d, p0/z, [%[in0], x4, lsl 3]\n\t"
        "cmplt   p1.d, p2/z, z0.d, z1.d\n\t"
        "mov     z0.d, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.d, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.d, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    3b\n\t"
        "2:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

template <> int Less<int32_t>(int32_t *input0, int32_t input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Less Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 2f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int32_t数据个数
        "cntw    x5\n\t"
        "mov     z1.s, %w[in1]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.s, xzr, %[len]\n\t"
        "3:\n\t"
        // 将input0， input1，加载至sve寄存器，并做less运算
        "ld1w    z0.s, p0/z, [%[in0], x4, lsl 2]\n\t"
        "cmplt   p1.s, p2/z, z0.s, z1.s\n\t"
        "mov     z0.s, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.s, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.s, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    3b\n\t"
        "2:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

// Less left
template <> int Less<int64_t>(int64_t input0, int64_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Less Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 2f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int64_t数据个数
        "cntd    x5\n\t"
        "mov     z1.d, %[in0]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.d, xzr, %[len]\n\t"
        "3:\n\t"
        // 将input0， input1，加载至sve寄存器，并做less运算
        "ld1d    z0.d, p0/z, [%[in1], x4, lsl 3]\n\t"
        "cmpgt   p1.d, p2/z, z0.d, z1.d\n\t"
        "mov     z0.d, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.d, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.d, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    3b\n\t"
        "2:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

template <> int Less<int32_t>(int32_t input0, int32_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Less Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 2f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int32_t数据个数
        "cntw    x5\n\t"
        "mov     z1.s, %w[in0]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.s, xzr, %[len]\n\t"
        "3:\n\t"
        // 将input0， input1，加载至sve寄存器，并做less运算
        "ld1w    z0.s, p0/z, [%[in1], x4, lsl 2]\n\t"
        "cmpgt   p1.s, p2/z, z0.s, z1.s\n\t"
        "mov     z0.s, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.s, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.s, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    3b\n\t"
        "2:\n\t"
        : [ in1 ] "r"(input1), [ in0 ] "r"(input0), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

// Greater
template <> int Greater<int64_t>(int64_t *input0, int64_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Greater Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 1f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int64_t数据个数
        "cntd    x5\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.d, xzr, %[len]\n\t"
        "2:\n\t"
        // 将input0， input1，加载至sve寄存器，并做Greater运算
        "ld1d    z0.d, p0/z, [%[in1], x4, lsl 3]\n\t"
        "ld1d    z1.d, p0/z, [%[in0], x4, lsl 3]\n\t"
        "cmpgt   p1.d, p2/z, z1.d, z0.d\n\t"
        "mov     z0.d, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.d, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.d, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    2b\n\t"
        "1:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

template <> int Greater<int32_t>(int32_t *input0, int32_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Greater Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 1f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int32_t数据个数
        "cntw    x5\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.s, xzr, %[len]\n\t"
        "2:\n\t"
        // 将input0， input1，加载至sve寄存器，并做Greater运算
        "ld1w    z0.s, p0/z, [%[in1], x4, lsl 2]\n\t"
        "ld1w    z1.s, p0/z, [%[in0], x4, lsl 2]\n\t"
        "cmpgt   p1.s, p2/z, z1.s, z0.s\n\t"
        "mov     z0.s, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.s, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.s, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    2b\n\t"
        "1:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

// Greater rigth
template <> int Greater<int64_t>(int64_t *input0, int64_t input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Greater Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 1f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int64_t数据个数
        "cntd    x5\n\t"
        "mov     z1.d, %[in1]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.d, xzr, %[len]\n\t"
        "2:\n\t"
        // 将input0， input1，加载至sve寄存器，并做Greater运算
        "ld1d    z0.d, p0/z, [%[in0], x4, lsl 3]\n\t"
        "cmpgt   p1.d, p2/z, z0.d, z1.d\n\t"
        "mov     z0.d, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.d, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.d, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    2b\n\t"
        "1:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

template <> int Greater<int32_t>(int32_t *input0, int32_t input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input0 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Greater Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 1f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int32_t数据个数
        "cntw    x5\n\t"
        "mov     z1.s, %w[in1]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.s, xzr, %[len]\n\t"
        "2:\n\t"
        // 将input0， input1，加载至sve寄存器，并做Greater运算
        "ld1w    z0.s, p0/z, [%[in0], x4, lsl 2]\n\t"
        "cmpgt   p1.s, p2/z, z0.s, z1.s\n\t"
        "mov     z0.s, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.s, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.s, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    2b\n\t"
        "1:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

// Greater left
template <> int Greater<int64_t>(int64_t input0, int64_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Greater Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 1f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int64_t数据个数
        "cntd    x5\n\t"
        "mov     z1.d, %[in0]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.d, xzr, %[len]\n\t"
        "2:\n\t"
        // 将input0， input1，加载至sve寄存器，并做Greater运算
        "ld1d    z0.d, p0/z, [%[in1], x4, lsl 3]\n\t"
        "cmplt   p1.d, p2/z, z0.d, z1.d\n\t"
        "mov     z0.d, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.d, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.d, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    2b\n\t"
        "1:\n\t"
        : [ in0 ] "r"(input0), [ in1 ] "r"(input1), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}

template <> int Greater<int32_t>(int32_t input0, int32_t *input1, bool *output, const size_t length)
{
    if (OCK_PREDICT_FALSE(input1 == nullptr || output == nullptr)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter verification failed for the Greater Op.");
        return H_POINTER_NULL;
    }
    auto ret = ParamLengthCheck(length);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Parameter length check failed.");
        return ret;
    }
    asm volatile("cbz     %[len], 1f\n\t"
        "mov     x4, 0\n\t"
        // 获取单个sve寄存器可容纳的int32_t数据个数
        "cntw    x5\n\t"
        "mov     z1.s, %w[in0]\n\t"
        "ptrue   p2.b, all\n\t"
        "whilelo p0.s, xzr, %[len]\n\t"
        "2:\n\t"
        // 将input0， input1，加载至sve寄存器，并做Greater运算
        "ld1w    z0.s, p0/z, [%[in1], x4, lsl 2]\n\t"
        "cmplt   p1.s, p2/z, z0.s, z1.s\n\t"
        "mov     z0.s, p1/z, #1\n\t"
        // 存储比较结果至output
        "st1b    z0.s, p0, [%[out], x4]\n\t"
        "add     x4, x4, x5\n\t"
        // 更新下个循环将运算的数据索引
        "whilelo p0.s, x4, %[len]\n\t"
        // 若仍有元素未被运算，跳转至位置2：
        "b.any    2b\n\t"
        "1:\n\t"
        : [ in1 ] "r"(input1), [ in0 ] "r"(input0), [ out ] "r"(output)
        : [ len ] "r"(length)
        : "p0", "p1", "p2", "z0", "z1", "x4", "x5", "memory");
    return H_OK;
}
} // namespace ock

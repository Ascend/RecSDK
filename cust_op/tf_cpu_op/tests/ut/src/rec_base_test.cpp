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

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

#include "cmp.h"
#include "common.h"
#include "error_code.h"
#include "floor_mod.h"
#include "select.h"
#include "set_external_logger.h"

#include "rec_base_test.h"

using namespace ock;

template <typename T>
void GenKeys(T *arr, size_t n, T min = -100, T max = 100)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
    for (uint64_t i = 0; i < n; i++) {
        arr[i] = distribution(gen);
    }
}

template <typename T>
std::vector<T> GenKeys(size_t n, T min = -100, T max = 100)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
    std::vector<T> dataSet(n);
    for (T &x : dataSet) {
        x = distribution(gen);
    }
    return dataSet;
}

void GenKeys(bool *array, size_t n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    const double trueProbability = 0.5;
    std::bernoulli_distribution distribution(trueProbability);
    for (uint64_t i = 0; i < n; i++) {
        array[i] = distribution(gen);
    }
}

template <>
void GenKeys<double>(double *arr, size_t n, double min, double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(min, max);
    for (uint64_t i = 0; i < n; i++) {
        arr[i] = distribution(gen);
    }
}

template <>
void GenKeys<float>(float *arr, size_t n, float min, float max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);
    for (uint64_t i = 0; i < n; i++) {
        arr[i] = distribution(gen);
    }
}

template <typename T>
std::vector<std::vector<T>> convertToVector(T *arr, size_t rows, size_t cols)
{
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols, 0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = arr[i * cols + j];
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> multiplyMatrices(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B)
{
    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();

    std::vector<std::vector<T>> C(m, std::vector<T>(n, 0));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t h = 0; h < k; h++) {
                C[i][j] += static_cast<double>(A[i][h]) * static_cast<double>(B[h][j]);
            }
        }
    }
    return C;
}

template <typename T>
void convertToArray(const std::vector<std::vector<T>> &C, T *expected, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            expected[i * cols + j] = C[i][j];
        }
    }
}

void RecBaseTest::SetUpTestCase()
{
    SetExternalLogFunc(RecBaseLog);
}

void RecBaseTest::TearDownTestCase() {}

void RecBaseTest::SetUp() {}

void RecBaseTest::TearDown() {}

template <typename T>
void TestLessHelper(size_t length)
{
    ASSERT_GE(length, 0);
    auto *in0 = new T[length];
    auto *in1 = new T[length];
    auto *output = new bool[length];
    auto *answer = new bool[length];
    GenKeys(in0, length, static_cast<T>(0), static_cast<T>(length));
    GenKeys(in1, length, static_cast<T>(0), static_cast<T>(length));
    int ret = -1;
    ret = Less(in0, in1, output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = in0[i] < in1[i];
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_EQ(output[i], answer[i]);
    }

    // 空指针校验
    ret = Less(static_cast<T *>(nullptr), in1, output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Less(in0, static_cast<T *>(nullptr), output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Less(in0, in1, nullptr, length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    // right
    ret = Less(in0, in1[0], output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = in0[i] < in1[0];
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_EQ(output[i], answer[i]);
    }

    // 空指针校验
    ret = Less(static_cast<T *>(nullptr), in1[0], output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Less(in0, in1[0], nullptr, length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    // left
    ret = Less(in0[0], in1, output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = in0[0] < in1[i];
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_EQ(output[i], answer[i]);
    }

    // 空指针校验
    ret = Less(in0[0], static_cast<T *>(nullptr), output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Less(in0[0], in1, nullptr, length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    delete[] in0;
    delete[] in1;
    delete[] output;
    delete[] answer;
}

template <typename T>
void TestGreaterHelper(size_t length)
{
    ASSERT_GE(length, 0);
    auto *output = new bool[length];
    auto *answer = new bool[length];
    auto *in0 = new T[length];
    auto *in1 = new T[length];
    GenKeys(in1, length, static_cast<T>(0), static_cast<T>(length));
    GenKeys(in0, length, static_cast<T>(0), static_cast<T>(length));
    int ret = -1;
    ret = Greater(in0, in1, output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = in0[i] > in1[i];
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_EQ(output[i], answer[i]);
    }

    // 空指针校验
    ret = Greater(static_cast<T *>(nullptr), in1, output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Greater(in0, static_cast<T *>(nullptr), output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Greater(in0, in1, nullptr, length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    // right
    ret = Greater(in0, in1[0], output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = in0[i] > in1[0];
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_EQ(output[i], answer[i]);
    }

    // 空指针校验
    ret = Greater(static_cast<T *>(nullptr), in1[0], output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Greater(in0, in1[0], nullptr, length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    // left
    ret = Greater(in0[0], in1, output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = in0[0] > in1[i];
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_EQ(output[i], answer[i]);
    }

    // 空指针校验
    ret = Greater(in0[0], static_cast<T *>(nullptr), output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Greater(in0[0], in1, nullptr, length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    delete[] in1;
    delete[] in0;
    delete[] answer;
    delete[] output;
}

template <typename T>
void TestFloorModHelper(size_t length)
{
    ASSERT_GE(length, 0);
    auto *output = new T[length];
    auto *answer = new T[length];
    auto *input = new T[length];
    auto *mod = new T[length];
    const T err = std::is_same<T, float>::value ? 1e-4 : 1e-8;
    GenKeys(mod, length);
    GenKeys(input, length);

    // 避免除以0
    for (uint64_t idx = 0; idx < length; idx++) {
        if (0 <= mod[idx] && mod[idx] < err) {
            mod[idx] = err;
        } else if (-err < mod[idx] && mod[idx] < 0) {
            mod[idx] = -err;
        }
    }

    int ret = -1;
    ret = FloorMod(input, mod, output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = std::fmod(input[i], mod[i]);
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_NEAR(output[i], answer[i], std::max(err, std::abs(err * answer[i])));
    }

    // 空指针校验
    ret = FloorMod(static_cast<T *>(nullptr), mod, output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = FloorMod(input, static_cast<T *>(nullptr), output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = FloorMod(input, mod, static_cast<T *>(nullptr), length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    // right
    ret = FloorMod(input, mod[0], output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = std::fmod(input[i], mod[0]);
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_NEAR(output[i], answer[i], std::max(err, std::abs(err * answer[i])));
    }

    // 空指针校验
    ret = FloorMod(static_cast<T *>(nullptr), mod[0], output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = FloorMod(input, mod[0], static_cast<T *>(nullptr), length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    // left
    ret = FloorMod(input[0], mod, output, length);
    ASSERT_EQ(ret, H_OK);

    for (uint64_t i = 0; i < length; i++) {
        answer[i] = std::fmod(input[0], mod[i]);
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_NEAR(output[i], answer[i], std::max(err, std::abs(err * answer[i])));
    }

    // 空指针校验
    ret = FloorMod(input[0], static_cast<T *>(nullptr), output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = FloorMod(input[0], mod, static_cast<T *>(nullptr), length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    // 除零校验
    mod[0] = static_cast<T>(0.0);

    ret = FloorMod(input, mod, output, length);
    ASSERT_EQ(ret, H_DIV_BY_ZERO);

    ret = FloorMod(input, mod[0], output, length);
    ASSERT_EQ(ret, H_DIV_BY_ZERO);

    ret = FloorMod(input[0], mod, output, length);
    ASSERT_EQ(ret, H_DIV_BY_ZERO);

    delete[] output;
    delete[] input;
    delete[] answer;
    delete[] mod;
}

template <typename T>
void TestSelectHelper(size_t length)
{
    ASSERT_GE(length, 0);
    auto *then = new T[length];
    auto *el = new T[length];
    auto *answer = new T[length];
    auto *output = new T[length];
    auto *cond = new bool[length];

    GenKeys(el, length, static_cast<T>(0), static_cast<T>(length));
    GenKeys(then, length, static_cast<T>(0), static_cast<T>(length));
    GenKeys(cond, length);

    int ret = -1;
    ret = Select(cond, then, el, output, length);
    ASSERT_EQ(ret, H_OK);
    for (uint64_t i = 0; i < length; i++) {
        answer[i] = cond[i] ? then[i] : el[i];
    }
    for (size_t i = 0; i < length; i++) {
        ASSERT_EQ(output[i], answer[i]);
    }

    // 空指针校验
    ret = Select(nullptr, then, el, output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Select(cond, static_cast<T *>(nullptr), el, output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Select(cond, then, static_cast<T *>(nullptr), output, length);
    ASSERT_EQ(ret, H_POINTER_NULL);
    ret = Select(cond, then, el, static_cast<T *>(nullptr), length);
    ASSERT_EQ(ret, H_POINTER_NULL);

    delete[] cond;
    delete[] then;
    delete[] el;
    delete[] output;
    delete[] answer;
}

TEST_F(RecBaseTest, LESS_TEST)
{
    RecBaseLog(INFO, "===========LESS_TEST start=============");
    size_t length = 999;

    TestLessHelper<int32_t>(length);
    TestLessHelper<int64_t>(length);

    RecBaseLog(INFO, "===========LESS_TEST end=============");
}

TEST_F(RecBaseTest, GREATER_TEST)
{
    RecBaseLog(INFO, "===========GREATER_TEST start=============");
    size_t length = 999;

    TestGreaterHelper<int32_t>(length);
    TestGreaterHelper<int64_t>(length);

    RecBaseLog(INFO, "===========GREATER_TEST end=============");
}

TEST_F(RecBaseTest, FLOOR_MOD_TEST)
{
    RecBaseLog(INFO, "===========FLOOR_MOD_TEST start=============");
    size_t length = 999;

    TestFloorModHelper<float>(length);
    TestFloorModHelper<double>(length);

    RecBaseLog(INFO, "===========FLOOR_MOD_TEST end=============");
}

TEST_F(RecBaseTest, SELECT_TEST)
{
    RecBaseLog(INFO, "===========SELECT_TEST start=============");
    size_t length = 999;

    TestSelectHelper<int64_t>(length);

    RecBaseLog(INFO, "===========SELECT_TEST end=============");
}
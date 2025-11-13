/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "gtest/gtest.h"
#include "emock/emock.hpp"

/* 使用EMOCK打桩需使用本宏定义申明测试用例，用于 ARM/X86 平台兼容 */
#if defined(TEST_USE_EMOCK) && (TEST_USE_EMOCK == 1)
    #define TEST_EMOCK_F(test_fixture, test_name) TEST_F(test_fixture, test_name)
#else
    #define TEST_EMOCK_F(test_fixture, test_name) \
        TEST_F(test_fixture, test_name) { \
            GTEST_SKIP(); \
        } \
        static void dummy_function_##test_fixture##test_name()
#endif

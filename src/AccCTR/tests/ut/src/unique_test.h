/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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

#ifndef OCK_UNIQUE_TEST_H
#define OCK_UNIQUE_TEST_H

#include <gtest/gtest.h>
#include <vector>
#include <unordered_set>
#include <map>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "unique.h"

using namespace std;
using namespace ock::ctr;


class UniqueTest : public testing::Test {
protected:
    UniqueTest() {};
    ~UniqueTest() {};
    static void SetUpTestCase();
    static void TearDownTestCase();


    void SetUp() {}

    void TearDown() {}
};


#endif // OCK_UNIQUE_TEST_H

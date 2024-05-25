/* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

#ifndef CTR_EMB_CACHE_TEST_H
#define CTR_EMB_CACHE_TEST_H

#include <vector>
#include <unordered_set>
#include <map>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "factory.h"
#include "embedding_cache.h"


class EmbCacheTest : public testing::Test {
protected:
    EmbCacheTest(){};
    ~EmbCacheTest(){};
    static void SetUpTestCase();
    static void TearDownTestCase();


    void SetUp() override;

    void TearDown() override;

    static ock::ctr::EmbCacheManagerPtr SimpleCreateTable(std::string tableName, uint32_t hostVocabSize, uint32_t embeddingSize,
        uint32_t extEmbeddingSize, uint32_t devVocabSize, std::pair<float, float> normalPara = { 0, 0.05 },
        float constPara = 0.233);

    static ock::ctr::EmbCacheManagerPtr ConstZeroCreateTable(std::string tableName, uint32_t hostVocabSize,
        uint32_t embeddingSize, uint32_t extEmbeddingSize, uint32_t devVocabSize, uint64_t prefillBufferSize = 50000,
        uint8_t prefillThreadNum = 1);

    std::string tooLongTableName =
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100012";
};

#endif // CTR_EMB_CACHE_TEST_H

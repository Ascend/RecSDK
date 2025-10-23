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

#include <gtest/gtest.h>

#include "l3_storage/l3_storage.h"

using namespace MxRec;
using namespace testing;

constexpr uint64_t MAX_TABLE_SIZE = 1;

class L3StorageTest : public testing::Test {
public:
    L3Storage m_l3Storage;
    std::string m_embTableName = "table1";
};

TEST_F(L3StorageTest, IsTableExist)
{
    auto isExist = m_l3Storage.IsTableExist(m_embTableName);
    EXPECT_EQ(isExist, false);
}

TEST_F(L3StorageTest, IsKeyExist)
{
    emb_cache_key_t key{1};
    auto isExist = m_l3Storage.IsKeyExist(m_embTableName, key);
    EXPECT_EQ(isExist, false);
}

TEST_F(L3StorageTest, CreateAndDelTable)
{
    std::vector<string> savePaths;
    m_l3Storage.CreateTable(m_embTableName, savePaths, MAX_TABLE_SIZE);
    auto space = m_l3Storage.GetTableAvailableSpace(m_embTableName);
    EXPECT_EQ(space, 0);

    std::vector<emb_cache_key_t> keys;
    std::vector<float*> embeddingsAddr;
    m_l3Storage.InsertEmbeddingsByAddr(m_embTableName, keys, embeddingsAddr, MAX_TABLE_SIZE);
    auto ret = m_l3Storage.FetchEmbeddings(m_embTableName, keys);
    EXPECT_EQ(ret.size(), 0);

    auto usage = m_l3Storage.GetTableUsage(m_embTableName);
    EXPECT_EQ(usage, 0);

    auto table = m_l3Storage.ExportTableKey();
    EXPECT_EQ(table.size(), 0);

    m_l3Storage.DeleteEmbeddings(m_embTableName, keys);
}

TEST_F(L3StorageTest, SaveAndLoad)
{
    int step = 0;
    std::map<string, map<emb_key_t, KeyInfo>> keyInfoMap;
    m_l3Storage.Start();
    m_l3Storage.Save(step);
    m_l3Storage.Save(step, keyInfoMap);

    std::vector<string> savePaths;
    m_l3Storage.Load(m_embTableName, savePaths, MAX_TABLE_SIZE, step);

    auto table = m_l3Storage.ExportTableKey();
    EXPECT_EQ(table.size(), 0);
}
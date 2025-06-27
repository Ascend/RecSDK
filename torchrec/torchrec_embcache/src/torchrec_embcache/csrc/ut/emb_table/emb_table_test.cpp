/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <memory>

#include "../common_main.h"

#include "emb_table/emb_table.h"
#include "utils/string_tools.h"

using namespace Embcache;

class EmbTableTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        embConfig.tableName = "user_table";
        embConfig.embDim = 8;
        embConfig.optimNum = 2;
        embConfig.weightInitMin = -0.5;
        embConfig.weightInitMax = 0.5;
    }

    void TearDown() override {}

    EmbConfig embConfig;
    std::shared_ptr<EmbTable> unorderedMapTablePtr;
    std::shared_ptr<EmbTable> fastHashMapTablePtr;
};

TEST_F(EmbTableTest, EmbTableUnorderedMap_FindOrInsert)
{
    unorderedMapTablePtr = std::make_shared<EmbTableUnorderedMap>(embConfig);

    std::vector<int64_t> keys{1, 1};
    std::vector<float> outEmbs(keys.size() * embConfig.embDim);
    std::vector<float> outOptims(keys.size() * embConfig.optimNum * embConfig.embDim);

    unorderedMapTablePtr->FindOrInsert(keys, outEmbs.data(), outOptims.data());

    LOG(INFO) << "keys.size():" << keys.size() << ", embDim:" << embConfig.embDim
              << ", optimNum:" << embConfig.optimNum;
    LOG(INFO) << "outEmbs.size():" << outEmbs.size() << ", outEmbs:" << StringTools::ToString(outEmbs);
    LOG(INFO) << "outOptims.size():" << outOptims.size() << ", outOptims:" << StringTools::ToString(outOptims);

    ASSERT_EQ(outEmbs.size(), keys.size() * embConfig.embDim);
    ASSERT_EQ(outOptims.size(), keys.size() * embConfig.optimNum * embConfig.embDim);
}

TEST_F(EmbTableTest, EmbTableUnorderedMap_InsertOrAssign)
{
    unorderedMapTablePtr = std::make_shared<EmbTableUnorderedMap>(embConfig);

    std::vector<int64_t> keys{1, 1};
    std::vector<float> inEmbs(embConfig.embDim, 0.51);
    std::vector<float> finalEmbs(embConfig.embDim, 0.52);
    std::copy(finalEmbs.begin(), finalEmbs.end(), std::back_inserter(inEmbs));

    std::vector<float> inOptims(embConfig.optimNum * embConfig.embDim, 0.81);
    std::vector<float> finalOptims(embConfig.optimNum * embConfig.embDim, 0.82);
    std::copy(finalOptims.begin(), finalOptims.end(), std::back_inserter(inOptims));

    unorderedMapTablePtr->InsertOrAssign(keys, inEmbs.data(), inOptims.data());

    // to check
    std::vector<int64_t> key_to_check{1};
    std::vector<float> outEmbs(embConfig.embDim);
    std::vector<float> outOptims(embConfig.optimNum * embConfig.embDim);
    unorderedMapTablePtr->FindOrInsert(key_to_check, outEmbs.data(), outOptims.data());

    LOG(INFO) << "outEmbs:" << StringTools::ToString(outEmbs);
    LOG(INFO) << "outOptims:" << StringTools::ToString(outOptims);

    ASSERT_EQ(finalEmbs, outEmbs);
    ASSERT_EQ(finalOptims, outOptims);
}

TEST_F(EmbTableTest, EmbTableFastHashMap_FindOrInsert)
{
    fastHashMapTablePtr = std::make_shared<EmbTableFastHashMap>(embConfig);

    std::vector<int64_t> keys{1, 1};
    std::vector<float> outEmbs(keys.size() * embConfig.embDim);
    std::vector<float> outOptims(keys.size() * embConfig.optimNum * embConfig.embDim);

    fastHashMapTablePtr->FindOrInsert(keys, outEmbs.data(), outOptims.data());

    LOG(INFO) << "keys.size():" << keys.size() << ", embDim:" << embConfig.embDim
              << ", optimNum:" << embConfig.optimNum;
    LOG(INFO) << "outEmbs.size():" << outEmbs.size() << ", outEmbs:" << StringTools::ToString(outEmbs);
    LOG(INFO) << "outOptims.size():" << outOptims.size() << ", outOptims:" << StringTools::ToString(outOptims);

    ASSERT_EQ(outEmbs.size(), keys.size() * embConfig.embDim);
    ASSERT_EQ(outOptims.size(), keys.size() * embConfig.optimNum * embConfig.embDim);
}

TEST_F(EmbTableTest, EmbTableFastHashMap_InsertOrAssign)
{
    fastHashMapTablePtr = std::make_shared<EmbTableFastHashMap>(embConfig);

    std::vector<int64_t> keys{1, 1};
    std::vector<float> inEmbs(embConfig.embDim, 0.51);
    std::vector<float> finalEmbs(embConfig.embDim, 0.52);
    std::copy(finalEmbs.begin(), finalEmbs.end(), std::back_inserter(inEmbs));

    std::vector<float> inOptims(embConfig.optimNum * embConfig.embDim, 0.81);
    std::vector<float> finalOptims(embConfig.optimNum * embConfig.embDim, 0.82);
    std::copy(finalOptims.begin(), finalOptims.end(), std::back_inserter(inOptims));

    fastHashMapTablePtr->InsertOrAssign(keys, inEmbs.data(), inOptims.data());

    // to check
    std::vector<int64_t> key_to_check{1};
    std::vector<float> outEmbs(embConfig.embDim);
    std::vector<float> outOptims(embConfig.optimNum * embConfig.embDim);
    fastHashMapTablePtr->FindOrInsert(key_to_check, outEmbs.data(), outOptims.data());

    LOG(INFO) << "outEmbs:" << StringTools::ToString(outEmbs);
    LOG(INFO) << "outOptims:" << StringTools::ToString(outOptims);

    ASSERT_EQ(finalEmbs, outEmbs);
    ASSERT_EQ(finalOptims, outOptims);
}

int main(int argc, char* argv[])
{
    return CommonMain(argc, argv);
}
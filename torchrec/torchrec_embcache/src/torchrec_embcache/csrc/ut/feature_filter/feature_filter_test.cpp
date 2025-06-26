/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <vector>

#include "feature_filter/feature_filter.h"
#include "../common_main.h"

using namespace Embcache;
using namespace std;

class FeatureFilterTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    std::unique_ptr<FeatureFilter> featureFilterPtr;
};

TEST_F(FeatureFilterTest, StatisticsAndFilterTest)
{
    featureFilterPtr = std::make_unique<FeatureFilter>(3, 0);
    int64_t keyData[12] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 1, 2, 3};
    int64_t start = 0;
    int64_t end = 12;
    featureFilterPtr->StatisticsAndFilter(keyData, start, end);
    int64_t expectData[12] = {1, 2, 3, -1, -1, -1, 1, 2, 3, 1, 2, 3};
    for (int i = 0; i < 12; ++i) {
        ASSERT_EQ(keyData[i], expectData[i]);
    }
}

int main(int argc, char* argv[])
{
    return CommonMain(argc, argv);
}

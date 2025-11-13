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

#include <vector>

#include "gtest/gtest.h"
#include "emock_def.h"

#define private public
#include "common/types.h"
#include "feature/count_filter.h"
#include "hdc/acl_channel.h"
#include "spdlog/spdlog.h"

namespace rec_sdk {
namespace feature {

using std::vector;

using common::emb_key_t;
using common::i32;

class CountFilterTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        testFilePath = "./test_count_filter.bin";
    }
    void TearDown() override {}

private:
    std::string testFilePath;
};

TEST_F(CountFilterTest, FilterOutAllKeys)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{0, 1, 2, 3, 4, 5};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    const auto expected = vector<emb_key_t>{-1, -1, -1, -1, -1, -1};

    countFilter.Filter(keys, cnts);
    EXPECT_EQ(keys, expected);
}

TEST_F(CountFilterTest, FilterOutSomeKeys)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3, 1, 2, 4};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    const auto expected = vector<emb_key_t>{1, 2, -1, 1, 2, -1};

    countFilter.Filter(keys, cnts);
    EXPECT_EQ(keys, expected);
}

TEST_F(CountFilterTest, FilterOutNoKey)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{1, 1, 2, 2, 3, 3};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    const auto expected = vector<emb_key_t>{1, 1, 2, 2, 3, 3};

    countFilter.Filter(keys, cnts);
    EXPECT_EQ(keys, expected);
}

TEST_F(CountFilterTest, SaveAndLoadOK)
{
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);

    auto keys = vector<emb_key_t>{1, 2, 3, 1, 2, 4};
    auto cnts = vector<i32>{1, 1, 1, 1, 1, 1};
    countFilter.Filter(keys, cnts);

    countFilter.Save(testFilePath);
    countFilter.histVisitedCnts_.clear();
    
    countFilter.Load(testFilePath);
    EXPECT_EQ(countFilter.histVisitedCnts_.size(), 4);
}

TEST_EMOCK_F(CountFilterTest, UT_GetDeviceToHostChannelName_Match)
{
    std::string tab00Name = "Table00";
    EMOCK(acltdtCreateChannelWithCapacity).stubs().with(any()).will(returnValue((acltdtChannelHandle*)0x1234));
    EMOCK(acltdtStopChannel).stubs().with(any()).will(returnValue(ACL_SUCCESS));
    EMOCK(acltdtDestroyChannel).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    auto countFilter = CountFilter(0, tab00Name, 10);
    auto d2HTransAclCh = dynamic_cast<rec_sdk::hdc::AclChannel *>(countFilter.d2hTransporter_.get());
    auto h2dTransAclCh = dynamic_cast<rec_sdk::hdc::AclChannel *>(countFilter.h2dTransporter_.get());
    if ((d2HTransAclCh == nullptr) || (h2dTransAclCh == nullptr)) {
        FAIL() << "nullptr ! dynamic_cast hdc::AclChannel error.";
    }

    auto d2hChName = d2HTransAclCh->channelName_;
    auto h2dChName = h2dTransAclCh->channelName_;

    auto exptD2hChName = fmt::format("{}_{}", tab00Name, CountFilter::D2H_CHANNEL_SUFFIX);
    auto exptH2dChName = fmt::format("{}_{}", tab00Name, CountFilter::H2D_CHANNEL_SUFFIX);

    EXPECT_EQ(exptD2hChName, d2hChName);
    EXPECT_EQ(exptH2dChName, h2dChName);
}

TEST_F(CountFilterTest, UT_SaveFlatHashMapToBinaryFile_Fail)
{
    std::string filePath = "";
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);
    EXPECT_THROW(countFilter.Save(filePath), std::runtime_error);
}

TEST_F(CountFilterTest, UT_LoadFlatHashMapFromBinaryFile_Fail)
{
    std::string filePath = "";
    auto countFilter = CountFilter(0, 2, nullptr, nullptr);
    EXPECT_THROW(countFilter.Load(filePath), std::runtime_error);
}

}  // namespace feat
}  // namespace rec_sdk

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

#include <gtest/gtest.h>
#include "utils/common.h"

TEST(TestVectorToString, Basic)
{
    std::vector<int> vec = {1, 2, 3};
    EXPECT_EQ(MxRec::VectorToString(vec), "[1, 2, 3]");
}

TEST(TestMapToString, Basic)
{
    std::map<int, int> map = {{1, 2}, {3, 4}};
    EXPECT_EQ(MxRec::MapToString(map), "{1: 2, 3: 4}");
}

// 测试 MapToString 函数可以处理 absl::flat_hash_map
TEST(TestMapToString, AbseilFlatHashMap)
{
    absl::flat_hash_map<int, int> map = {{1, 2}, {3, 4}};
    std::string result = MxRec::MapToString(map);
    EXPECT_TRUE(result.find("1: 2") != std::string::npos);
    EXPECT_TRUE(result.find("3: 4") != std::string::npos);
}

TEST(TestVec2TensorI32, Basic)
{
    std::vector<int> vec = {1, 2, 3};
    tensorflow::Tensor tensor = MxRec::Vec2TensorI32(vec);
    auto tensor_data = tensor.flat<tensorflow::int32>();
    for (int i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(tensor_data(i), vec[i]);
    }
}

TEST(TestVec2TensorI64, Basic) {
    std::vector<int64_t> vec = {1, 2, 3};
    tensorflow::Tensor tensor = MxRec::Vec2TensorI64(vec);
    auto tensor_data = tensor.flat<tensorflow::int64>();
    for (int i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(tensor_data(i), vec[i]);
    }
}

TEST(TestGetUBSize, InvalidDeviceID)
{
    EXPECT_THROW(MxRec::GetUBSize(999), std::runtime_error);
}

// 测试 Batch 结构的 Size 和 UnParse 方法
TEST(TestBatch, SizeAndUnParse)
{
    MxRec::Batch<int> batch;
    batch.sample = {1, 2, 3};
    EXPECT_EQ(batch.Size(), 3);
    EXPECT_EQ(batch.UnParse(), "1 2 3 ");
}

// 测试 RankInfo 结构的默认构造函数
TEST(TestRankInfo, DefaultConstructor)
{
    MxRec::RankInfo rankInfo;
}

// 测试 ThresholdValue 结构的默认构造函数
TEST(TestThresholdValue, DefaultConstructor)
{
    MxRec::ThresholdValue thresholdValue;
}

// 测试 FeatureItemInfo 结构的默认构造函数和带参数的构造函数
TEST(TestFeatureItemInfo, Constructors)
{
    MxRec::FeatureItemInfo featureItemInfo1;

    MxRec::FeatureItemInfo featureItemInfo2(123, 456);
}

// 测试 AdmitAndEvictData 结构的默认构造函数
TEST(TestAdmitAndEvictData, DefaultConstructor)
{
    MxRec::AdmitAndEvictData admitAndEvictData;
}

// 测试 EmbInfo 结构的默认构造函数
TEST(TestEmbInfo, DefaultConstructor)
{
    MxRec::EmbInfo embInfo;
}

// 测试 HostEmbTable 结构的默认构造函数
TEST(TestHostEmbTable, DefaultConstructor)
{
    MxRec::HostEmbTable hostEmbTable;
}

// 测试 All2AllInfo 结构的默认构造函数
TEST(TestAll2AllInfo, DefaultConstructor)
{
    MxRec::All2AllInfo all2AllInfo;
}

// 测试 UniqueInfo 结构的默认构造函数
TEST(TestUniqueInfo, DefaultConstructor)
{
    MxRec::UniqueInfo uniqueInfo;
}

// 测试 KeySendInfo 结构的默认构造函数
TEST(TestKeySendInfo, DefaultConstructor)
{
    MxRec::KeySendInfo keySendInfo;
}

// 测试 CkptTransData 结构的默认构造函数
TEST(TestCkptTransData, DefaultConstructor)
{
    MxRec::CkptTransData ckptTransData;
}

TEST(TestCTRLog, DifferentLevel)
{
    int invilid = -1;
    MxRec::CTRLog(MxRec::CTRLogLevel::DEBUG, "test message");
    MxRec::CTRLog(MxRec::CTRLogLevel::INFO, "test message");
    MxRec::CTRLog(MxRec::CTRLogLevel::WARN, "test message");
    MxRec::CTRLog(MxRec::CTRLogLevel::ERROR, "test message");
    EXPECT_NO_THROW(MxRec::CTRLog(invilid, "test message"));
}
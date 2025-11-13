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

#include <stdexcept>

#include "gtest/gtest.h"
#include "tensorflow/core/framework/tensor.h"

#include "feature/conv_util.h"
#include "common/types.h"

namespace rec_sdk {
namespace common  {


struct UtDataTypesMap {
    aclDataType aclDType;
    tensorflow::DataType tfDType;
    std::string aclDTypeStr;
};

struct UtAclTensorTypeStrMap {
    acltdtTensorType tensorType;
    std::string tensorTypeStr;
};

#define UT_DATA_TYPES_MAP_DEF(aclDType, tfDType) {(aclDType), (tfDType), (#aclDType)}
std::vector<struct UtDataTypesMap> g_utDataTypesMap = {
    UT_DATA_TYPES_MAP_DEF(ACL_FLOAT16, tensorflow::DT_HALF),
    UT_DATA_TYPES_MAP_DEF(ACL_FLOAT, tensorflow::DT_FLOAT),
    UT_DATA_TYPES_MAP_DEF(ACL_DOUBLE, tensorflow::DT_DOUBLE),
    UT_DATA_TYPES_MAP_DEF(ACL_INT8, tensorflow::DT_INT8),
    UT_DATA_TYPES_MAP_DEF(ACL_INT16, tensorflow::DT_INT16),
    UT_DATA_TYPES_MAP_DEF(ACL_INT32, tensorflow::DT_INT32),
    UT_DATA_TYPES_MAP_DEF(ACL_INT64, tensorflow::DT_INT64),
    UT_DATA_TYPES_MAP_DEF(ACL_UINT8, tensorflow::DT_UINT8),
    UT_DATA_TYPES_MAP_DEF(ACL_UINT16, tensorflow::DT_UINT16),
    UT_DATA_TYPES_MAP_DEF(ACL_UINT32, tensorflow::DT_UINT32),
    UT_DATA_TYPES_MAP_DEF(ACL_UINT64, tensorflow::DT_UINT64),
    UT_DATA_TYPES_MAP_DEF(ACL_BOOL, tensorflow::DT_BOOL),
};

#define UT_ACL_TENSOR_TYPE_STR_MAP_DEF(aclTensorType) {(aclTensorType), (#aclTensorType)}
std::vector<struct UtAclTensorTypeStrMap> g_utAclTensorTypeStrMap = {
    UT_ACL_TENSOR_TYPE_STR_MAP_DEF(ACL_TENSOR_DATA_TENSOR),
    UT_ACL_TENSOR_TYPE_STR_MAP_DEF(ACL_TENSOR_DATA_SLICE_TENSOR),
    UT_ACL_TENSOR_TYPE_STR_MAP_DEF(ACL_TENSOR_DATA_ABNORMAL),
    UT_ACL_TENSOR_TYPE_STR_MAP_DEF(ACL_TENSOR_DATA_END_TENSOR),
    UT_ACL_TENSOR_TYPE_STR_MAP_DEF(ACL_TENSOR_DATA_END_OF_SEQUENCE)
};

TEST(UtTypesTest, TEST_GetAclTensorTypeStrLoopCheckOk)
{
    for (auto it : g_utAclTensorTypeStrMap) {
        auto tensorTypeStr = GetAclTensorTypeStr(it.tensorType);
        EXPECT_EQ(tensorTypeStr, it.tensorTypeStr);
    }
}

TEST(UtTypesTest, TEST_GetAclDataTypeStrLoopCheckOk)
{
    for (auto it : g_utDataTypesMap) {
        auto dataTypeStr = GetAclDataTypeStr(it.aclDType);
        EXPECT_EQ(dataTypeStr, it.aclDTypeStr);
    }
}

TEST(UtTypesTest, TEST_MapDataTypeFromTFToAclLoopCheckOk)
{
    for (auto it : g_utDataTypesMap) {
        auto aclDType = MapDataTypeFromTFToAcl(it.tfDType);
        EXPECT_EQ(aclDType, it.aclDType);
    }
}

TEST(UtTypesTest, TEST_MapDataTypeFromAclToTFLoopCheckOk)
{
    for (auto it : g_utDataTypesMap) {
        auto tfDType = MapDataTypeFromAclToTF(it.aclDType);
        EXPECT_EQ(tfDType, it.tfDType);
    }
}

TEST(UtTypesTest, TEST_GetAclDataTypeStr_RetError)
{
    auto retStr = GetAclDataTypeStr(ACL_DT_UNDEFINED);
    EXPECT_EQ(retStr, "ACL_DT_UNDEFINED");
}

TEST(UtTypesTest, TEST_GetAclTensorTypeStr_RetError)
{
    auto retStr = GetAclTensorTypeStr(ACL_TENSOR_DATA_UNDEFINED);
    EXPECT_EQ(retStr, "ACL_TENSOR_DATA_UNDEFINED");
}

TEST(UtTypesTest, TEST_MapDataTypeFromTFToAcl_RetError)
{
    auto ret = MapDataTypeFromTFToAcl(tensorflow::DT_INVALID);
    EXPECT_EQ(ret, ACL_DT_UNDEFINED);
}

TEST(UtTypesTest, TEST_MapDataTypeFromAclToTF_RetError)
{
    auto ret = MapDataTypeFromAclToTF(ACL_DT_UNDEFINED);
    EXPECT_EQ(ret, tensorflow::DT_INVALID);
}

}  // namespace common
}  // namespace rec_sdk

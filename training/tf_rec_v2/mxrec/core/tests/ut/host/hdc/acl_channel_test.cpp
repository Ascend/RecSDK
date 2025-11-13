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

#include <array>
#include <cstdint>

#include "gtest/gtest.h"
#include "acl/acl_base.h"
#include "tensorflow/core/framework/tensor_shape.h"

#define private public
#include "hdc/acl_channel.h"
#include "common/types.h"
#include "emock_def.h"

namespace rec_sdk {
namespace hdc {

using std::array;
using common::i64;
using tensorflow::Tensor;

class UT_AclChannelTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        EMOCK(acltdtCreateChannelWithCapacity)
            .stubs().with(any()).will(returnValue(reinterpret_cast<acltdtChannelHandle*>(0x1234)));
        EMOCK(acltdtStopChannel).stubs().with(any()).will(returnValue(ACL_SUCCESS));
        EMOCK(acltdtDestroyChannel).stubs().with(any()).will(returnValue(ACL_SUCCESS));
    }
    void TearDown() override
    {
        GlobalMockObject::reset();
    }
};

TEST_F(UT_AclChannelTest, DataItemToTensorInt64)
{
    constexpr auto dims = array<int64_t, 1>{7};
    auto nums = array<i64, 7>{1, 2, 3, 4, 5, 6, 7};
    auto dataItem =
        acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, dims.data(), dims.size(), ACL_INT64, nums.data(), nums.size());

    auto tensor = AssembleDataItemToTensor(dataItem);
    auto tensorData = tensor.flat<i64>();

    ASSERT_EQ(tensor.dtype(), tensorflow::DT_INT64);
    ASSERT_EQ(tensor.dim_size(0), nums.size());

    for (size_t i = 0; i < nums.size(); ++i) {
        EXPECT_EQ(tensorData(i), nums[i]);
    }

    acltdtDestroyDataItem(dataItem);
}

TEST_F(UT_AclChannelTest, TensorToDataItemInt64)
{
    auto tensorShape = tensorflow::TensorShape({7});
    auto tensor = tensorflow::Tensor(tensorflow::DT_INT64, tensorShape);
    auto tensorData = tensor.flat<i64>();

    auto nums = array<i64, 7>{1, 2, 3, 4, 5, 6, 7};
    for (size_t i = 0; i < nums.size(); ++i) {
        tensorData(i) = nums[i];
    }

    auto dataItem = AssembleTensorToDataItem(std::move(tensor));

    ASSERT_EQ(acltdtGetDataTypeFromItem(dataItem), ACL_INT64);
    ASSERT_EQ(acltdtGetDimNumFromItem(dataItem), tensor.dims());

    auto data = reinterpret_cast<i64*>(acltdtGetDataAddrFromItem(dataItem));
    for (size_t i = 0; i < nums.size(); ++i) {
        EXPECT_EQ(data[i], nums[i]);
    }

    acltdtDestroyDataItem(dataItem);
}

// 测试SendTensors - 正常情况
TEST_EMOCK_F(UT_AclChannelTest, SendTensors_Normal)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_0";
    AclChannel channel(deviceId, channelName);

    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
    tensor.flat<int32_t>().setValues({1, 2, 3, 4});
    std::vector<tensorflow::Tensor> tensors = {tensor};

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0x1234)));

    EMOCK(acltdtAddDataItem).stubs().with(any(), any()).will(returnValue(ACL_SUCCESS));

    EMOCK(acltdtSendTensor).stubs().with(any(), any(), (int32_t)AclChannel::BLOCK_TIMEOUT)
        .will(returnValue(ACL_SUCCESS));

    EMOCK(acltdtDestroyDataset).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    EXPECT_NO_THROW(channel.SendTensors(std::move(tensors)));
}

// 测试SendTensors - 数据集创建失败
TEST_EMOCK_F(UT_AclChannelTest, SendTensors_CreateDatasetFailed)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_1";
    AclChannel channel(deviceId, channelName);

    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
    tensor.flat<int32_t>().setValues({1, 2, 3, 4});
    std::vector<tensorflow::Tensor> tensors = {tensor};

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0)));

    EXPECT_THROW(channel.SendTensors(std::move(tensors)), std::runtime_error);
}

// 测试RecvTensors - 正常情况
TEST_EMOCK_F(UT_AclChannelTest, RecvTensors_Normal)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_2";
    AclChannel channel(deviceId, channelName);
    constexpr auto dims = array<int64_t, 1>{7};
    auto nums = array<i64, 7>{1, 2, 3, 4, 5, 6, 7};
    auto dataItem =
        acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, dims.data(), dims.size(), ACL_INT64, nums.data(), nums.size());

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0x1234)));
    EMOCK(acltdtReceiveTensor).stubs().with(any(), any(), (int32_t)AclChannel::BLOCK_TIMEOUT)
        .will(returnValue(ACL_SUCCESS));

    EMOCK(acltdtGetDatasetSize).stubs().with(any()).will(returnValue(1));
    EMOCK(acltdtGetDataItem).stubs().with(any(), 0).will(returnValue(dataItem));
    EMOCK(acltdtGetTensorTypeFromItem).stubs().with(any()).will(returnValue(ACL_TENSOR_DATA_TENSOR));
    EMOCK(acltdtGetDataTypeFromItem).stubs().with(any()).will(returnValue(ACL_UINT32));
    EMOCK(acltdtGetDimNumFromItem).stubs().with(any()).will(returnValue(2));
    EMOCK(acltdtDestroyDataset).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    std::vector<Tensor> tensors;
    EXPECT_NO_THROW({
        tensors = channel.RecvTensors();
    });

    EXPECT_EQ(tensors.size(), 1);
}

// 测试RecvTensors - 数据集创建失败
TEST_EMOCK_F(UT_AclChannelTest, RecvTensors_CreateDatasetFailed)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_3";
    AclChannel channel(deviceId, channelName);

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0)));

    EXPECT_THROW(channel.RecvTensors(), std::runtime_error);
}

// 测试SendTensors - 添加数据项失败
TEST_EMOCK_F(UT_AclChannelTest, SendTensors_AddDataItemFailed)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_4";
    AclChannel channel(deviceId, channelName);

    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
    tensor.flat<int32_t>().setValues({1, 2, 3, 4});
    std::vector<tensorflow::Tensor> tensors = {tensor};

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0x1234)));
    EMOCK(acltdtAddDataItem).stubs().with(any(), any()).will(returnValue(ACL_ERROR_INVALID_PARAM));
    EMOCK(acltdtDestroyDataset).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    EXPECT_THROW(channel.SendTensors(std::move(tensors)), std::runtime_error);
}

// 测试RecvTensors - 接收失败
TEST_EMOCK_F(UT_AclChannelTest, RecvTensors_ReceiveFailed)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_5";
    AclChannel channel(deviceId, channelName);

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0x1234)));
    EMOCK(acltdtReceiveTensor)
        .stubs()
        .with(any(), any(), (int32_t)AclChannel::BLOCK_TIMEOUT)
        .will(returnValue(ACL_ERROR_UNINITIALIZE));
    EMOCK(acltdtDestroyDataset).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    EXPECT_THROW(channel.RecvTensors(), std::runtime_error);
}

TEST_EMOCK_F(UT_AclChannelTest, AclChannel_InitacltdtCreateChannelWithCapacityFail)
{
    GlobalMockObject::reset();

    EMOCK(acltdtCreateChannelWithCapacity)
        .stubs()
        .with(any())
        .will(returnValue(reinterpret_cast<acltdtChannelHandle*>(0)));
    EMOCK(acltdtStopChannel).stubs().with(any()).will(returnValue(ACL_SUCCESS));
    EMOCK(acltdtDestroyChannel).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_6";
    EXPECT_THROW(AclChannel channel(deviceId, channelName), std::runtime_error);
}

TEST_EMOCK_F(UT_AclChannelTest, AclChannel_ReInitacltdtCreateChannelWithCapacityFail)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel_7";
    AclChannel channel(deviceId, channelName);
    EXPECT_THROW(AclChannel channel(deviceId, channelName), std::runtime_error);
}

// 测试SendTensors - 队列满重试成功
TEST_EMOCK_F(UT_AclChannelTest, SendTensors_RetrySuccess)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel";
    AclChannel channel(deviceId, channelName);

    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
    tensor.flat<int32_t>().setValues({1, 2, 3, 4});
    std::vector<tensorflow::Tensor> tensors = {tensor};

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0x1234)));
    EMOCK(acltdtAddDataItem).stubs().with(any(), any()).will(returnValue(ACL_SUCCESS));
    EMOCK(acltdtSendTensor).stubs().with(any(), any(), (int32_t)AclChannel::BLOCK_TIMEOUT)
        .will(returnValue(ACL_ERROR_RT_QUEUE_FULL))
        .then(returnValue(ACL_SUCCESS));
    EMOCK(acltdtDestroyDataset).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    channel.SendTensors(std::move(tensors));
}

// 测试SendTensors - 发送失败
TEST_EMOCK_F(UT_AclChannelTest, SendTensors_SendFailed)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel";
    AclChannel channel(deviceId, channelName);

    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
    tensor.flat<int32_t>().setValues({1, 2, 3, 4});
    std::vector<tensorflow::Tensor> tensors = {tensor};

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0x1234)));
    EMOCK(acltdtAddDataItem).stubs().with(any(), any()).will(returnValue(ACL_SUCCESS));
    EMOCK(acltdtSendTensor).stubs().with(any(), any(), (int32_t)AclChannel::BLOCK_TIMEOUT)
        .will(returnValue(ACL_ERROR_RT_OVER_LIMIT));
    EMOCK(acltdtDestroyDataset).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    EXPECT_THROW({
        channel.SendTensors(std::move(tensors));
    }, std::runtime_error);
}

// 测试RecvTensors - 解析数据项失败
TEST_EMOCK_F(UT_AclChannelTest, RecvTensors_ParseDataItemFailed)
{
    const uint32_t deviceId = 0;
    const std::string channelName = "test_channel";
    AclChannel channel(deviceId, channelName);

    EMOCK(acltdtCreateDataset).stubs().will(returnValue(reinterpret_cast<acltdtDataset*>(0x1234)));
    EMOCK(acltdtReceiveTensor).stubs().with(any(), any(), (int32_t)AclChannel::BLOCK_TIMEOUT)
        .will(returnValue(ACL_SUCCESS));
    EMOCK(acltdtGetDatasetSize).stubs().with(any()).will(returnValue(1));
    EMOCK(acltdtGetDataItem).stubs().with(any(), 0).will(returnValue(reinterpret_cast<acltdtDataItem*>(0)));
    EMOCK(acltdtDestroyDataset).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    EXPECT_THROW(channel.RecvTensors(), std::runtime_error);
}

TEST_EMOCK_F(UT_AclChannelTest, InitDataItem_Uint32_Success)
{
    tensorflow::Tensor tensor(tensorflow::DT_UINT32, tensorflow::TensorShape({2, 2}));
    tensor.flat<tensorflow::uint32>().setValues({1, 2, 3, 4});
    aclDataType aclDType = ACL_UINT32;
    acltdtDataItem* dataItem = nullptr;

    EMOCK(acltdtCreateDataItem).stubs().will(returnValue(reinterpret_cast<acltdtDataItem*>(0x1234)));

    InitDataItem(dataItem, aclDType, tensor);

    EXPECT_NE(dataItem, nullptr);
}

TEST_EMOCK_F(UT_AclChannelTest, InitDataItem_Uint64_Success)
{
    tensorflow::Tensor tensor(tensorflow::DT_UINT64, tensorflow::TensorShape({2, 2}));
    tensor.flat<tensorflow::uint64>().setValues({1, 2, 3, 4});
    aclDataType aclDType = ACL_UINT64;
    acltdtDataItem* dataItem = nullptr;

    EMOCK(acltdtCreateDataItem).stubs().will(returnValue(reinterpret_cast<acltdtDataItem*>(0x1234)));

    InitDataItem(dataItem, aclDType, tensor);

    EXPECT_NE(dataItem, nullptr);
}

TEST_F(UT_AclChannelTest, InitDataItem_UnsupportedType_ThrowException)
{
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 2}));
    aclDataType aclDType = ACL_FLOAT;
    acltdtDataItem* dataItem = nullptr;

    EXPECT_THROW({
        InitDataItem(dataItem, aclDType, tensor);
    }, std::runtime_error);
}


// 测试AssembleDataItemToTensor - 数据项为空
TEST_F(UT_AclChannelTest, AssembleDataItemToTensor_NullDataItem)
{
    EXPECT_THROW(AssembleDataItemToTensor(nullptr), std::runtime_error);
}

// 测试AssembleDataItemToTensor - 数据项类型不正确
TEST_EMOCK_F(UT_AclChannelTest, AssembleDataItemToTensor_InvalidTensorType)
{
    acltdtDataItem* dataItem = reinterpret_cast<acltdtDataItem*>(0x1234);
    EMOCK(acltdtGetTensorTypeFromItem)
        .stubs()
        .with(reinterpret_cast<const acltdtDataItem*>(dataItem))
        .will(returnValue(ACL_TENSOR_DATA_UNDEFINED));

    EXPECT_THROW(AssembleDataItemToTensor(dataItem), std::runtime_error);
}

// 测试AssembleDataItemToTensor - 数据类型不支持
TEST_EMOCK_F(UT_AclChannelTest, AssembleDataItemToTensor_UnsupportedDataType)
{
    acltdtDataItem* dataItem = reinterpret_cast<acltdtDataItem*>(0x1234);
    EMOCK(acltdtGetTensorTypeFromItem)
        .stubs()
        .with(reinterpret_cast<const acltdtDataItem*>(dataItem))
        .will(returnValue(ACL_TENSOR_DATA_TENSOR));
    EMOCK(acltdtGetDataTypeFromItem)
        .stubs()
        .with(reinterpret_cast<const acltdtDataItem*>(dataItem))
        .will(returnValue(ACL_DT_UNDEFINED));

    EXPECT_THROW(AssembleDataItemToTensor(dataItem), std::runtime_error);
}

// 测试AssembleDataItemToTensor - 获取维度失败
TEST_EMOCK_F(UT_AclChannelTest, AssembleDataItemToTensor_GetDimsFailed)
{
    acltdtDataItem* dataItem = reinterpret_cast<acltdtDataItem*>(0x1234);
    EMOCK(acltdtGetTensorTypeFromItem)
        .stubs().with(reinterpret_cast<const acltdtDataItem*>(dataItem)).will(returnValue(ACL_TENSOR_DATA_TENSOR));
    EMOCK(acltdtGetDataTypeFromItem)
        .stubs().with(reinterpret_cast<const acltdtDataItem*>(dataItem)).will(returnValue(ACL_INT32));
    EMOCK(acltdtGetDimNumFromItem)
        .stubs().with(reinterpret_cast<const acltdtDataItem*>(dataItem)).will(returnValue(2));
    EMOCK(acltdtGetDimsFromItem)
        .stubs().with(reinterpret_cast<const acltdtDataItem*>(dataItem), any(), 2)
        .will(returnValue(ACL_ERROR_INVALID_PARAM));

    EXPECT_THROW(AssembleDataItemToTensor(dataItem), std::runtime_error);
}

// 测试AssembleTensorToDataItem - Tensor为空
TEST_F(UT_AclChannelTest, AssembleTensorToDataItem_EmptyTensor)
{
    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({0}));

    EXPECT_THROW(AssembleTensorToDataItem(std::move(tensor)), std::runtime_error);
}

// 测试AssembleTensorToDataItem - 数据类型无法映射
TEST_F(UT_AclChannelTest, AssembleTensorToDataItem_UnmappedDataType)
{
    tensorflow::Tensor tensor(tensorflow::DT_STRING, tensorflow::TensorShape({2, 2}));

    EXPECT_THROW(AssembleTensorToDataItem(std::move(tensor)), std::runtime_error);
}

// 测试AssembleTensorToDataItem - 初始化数据项失败
TEST_EMOCK_F(UT_AclChannelTest, AssembleTensorToDataItem_InitFailed)
{
    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
    tensor.flat<int32_t>().setValues({1, 2, 3, 4});

    EMOCK(acltdtCreateDataItem).stubs().will(returnValue(reinterpret_cast<acltdtDataItem*>(0)));

    EXPECT_THROW(AssembleTensorToDataItem(std::move(tensor)), std::runtime_error);
}

// 测试DT_INT32情况
TEST_EMOCK_F(UT_AclChannelTest, InitTensor_INT32)
{
    tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
    std::vector<int32_t> data = {1, 2};
    acltdtDataItem* dataItem = reinterpret_cast<acltdtDataItem*>(0x1234);

    EMOCK(acltdtGetDataAddrFromItem)
        .stubs().with(reinterpret_cast<const acltdtDataItem*>(dataItem)).will(returnValue((void*)data.data()));

    InitTensor(tensor, dataItem);

    EXPECT_EQ(tensor.flat<tensorflow::int32>()(0), 1);
    EXPECT_EQ(tensor.flat<tensorflow::int32>()(1), 2);
}

// 测试DT_UINT64情况
TEST_EMOCK_F(UT_AclChannelTest, InitTensor_UINT64)
{
    tensorflow::Tensor tensor(tensorflow::DT_UINT64, tensorflow::TensorShape({2}));
    std::vector<uint64_t> data = {0x1122334455667788u, 0x8877665544332211u};
    acltdtDataItem* dataItem = reinterpret_cast<acltdtDataItem*>(0x1234);

    EMOCK(acltdtGetDataAddrFromItem)
        .stubs().with(reinterpret_cast<const acltdtDataItem*>(dataItem)).will(returnValue((void*)data.data()));

    InitTensor(tensor, dataItem);

    EXPECT_EQ(tensor.flat<tensorflow::uint64>()(0), 0x1122334455667788u);
    EXPECT_EQ(tensor.flat<tensorflow::uint64>()(1), 0x8877665544332211u);
}

// 测试默认情况，抛出异常
TEST_F(UT_AclChannelTest, InitTensor_UnsupportedType)
{
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({2}));
    acltdtDataItem* dataItem = reinterpret_cast<acltdtDataItem*>(0);

    EXPECT_THROW(InitTensor(tensor, dataItem), std::runtime_error);
}


}  // namespace trans
}  // namespace rec_sdk

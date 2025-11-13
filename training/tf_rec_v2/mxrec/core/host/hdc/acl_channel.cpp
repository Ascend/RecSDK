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

#include "acl_channel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "spdlog/spdlog.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "common/types.h"
#include "common/version.h"

namespace rec_sdk {
namespace hdc {

using common::emb_key_t;
using std::runtime_error;
using std::vector;
using tensorflow::Tensor;

std::unordered_set<std::string> AclChannel::channelNames_ = {};
std::mutex AclChannel::channelNamesMutex_ = {};

AclChannel::AclChannel(uint32_t deviceId, const std::string& channelName)
    : deviceId_(deviceId),
      channelName_(channelName)
{
    {
        std::lock_guard<std::mutex> _(channelNamesMutex_);
        if (channelNames_.find(channelName_) != channelNames_.end()) {
            spdlog::error("Channel name already exists ==> deviceId={}, channelName={} <==", deviceId_, channelName_);
            throw runtime_error("channel name already exists");
        }
        channelNames_.insert(channelName_);
    }

    channelHandle_ = acltdtCreateChannelWithCapacity(deviceId_, channelName_.c_str(), AclChannel::DEFAULT_CAP);
    if (channelHandle_ == nullptr) {
        spdlog::error("Failed to create acl channel! ==> deviceId={}, channelName={} <==", deviceId_, channelName_);
        throw runtime_error("acl create channel failed.");
    }
}

AclChannel::~AclChannel()
{
    auto ret = acltdtStopChannel(channelHandle_);
    if (ret != ACL_SUCCESS) {
        spdlog::error("Failed to stop acl channel! ==> aclError={}, deviceId={}, channelName={} <==", ret, deviceId_,
                      channelName_);
        std::terminate();
    }

    ret = acltdtDestroyChannel(channelHandle_);
    if (ret != ACL_SUCCESS) {
        spdlog::error("Failed to destroy acl channel! ==> aclError={}, deviceId={}, channelName={} <==", ret, deviceId_,
                      channelName_);
        std::terminate();
    }

    {
        std::lock_guard<std::mutex> _(channelNamesMutex_);
        channelNames_.erase(channelName_);
    }
    hdcRunning_ = false;
}

void AclChannel::SendTensors(std::vector<Tensor>&& tensors) const
{
    spdlog::debug("Start sending tensors through acl channel. ==> deviceId={}, channelName={} <==", deviceId_,
                  channelName_);

    auto dataset = acltdtCreateDataset();
    if (dataset == nullptr) {
        spdlog::error("Failed to create acl dataset! ==> deviceId={}, channelName={} <==", deviceId_, channelName_);
        throw runtime_error("acl create dataset failed");
    }

    for (auto& tensor : tensors) {
        acltdtDataItem* dataItem = nullptr;
        try {
            dataItem = AssembleTensorToDataItem(std::move(tensor));
        } catch (const std::runtime_error& err) {
            acltdtDestroyDataset(dataset);
            throw err;
        }

        auto ret = acltdtAddDataItem(dataset, dataItem);
        if (ret != ACL_SUCCESS) {
            spdlog::error("Failed to add acl data item to acl dataset! ==> deviceId={}, channelName={} <==", deviceId_,
                          channelName_);
            acltdtDestroyDataset(dataset);
            throw runtime_error("acl add data item failed");
        }
    }

    auto ret = acltdtSendTensor(channelHandle_, dataset, AclChannel::BLOCK_TIMEOUT);
    while (ret == ACL_ERROR_RT_QUEUE_FULL) {
        ret = acltdtSendTensor(channelHandle_, dataset, AclChannel::BLOCK_TIMEOUT);
        spdlog::trace("Retry to send tensor as the queue of acl channel is full. ==> deviceId={}, channelName={} <==",
                      deviceId_, channelName_);
    }
    if (ret != ACL_SUCCESS) {
        spdlog::error("Failed to send acl tensor! ==> aclError={}  deviceId={}, channelName={} <==", ret, deviceId_,
                      channelName_);
        acltdtDestroyDataset(dataset);
        throw runtime_error("acl send tensor failed");
    }

    ret = acltdtDestroyDataset(dataset);
    if (ret != ACL_SUCCESS) {
        spdlog::error("Failed to destroy acl dataset! ==> aclError={} <==", ret);
        throw runtime_error("acl destroy dataset failed");
    }

    spdlog::debug("Finish sending tensors through acl channel. ==> deviceId={}, channelName={} <==", deviceId_,
                  channelName_);
}

vector<Tensor> AclChannel::RecvTensors() const
{
    spdlog::debug("Start receiving tensors through acl channel. ==> deviceId={}, channelName={} <==", deviceId_,
                  channelName_);

    auto tensors = vector<Tensor>();

    const auto dataset = acltdtCreateDataset();
    if (dataset == nullptr) {
        spdlog::error("Failed to create acl dataset! ==> deviceId={}, channelName={} <==", deviceId_, channelName_);
        throw runtime_error("acl create dataset failed");
    }

    auto ret = acltdtReceiveTensor(channelHandle_, dataset, AclChannel::BLOCK_TIMEOUT);
    if (ret != ACL_SUCCESS) {
        spdlog::error("Failed to receive acl tensor! ==> aclError={}, deviceId={}, channelName={} <==", ret, deviceId_,
                      channelName_);
        acltdtDestroyDataset(dataset);
        throw runtime_error("acl receive tensor failed");
    }

    auto datasetLen = acltdtGetDatasetSize(dataset);
    for (size_t i = 0; i < datasetLen; i++) {
        auto dataItem = acltdtGetDataItem(dataset, i);
        if (dataItem == nullptr) {
            spdlog::error("Failed to get acl data item from acl dataset. ==> deviceId={}, channelName={} <==",
                          deviceId_, channelName_);
            acltdtDestroyDataset(dataset);
            throw runtime_error("acl get data item from dataset failed");
        }

        auto tensor = Tensor();
        try {
            tensor = AssembleDataItemToTensor(dataItem);
        } catch (const std::runtime_error& err) {
            acltdtDestroyDataset(dataset);
            throw err;
        }

        tensors.push_back(tensor);
    }

    ret = acltdtDestroyDataset(dataset);
    if (ret != ACL_SUCCESS) {
        spdlog::error("Failed to destroy acl dataset! ==> aclError={} <==", ret);
        throw runtime_error("acl destroy dataset failed");
    }

    spdlog::debug("Finish receiving tensors through acl channel. ==> deviceId={}, channelName={} <==", deviceId_,
                  channelName_);

    return tensors;
};

Tensor AssembleDataItemToTensor(acltdtDataItem* dataItem)
{
    if (dataItem == nullptr) {
        spdlog::error("Got a null dataItem.");
        throw runtime_error("null dataItem");
    }

    auto aclTensorType = acltdtGetTensorTypeFromItem(dataItem);
    if (aclTensorType != ACL_TENSOR_DATA_TENSOR) {
        auto typeStr = common::GetAclTensorTypeStr(aclTensorType);
        spdlog::error("Got an unexpected acl tensor type! ==> aclTensorType={} <==", typeStr);
        throw runtime_error("unexpected acl tensor type");
    }

    auto aclDType = acltdtGetDataTypeFromItem(dataItem);
    auto tfDType = common::MapDataTypeFromAclToTF(aclDType);
    if (!tensorflow::DataTypeCanUseMemcpy(tfDType)) {
        auto tfTypeStr = tensorflow::DataTypeString(tfDType);
        spdlog::error("Got uncopiable tf data type! ==> tfDataType={} <==", tfTypeStr);
        throw runtime_error("uncopiable tf data type");
    }

    auto dataItemDimNum = acltdtGetDimNumFromItem(dataItem);
    auto dataItemDims = std::vector<int64_t>(dataItemDimNum);
    auto ret = acltdtGetDimsFromItem(dataItem, dataItemDims.data(), dataItemDimNum);
    if (ret != ACL_SUCCESS) {
        spdlog::error("Failed to get dims from data item! ==> aclError={} <==", ret);
        throw runtime_error("acl get dims failed.");
    }

    auto tensorShape = tensorflow::TensorShape();
    std::for_each(dataItemDims.begin(), dataItemDims.end(), [&](int64_t x) { tensorShape.AddDim(x); });

    auto tensor = Tensor(tfDType, tensorShape);

#ifdef TF_1
    InitTensor(tensor, dataItem);
#else
    auto dataItemPtr = static_cast<char*>(acltdtGetDataAddrFromItem(dataItem));
    auto tensorTotalBytes = tensor.TotalBytes();
    std::copy(dataItemPtr, dataItemPtr + tensorTotalBytes, static_cast<char*>(tensor.data()));
#endif

    return tensor;
}

void InitTensor(Tensor& tensor, acltdtDataItem* dataItem)
{
    auto initTensor = [&tensor, &dataItem](auto type) {
        using T = decltype(type);
        auto dataItemPtr = static_cast<T*>(acltdtGetDataAddrFromItem(dataItem));
        auto flatTensor = tensor.flat<T>();
        std::copy(dataItemPtr, dataItemPtr + flatTensor.size(), flatTensor.data());
    };

    switch (tensor.dtype()) {
        case tensorflow::DT_INT32: {
            initTensor(common::i32());
            break;
        }
        case tensorflow::DT_INT64: {
            initTensor(common::i64());
            break;
        }
        case tensorflow::DT_UINT32: {
            initTensor(common::u32());
            break;
        }
        case tensorflow::DT_UINT64: {
            initTensor(common::u64());
            break;
        }
        default: {
            auto tfTypeStr = tensorflow::DataTypeString(tensor.dtype());
            spdlog::error("Got unsupported tf data type, only ['int32', 'int64', 'uint32', 'uint64'] available! ==> "
                          "tfDataType={} <==",
                          tfTypeStr);
            throw runtime_error("unsupported tf data type");
        }
    }
}

acltdtDataItem* AssembleTensorToDataItem(Tensor&& tensor)
{
    if (tensor.NumElements() == 0) {
        spdlog::error("Got an empty tensor.");
        throw runtime_error("empty tensor");
    }

    auto aclDType = common::MapDataTypeFromTFToAcl(tensor.dtype());
    if (aclDType == ACL_DT_UNDEFINED) {
        spdlog::error("Got undefined acl data type from tf data type! ==> tfDataType={} <==",
                      tensorflow::DataTypeString(tensor.dtype()));
        throw runtime_error("undeined acl data type from tf data type.");
    }

    acltdtDataItem* dataItem = nullptr;

#ifdef TF_1
    InitDataItem(dataItem, aclDType, tensor);
#else
    auto tensorDims = tensor.shape().dim_sizes();
    auto tensorPtr = tensor.data();
    dataItem = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, tensorDims.data(), tensorDims.size(), aclDType, tensorPtr,
                                    tensor.TotalBytes());
#endif
    if (dataItem == nullptr) {
        spdlog::error("Failed to create acl data item!");
        throw runtime_error("acl create data item failed");
    }

    return dataItem;
}

void InitDataItem(acltdtDataItem*& dataItem, aclDataType aclDType, tensorflow::Tensor& tensor)
{
    auto initDataItem = [&dataItem, &aclDType, &tensor](auto type) {
        using T = decltype(type);
        auto tensorDims = tensor.shape().dim_sizes();
        auto tensorDimsPtr = reinterpret_cast<int64_t*>(tensorDims.data());
        auto flatTensor = tensor.flat<T>();
        dataItem = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, tensorDimsPtr, tensorDims.size(), aclDType,
                                        flatTensor.data(), tensor.TotalBytes());
    };

    switch (tensor.dtype()) {
        case tensorflow::DT_INT32: {
            initDataItem(common::i32());
            break;
        }
        case tensorflow::DT_INT64: {
            initDataItem(common::i64());
            break;
        }
        case tensorflow::DT_UINT32: {
            initDataItem(common::u32());
            break;
        }
        case tensorflow::DT_UINT64: {
            initDataItem(common::u64());
            break;
        }
        default: {
            auto tfTypeStr = tensorflow::DataTypeString(tensor.dtype());
            spdlog::error("Got unsupported tf data type, only ['int32', 'int64', 'uint32', 'uint64'] available! ==> "
                          "tfDataType={} <==",
                          tfTypeStr);
            throw runtime_error("unsupported tf data type");
        }
    }
}

}  // namespace hdc
}  // namespace rec_sdk

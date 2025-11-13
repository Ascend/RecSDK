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

#ifndef REC_SDK_ACL_CHANNEL_H
#define REC_SDK_ACL_CHANNEL_H

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "acl/acl_tdt.h"

#include "transporter.h"

namespace rec_sdk {
namespace hdc {

class AclChannel : public Transporter {
public:
    explicit AclChannel() = delete;
    explicit AclChannel(uint32_t deviceId, const std::string& channelName);

    ~AclChannel() override;

    AclChannel(const AclChannel& other) = delete;
    AclChannel(AclChannel&& other) noexcept = delete;

    AclChannel& operator=(const AclChannel& other) = delete;
    AclChannel& operator=(AclChannel&& other) noexcept = delete;

    void SendTensors(std::vector<tensorflow::Tensor>&& tensors) const override;

    std::vector<tensorflow::Tensor> RecvTensors() const override;

private:
    static constexpr uint32_t DEFAULT_CAP = 100;
    static constexpr uint32_t BLOCK_TIMEOUT = -1;

    static std::mutex channelNamesMutex_;
    static std::unordered_set<std::string> channelNames_;

    uint32_t deviceId_;
    std::string channelName_;
    acltdtChannelHandle* channelHandle_;
};

tensorflow::Tensor AssembleDataItemToTensor(acltdtDataItem* dataItem);

void InitTensor(tensorflow::Tensor& tensor, acltdtDataItem* dataItem);

acltdtDataItem* AssembleTensorToDataItem(tensorflow::Tensor&& tensor);

void InitDataItem(acltdtDataItem*& dataItem, aclDataType aclDType, tensorflow::Tensor& tensor);

}  // namespace hdc
}  // namespace rec_sdk

#endif // REC_SDK_ACL_CHANNEL_H

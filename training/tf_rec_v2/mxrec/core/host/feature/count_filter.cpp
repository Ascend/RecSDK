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

#include "count_filter.h"

#include <cstdint>
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "spdlog/spdlog.h"
#include "tensorflow/core/framework/tensor.h"

#include "hdc/acl_channel.h"
#include "common/types.h"
#include "common/constants.h"
#include "conv_util.h"
#include "io_util.h"

namespace rec_sdk {
namespace feature {

using std::string;
using std::vector;

using common::emb_key_t;
using common::i32;

CountFilter::CountFilter(const int64_t deviceId, const string& tableName, const i32 minUsedTimes)
    : deviceId_(deviceId),
      tableName_(tableName),
      minUsedTimes_(minUsedTimes)
{
    d2hTransporter_ = std::make_unique<hdc::AclChannel>(deviceId, GetDeviceToHostChannelName());
    h2dTransporter_ = std::make_unique<hdc::AclChannel>(deviceId, GetHostToDeviceChannelName());
}

void CountFilter::Start()
{
    future_ = std::async(std::launch::async, [this]() {
        while (d2hTransporter_->hdcRunning_) {
            auto recvTensors = d2hTransporter_->RecvTensors();

            auto keys = ConvertTensorToVector1D<emb_key_t>(recvTensors[0]);
            auto cnts = ConvertTensorToVector1D<i32>(recvTensors[1]);

            Filter(keys, cnts);

            auto validKeyTensor = ConvertVectorToTensor1D(std::move(keys));
            auto sendTensors = vector<tensorflow::Tensor>{validKeyTensor};

            h2dTransporter_->SendTensors(std::move(sendTensors));
        }
    });
}

void CountFilter::Save(const string& filePath)
{
    auto result = SaveBinaryFileWithTensorKV(histVisitedCnts_, filePath);
    if (!result) {
        spdlog::error("Failed to save count filter data for table ==> {} <== to path ==> {} <==.", tableName_,
                      filePath);
        throw std::runtime_error("failed to save count filter data");
    }
}

void CountFilter::Load(const string& filePath)
{
    auto result = LoadBinaryFileWithTensorKV(histVisitedCnts_, filePath);
    if (!result) {
        spdlog::error("Failed to load count filter data for table ==> {} <== from path ==> {} <==.", tableName_,
                      filePath);
        throw std::runtime_error("failed to load count filter data");
    }
}

void CountFilter::Filter(vector<emb_key_t>& keys, const vector<i32>& cnts)
{
    for (size_t i = 0; i < keys.size(); i++) {
        auto& key = keys[i];
        auto& cnt = cnts[i];
        if (histVisitedCnts_.find(key) == histVisitedCnts_.end()) {
            histVisitedCnts_[key] = cnt;
        } else {
            histVisitedCnts_[key] += cnt;
        }
    }

    for (auto& key : keys) {
        if (histVisitedCnts_[key] < minUsedTimes_) {
            key = common::INVALID_KEY;
        }
    }
}

string CountFilter::GetDeviceToHostChannelName() const
{
    return fmt::format("{}_{}", tableName_, CountFilter::D2H_CHANNEL_SUFFIX);
}

string CountFilter::GetHostToDeviceChannelName() const
{
    return fmt::format("{}_{}", tableName_, CountFilter::H2D_CHANNEL_SUFFIX);
}

}  // namespace feat
}  // namespace rec_sdk

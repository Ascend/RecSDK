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

#include "time_evictor.h"

#include <algorithm>
#include <future>
#include <memory>
#include <vector>
#include <string>

#include "absl/container/flat_hash_map.h"

#include "hdc/acl_channel.h"
#include "common/types.h"
#include "conv_util.h"
#include "io_util.h"

namespace rec_sdk {
namespace feature {

using std::string;
using std::vector;

using common::emb_key_t;

TimeEvictor::TimeEvictor(const int64_t deviceId, const string& tableName, const uint64_t maxColdSecs)
    : deviceId_(deviceId),
      tableName_(tableName),
      maxColdSecs_(maxColdSecs)
{
    d2hTransporter_ = std::make_unique<hdc::AclChannel>(deviceId_, GetDeviceToHostChannelName());
}

void TimeEvictor::Start()
{
    future_ = std::async(std::launch::async, [this]() {
        while (d2hTransporter_->hdcRunning_) {
            auto recv_tensors = d2hTransporter_->RecvTensors();
            auto keys = ConvertTensorToVector1D<emb_key_t>(std::move(recv_tensors[0]));
            this->Update(keys);
        }
    });
}

vector<emb_key_t> TimeEvictor::Evict()
{
    auto evictedKeys = vector<emb_key_t>();
    const auto now = std::chrono::system_clock::now();

    for (const auto& [key, lastVisitedTime] : lastVisitedTimes_) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastVisitedTime).count();
        auto coldSecs = static_cast<uint64_t>(duration);

        if (coldSecs > maxColdSecs_) {
            evictedKeys.push_back(key);
        }
    }

    for (const auto& key : evictedKeys) {
        lastVisitedTimes_.erase(key);
    }

    return evictedKeys;
}

void TimeEvictor::Save(const string& filePath)
{
    auto result = SaveFlatHashMapToBinaryFile(lastVisitedTimes_, filePath);
    if (!result) {
        spdlog::error("Failed to save time evictor data for table ==> {} <== to path ==> {} <==.", tableName_,
                      filePath);
        throw std::runtime_error("failed to save time evictor data");
    }
}

void TimeEvictor::Load(const string& filePath)
{
    auto result = LoadFlatHashMapFromBinaryFile(lastVisitedTimes_, filePath);
    if (!result) {
        spdlog::error("Failed to load time evictor data for table ==> {} <== from path ==> {} <==.", tableName_,
                      filePath);
        throw std::runtime_error("failed to load time evictor data");
    }
}

void TimeEvictor::Update(const vector<emb_key_t>& keys)
{
    std::for_each(keys.begin(), keys.end(),
                  [&](emb_key_t key) { lastVisitedTimes_[key] = std::chrono::system_clock::now(); });
}

string TimeEvictor::GetDeviceToHostChannelName() const
{
    return fmt::format("{}_{}", tableName_, TimeEvictor::D2H_CHANNEL_SUFFIX);
}

}  // namespace feature
}  // namespace rec_sdk

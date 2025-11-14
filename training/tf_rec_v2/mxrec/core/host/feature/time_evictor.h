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

#ifndef REC_SDK_TIME_EVICTOR_H
#define REC_SDK_TIME_EVICTOR_H

#include <chrono>
#include <cstdint>
#include <future>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "hdc/transporter.h"
#include "common/types.h"

namespace rec_sdk {
namespace feature {

class TimeEvictor {
public:
    explicit TimeEvictor() = delete;
    explicit TimeEvictor(const int64_t deviceId, const std::string& tableName, const uint64_t maxColdSecs);
    explicit TimeEvictor(const int64_t deviceId, const uint64_t maxColdSecs,
                         std::unique_ptr<hdc::Transporter> d2hTransporter)
        : deviceId_(deviceId),
          maxColdSecs_(maxColdSecs),
          d2hTransporter_(std::move(d2hTransporter))
    {
    }

    TimeEvictor(const TimeEvictor& rhs) = delete;
    TimeEvictor(TimeEvictor&& rhs) noexcept = delete;

    TimeEvictor& operator=(const TimeEvictor& rhs) = delete;
    TimeEvictor& operator=(TimeEvictor&& rhs) noexcept = delete;

    ~TimeEvictor() noexcept = default;

    void Start();
    void Save(const std::string& filePath);
    void Load(const std::string& filePath);
    std::vector<common::emb_key_t> Evict();

private:
    void Update(const std::vector<common::emb_key_t>& keys);

    std::string GetDeviceToHostChannelName() const;

private:
    static constexpr std::string_view D2H_CHANNEL_SUFFIX = "time_evictor_d2h";
    static constexpr std::string_view H2D_CHANNEL_SUFFIX = "time_evictor_h2d";

    int64_t deviceId_;
    std::string tableName_;
    uint64_t maxColdSecs_;
    std::future<void> future_;
    absl::flat_hash_map<common::emb_key_t, std::chrono::time_point<std::chrono::system_clock>> lastVisitedTimes_;

    std::unique_ptr<hdc::Transporter> d2hTransporter_;
};

}  // namespace feature
}  // namespace rec_sdk

#endif // REC_SDK_TIME_EVICTOR_H

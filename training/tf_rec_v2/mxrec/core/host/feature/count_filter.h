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

#ifndef REC_SDK_COUNT_FILTER_H
#define REC_SDK_COUNT_FILTER_H

#include <cstdint>
#include <future>
#include <memory>

#include "absl/container/flat_hash_map.h"

#include "hdc/transporter.h"
#include "common/types.h"

namespace rec_sdk {
namespace feature {

class CountFilter {
public:
    explicit CountFilter() = delete;
    explicit CountFilter(const int64_t deviceId, const std::string& tableName, const common::i32 minUsedTimes);
    explicit CountFilter(const int64_t deviceId, const common::i32 minUsedTimes,
                         std::unique_ptr<hdc::Transporter> d2hTransporter,
                         std::unique_ptr<hdc::Transporter> h2dTransporter)
        : deviceId_(deviceId),
          minUsedTimes_(minUsedTimes),
          d2hTransporter_(std::move(d2hTransporter)),
          h2dTransporter_(std::move(h2dTransporter))
    {
    }

    CountFilter(const CountFilter& rhs) = delete;
    CountFilter(CountFilter&& rhs) noexcept = delete;

    CountFilter& operator=(const CountFilter& rhs) = delete;
    CountFilter& operator=(CountFilter&& rhs) noexcept = delete;

    ~CountFilter() noexcept = default;

    void Start();
    void Save(const std::string& filePath);
    void Load(const std::string& filePath);

private:
    void Filter(std::vector<common::emb_key_t>& keys, const std::vector<common::i32>& cnts);

    std::string GetDeviceToHostChannelName() const;
    std::string GetHostToDeviceChannelName() const;

private:
    static constexpr std::string_view D2H_CHANNEL_SUFFIX = "count_filter_d2h";
    static constexpr std::string_view H2D_CHANNEL_SUFFIX = "count_filter_h2d";

    int64_t deviceId_;
    std::string tableName_;
    common::i32 minUsedTimes_;
    std::future<void> future_;
    absl::flat_hash_map<common::emb_key_t, common::i32> histVisitedCnts_;

    std::unique_ptr<hdc::Transporter> d2hTransporter_;
    std::unique_ptr<hdc::Transporter> h2dTransporter_;
};

}  // namespace feat
}  // namespace rec_sdk

#endif // REC_SDK_COUNT_FILTER_H

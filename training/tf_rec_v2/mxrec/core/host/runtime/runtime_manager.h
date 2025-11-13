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

#ifndef REC_SDK_RUNTIME_MANAGER_H
#define REC_SDK_RUNTIME_MANAGER_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/types.h"
#include "feature/count_filter.h"
#include "feature/time_evictor.h"

namespace rec_sdk {
namespace runtime {

class RuntimeManager {
public:
    explicit RuntimeManager() = delete;
    explicit RuntimeManager(int32_t deviceId);

    RuntimeManager(const RuntimeManager& rhs) = delete;
    RuntimeManager(const RuntimeManager&& rhs) noexcept = delete;

    RuntimeManager operator=(const RuntimeManager& rhs) = delete;
    RuntimeManager operator=(const RuntimeManager&& rhs) noexcept = delete;

    ~RuntimeManager() noexcept;

    void StartCountFilter(const std::string tableName, common::i32 minUsedTimes);
    void StartTimeEvictor(const std::string tableName, common::u64 maxColdSecs);

    void SaveCountFilter(const std::string tableName, const std::string filePath) const;
    void SaveTimeEvictor(const std::string tableName, const std::string filePath) const;

    void LoadCountFilter(const std::string tableName, const std::string filePath) const;
    void LoadTimeEvictor(const std::string tableName, const std::string filePath) const;

    std::vector<common::emb_key_t> GetEvictedKeys(const std::string tableName) const;

private:
    int32_t deviceId_;
    std::unordered_map<std::string, std::unique_ptr<feature::CountFilter>> countFilters_;
    std::unordered_map<std::string, std::unique_ptr<feature::TimeEvictor>> timeFilters_;
};

}  // namespace runtime
}  // namespace rec_sdk

#endif // REC_SDK_RUNTIME_MANAGER_H

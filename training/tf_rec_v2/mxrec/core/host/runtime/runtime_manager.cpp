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

#include "runtime_manager.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "acl/acl.h"
#include "spdlog/spdlog.h"

#include "common/types.h"

namespace rec_sdk {
namespace runtime {

using std::string;

using common::emb_key_t;
using common::i32;
using common::u64;

RuntimeManager::RuntimeManager(int32_t deviceId) : deviceId_(deviceId)
{
    auto err = aclInit(nullptr);
    if (err != ACL_SUCCESS) {
        spdlog::error("Failed to exec acl initialization, error code: {}.", err);
        throw std::runtime_error("failed to exec acl initialization.");
    }

    err = aclrtSetDevice(deviceId_);
    if (err != ACL_SUCCESS) {
        spdlog::error("Failed to set device, error code: {}.", err);
        throw std::runtime_error("failed to set device.");
    }
};

RuntimeManager::~RuntimeManager() noexcept
{
    auto err = aclrtResetDevice(deviceId_);
    if (err != ACL_SUCCESS) {
        spdlog::error("Failed to reset device, error code: {}.", err);
    }

    err = aclFinalize();
    if (err != ACL_SUCCESS) {
        spdlog::error("Failed to finalize acl, error code: {}.", err);
    }
}

void RuntimeManager::StartCountFilter(const string tableName, i32 minUsedTimes)
{
    auto countFilter = std::make_unique<feature::CountFilter>(deviceId_, tableName, minUsedTimes);

    countFilter->Start();

    countFilters_.emplace(tableName, std::move(countFilter));
}

void RuntimeManager::StartTimeEvictor(const string tableName, u64 maxColdSecs)
{
    auto timeEvictor = std::make_unique<feature::TimeEvictor>(deviceId_, tableName, maxColdSecs);

    timeEvictor->Start();

    timeFilters_.emplace(tableName, std::move(timeEvictor));
}

void RuntimeManager::LoadCountFilter(const std::string tableName, const std::string filePath) const
{
    auto it = countFilters_.find(tableName);
    if (it == countFilters_.end()) {
        spdlog::error("Count filter for table {} doesn't exist.", tableName);
        throw std::invalid_argument("count filter doesn't exist");
    }

    it->second->Load(filePath);
}

void RuntimeManager::LoadTimeEvictor(const std::string tableName, const std::string filePath) const
{
    auto it = timeFilters_.find(tableName);
    if (it == timeFilters_.end()) {
        spdlog::error("Time evictor for table {} doesn't exist.", tableName);
        throw std::invalid_argument("time evictor doesn't exist");
    }

    it->second->Load(filePath);
}

void RuntimeManager::SaveCountFilter(const string tableName, const string filePath) const
{
    auto it = countFilters_.find(tableName);
    if (it == countFilters_.end()) {
        spdlog::error("Count filter for table {} doesn't exist.", tableName);
        throw std::invalid_argument("count filter doesn't exist");
    }

    it->second->Save(filePath);
}

void RuntimeManager::SaveTimeEvictor(const string tableName, const string filePath) const
{
    auto it = timeFilters_.find(tableName);
    if (it == timeFilters_.end()) {
        spdlog::error("Time evictor for table {} doesn't exist.", tableName);
        throw std::invalid_argument("time evictor doesn't exist");
    }

    it->second->Save(filePath);
}

std::vector<common::emb_key_t> RuntimeManager::GetEvictedKeys(const string tableName) const
{
    auto it = timeFilters_.find(tableName);
    if (it == timeFilters_.end()) {
        spdlog::error("Time evictor for table {} doesn't exist.", tableName);
        throw std::invalid_argument("time evictor doesn't exist");
    }

    return it->second->Evict();
}

}  // namespace runtime
}  // namespace rec_sdk

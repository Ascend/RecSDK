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

#ifndef REC_SDK_IO_UTIL_H
#define REC_SDK_IO_UTIL_H

#include <chrono>
#include <fstream>
#include <string>
#include <fcntl.h>

#include "absl/container/flat_hash_map.h"
#include "spdlog/spdlog.h"

#include "common/types.h"

namespace rec_sdk {
namespace feature {

constexpr int FILE_RIGHT_MODE = 0640;
constexpr long long FILE_MAX_SIZE = 1LL << 40;

size_t GetFileSize(const std::string& filePath);
bool ValidateReadFile(const std::string& filePath);

bool SaveBinaryFileWithTensorKV(const absl::flat_hash_map<common::emb_key_t, common::i32>& map,
                                const std::string& filePath);

bool LoadBinaryFileWithTensorKV(absl::flat_hash_map<common::emb_key_t, common::i32>& map,
                                const std::string& filePath);

bool SaveFlatHashMapToBinaryFile(
    const absl::flat_hash_map<common::emb_key_t, std::chrono::time_point<std::chrono::system_clock>>& map,
    const std::string& filePath);

bool LoadFlatHashMapFromBinaryFile(
    absl::flat_hash_map<common::emb_key_t, std::chrono::time_point<std::chrono::system_clock>>& map,
    const std::string& filePath);

}  // namespace feature
}  // namespace rec_sdk

#endif // REC_SDK_IO_UTIL_H

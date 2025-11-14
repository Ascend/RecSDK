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

#include "io_util.h"

#include <experimental/filesystem>
#include <sys/stat.h>

namespace rec_sdk {
namespace feature {

namespace fs = std::experimental::filesystem;

size_t GetFileSize(const std::string& filePath)
{
    std::ifstream readFile;
    readFile.open(filePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!readFile.is_open()) {
        spdlog::error("Open file {} to get file size failed.", filePath);
        throw std::runtime_error("failed to get file size");
    }
    size_t fileSize = static_cast<size_t>(readFile.tellg());
    readFile.close();
    return fileSize;
}

bool ValidateReadFile(const std::string& filePath)
{
    // validate soft link
    struct stat fileInfo;
    if (lstat(filePath.c_str(), &fileInfo) == -1) {
        spdlog::error("Failed to lstat {}", filePath);
        return false;
    }
    if (S_ISLNK(fileInfo.st_mode)) {
        spdlog::error("Found soft link in path: {}.", filePath);
        return false;
    }

    // validate file size
    size_t fileSize;
    try {
        fileSize = GetFileSize(filePath);
    } catch (std::runtime_error& e) {
        spdlog::error("Failed to GetFileSize {}", filePath);
        return false;
    }
    if (fileSize > FILE_MAX_SIZE) {
        spdlog::error("The reading file {} size is invalid FILE_MAX_SIZE:{}", filePath, FILE_MAX_SIZE);
        return false;
    }

    // validate file privilege
    fs::perms permissions = fs::status(filePath).permissions();
    if ((permissions & fs::perms::owner_read) == fs::perms::none) {
        spdlog::error("no read permission for file: {}.", filePath);
        return false;
    }
    return true;
}

bool SaveBinaryFileWithTensorKV(const absl::flat_hash_map<common::emb_key_t, common::i32>& map,
                                const std::string& filePath)
{
    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, FILE_RIGHT_MODE);
    if (fd == -1) {
        spdlog::error("Failed to open file: {}, error info: {}.", filePath, strerror(errno));
        return false;
    }

    size_t len = map.size();
    if (write(fd, &len, sizeof(len)) != sizeof(len)) {
        spdlog::error("Failed to write length: {}, file: {}, error info: {}.", len, filePath, strerror(errno));
        close(fd);
        return false;
    }

    for (const auto& [key, value] : map) {
        if (write(fd, &key, sizeof(common::emb_key_t)) != sizeof(common::emb_key_t)) {
            spdlog::error("Failed to write key: {}, file: {}, error info: {}.", key, filePath, strerror(errno));
            close(fd);
            return false;
        }

        if (write(fd, &value, sizeof(common::i32)) != sizeof(common::i32)) {
            spdlog::error("Failed to write value: {}, file: {}, error info: {}.", value, filePath, strerror(errno));
            close(fd);
            return false;
        }
    }
    close(fd);
    return true;
}

bool LoadBinaryFileWithTensorKV(absl::flat_hash_map<common::emb_key_t, common::i32>& map,
                                const std::string& filePath)
{
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        spdlog::error("Failed to open file: {}, error info: {}.", filePath, strerror(errno));
        return false;
    }

    if (ValidateReadFile(filePath) != true) {
        spdlog::error("ValidateReadFile is false file: {}.", filePath);
        close(fd);
        return false;
    }

    size_t len = 0;
    if (read(fd, &len, sizeof(len)) != sizeof(len)) {
        spdlog::error("Failed to read length from file: {}, error info: {}.", filePath, strerror(errno));
        close(fd);
        return false;
    }

    map.clear();
    for (size_t i = 0; i < len; ++i) {
        common::emb_key_t key;
        common::i32 value;

        if (read(fd, &key, sizeof(common::emb_key_t)) != sizeof(common::emb_key_t)) {
            spdlog::error("Failed to read key from file: {}, error info: {}.", filePath, strerror(errno));
            close(fd);
            return false;
        }
        if (read(fd, &value, sizeof(common::i32)) != sizeof(common::i32)) {
            spdlog::error("Failed to read value from file: {}, error info: {}.", filePath, strerror(errno));
            close(fd);
            return false;
        }

        map.emplace(key, value);
    }

    close(fd);
    return true;
}

bool SaveFlatHashMapToBinaryFile(
    const absl::flat_hash_map<common::emb_key_t, std::chrono::time_point<std::chrono::system_clock>>& map,
    const std::string& filePath)
{
    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, FILE_RIGHT_MODE);
    if (fd == -1) {
        spdlog::error("Failed to open file: {}, error info: {}.", filePath, strerror(errno));
        return false;
    }

    size_t len = map.size();
    if (write(fd, &len, sizeof(len)) != sizeof(len)) {
        spdlog::error("Failed to write length: {}, file: {}, error info: {}.", len, filePath, strerror(errno));
        close(fd);
        return false;
    }

    for (const auto& [key, value] : map) {
        if (write(fd, &key, sizeof(common::emb_key_t)) != sizeof(common::emb_key_t)) {
            spdlog::error("Failed to write key: {}, file: {}, error info: {}.", key, filePath, strerror(errno));
            close(fd);
            return false;
        }

        auto duration_since_epoch = value.time_since_epoch();
        auto count = std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_epoch).count();
        if (write(fd, &count, sizeof(count)) != sizeof(count)) {
            spdlog::error("Failed to write value: {}, file: {}, error info: {}.", key, filePath, strerror(errno));
            close(fd);
            return false;
        }
    }

    close(fd);
    return true;
}

bool LoadFlatHashMapFromBinaryFile(
    absl::flat_hash_map<common::emb_key_t, std::chrono::time_point<std::chrono::system_clock>>& map,
    const std::string& filePath)
{
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        spdlog::error("Failed to open file: {}, error info: {}.", filePath, strerror(errno));
        return false;
    }

    if (ValidateReadFile(filePath) != true) {
        spdlog::error("ValidateReadFile is false file: {}.", filePath);
        close(fd);
        return false;
    }

    size_t len = 0;
    if (read(fd, &len, sizeof(len)) != sizeof(len)) {
        spdlog::error("Failed to read length from file: {}, error info: {}.", filePath, strerror(errno));
        close(fd);
        return false;
    }

    map.clear();
    for (size_t i = 0; i < len; ++i) {
        common::emb_key_t key;
        auto count = 0;

        if (read(fd, &key, sizeof(common::emb_key_t)) != sizeof(common::emb_key_t)) {
            spdlog::error("Failed to read key from file: {}, error info: {}.", filePath, strerror(errno));
            close(fd);
            return false;
        }
        if (read(fd, &count, sizeof(count)) != sizeof(count)) {
            spdlog::error("Failed to read value from file: {}, error info: {}.", filePath, strerror(errno));
            close(fd);
            return false;
        }

        auto value = std::chrono::time_point<std::chrono::system_clock>(std::chrono::milliseconds(count));
        map.emplace(key, value);
    }

    close(fd);
    return true;
}

}  // namespace feature
}  // namespace rec_sdk
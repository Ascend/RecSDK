/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef MX_REC_FILE_SYSTEM_H
#define MX_REC_FILE_SYSTEM_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

#include "utils/logger.h"

using namespace Embcache;

namespace MxRec {
using namespace std;

constexpr long long FILE_MAX_SIZE = 1LL << 40;  // 1TB
constexpr int FILE_MIN_SIZE = 0;

class FileSystem {
public:
    FileSystem() = default;
    virtual ~FileSystem() = default;

    virtual void CreateDir(const string& dirName) = 0;
    virtual vector<string> ListDir(const string& dirName) = 0;
    virtual size_t GetFileSize(const string& filePath) = 0;

    virtual ssize_t Write(const string& filePath, const char* fileContent, size_t dataSize) = 0;

    virtual ssize_t Write(const string& filePath, const char* fileContent, size_t dataSize, int fd) = 0;

    virtual ssize_t Read(const string& filePath, char* fileContent, size_t datasetSize) = 0;
    virtual ssize_t Read(const string& filePath, vector<vector<float>>& fileContent, int64_t contentOffset,
                         vector<int64_t> offsetArr, const size_t& embeddingSize) = 0;
    virtual void CreateFileDir(const string& filePath) = 0;
    virtual void Valid4WriteDir(const string& fileDirPath) = 0;
    // The parameter oneTimeReadWriteLen specifies the maximum length of a file read or write at a time.
    // The parameter can be adjusted based on the service requirements.
    const size_t oneTimeReadWriteLen = 32768;
    const int embHashNum = 1;
    const int keyAddrElem = 1;
    std::vector<char> buffer;
    std::vector<char> writeBuffer;
};
}  // namespace MxRec

#endif  // MX_REC_FILE_SYSTEM_H

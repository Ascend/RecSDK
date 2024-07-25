/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <gtest/gtest.h>

#include "file_system/file_system_handler.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(LocalFileSystem, WriteAndReadFile)
{
    string filePath = "./write.data";
    vector<int64_t> writeData = {0, 1, 2, 3, 4, 5};
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    ssize_t res = fileSystemPtr->Write(filePath, reinterpret_cast<const char *>(writeData.data()),
                                    writeData.size() * sizeof(int64_t));

    ASSERT_EQ(writeData.size() * sizeof(int64_t), res);
    vector<int64_t> readData = {};
    readData.reserve(6);
    res = fileSystemPtr->Read(filePath, reinterpret_cast<char *>(readData.data()),
                                      writeData.size() * sizeof(int64_t));
    ASSERT_EQ(writeData.size() * sizeof(int64_t), res);
}

TEST(LocalFileSystem, WriteEmbedding)
{
    string filePath = "./write.data";
    vector<float> writeData = {1.1, 2.2, 3.3, 4.4, 5.5};
    vector<vector<float>> writeData1 = {writeData};
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    ssize_t res = fileSystemPtr->Write(filePath, writeData1, sizeof(float));
    ASSERT_EQ(writeData.size() * sizeof(float), res);
}

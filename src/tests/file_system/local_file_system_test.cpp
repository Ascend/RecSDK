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
#include <emock/emock.hpp>
#include <vector>

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

TEST(LocalFileSystem, WriteEmbedding_Error)
{
    string filePath = "./not/exist";
    int embSize = 1;
    vector<int64_t> addr = {1, 2, 3, 4, 5};
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_THROW(fileSystemPtr->WriteEmbedding(filePath, embSize, addr, 0), std::runtime_error);
}

TEST(LocalFileSystem, ReadEmbedding)
{
    string filePath = "./test/path";
    EmbeddingSizeInfo embedSizeInfo;
    int64_t faddr = 1;
    int deviceId = 0;
    vector<int64_t> offsetArr = {0, 1, 2, 3, 4};
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_NO_THROW(fileSystemPtr->ReadEmbedding(filePath, embedSizeInfo, faddr, deviceId, offsetArr));
}

TEST(LocalFileSystem, Read_InvalidPath_Error)
{
    string filePath = "./not/exist";
    char buffer[1024];
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_THROW(fileSystemPtr->Read(filePath, buffer, sizeof(buffer)), std::runtime_error);
}

TEST(LocalFileSystem, Create_FilePathEmpty_Error)
{
    string filePath = "";
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    EXPECT_THROW(fileSystemHandler->Create(filePath), std::runtime_error);
}

TEST(LocalFileSystem, CreateDir_MaxDepth_Error)
{
    string filePath;
    int maxDepth = 101;
    for (int i = 0; i < maxDepth; ++i) {
        filePath += "/level" + std::to_string(i);
    }
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_THROW(fileSystemPtr->CreateDir(filePath), std::runtime_error);
}

TEST(LocalFileSystem, ListDir_NullptrDir)
{
    string filePath = "./not/exist";
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    auto res = fileSystemPtr->ListDir(filePath);
    EXPECT_TRUE(res.empty());
}

TEST(LocalFileSystem, CheckOpenFile4Write_OpenError)
{
    string filePath = "./not/exist";
    auto fileSystemPtr = make_unique<LocalFileSystem>();
    EXPECT_THROW(fileSystemPtr->CheckOpenFile4Write(filePath, -1), std::runtime_error);
}

TEST(LocalFileSystem, CheckOpenFileRet_NullptrError)
{
    string filePath = "./test/path";
    FILE* fp = nullptr;
    auto fileSystemPtr = make_unique<LocalFileSystem>();
    EXPECT_THROW(fileSystemPtr->CheckOpenFileRet(fp, filePath), std::runtime_error);
}
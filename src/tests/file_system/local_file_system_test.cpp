/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */

#include <gtest/gtest.h>

#include "file_system/file_system_handler.h"
#include "file_system/local_file_system/local_file_system.h"

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


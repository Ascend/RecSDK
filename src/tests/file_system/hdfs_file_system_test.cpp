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

#include "file_system/file_system_handler.h"
#include "file_system/hdfs_file_system/hdfs_file_system.h"
#include "file_system/hdfs_file_system/hdfs_wrapper.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace emock;

void MockHdfs()
{
    hdfsFS ConnectFs;
    hdfsFile hdfsFileHandler;
    hdfsFileInfo* fileInfo;
    EMOCK(&HdfsWrapper::LoadHdfsLib).stubs().will(ignoreReturnValue());
    EMOCK(&HdfsWrapper::CloseHdfsLib).stubs().will(ignoreReturnValue());
    EMOCK(&HdfsWrapper::Connect).stubs().will(returnValue(ConnectFs));
    EMOCK(&HdfsWrapper::Disconnect).stubs().will(returnValue(1));
    EMOCK(&HdfsWrapper::ListDirectory).stubs().will(returnValue(fileInfo));
    EMOCK(&HdfsWrapper::FreeFileInfo).stubs().will(ignoreReturnValue());
    EMOCK(&HdfsWrapper::OpenFile).stubs().will(returnValue(hdfsFileHandler));
    EMOCK(&HdfsWrapper::CloseFile).stubs().will(returnValue(1));
    EMOCK(&HdfsWrapper::Seek).stubs().will(returnValue(1));
}


class HdfsFileSystemTest : public testing::Test {
protected:
    HdfsFileSystemTest() {
    }

    void SetUp()
    {
        MockHdfs();
    }

    void TearDown() {
    }
};

TEST_F(HdfsFileSystemTest, CreateDir)
{
    EMOCK(&HdfsWrapper::CreateDirectory).stubs().will(returnValue(1));
    string filePath = "hdfs://master:9000/test_dir/";
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_NO_THROW(fileSystemPtr->CreateDir(filePath));
}

TEST_F(HdfsFileSystemTest, CreateDirFailed)
{
    EMOCK(&HdfsWrapper::CreateDirectory).stubs().will(returnValue(-1));
    string filePath = "hdfs://master:9000/test_dir/";
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_NO_THROW(fileSystemPtr->CreateDir(filePath));
}

TEST_F(HdfsFileSystemTest, GetFileSize)
{
    hdfsFileInfo* fileInfo;
    EMOCK(&HdfsWrapper::GetPathInfo).stubs().will(returnValue(fileInfo));
    string filePath = "hdfs://master:9000/test_dir/";
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_NO_THROW(fileSystemPtr->GetFileSize(filePath));
}


/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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
#include "file_system/hdfs_file_system/hdfs_wrapper.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace emock;

using WriteCharFunc = tSize (HdfsWrapper::*)(hdfsFS, hdfsFile, const char*, tSize) const;
using WriteFloatFunc = tSize (HdfsWrapper::*)(hdfsFS, hdfsFile, const float*, tSize) const;
using ReadCharFunc = tSize (HdfsWrapper::*)(hdfsFS, hdfsFile, char*, tSize) const;
using ReadFloatFunc = tSize (HdfsWrapper::*)(hdfsFS, hdfsFile, float*, tSize) const;

void MockHdfs()
{
    EMOCK(&HdfsWrapper::LoadHdfsLib).stubs().will(ignoreReturnValue());
    hdfsFS ConnectFs;
    hdfsFile hdfsFileHandler;
    hdfsFileInfo* fileInfo;
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

    void TearDown()
    {
        GlobalMockObject::reset();
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
    EXPECT_THROW(fileSystemPtr->CreateDir(filePath), std::runtime_error);
}

TEST_F(HdfsFileSystemTest, GetFileSize)
{
    std::unique_ptr<hdfsFileInfo> fileInfo = std::make_unique<hdfsFileInfo>();
    EMOCK(&HdfsWrapper::GetPathInfo).stubs().will(returnValue(fileInfo.get()));
    string filePath = "hdfs://master:9000/test_dir/";
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_NO_THROW(fileSystemPtr->GetFileSize(filePath));
}

TEST_F(HdfsFileSystemTest, GetFileSize_Error)
{
    std::unique_ptr<hdfsFileInfo> fileInfo = nullptr;
    EMOCK(&HdfsWrapper::GetPathInfo).stubs().will(returnValue(fileInfo.get()));
    string filePath = "hdfs://master:9000/test_dir/";
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    auto fileSystemPtr = fileSystemHandler->Create(filePath);
    EXPECT_THROW(fileSystemPtr->GetFileSize(filePath), std::runtime_error);
}

TEST_F(HdfsFileSystemTest, ListDir)
{
    EMOCK(&HdfsWrapper::CreateDirectory).stubs().will(returnValue(-1));
    string dirName = "testName";
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_NO_THROW(fileSystemPtr->ListDir(dirName));
}


TEST_F(HdfsFileSystemTest, WriteReadChar)
{
    string filePath = "testPath";
    const char* fileContent = "test";
    char* readContent = "test";
    size_t dataSize = 1;
    EMOCK(&HdfsFileSystem::CheckOpenHdfsFileRet).stubs();
    EMOCK(static_cast<WriteCharFunc>(&HdfsWrapper::Write)).stubs().will(returnValue(1));
    EMOCK(static_cast<ReadCharFunc>(&HdfsWrapper::Read)).stubs().will(returnValue(1));
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_EQ(fileSystemPtr->Write(filePath, fileContent, dataSize), 1);
    EXPECT_EQ(fileSystemPtr->Read(filePath, readContent, dataSize), 1);
}

TEST_F(HdfsFileSystemTest, WriteReadChar_ReturnRes)
{
    string filePath = "testPath";
    const char* fileContent = "test";
    char* readContent = "test";
    size_t dataSize = 1;
    EMOCK(&HdfsFileSystem::CheckOpenHdfsFileRet).stubs();
    EMOCK(static_cast<WriteCharFunc>(&HdfsWrapper::Write)).stubs().will(returnValue(-1));
    EMOCK(static_cast<ReadCharFunc>(&HdfsWrapper::Read)).stubs().will(returnValue(-1));
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_EQ(fileSystemPtr->Write(filePath, fileContent, dataSize), -1);
    EXPECT_EQ(fileSystemPtr->Read(filePath, readContent, dataSize), -1);
}

TEST_F(HdfsFileSystemTest, WriteReadFloat)
{
    string filePath = "testPath";
    vector<vector<float>> fileContent = {{1.1f, 2.2f, 3.3f}};
    size_t dataSize = 1;
    int64_t contentOffset = 1;
    vector<int64_t> offsetArr = {1};
    EMOCK(&HdfsFileSystem::CheckOpenHdfsFileRet).stubs();
    EMOCK(static_cast<WriteFloatFunc>(&HdfsWrapper::Write)).stubs().will(returnValue(1));
    EMOCK(static_cast<ReadFloatFunc>(&HdfsWrapper::Read)).stubs().will(returnValue(1));
    EMOCK(&HdfsWrapper::Seek).stubs().will(returnValue(1));
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_EQ(fileSystemPtr->Write(filePath, fileContent, dataSize), 1);
    EXPECT_EQ(fileSystemPtr->Read(filePath, fileContent, contentOffset, offsetArr, dataSize), 1);
}

TEST_F(HdfsFileSystemTest, WriteReadFloat_ReturnRes)
{
    string filePath = "testPath";
    vector<vector<float>> fileContent = {{1.1f, 2.2f, 3.3f}};
    size_t dataSize = 1;
    int64_t contentOffset = 1;
    vector<int64_t> offsetArr = {1};
    EMOCK(&HdfsFileSystem::CheckOpenHdfsFileRet).stubs();
    EMOCK(static_cast<WriteFloatFunc>(&HdfsWrapper::Write)).stubs().will(returnValue(-1));
    EMOCK(static_cast<ReadFloatFunc>(&HdfsWrapper::Read)).stubs().will(returnValue(-1));
    EMOCK(&HdfsWrapper::Seek).stubs().will(returnValue(1));
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_EQ(fileSystemPtr->Write(filePath, fileContent, dataSize), -1);
    EXPECT_EQ(fileSystemPtr->Read(filePath, fileContent, contentOffset, offsetArr, dataSize), -1);
}

TEST_F(HdfsFileSystemTest, WriteFloat_SizeZero)
{
    string filePath = "testPath";
    vector<vector<float>> fileContent = {};
    size_t dataSize = 1;
    EMOCK(&HdfsFileSystem::CheckOpenHdfsFileRet).stubs();
    EMOCK(static_cast<WriteFloatFunc>(&HdfsWrapper::Write)).stubs().will(returnValue(1));
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_EQ(fileSystemPtr->Write(filePath, fileContent, dataSize), 0);
}

TEST_F(HdfsFileSystemTest, WriteEmbedding)
{
    string filePath = "testPath";
    int embeddingSize = 1;
    vector<int64_t> addr = {1, 2, 3, 4, 5};
    int deviceId = 0;
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EMOCK(&HdfsFileSystem::CheckOpenHdfsFileRet).stubs();
    EXPECT_NO_THROW(fileSystemPtr->WriteEmbedding(filePath, embeddingSize, addr, deviceId));
}

TEST_F(HdfsFileSystemTest, WriteEmbedding_Error)
{
    string filePath = "testPath";
    int embeddingSize = 1;
    vector<int64_t> addr = {1, 2, 3, 4, 5};
    int deviceId = 0;
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_THROW(fileSystemPtr->WriteEmbedding(filePath, embeddingSize, addr, deviceId), std::runtime_error);
}

TEST_F(HdfsFileSystemTest, ReadEmbedding)
{
    string filePath = "testPath";
    EmbeddingSizeInfo embedSizeInfo;
    int64_t faddr = 1;
    int deviceId = 0;
    vector<int64_t> offsetArr = {1, 2, 3, 4, 5};
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_NO_THROW(fileSystemPtr->ReadEmbedding(filePath, embedSizeInfo, faddr, deviceId, offsetArr));
}

TEST_F(HdfsFileSystemTest, CheckHdfsReadRet)
{
    hdfsFile file;
    tSize res = 1;
    size_t expectReadBytes = 1;
    const string filePath = "testPath";
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_NO_THROW(fileSystemPtr->CheckHdfsReadRet(file, res, expectReadBytes, filePath));
}

TEST_F(HdfsFileSystemTest, CheckHdfsReadRet_ReadingFileError)
{
    hdfsFile file;
    tSize res = -1;
    size_t expectReadBytes = 1;
    const string filePath = "testPath";
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_THROW(fileSystemPtr->CheckHdfsReadRet(file, res, expectReadBytes, filePath), std::runtime_error);
}

TEST_F(HdfsFileSystemTest, CheckHdfsReadRet_ReadByteError)
{
    hdfsFile file;
    tSize res = 0;
    size_t expectReadBytes = 1;
    const string filePath = "testPath";
    auto fileSystemPtr = make_unique<HdfsFileSystem>();
    EXPECT_THROW(fileSystemPtr->CheckHdfsReadRet(file, res, expectReadBytes, filePath), std::runtime_error);
}
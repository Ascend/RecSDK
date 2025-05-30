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

#include <random>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <limits>
#include <mpi.h>

#include "utils/common.h"
#include "utils/logger.h"
#include "utils/error.h"
#include "emb_table/embedding_ddr.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingDDRTest : public testing::Test {
protected:
    EmbeddingDDRTest()
    {
        int embSize = 1000;
        int extEmbSize = 2000;
        struct EmbInfoParams embParam(string("test1"), 0, embSize, extEmbSize, true, true, false, false);
        std::vector<size_t> vocabsize = {100, 100, 100};
        vector<EmbCache::InitializerInfo> initializeInfos = {};
        std::vector<std::string> ssdDataPath = {""};
        std::vector<int64_t> paddingKeys = {1};
        vector<int> maxStep = {1000};
        embInfo_ = EmbInfo(embParam, vocabsize, initializeInfos, ssdDataPath, paddingKeys);
        int rankId;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        rankInfo_ = RankInfo(rankId, 0, 0, 1, maxStep);
        savePath << "test_dir/device" << rankInfo_.rankId;
    }

    void SetUp() {
    }
    void TearDown() {
    }

    static void SetupTestCase()
    {
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
    }

    static void TearDownTestCase()
    {
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
    }

    EmbInfo embInfo_;
    RankInfo rankInfo_;
    stringstream savePath;
};

TEST_F(EmbeddingDDRTest, TestEmptyFuncNoReturn)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }

    table->Key2Offset(testKeys, TRAIN_CHANNEL_ID);
    table->Key2OffsetForDp(testKeys, EVAL_CHANNEL_ID);
    table->EvictKeys(testKeys);
    EXPECT_EQ(testKeys.size(), testNum);
}

TEST_F(EmbeddingDDRTest, EvictDeleteEmb)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    table->EvictDeleteEmb(testKeys);
    EXPECT_EQ(testKeys.size(), 100);
}

TEST_F(EmbeddingDDRTest, TestSaveShouldCreateFileWhenSaveDeltaIsFalse)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    int result = ock::ctr::Factory::Create(factory);
    EXPECT_EQ(result, 0);
    auto embCache = std::shared_ptr<EmbCache::EmbCacheManager>();
    factory->CreateEmbCacheManager(embCache);
    table->SetEmbCache(embCache);

    stringstream savePathTwo;
    savePathTwo << "test_dir/SaveDeltaFalse" << rankInfo_.rankId;
    map<emb_key_t, KeyInfo> keyInfo;
    table->SetFileSystemPtr(savePathTwo.str());
    table->Save(savePathTwo.str(), 1, false, keyInfo);
    stringstream saveKeyPathTwo;
    saveKeyPathTwo << savePathTwo.str() << "/" << tableName << "/key";
    stringstream saveEmbeddingPathTwo;
    saveEmbeddingPathTwo << savePathTwo.str() << "/" << tableName << "/embedding";

    EXPECT_EQ(access(saveKeyPathTwo.str().c_str(), F_OK), 0);
    EXPECT_EQ(access(saveEmbeddingPathTwo.str().c_str(), F_OK), 0);
    embCache->Destroy();
}

TEST_F(EmbeddingDDRTest, TestSaveShouldCreateFileWhenSaveDeltaIsTrue)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    factory->CreateEmbCacheManager(table->embCache);

    stringstream savePathOne;
    savePathOne << "test_dir/SaveDeltaTrue" << rankInfo_.rankId;
    map<emb_key_t, KeyInfo> keyInfo;
    table->SetFileSystemPtr(savePathOne.str());
    table->Save(savePathOne.str(), 1, true, keyInfo);
    stringstream saveKeyPathOne;
    saveKeyPathOne << savePathOne.str() << "/" << tableName << "/key";
    stringstream saveEmbeddingPathOne;
    saveEmbeddingPathOne << savePathOne.str() << "/" << tableName << "/embedding";

    EXPECT_EQ(access(saveKeyPathOne.str().c_str(), F_OK), 0);
    EXPECT_EQ(access(saveEmbeddingPathOne.str().c_str(), F_OK), 0);
}


TEST_F(EmbeddingDDRTest, TestLoadShouldThrowOckErrorWhenEmbCacheNoData)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    factory->CreateEmbCacheManager(table->embCache);
    stringstream savePathTwo;
    savePathTwo << "test_dir/SaveDeltaFalse" << rankInfo_.rankId;

    stringstream fileKeyPath;
    fileKeyPath << savePathTwo.str() << "/" << tableName << "/key" << "/" << "slice_" << rankInfo_.rankId << ".data";
    stringstream newFileKeyPath;
    newFileKeyPath << savePathTwo.str() << "/" << tableName << "/key" << "/" << "slice.data";
    RenameFilePath(fileKeyPath.str(), newFileKeyPath.str());

    stringstream fileEmbeddingPath;
    fileEmbeddingPath << savePathTwo.str() << "/" << tableName << "/embedding/slice_" << rankInfo_.rankId << ".data";
    stringstream newFileEmbeddingPath;
    newFileEmbeddingPath << savePathTwo.str() << "/" << tableName << "/embedding" << "/" << "slice.data";
    RenameFilePath(fileEmbeddingPath.str(), newFileEmbeddingPath.str());

    map<emb_key_t, KeyInfo> keyInfo;
    table->SetFileSystemPtr(savePathTwo.str());
    map<string, unordered_set<emb_cache_key_t>> trainKeySetTwo;
    vector<string> warmStartTablesTwo;
    EXPECT_THROW(table->Load(savePathTwo.str(), trainKeySetTwo, warmStartTablesTwo), std::invalid_argument);
}

TEST_F(EmbeddingDDRTest, TestSaveKeyShouldCreateFileWhenNoError)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t keySize = 5;
    vector<emb_cache_key_t> testKeys = {4, 8, 12, 16, 20};

    table->SetFileSystemPtr(savePath.str());
    table->SaveKey(savePath.str(), testKeys);

    stringstream saveKeyPath;
    saveKeyPath << savePath.str() << "/" << tableName << "/key";
    EXPECT_EQ(access(saveKeyPath.str().c_str(), F_OK), 0);
}

TEST_F(EmbeddingDDRTest, TestSaveEmbeddingShouldCreateFileWhenNoError)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t keySize = 5;
    const size_t embSize = 10;
    table->embSize_ = embSize;
    vector<vector<float>> testEmbeddings(keySize, vector<float>(table->embSize_));
    for (size_t i = 0; i < keySize; ++i) {
        for (size_t j = 0; j < table->embSize_; ++j) {
            testEmbeddings[i][j] = float(i) ;
        }
    }

    table->SetFileSystemPtr(savePath.str());
    table->SaveEmbedding(savePath.str(), testEmbeddings);

    stringstream saveEmbeddingPath;
    saveEmbeddingPath << savePath.str() << "/" << tableName << "/embedding";
    EXPECT_EQ(access(saveEmbeddingPath.str().c_str(), F_OK), 0);
}

TEST_F(EmbeddingDDRTest, TestSaveOptimizerSlotShouldCreateFileWhenNoError)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    OptimizerInfo info;
    info.optimName = "Adam";
    info.optimParams = {"momentum", "velocity"};
    table->SetOptimizerInfo(info);

    const size_t keySize = 5;
    const size_t embSize = 10;
    table->embSize_ = embSize;
    const size_t doubleEmbDim = 20;
    vector<vector<float>> optimizerSlots(keySize, vector<float>(doubleEmbDim));
    for (size_t i = 0; i < keySize; ++i) {
        for (size_t j = 0; j < doubleEmbDim; ++j) {
            optimizerSlots[i][j] = float(i) ;
        }
    }

    table->SetFileSystemPtr(savePath.str());
    table->SaveOptimizerSlot(savePath.str(), optimizerSlots, keySize);

    stringstream saveMomentumPath;
    saveMomentumPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[0];
    stringstream saveVelocityPath;
    saveVelocityPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[1];
    EXPECT_EQ(access(saveMomentumPath.str().c_str(), F_OK), 0);
    EXPECT_EQ(access(saveVelocityPath.str().c_str(), F_OK), 0);
}

TEST_F(EmbeddingDDRTest, TestSaveOptimizerSlotShouldNoReturnWhenSizeIsZero)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t keySize = 5;
    vector<vector<float>> optimizerSlots;
    table->SaveOptimizerSlot(savePath.str(), optimizerSlots, keySize);
}

TEST_F(EmbeddingDDRTest, TestSaveOptimizerSlotShouldThrowErrorWhenSizeNoEqual)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t keySize = 5;
    vector<vector<float>> optimizerSlots = {{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_THROW(table->SaveOptimizerSlot(savePath.str(), optimizerSlots, keySize), std::runtime_error);
}

TEST_F(EmbeddingDDRTest, TestSetOptimizerInfo)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    OptimizerInfo info;
    info.optimName = "Adam";
    info.optimParams = {"momentum", "velocity"};
    table->SetOptimizerInfo(info);
    EXPECT_EQ(table->optimName, info.optimName);
    EXPECT_EQ(table->optimParams, info.optimParams);
}

TEST_F(EmbeddingDDRTest, TestLoadKeyShouldThrowErrorWhenFileNoExist)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    vector<emb_cache_key_t> keys;
    table->SetFileSystemPtr(savePath.str());
    EXPECT_THROW(table->LoadKey(savePath.str(), keys), std::runtime_error);
}

TEST_F(EmbeddingDDRTest, TestLoadKeyShouldLoadKeySizeEqualSaveKeySizeWhenNoError)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    stringstream filePath;
    filePath << savePath.str() << "/" << tableName << "/key" << "/" << "slice_" << rankInfo_.rankId << ".data";
    stringstream newFilePath;
    newFilePath << savePath.str() << "/" << tableName << "/key" << "/" << "slice.data";
    RenameFilePath(filePath.str(), newFilePath.str());

    if (rankInfo_.rankId == 0) {
        const size_t keySize = 5;
        vector<emb_cache_key_t> keys;
        table->SetFileSystemPtr(savePath.str());
        table->LoadKey(savePath.str(), keys);

        EXPECT_EQ(keys.size(), keySize);
        EXPECT_EQ(table->hostLoadOffset.size(), keySize);
    }
}

TEST_F(EmbeddingDDRTest, TestLoadEmbeddingShouldEmbeddingSizeEqualSaveKeySizeWhenNoError)
{
    const string tableName = "test1";
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    stringstream filePath;
    filePath << savePath.str() << "/" << tableName << "/embedding" << "/" << "slice_" << rankInfo_.rankId << ".data";
    stringstream newFilePath;
    newFilePath << savePath.str() << "/" << tableName << "/embedding" << "/" << "slice.data";
    RenameFilePath(filePath.str(), newFilePath.str());

    const size_t keySize = 5;
    const size_t embSize = 10;
    table->embSize_ = embSize;
    table->hostLoadOffset = {0, 1, 2, 3, 4};
    vector<vector<float>> embeddings;
    table->SetFileSystemPtr(savePath.str());
    table->LoadEmbedding(savePath.str(), embeddings);

    EXPECT_EQ(embeddings.size(), keySize);
}

TEST_F(EmbeddingDDRTest, TestLoadOptimizerSlotShouldNoReturnWhenOptimParamsSizeIsZero)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    vector<vector<float>> optimizerSlots;
    table->LoadOptimizerSlot(savePath.str(), optimizerSlots);
}

TEST_F(EmbeddingDDRTest, TestLoadOptimizerSlotShouldOptimizerSlotsSizeEqualSavekeySizeWhenNoError)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    OptimizerInfo info;
    info.optimName = "Adam";
    info.optimParams = {"momentum", "velocity"};
    table->SetOptimizerInfo(info);

    stringstream filePath1;
    filePath1 << savePath.str() << "/test1/" << info.optimName + "_" + info.optimParams[0]
              << "/" << "slice_" << rankInfo_.rankId << ".data";
    stringstream newFilePath1;
    newFilePath1 << savePath.str() << "/test1/" << info.optimName + "_" + info.optimParams[0] << "/" << "slice.data";
    stringstream filePath2;
    filePath2 << savePath.str() << "/test1/" << info.optimName + "_" + info.optimParams[1]
              << "/" << "slice_" << rankInfo_.rankId << ".data";
    stringstream newFilePath2;
    newFilePath2 << savePath.str() << "/test1/" << info.optimName + "_" + info.optimParams[1] << "/" << "slice.data";
    RenameFilePath(filePath1.str(), newFilePath1.str());
    RenameFilePath(filePath2.str(), newFilePath2.str());

    const size_t keySize = 5;
    const size_t embSize = 10;
    table->embSize_ = embSize;
    table->hostLoadOffset = {0, 1, 2, 3, 4};
    vector<vector<float>> optimizerSlots;
    table->SetFileSystemPtr(savePath.str());
    table->LoadOptimizerSlot(savePath.str(), optimizerSlots);
    EXPECT_EQ(optimizerSlots.size(), keySize);
}

TEST_F(EmbeddingDDRTest, TestGetDeviceOffsetShouldThrowError)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    EXPECT_THROW(table->GetDeviceOffset(), std::runtime_error);
}

TEST_F(EmbeddingDDRTest, TestGetTableInfo)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    TableInfo tableInfo = table->GetTableInfo();
    EXPECT_EQ(tableInfo.devVocabSize, embInfo_.devVocabSize);
    EXPECT_EQ(tableInfo.hostVocabSize, embInfo_.hostVocabSize);
    EXPECT_EQ(tableInfo.name, embInfo_.name);
}

TEST_F(EmbeddingDDRTest, TestCapacity)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    EXPECT_EQ(table->capacity(), 0);
}

TEST_F(EmbeddingDDRTest, TestSyncLatestEmbeddingShouldThrowOckErrorWhenEmbCacheNoData)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    factory->CreateEmbCacheManager(table->embCache);
    shared_ptr<HDTransfer> hdTransfer = std::make_shared<HDTransfer>();
    table->SetHDTransfer(hdTransfer.get());
    EXPECT_THROW(table->SyncLatestEmbedding(1), std::invalid_argument);
    hdTransfer->Destroy();
}

TEST_F(EmbeddingDDRTest, TestBackUpTrainStatusShouldThrowError)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    factory->CreateEmbCacheManager(table->embCache);
    EXPECT_THROW(table->BackUpTrainStatus(), std::runtime_error);
}

TEST_F(EmbeddingDDRTest, TestRecoverTrainStatusShouldThrowError)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    factory->CreateEmbCacheManager(table->embCache);
    EXPECT_THROW(table->RecoverTrainStatus(), std::runtime_error);
}
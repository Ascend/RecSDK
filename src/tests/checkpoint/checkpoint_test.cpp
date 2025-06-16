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

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <mpi.h>
#include <gtest/gtest.h>
#include <emock/emock.hpp>

#include "checkpoint/checkpoint.h"
#include "ckpt_data_handler/key_count_map_ckpt/key_count_map_ckpt.h"
#include "utils/common.h"

using namespace std;
using namespace MxRec;

const float MEM_INIT_VALUE = 0.5;
const size_t TEST_SIZE = 4;
const ssize_t RETURN_SIZE = -2;

class CheckpointTest : public testing::Test {
protected:
    string testPath{"./ckpt_mgmt_test"};
    int rankId;

    int floatBytes{4};
    int int32Bytes{4};
    int int64Bytes{8};

    int64_t int64Min{static_cast<int64_t>(UINT32_MAX)};

    int maxChannelNum = MAX_CHANNEL_NUM;
    int keyProcessThread = 1;

    int embInfoNum{10};

    float floatMem{MEM_INIT_VALUE};
    int64_t featMem{static_cast<int64_t>(UINT32_MAX)};
    int32_t offsetMem{0};
    int32_t maxOffsetMem{16};

    string name{"table"};
    int sendCount{8};
    int embeddingSize{100};
    int devVocabSize{8};
    int hostVocabSize{16};

    vector<EmbInfo> testEmbInfos;
    RankInfo rankInfo;

    void SetUp()
    {
        int claimed;

        MPI_Query_thread(&claimed);
        ASSERT_EQ(claimed, MPI_THREAD_MULTIPLE);
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        rankInfo.rankId = rankId;
        rankInfo.useDynamicExpansion = false;

        emock::GlobalMockObject::reset();
    }

    void SetEmbInfo()
    {
        int idx{0};
        testEmbInfos.resize(embInfoNum);
        for (auto& testEmbInfo : testEmbInfos) {
            testEmbInfo.name = name + to_string(idx);
            testEmbInfo.sendCount = sendCount;
            testEmbInfo.extEmbeddingSize = embeddingSize;
            testEmbInfo.devVocabSize = devVocabSize;
            testEmbInfo.hostVocabSize = hostVocabSize;
            testEmbInfo.isSave = true;
            ++idx;
        }
    }

    void SetEmbData(vector<vector<float>>& testEmbData)
    {
        testEmbData.resize(hostVocabSize);
        floatMem = MEM_INIT_VALUE;
        for (auto& testData : testEmbData) {
            testData.resize(embeddingSize);
            for (auto& testValue : testData) {
                testValue = floatMem;
                floatMem++;
            }
        }
    }

    void SetHostEmbs(std::shared_ptr<EmbMemT> testHostEmbs)
    {
        vector<vector<float>> testEmbData;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetEmbData(testEmbData);
            HostEmbTable embTable{testEmbInfo, move(testEmbData)};
            testHostEmbs->insert({testEmbInfo.name, move(embTable)});  // set test input data
        }
    }

    void SetHostEmptyEmbs(std::shared_ptr<EmbMemT> loadHostEmbs)
    {
        vector<vector<float>> testEmbData;
        for (const auto& testEmbInfo : testEmbInfos) {
            // SetEmbData
            testEmbData.resize(hostVocabSize);
            for (auto& testData : testEmbData) {
                testData.resize(embeddingSize);
                for (auto& testValue : testData) {
                    testValue = 0;
                }
            }
            HostEmbTable embTable{testEmbInfo, move(testEmbData)};
            loadHostEmbs->insert({testEmbInfo.name, move(embTable)});  // set test input data
        }
    }

    void SetHashMapInfo(absl::flat_hash_map<emb_key_t, size_t>& testHash, vector<int32_t>& testDev2B,
                        vector<int64_t>& testDev2K)
    {
        testDev2B.resize(devVocabSize);
        testDev2K.resize(devVocabSize);
        for (int i{0}; i < devVocabSize; ++i) {
            testDev2K.at(i) = offsetMem;
            testHash[featMem] = offsetMem;

            featMem++;
            offsetMem++;
        }
        fill(testDev2B.begin(), testDev2B.end(), -1);
    }

    void SetKeyOffsetMap(absl::flat_hash_map<emb_key_t, int64_t>& testKeyOffsetMap)
    {
        for (int64_t i{0}; i < hostVocabSize; ++i) {
            testKeyOffsetMap[featMem] = i;
            featMem++;
        }
    }

    void SetDDRKeyFreqMap(unordered_map<emb_cache_key_t, freq_num_t>& testDDRKeyFreqMap)
    {
        for (int64_t i{0}; i < hostVocabSize; ++i) {
            testDDRKeyFreqMap[featMem] = i;
            featMem++;
        }
    }

    void SetKeyCountMap(absl::flat_hash_map<emb_key_t, size_t>& testKeyCountMap)
    {
        for (int64_t i{0}; i < hostVocabSize; ++i) {
            testKeyCountMap[featMem] = i;
            featMem++;
        }
    }

    void SetExcludeDDRKeyFreqMap(unordered_map<emb_cache_key_t, freq_num_t>& testExcludeDDRKeyFreqMap)
    {
        for (int64_t i{0}; i < hostVocabSize; ++i) {
            testExcludeDDRKeyFreqMap[featMem] = i;
            featMem++;
        }
    }

    void SetDDRKeyFreqMaps(KeyFreqMemT& testDDRKeyFreqMaps)
    {
        unordered_map<emb_cache_key_t, freq_num_t> testDDRKeyFreqMap;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetDDRKeyFreqMap(testDDRKeyFreqMap);
            testDDRKeyFreqMaps[testEmbInfo.name] = std::move(testDDRKeyFreqMap);
        }
    }

    void SetKeyCountMaps(KeyCountMemT& testKeyCountMaps)
    {
        absl::flat_hash_map<emb_key_t, size_t> testKeyCountMap;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetKeyCountMap(testKeyCountMap);
            testKeyCountMaps[testEmbInfo.name] = std::move(testKeyCountMap);
        }
    }

    void SetExcludeDDRKeyFreqMaps(KeyFreqMemT& testExcludeDDRKeyFreqMaps)
    {
        unordered_map<emb_cache_key_t, freq_num_t> testExcludeDDRKeyFreqMap;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetExcludeDDRKeyFreqMap(testExcludeDDRKeyFreqMap);
            testExcludeDDRKeyFreqMaps[testEmbInfo.name] = std::move(testExcludeDDRKeyFreqMap);
        }
    }

    void SetHistRec(AdmitAndEvictData& histRec)
    {
        int64_t featureId{int64Min};
        int count{1};
        time_t lastTime{1000};
        time_t timeStamp{10000};

        for (const auto& testEmbInfo : testEmbInfos) {
            auto& historyRecords{histRec.historyRecords[testEmbInfo.name]};
            auto& timestamps{histRec.timestamps[testEmbInfo.name]};

            timestamps = timeStamp;

            for (int i = 0; i < count; ++i) {
                historyRecords[featureId].count = count;
                historyRecords[featureId].lastTime = lastTime;

                featureId++;
            }

            count++;
            lastTime++;
            timeStamp++;
        }
    }

    void SetTable2Threshold(Table2ThreshMemT& testTable2Threshold)
    {
        for (const auto& testEmbInfo : testEmbInfos) {
            ThresholdValue val;
            val.tableName = testEmbInfo.name;
            val.countThreshold = offsetMem;
            val.timeThreshold = offsetMem;
            val.faaeCoefficient = 1;
            val.isEnableSum = true;

            offsetMem++;

            testTable2Threshold[testEmbInfo.name] = move(val);
        }
    }

    void SetHistRecCombine(AdmitAndEvictData& histRec)
    {
        int64_t featureId{int64Min};
        int count{1};
        time_t lastTime{1000};
        time_t timeStamp{10000};

        auto& historyRecords{histRec.historyRecords[COMBINE_HISTORY_NAME]};
        auto& timestamps{histRec.timestamps[COMBINE_HISTORY_NAME]};

        timestamps = timeStamp;

        for (int i = 0; i < count; ++i) {
            historyRecords[featureId].count = count;
            historyRecords[featureId].lastTime = lastTime;

            featureId++;
        }

        count++;
        lastTime++;
        timeStamp++;
    }
};

TEST_F(CheckpointTest, KeyFreqMaps)
{
    KeyFreqMemT testDDRKeyFreqMaps;
    KeyFreqMemT validDDRKeyFreqMaps;
    KeyFreqMemT testExcludeDDRKeyFreqMaps;
    KeyFreqMemT validExcludeDDRKeyFreqMaps;

    SetEmbInfo();
    SetDDRKeyFreqMaps(testDDRKeyFreqMaps);
    SetExcludeDDRKeyFreqMaps(testExcludeDDRKeyFreqMaps);
    validDDRKeyFreqMaps = testDDRKeyFreqMaps;
    validExcludeDDRKeyFreqMaps = testExcludeDDRKeyFreqMaps;

    CkptData testSaveData;
    CkptData validLoadData;
    CkptData testLoadData;

    testSaveData.ddrKeyFreqMaps = std::move(testDDRKeyFreqMaps);
    testSaveData.excludeDDRKeyFreqMaps = std::move(testExcludeDDRKeyFreqMaps);
    validLoadData.ddrKeyFreqMaps = std::move(validDDRKeyFreqMaps);
    Checkpoint testCkpt;
    testCkpt.SaveModel(testPath, testSaveData, rankInfo, testEmbInfos);
    bool fileExist = false;
    if (access("./ckpt_mgmt_test/table0/ddr_key_freq_map", F_OK) == 0) {
        fileExist = true;
    }
    EXPECT_EQ(fileExist, true);
}

TEST_F(CheckpointTest, KeyCountMapCkpt)
{
    KeyCountMemT testKeyCountMaps;
    KeyCountMemT validKeyCountMaps;

    SetEmbInfo();
    SetKeyCountMaps(testKeyCountMaps);

    validKeyCountMaps = testKeyCountMaps;

    CkptData testSaveData;
    CkptData validLoadData;
    CkptData testLoadData;

    testSaveData.keyCountMap = std::move(testKeyCountMaps);
    validLoadData.keyCountMap = std::move(validKeyCountMaps);

    Checkpoint testCkpt;
    testCkpt.SaveModel(testPath, testSaveData, rankInfo, testEmbInfos);
    bool fileExist = false;
    if (access("./ckpt_mgmt_test/table0/key_count_map", F_OK) == 0) {
        fileExist = true;
    }
    EXPECT_EQ(fileExist, true);
}

TEST_F(CheckpointTest, FeatAdmitNEvict)
{
    Table2ThreshMemT testTrens2Thresh;
    Table2ThreshMemT validTrens2Thresh;
    AdmitAndEvictData testHistRec;
    AdmitAndEvictData validHistRec;

    SetEmbInfo();
    SetTable2Threshold(testTrens2Thresh);
    validTrens2Thresh = testTrens2Thresh;
    bool isCombine = false;

    if (isCombine) {
        SetHistRecCombine(testHistRec);
    } else {
        SetHistRec(testHistRec);
    }

    validHistRec = testHistRec;

    CkptData testSaveData;
    CkptData validLoadData;
    CkptData testLoadData;

    testSaveData.table2Thresh = testTrens2Thresh;
    testSaveData.histRec.timestamps = testHistRec.timestamps;
    testSaveData.histRec.historyRecords = testHistRec.historyRecords;
    validLoadData.table2Thresh = validTrens2Thresh;
    validLoadData.histRec = validHistRec;
    validLoadData.histRec.timestamps = validHistRec.timestamps;
    validLoadData.histRec.historyRecords = validHistRec.historyRecords;

    Checkpoint testCkpt;
    testCkpt.SaveModel(testPath, testSaveData, rankInfo, testEmbInfos);
    bool fileExist = false;
    if (access("./ckpt_mgmt_test/table0/history_record", F_OK) == 0) {
        fileExist = true;
    }
    EXPECT_EQ(fileExist, true);
}

TEST_F(CheckpointTest, LoadModelOk)
{
    auto savePath = "./ckpt_mgmt_test"s;
    auto ckptData = CkptData();
    auto rankInfo = RankInfo();
    auto embInfos = vector<EmbInfo>();
    auto featTypes = vector<CkptFeatureType>();

    auto embInfo = EmbInfo();
    embInfo.name = "table0"s;
    embInfos.push_back(embInfo);

    auto ckpt = Checkpoint();
    ckpt.LoadModel(savePath, ckptData, rankInfo, embInfos, featTypes);

    ckptData.keyCountMap.emplace("testKey"s, absl::flat_hash_map<emb_key_t, size_t>());

    ckpt.SetDataHandler(ckptData);
    ckpt.LoadProcess(ckptData);
    EXPECT_EQ(ckpt.GetEmbeddingSize("table0"s).extEmbSize, 0);

    auto rankSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &rankSize);

    auto res = ckpt.GetTableLayerLoadDir();
    std::sort(res.begin(), res.end());
    EXPECT_EQ(res.size(), 10);
}

TEST_F(CheckpointTest, LoadDatasetOk)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&Checkpoint::ReadStream).expects(exactly(2));
    EMOCK(&KeyCountMapCkpt::SetDatasetForLoadEmb).expects(once());

    auto ckpt = Checkpoint();

    auto embNames = vector<string>{"test"};
    auto dataTypes = vector<CkptDataType>{CkptDataType::EMB_INFO};
    auto dataHandler = make_unique<KeyCountMapCkpt>();
    auto ckptData = CkptData();

    ckpt.LoadDataset(embNames, dataTypes, std::move(dataHandler), ckptData);
}

TEST_F(CheckpointTest, LoadDataset_Table2Thresh_PathNotExists)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&Checkpoint::ReadStream).expects(exactly(2));
    EMOCK(&KeyCountMapCkpt::SetDatasetForLoadEmb).expects(once());
    EMOCK(CheckFileExist).expects(once()).will(returnValue(false));

    auto ckpt = Checkpoint();
    auto embNames = vector<string>{"test"};
    auto dataTypes = vector<CkptDataType>{CkptDataType::TABLE_2_THRESH};
    auto dataHandler = make_unique<KeyCountMapCkpt>();
    auto ckptData = CkptData();

    ckpt.LoadDataset(embNames, dataTypes, std::move(dataHandler), ckptData);
}

TEST_F(CheckpointTest, LoadDataset_HistRec_PathNotExists)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&Checkpoint::ReadStream).expects(exactly(2));
    EMOCK(&KeyCountMapCkpt::SetDatasetForLoadEmb).expects(once());
    EMOCK(CheckFileExist).expects(once()).will(returnValue(false));

    auto ckpt = Checkpoint();
    auto embNames = vector<string>{"test"};
    auto dataTypes = vector<CkptDataType>{CkptDataType::HIST_REC};
    auto dataHandler = make_unique<KeyCountMapCkpt>();
    auto ckptData = CkptData();

    ckpt.LoadDataset(embNames, dataTypes, std::move(dataHandler), ckptData);
}

TEST_F(CheckpointTest, LoadDataset_EmbData)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&Checkpoint::ReadStream).expects(once());
    EMOCK(&Checkpoint::ReadStreamForEmbData).expects(once());
    EMOCK(&KeyCountMapCkpt::SetDatasetForLoadEmb).expects(once());

    auto ckpt = Checkpoint();
    auto embNames = vector<string>{"test"};
    auto dataTypes = vector<CkptDataType>{CkptDataType::EMB_DATA};
    auto dataHandler = make_unique<KeyCountMapCkpt>();
    auto ckptData = CkptData();

    ckpt.LoadDataset(embNames, dataTypes, std::move(dataHandler), ckptData);
}

TEST_F(CheckpointTest, LoadDataset_Attribute)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&Checkpoint::ReadStream).expects(exactly(2));
    EMOCK(&Checkpoint::ReadStreamForEmbData).expects(once());
    EMOCK(&KeyCountMapCkpt::SetDatasetForLoadEmb).expects(once());

    auto ckpt = Checkpoint();
    auto embNames = vector<string>{"test"};
    auto dataTypes = vector<CkptDataType>{CkptDataType::ATTRIBUTE};
    auto dataHandler = make_unique<KeyCountMapCkpt>();
    auto ckptData = CkptData();

    ckpt.LoadDataset(embNames, dataTypes, std::move(dataHandler), ckptData);
}

TEST_F(CheckpointTest, ReadStreamOk)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&LocalFileSystem::GetFileSize).expects(once()).will(returnValue(static_cast<size_t>(0)));
    EMOCK(&Checkpoint::SetTransDataSize).expects(once());
    EMOCK((ssize_t(LocalFileSystem::*)(const string&, char*, size_t))(&LocalFileSystem::Read))
        .expects(once())
        .will(returnValue(0));

    auto ckpt = Checkpoint();

    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::EMB_INFO;
    auto dataElemBytes = 1;

    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());
    ckpt.ReadStream(transData, dataDir, dataType, dataElemBytes);
}

TEST_F(CheckpointTest, ReadStream_DataElmtBytesZero)
{
    auto ckpt = Checkpoint();
    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::EMB_INFO;
    auto dataElemBytes = 0;

    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());
    ckpt.ReadStream(transData, dataDir, dataType, dataElemBytes);
}

TEST_F(CheckpointTest, ReadStream_DataIsMissing)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&LocalFileSystem::GetFileSize).expects(once()).will(returnValue(static_cast<size_t>(TEST_SIZE)));
    EMOCK(&Checkpoint::SetTransDataSize).expects(once());

    auto ckpt = Checkpoint();
    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::EMB_INFO;
    auto dataElemBytes = 3;

    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());
    EXPECT_THROW(ckpt.ReadStream(transData, dataDir, dataType, dataElemBytes), std::runtime_error);
}

TEST_F(CheckpointTest, ReadStream_InputInt64_LoadDataFailed_Error)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&LocalFileSystem::GetFileSize).expects(once()).will(returnValue(static_cast<size_t>(0)));
    EMOCK(&Checkpoint::SetTransDataSize).expects(once());
    EMOCK((ssize_t(LocalFileSystem::*)(const string&, char*, size_t))(&LocalFileSystem::Read))
        .expects(once())
        .will(returnValue(-1));

    auto ckpt = Checkpoint();
    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::EMB_HASHMAP;
    auto dataElemBytes = 1;

    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());
    EXPECT_THROW(ckpt.ReadStream(transData, dataDir, dataType, dataElemBytes), std::runtime_error);
}

TEST_F(CheckpointTest, ReadStream_InputAttribute_LoadDataFailed_Error)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&LocalFileSystem::GetFileSize).expects(once()).will(returnValue(static_cast<size_t>(0)));
    EMOCK(&Checkpoint::SetTransDataSize).expects(once());
    EMOCK((ssize_t(LocalFileSystem::*)(const string&, char*, size_t))(&LocalFileSystem::Read))
        .expects(once())
        .will(returnValue(RETURN_SIZE));

    auto ckpt = Checkpoint();
    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::ATTRIBUTE;
    auto dataElemBytes = 1;

    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());
    EXPECT_THROW(ckpt.ReadStream(transData, dataDir, dataType, dataElemBytes), std::runtime_error);
}

TEST_F(CheckpointTest, ReadStream_InputUnknowType_Error)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&LocalFileSystem::GetFileSize).expects(once()).will(returnValue(static_cast<size_t>(0)));
    EMOCK(&Checkpoint::SetTransDataSize).expects(once());
    EMOCK((ssize_t(LocalFileSystem::*)(const string&, char*, size_t))(&LocalFileSystem::Read))
        .expects(once())
        .will(returnValue(0));

    auto ckpt = Checkpoint();
    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::EMB_DATA;
    auto dataElemBytes = 1;

    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());
    EXPECT_THROW(ckpt.ReadStream(transData, dataDir, dataType, dataElemBytes), std::runtime_error);
}

TEST_F(CheckpointTest, ReadStreamForEmbDataOk)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(once()).will(returnValue(true));
    EMOCK(&LocalFileSystem::GetFileSize).expects(once()).will(returnValue(static_cast<size_t>(0)));

    auto transData = CkptTransData();
    transData.attribute.push_back(32);

    auto dataDir = "./test"s;
    uint32_t dataElemBytes = 32;
    auto embName = "test"s;

    auto ckptData = CkptData();
    auto embMem = new EmbMemT();
    embMem->emplace(embName, HostEmbTable());
    ckptData.hostEmbs = embMem;

    auto ckpt = Checkpoint();
    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());
    ckpt.ReadStreamForEmbData(transData, dataDir, dataElemBytes, ckptData, embName);

    EXPECT_NE(ckptData.hostEmbs->at("test").embData.size(), 1);

    delete embMem;
}

TEST_F(CheckpointTest, ReadStreamForEmbDataErr)
{
    EMOCK(&Checkpoint::CheckFileSystemPtr).expects(exactly(2)).will(returnValue(true));

    auto transData = CkptTransData();
    transData.attribute.push_back(-1);

    auto dataDir = "./test"s;
    uint32_t dataElemBytes = 32;
    auto embName = "test"s;

    auto ckptData = CkptData();

    auto ckpt = Checkpoint();
    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());

    EXPECT_THROW(ckpt.ReadStreamForEmbData(transData, dataDir, dataElemBytes, ckptData, embName), std::runtime_error);

    EMOCK(&LocalFileSystem::GetFileSize).expects(once()).will(returnValue(static_cast<size_t>(1)));

    auto embMem = new EmbMemT();
    embMem->emplace(embName, HostEmbTable());
    ckptData.hostEmbs = embMem;

    transData.attribute.clear();
    transData.attribute.push_back(MAX_VOCABULARY_SIZE);

    EXPECT_THROW(ckpt.ReadStreamForEmbData(transData, dataDir, dataElemBytes, ckptData, embName), std::runtime_error);

    delete embMem;
}

TEST_F(CheckpointTest, ReadStreamForEmbData_EmptyDataElmtByte)
{
    auto transData = CkptTransData();
    transData.attribute.push_back(32);
    auto dataDir = "./test"s;
    uint32_t dataElemBytes = 0;
    auto embName = "test"s;
    auto ckptData = CkptData();
    auto ckpt = Checkpoint();
    
    ckpt.ReadStreamForEmbData(transData, dataDir, dataElemBytes, ckptData, embName);
}

TEST_F(CheckpointTest, SetTransDataSizeOk)
{
    auto ckpt = Checkpoint();

    auto transData = CkptTransData();
    size_t datasetSize = 0;

    auto dataType = CkptDataType::EMB_INFO;
    ckpt.SetTransDataSize(transData, datasetSize, dataType);

    dataType = CkptDataType::EMB_HASHMAP;
    ckpt.SetTransDataSize(transData, datasetSize, dataType);

    dataType = CkptDataType::ATTRIBUTE;
    ckpt.SetTransDataSize(transData, datasetSize, dataType);

    dataType = static_cast<CkptDataType>(-1);
    EXPECT_THROW(ckpt.SetTransDataSize(transData, datasetSize, dataType), std::runtime_error);
}

TEST_F(CheckpointTest, SetDataHandler_InputFeatureTypes)
{
    vector<CkptFeatureType> featTypes = {
        CkptFeatureType::FEAT_ADMIT_N_EVICT,
        CkptFeatureType::DDR_KEY_FREQ_MAP,
        CkptFeatureType::KEY_COUNT_MAP
    };
    auto ckpt = Checkpoint();

    ckpt.SetDataHandler(featTypes);
}

TEST_F(CheckpointTest, GetEmbeddingSize_mgmtEmbInfoEmpty)
{
    auto ckpt = Checkpoint();

    auto res = ckpt.GetEmbeddingSize("test");
    EXPECT_EQ(res.embSize, 0);
}

TEST_F(CheckpointTest, GetEmbeddingSize_mgmtEmbInfoEmbNameNotfound)
{
    SetEmbInfo();
    auto ckpt = Checkpoint();
    ckpt.mgmtEmbInfo = testEmbInfos;

    auto res = ckpt.GetEmbeddingSize("test");
    EXPECT_EQ(res.embSize, 0);
}

TEST_F(CheckpointTest, CheckEmbNames_NameNotMatch_ReturnsFalse)
{
    SetEmbInfo();
    auto ckpt = Checkpoint();
    ckpt.mgmtEmbInfo = testEmbInfos;

    EXPECT_FALSE(ckpt.CheckEmbNames("target"));
}

TEST_F(CheckpointTest, CheckEmbNames_MatchButNotSave_ReturnsFalse)
{
    SetEmbInfo();
    testEmbInfos[0].isSave = false;
    testEmbInfos[0].name = "target";
    auto ckpt = Checkpoint();
    ckpt.mgmtEmbInfo = testEmbInfos;

    EXPECT_FALSE(ckpt.CheckEmbNames("target"));
}

TEST_F(CheckpointTest, SaveDataset_CheckEmbNamesFalse)
{
    SetEmbInfo();
    auto embNames = vector<string>{"test"};
    auto dataTypes = vector<CkptDataType>{CkptDataType::EMB_INFO};
    unique_ptr<CkptDataHandler> dataHandler;
    auto ckpt = Checkpoint();
    ckpt.mgmtEmbInfo = testEmbInfos;

    ckpt.SaveDataset(embNames, dataTypes, dataHandler);
}

TEST_F(CheckpointTest, WriteStream_DataTypeAttribute_SaveDataError1)
{
    EMOCK(static_cast<ssize_t (LocalFileSystem::*)(const string&, const char*, size_t)>(&LocalFileSystem::Write))
        .expects(once())
        .will(returnValue(-1));

    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::ATTRIBUTE;
    auto ckpt = Checkpoint();
    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());

    EXPECT_THROW(ckpt.WriteStream(transData, dataDir, transData.datasetSize, dataType), std::runtime_error);
}

TEST_F(CheckpointTest, WriteStream_DataTypeAttribute_SaveDataError2)
{
    EMOCK(static_cast<ssize_t (LocalFileSystem::*)(const string&, const char*, size_t)>(&LocalFileSystem::Write))
        .expects(once())
        .will(returnValue(RETURN_SIZE));

    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::ATTRIBUTE;
    auto ckpt = Checkpoint();
    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());

    EXPECT_THROW(ckpt.WriteStream(transData, dataDir, transData.datasetSize, dataType), std::runtime_error);
}

TEST_F(CheckpointTest, WriteStream_UnknowType_Error)
{
    EMOCK(static_cast<ssize_t (LocalFileSystem::*)(const string&, const char*, size_t)>(&LocalFileSystem::Write))
        .expects(once())
        .will(returnValue(1));

    auto transData = CkptTransData();
    auto dataDir = "./test"s;
    auto dataType = CkptDataType::EMB_DATA;
    auto ckpt = Checkpoint();
    ckpt.fileSystemPtr = std::make_unique<LocalFileSystem>(LocalFileSystem());

    EXPECT_THROW(ckpt.WriteStream(transData, dataDir, transData.datasetSize, dataType), std::runtime_error);
}

TEST_F(CheckpointTest, GetEmbedTableNames_IsSave)
{
    SetEmbInfo();
    auto ckpt = Checkpoint();
    ckpt.mgmtEmbInfo = testEmbInfos;

    ckpt.GetEmbedTableNames();
}

TEST_F(CheckpointTest, CheckFileSystemPtr_NullptrFileSystemPtr)
{
    auto ckpt = Checkpoint();

    EXPECT_THROW(ckpt.CheckFileSystemPtr(), std::runtime_error);
}

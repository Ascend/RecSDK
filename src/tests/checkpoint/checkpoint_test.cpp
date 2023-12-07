/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-15
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "checkpoint/checkpoint.h"
#include "ckpt_data_handler/nddr_feat_map_ckpt/nddr_feat_map_ckpt.h"


using namespace std;
using namespace MxRec;

const float MEM_INIT_VALUE = 0.5;

class CheckpointTest : public testing::Test {
protected:
    string testPath { "./ckpt_mgmt_test" };
    int rankId;

    int floatBytes { 4 };
    int int32Bytes { 4 };
    int int64Bytes { 8 };

    int64_t int64Min { static_cast<int64_t>(UINT32_MAX) };

    int maxChannelNum = MAX_CHANNEL_NUM;
    int keyProcessThread = 1;

    int embInfoNum { 10 };

    float floatMem { MEM_INIT_VALUE };
    int64_t featMem { static_cast<int64_t>(UINT32_MAX) };
    int32_t offsetMem { 0 };
    int32_t maxOffsetMem { 16 };

    string name { "table" };
    int sendCount { 8 };
    int embeddingSize { 100 };
    int devVocabSize { 8 };
    int hostVocabSize { 16 };

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
    }

    void SetEmbInfo()
    {
        int idx { 0 };
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
            HostEmbTable embTable { testEmbInfo, move(testEmbData) };
            testHostEmbs->insert({testEmbInfo.name,  move(embTable)}); // set test input data
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
            HostEmbTable embTable { testEmbInfo, move(testEmbData) };
            loadHostEmbs->insert({testEmbInfo.name,  move(embTable)}); // set test input data
        }
    }

    void SetHashMapInfo(absl::flat_hash_map<emb_key_t, size_t>& testHash,
                        vector<int32_t>& testDev2B,
                        vector<int64_t>& testDev2K)
    {
        testDev2B.resize(devVocabSize);
        testDev2K.resize(devVocabSize);
        for (int i { 0 }; i < devVocabSize; ++i) {
            testDev2K.at(i) = offsetMem;
            testHash[featMem] = offsetMem;

            featMem++;
            offsetMem++;
        }
        fill(testDev2B.begin(), testDev2B.end(), -1);
    }

    void SetEmbHashMaps(EmbHashMemT& testEmbHashMaps)
    {
        EmbHashMapInfo embHashInfo;
        absl::flat_hash_map<emb_key_t, size_t> testHash;
        vector<int32_t> testDev2B;
        vector<int64_t> testDev2K;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetHashMapInfo(testHash, testDev2B, testDev2K);

            embHashInfo.hostHashMap = std::move(testHash);

            embHashInfo.devOffset2Batch = move(testDev2B);
            embHashInfo.devOffset2Key = move(testDev2K);

            embHashInfo.currentUpdatePos = 0;
            embHashInfo.hostVocabSize = hostVocabSize;
            embHashInfo.devVocabSize = devVocabSize;

            testEmbHashMaps[testEmbInfo.name] = move(embHashInfo);
        }
    }

    void SetMaxOffset(OffsetMemT& testMaxOffset)
    {
        for (const auto& testEmbInfo : testEmbInfos) {
            testMaxOffset[testEmbInfo.name] = maxOffsetMem;
        }
    }

    void SetKeyOffsetMap(absl::flat_hash_map<emb_key_t, int64_t>& testKeyOffsetMap)
    {
        for (int64_t i { 0 }; i < hostVocabSize; ++i) {
            testKeyOffsetMap[featMem] = i;
            featMem++;
        }
    }

    void SetKeyOffsetMaps(KeyOffsetMemT& testKeyOffsetMaps)
    {
        absl::flat_hash_map<emb_key_t, int64_t> testKeyOffsetMap;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetKeyOffsetMap(testKeyOffsetMap);
            testKeyOffsetMaps[testEmbInfo.name] = std::move(testKeyOffsetMap);
        }
    }

    void SetDDRKeyFreqMap(unordered_map<emb_key_t, freq_num_t>& testDDRKeyFreqMap)
    {
        for (int64_t i { 0 }; i < hostVocabSize; ++i) {
            testDDRKeyFreqMap[featMem] = i;
            featMem++;
        }
    }

    void SetKeyCountMap(absl::flat_hash_map<emb_key_t, size_t>& testKeyCountMap)
    {
        for (int64_t i { 0 }; i < hostVocabSize; ++i) {
            testKeyCountMap[featMem] = i;
            featMem++;
        }
    }

    void SetExcludeDDRKeyFreqMap(unordered_map<emb_key_t, freq_num_t>& testExcludeDDRKeyFreqMap)
    {
        for (int64_t i { 0 }; i < hostVocabSize; ++i) {
            testExcludeDDRKeyFreqMap[featMem] = i;
            featMem++;
        }
    }

    void SetDDRKeyFreqMaps(KeyFreqMemT& testDDRKeyFreqMaps)
    {
        unordered_map<emb_key_t, freq_num_t> testDDRKeyFreqMap;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetDDRKeyFreqMap(testDDRKeyFreqMap);
            testDDRKeyFreqMaps[testEmbInfo.name] = std::move(testDDRKeyFreqMap);
        }
    }

    void SetKeyCountMaps(KeyCountMemT & testKeyCountMaps)
    {
        absl::flat_hash_map<emb_key_t, size_t> testKeyCountMap;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetKeyCountMap(testKeyCountMap);
            testKeyCountMaps[testEmbInfo.name] = std::move(testKeyCountMap);
        }
    }

    void SetExcludeDDRKeyFreqMaps(KeyFreqMemT& testExcludeDDRKeyFreqMaps)
    {
        unordered_map<emb_key_t, freq_num_t> testExcludeDDRKeyFreqMap;
        for (const auto& testEmbInfo : testEmbInfos) {
            SetExcludeDDRKeyFreqMap(testExcludeDDRKeyFreqMap);
            testExcludeDDRKeyFreqMaps[testEmbInfo.name] = std::move(testExcludeDDRKeyFreqMap);
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

    void SetHistRec(AdmitAndEvictData& histRec)
    {
        int64_t featureId { int64Min };
        int count { 1 };
        time_t lastTime { 1000 };
        time_t timeStamp { 10000 };

        for (const auto& testEmbInfo : testEmbInfos) {
            auto& historyRecords { histRec.historyRecords[testEmbInfo.name] };
            auto& timestamps { histRec.timestamps[testEmbInfo.name] };

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

    void SetHistRecCombine(AdmitAndEvictData& histRec)
    {
        int64_t featureId { int64Min };
        int count { 1 };
        time_t lastTime { 1000 };
        time_t timeStamp { 10000 };

        auto& historyRecords { histRec.historyRecords[COMBINE_HISTORY_NAME] };
        auto& timestamps { histRec.timestamps[COMBINE_HISTORY_NAME] };

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

TEST_F(CheckpointTest, HostEmbs)
{
    std::shared_ptr<EmbMemT> testHostEmbs = std::make_shared<EmbMemT>();
    SetEmbInfo();
    SetHostEmbs(testHostEmbs);
    shared_ptr<EmbMemT> validHostEmbs = std::make_shared<EmbMemT>();
    SetHostEmbs(validHostEmbs);
    shared_ptr<EmbMemT> loadHostEmbs = std::make_shared<EmbMemT>();
    SetHostEmptyEmbs(loadHostEmbs);

    CkptData testSaveData;
    CkptData validLoadData;
    CkptData testLoadData;

    testSaveData.hostEmbs = testHostEmbs.get();
    validLoadData.hostEmbs = validHostEmbs.get();
    testLoadData.hostEmbs = loadHostEmbs.get();

    Checkpoint testCkpt;
    testCkpt.SaveModel(testPath, testSaveData, rankInfo, testEmbInfos);
    testCkpt.LoadModel(testPath, testLoadData, rankInfo, testEmbInfos,
                       { CkptFeatureType::HOST_EMB });

    for (const auto& it : *validLoadData.hostEmbs) {
        const auto& embInfo = testLoadData.hostEmbs->at(it.first).hostEmbInfo;
        const auto& embData = testLoadData.hostEmbs->at(it.first).embData;

        EXPECT_EQ(it.second.hostEmbInfo.name, embInfo.name);
        EXPECT_EQ(it.second.hostEmbInfo.sendCount, embInfo.sendCount);
        EXPECT_EQ(it.second.hostEmbInfo.extEmbeddingSize, embInfo.extEmbeddingSize);
        EXPECT_EQ(it.second.hostEmbInfo.devVocabSize, embInfo.devVocabSize);
        EXPECT_EQ(it.second.hostEmbInfo.hostVocabSize, embInfo.hostVocabSize);

        EXPECT_EQ(it.second.embData, embData);
    }
}

TEST_F(CheckpointTest, EmbHashMaps)
{
    EmbHashMemT testEmbHashMaps;
    EmbHashMemT validEmbHashMaps;

    SetEmbInfo();
    SetEmbHashMaps(testEmbHashMaps);
    validEmbHashMaps = testEmbHashMaps;

    CkptData testSaveData;
    CkptData validLoadData;
    CkptData testLoadData;

    testSaveData.embHashMaps = std::move(testEmbHashMaps);
    validLoadData.embHashMaps = std::move(validEmbHashMaps);

    Checkpoint testCkpt;
    testCkpt.SaveModel(testPath, testSaveData, rankInfo, testEmbInfos);
    testCkpt.LoadModel(testPath, testLoadData, rankInfo, testEmbInfos, { CkptFeatureType::EMB_HASHMAP });

    EXPECT_EQ(validLoadData.embHashMaps.size(), testLoadData.embHashMaps.size());
    for (const auto& it : validLoadData.embHashMaps) {
        EXPECT_EQ(1, testLoadData.embHashMaps.count(it.first));

        const auto& hostHashMap = testLoadData.embHashMaps.at(it.first).hostHashMap;
        const auto& devOffset2Batch = testLoadData.embHashMaps.at(it.first).devOffset2Batch;
        const auto& devOffset2Key = testLoadData.embHashMaps.at(it.first).devOffset2Key;
        const auto& currentUpdatePos = testLoadData.embHashMaps.at(it.first).currentUpdatePos;
        const auto& hostVocabSize = testLoadData.embHashMaps.at(it.first).hostVocabSize;
        const auto& devVocabSize = testLoadData.embHashMaps.at(it.first).devVocabSize;

        EXPECT_EQ(it.second.hostHashMap, hostHashMap);

        EXPECT_EQ(it.second.devOffset2Batch, devOffset2Batch);
        EXPECT_EQ(it.second.devOffset2Key, devOffset2Key);

        EXPECT_EQ(it.second.currentUpdatePos, currentUpdatePos);
        EXPECT_EQ(it.second.hostVocabSize, hostVocabSize);
        EXPECT_EQ(it.second.devVocabSize, devVocabSize);
    }
}

TEST_F(CheckpointTest, KeyOffsetMaps)
{
    KeyOffsetMemT testKeyOffsetMaps;
    KeyOffsetMemT validKeyOffsetMaps;

    SetEmbInfo();
    SetKeyOffsetMaps(testKeyOffsetMaps);
    validKeyOffsetMaps = testKeyOffsetMaps;

    CkptData testSaveData;
    CkptData validLoadData;
    CkptData testLoadData;

    testSaveData.keyOffsetMap = std::move(testKeyOffsetMaps);
    validLoadData.keyOffsetMap = std::move(validKeyOffsetMaps);

    Checkpoint testCkpt;
    testCkpt.SaveModel(testPath, testSaveData, rankInfo, testEmbInfos);
    testCkpt.LoadModel(testPath, testLoadData, rankInfo, testEmbInfos, { CkptFeatureType::KEY_OFFSET_MAP });

    EXPECT_EQ(validLoadData.keyOffsetMap.size(), testLoadData.keyOffsetMap.size());
    for (const auto& it : validLoadData.keyOffsetMap) {
        EXPECT_EQ(1, testLoadData.keyOffsetMap.count(it.first));
        const auto& keyOffsetMap = testLoadData.keyOffsetMap.at(it.first);
        const auto& validKeyOffsetMap = validLoadData.keyOffsetMap.at(it.first);
        for (const auto& key: keyOffsetMap) {
            EXPECT_EQ(validKeyOffsetMap.count(key.first), 1);
        }
    }
}

TEST_F(CheckpointTest, AllMgmt)
{
    OffsetMemT testMaxOffset;
    OffsetMemT validMaxOffset;
    KeyOffsetMemT testKeyOffsetMaps;
    KeyOffsetMemT validKeyOffsetMaps;

    SetEmbInfo();
    SetMaxOffset(testMaxOffset);
    validMaxOffset = testMaxOffset;
    SetKeyOffsetMaps(testKeyOffsetMaps);
    validKeyOffsetMaps = testKeyOffsetMaps;

    CkptData testSaveData;
    CkptData validLoadData;
    CkptData testLoadData;

    testSaveData.maxOffset = std::move(testMaxOffset);
    validLoadData.maxOffset = std::move(validMaxOffset);
    testSaveData.keyOffsetMap = std::move(testKeyOffsetMaps);
    validLoadData.keyOffsetMap = std::move(validKeyOffsetMaps);

    Checkpoint testCkpt;
    testCkpt.SaveModel(testPath, testSaveData, rankInfo, testEmbInfos);
    testCkpt.LoadModel(testPath,
        testLoadData,
        rankInfo,
        testEmbInfos,
        {CkptFeatureType::KEY_OFFSET_MAP });

    EXPECT_EQ(validLoadData.maxOffset.size(), testLoadData.maxOffset.size());
    for (const auto& it : validLoadData.maxOffset) {
        EXPECT_EQ(1, testLoadData.maxOffset.count(it.first));
        const auto& maxOffset = testLoadData.maxOffset.at(it.first);
        EXPECT_EQ(it.second, maxOffset);
    }

    EXPECT_EQ(validLoadData.keyOffsetMap.size(), testLoadData.keyOffsetMap.size());
    for (const auto& it : validLoadData.keyOffsetMap) {
        EXPECT_EQ(1, testLoadData.keyOffsetMap.count(it.first));
        const auto& keyOffsetMap = testLoadData.keyOffsetMap.at(it.first);
        const auto& validKeyOffsetMap = validLoadData.keyOffsetMap.at(it.first);
        for (const auto& key: keyOffsetMap) {
            EXPECT_EQ(validKeyOffsetMap.count(key.first), 1);
        }
    }
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
    testCkpt.LoadModel(testPath, testLoadData, rankInfo, testEmbInfos, { CkptFeatureType::FEAT_ADMIT_N_EVICT });

    EXPECT_EQ(validLoadData.table2Thresh.size(), testLoadData.table2Thresh.size());
    EXPECT_EQ(validLoadData.histRec.historyRecords.size(), testLoadData.histRec.historyRecords.size());
    for (const auto& it : validLoadData.table2Thresh) {
        EXPECT_EQ(1, testLoadData.table2Thresh.count(it.first));

        const auto& table2Thresh = testLoadData.table2Thresh.at(it.first);

        EXPECT_EQ(it.second.tableName, table2Thresh.tableName);
        EXPECT_EQ(it.second.countThreshold, table2Thresh.countThreshold);
        EXPECT_EQ(it.second.timeThreshold, table2Thresh.timeThreshold);
    }

    for (const auto& it : validLoadData.histRec.timestamps) {
        EXPECT_EQ(1, testLoadData.histRec.timestamps.count(it.first));
        EXPECT_EQ(1, testLoadData.histRec.historyRecords.count(it.first));

        const auto& timestamps = testLoadData.histRec.timestamps.at(it.first);
        const auto& historyRecords = testLoadData.histRec.historyRecords.at(it.first);
        const auto& validHistRec = validLoadData.histRec.historyRecords.at(it.first);

        EXPECT_EQ(it.second, timestamps);
        for (const auto& validHR : validHistRec) {
            const auto& testHR = historyRecords.at(validHR.first);

            EXPECT_EQ(validHR.second.count, testHR.count);
            EXPECT_EQ(validHR.second.lastTime, testHR.lastTime);
        }
    }
}


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
    testCkpt.LoadModel(testPath, testLoadData, rankInfo, testEmbInfos, { CkptFeatureType::DDR_KEY_FREQ_MAP });
    EXPECT_EQ(validLoadData.ddrKeyFreqMaps.size(), testLoadData.ddrKeyFreqMaps.size());

    for (const auto& it : validLoadData.ddrKeyFreqMaps) {
        EXPECT_EQ(1, testLoadData.ddrKeyFreqMaps.count(it.first));
        const auto& ddrKeyFreqMap = testLoadData.ddrKeyFreqMaps.at(it.first);
        EXPECT_EQ(it.second, ddrKeyFreqMap);
    }
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
    testCkpt.LoadModel(testPath, testLoadData, rankInfo, testEmbInfos, { CkptFeatureType::KEY_COUNT_MAP });
    EXPECT_EQ(validLoadData.keyCountMap.size(), testLoadData.keyCountMap.size());
}
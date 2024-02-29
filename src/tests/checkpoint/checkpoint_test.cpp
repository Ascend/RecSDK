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

            for (auto ko: testHash) {
                embHashInfo.hostHashMap[ko.first] = static_cast<int64_t>(ko.second);
            }

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
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

#include "ckpt_data_handler/feat_admit_n_evict_ckpt/feat_admit_n_evict_ckpt.h"
#include "ckpt_data_handler/key_count_map_ckpt/key_count_map_ckpt.h"
#include "ckpt_data_handler/key_freq_map_ckpt/key_freq_map_ckpt.h"
#include "utils/common.h"

using namespace std;
using namespace MxRec;

using valid_float_t = absl::flat_hash_map<string, vector<float>>;
using valid_int_t = absl::flat_hash_map<string, vector<int>>;
using valid_int64_t = absl::flat_hash_map<string, vector<int64_t>>;
using valie_dataset_t = absl::flat_hash_map<string, vector<trans_serialize_t>>;
using valid_attrib_t = absl::flat_hash_map<string, vector<size_t>>;

struct InputArgs {
    vector<string>& embNames;
    CkptData& validData;
    FeatAdmitNEvictCkpt& testCkpt;
    valid_int_t& validTrens2ThreshArr;
    valid_attrib_t& validTrens2ThreshAttrib;
    valid_attrib_t& validHistRecAttrib;
    valid_int64_t& validHistRecArr;
    CkptData& testData;
};

class CkptDataHandlerTest : public testing::Test {
protected:
    int floatBytes { 4 };
    int int32Bytes { 4 };
    int int64Bytes { 8 };

    int64_t int64Min { static_cast<int64_t>(UINT32_MAX) };

    int maxChannelNum { MAX_CHANNEL_NUM };
    int keyProcessThread { 6 };

    vector<EmbInfo> testEmbInfos;
    valid_int_t validEmbInfo;
    valid_attrib_t validEmbInfoAttrib;

    void SetEmbInfo()
    {
        int embInfoNum { 10 };

        string name { "table" };
        int sendCount { 8 };
        int embeddingSize { 100 };
        size_t devVocabSize { 8 };
        size_t hostVocabSize { 16 };

        int idx { 0 };
        testEmbInfos.resize(embInfoNum);
        for (auto& testEmbInfo : testEmbInfos) {
            testEmbInfo.name = name + to_string(idx);
            testEmbInfo.sendCount = sendCount;
            testEmbInfo.extEmbeddingSize = embeddingSize;
            testEmbInfo.devVocabSize = devVocabSize;
            testEmbInfo.hostVocabSize = hostVocabSize;

            vector<int> validInt { sendCount,
                embeddingSize,
                static_cast<int>(devVocabSize),
                static_cast<int>(hostVocabSize) };
            vector<size_t> validAttrib { static_cast<size_t>(int32Bytes), static_cast<size_t>(int32Bytes) };

            validEmbInfo[name + to_string(idx)] = move(validInt);
            validEmbInfoAttrib[name + to_string(idx)] = move(validAttrib);
            ++idx;
        }
    }

    void SetTable2Threshold(Table2ThreshMemT& testTable2Threshold,
                           valid_int_t& validArr,
                           valid_attrib_t& validAttrib)
    {
        int countThreshold { 20 };
        int timeThreshold { 100 };
        int isSum {1};

        for (const auto& testEmbInfo : testEmbInfos) {
            ThresholdValue val;
            val.tableName = testEmbInfo.name;
            val.countThreshold = countThreshold;
            val.timeThreshold = timeThreshold;
            val.isEnableSum = true;

            vector<int> valid { countThreshold, timeThreshold, isSum};

            countThreshold++;
            timeThreshold++;

            testTable2Threshold[testEmbInfo.name] = move(val);
            validArr[testEmbInfo.name] = move(valid);
            validAttrib[testEmbInfo.name].push_back(3); // 3 is element num in  one vector
            validAttrib[testEmbInfo.name].push_back(int32Bytes);
        }
    }

    void SetHistRec(AdmitAndEvictData& histRec, valid_int64_t& validArr, valid_attrib_t& validAttrib)
    {
        int64_t featureId { int64Min };
        int count { 1 };
        time_t lastTime { 1000 };
        time_t timeStamp { 10000 };

        for (const auto& testEmbInfo : testEmbInfos) {
            auto& validA { validArr[testEmbInfo.name] };
            auto& historyRecords { histRec.historyRecords[testEmbInfo.name] };
            auto& timestamps { histRec.timestamps[testEmbInfo.name] };

            timestamps = timeStamp;
            validA.push_back(timeStamp);

            for (int i = 0; i < count; ++i) {
                historyRecords[featureId].count = count;
                historyRecords[featureId].lastTime = lastTime;

                validA.push_back(featureId);
                validA.push_back(count);
                validA.push_back(lastTime);

                featureId++;
            }

            auto& attribute = validAttrib[testEmbInfo.name];
            attribute.push_back(count);
            attribute.push_back(int64Bytes);

            count++;
            lastTime++;
            timeStamp++;
        }
    }

    void SetHistRecCombine(AdmitAndEvictData& histRec, valid_int64_t& validArr, valid_attrib_t& validAttrib)
    {
        int64_t featureId { int64Min };
        int count { 1 };
        time_t lastTime { 1000 };
        time_t timeStamp { 10000 };

        auto& validA { validArr[COMBINE_HISTORY_NAME] };
        auto& historyRecords { histRec.historyRecords[COMBINE_HISTORY_NAME] };
        auto& timestamps { histRec.timestamps[COMBINE_HISTORY_NAME] };

        timestamps = timeStamp;
        validA.push_back(timeStamp);

        for (int i = 0; i < count; ++i) {
            historyRecords[featureId].count = count;
            historyRecords[featureId].lastTime = lastTime;

            validA.push_back(featureId);
            validA.push_back(count);
            validA.push_back(lastTime);

            featureId++;
        }

        auto& attribute = validAttrib[COMBINE_HISTORY_NAME];
        attribute.push_back(count);
        attribute.push_back(int64Bytes);

        count++;
        lastTime++;
        timeStamp++;
    }

    void TestForSave(InputArgs& args)
    {
        for (const auto& embName : args.embNames) {
            EXPECT_EQ(1, args.validData.table2Thresh.count(embName));

            CkptTransData testSaveData = args.testCkpt.GetDataset(CkptDataType::TABLE_2_THRESH, embName);
            EXPECT_EQ(args.validTrens2ThreshArr.at(embName), testSaveData.int32Arr); // need other test method
            EXPECT_EQ(args.validTrens2ThreshAttrib.at(embName), testSaveData.attribute);
            testSaveData = args.testCkpt.GetDataset(CkptDataType::HIST_REC, embName);
            bool isCombine = false;

            if (!isCombine) {
                EXPECT_EQ(1, args.validData.histRec.timestamps.count(embName));
                EXPECT_EQ(1, args.validData.histRec.historyRecords.count(embName));
                EXPECT_EQ(args.validHistRecAttrib.at(embName), testSaveData.attribute);
            } else {
                EXPECT_EQ(1, args.validData.histRec.timestamps.count(COMBINE_HISTORY_NAME));
                EXPECT_EQ(1, args.validData.histRec.historyRecords.count(COMBINE_HISTORY_NAME));
                EXPECT_EQ(args.validHistRecAttrib.at(COMBINE_HISTORY_NAME), testSaveData.attribute);
            }
        }
    }
    void TestForLoad(InputArgs& args)
    {
        CkptTransData testLoadData;
        for (const auto& embName : args.embNames) {
            testLoadData.int32Arr = args.validTrens2ThreshArr.at(embName);
            testLoadData.attribute = args.validTrens2ThreshAttrib.at(embName);
            args.testCkpt.SetDataset(CkptDataType::TABLE_2_THRESH, embName, testLoadData);
            bool isCombine = false;

            if (!isCombine) {
                testLoadData.int64Arr = args.validHistRecArr.at(embName);
                testLoadData.attribute = args.validHistRecAttrib.at(embName);
            } else {
                testLoadData.int64Arr = args.validHistRecArr.at(COMBINE_HISTORY_NAME);
                testLoadData.attribute = args.validHistRecAttrib.at(COMBINE_HISTORY_NAME);
            }
            args.testCkpt.SetDataset(CkptDataType::HIST_REC, embName, testLoadData);
        }
        args.testCkpt.GetProcessData(args.testData);

        EXPECT_EQ(args.validData.table2Thresh.size(), args.testData.table2Thresh.size());
        EXPECT_EQ(args.validData.histRec.historyRecords.size(), args.testData.histRec.historyRecords.size());
        for (const auto& it : args.validData.table2Thresh) {
            EXPECT_EQ(1, args.testData.table2Thresh.count(it.first));

            const auto& table2Thresh = args.testData.table2Thresh.at(it.first);

            EXPECT_EQ(it.second.tableName, table2Thresh.tableName);
            EXPECT_EQ(it.second.countThreshold, table2Thresh.countThreshold);
            EXPECT_EQ(it.second.timeThreshold, table2Thresh.timeThreshold);
        }

        for (const auto& it : args.validData.histRec.timestamps) {
            EXPECT_EQ(1, args.testData.histRec.timestamps.count(it.first));
            EXPECT_EQ(1, args.testData.histRec.historyRecords.count(it.first));

            const auto& historyRecords = args.testData.histRec.historyRecords.at(it.first);
            const auto& validHistRec = args.validData.histRec.historyRecords.at(it.first);

            for (const auto& validHR : validHistRec) {
                const auto& testHR = historyRecords.at(validHR.first);

                EXPECT_EQ(validHR.second.count, testHR.count);
                EXPECT_EQ(validHR.second.lastTime, testHR.lastTime);
            }
        }
    }
};

TEST_F(CkptDataHandlerTest, FeatAdmitNEvict)
{
    Table2ThreshMemT testTrens2Thresh;
    Table2ThreshMemT validTrens2Thresh;
    AdmitAndEvictData testHistRec;
    AdmitAndEvictData validHistRec;

    valid_int_t validTrens2ThreshArr;
    valid_int64_t validHistRecArr;
    valid_attrib_t validTrens2ThreshAttrib;
    valid_attrib_t validHistRecAttrib;

    SetEmbInfo();
    SetTable2Threshold(testTrens2Thresh, validTrens2ThreshArr, validTrens2ThreshAttrib);
    validTrens2Thresh = testTrens2Thresh;
    bool isCombine = false;

    if (isCombine) {
        SetHistRecCombine(testHistRec, validHistRecArr, validHistRecAttrib);
    } else {
        SetHistRec(testHistRec, validHistRecArr, validHistRecAttrib);
    }
    validHistRec = testHistRec;

    CkptData testData;
    CkptData validData;
    FeatAdmitNEvictCkpt testCkpt;

    testData.table2Thresh = testTrens2Thresh;
    testData.histRec.timestamps = testHistRec.timestamps;
    testData.histRec.historyRecords = testHistRec.historyRecords;
    validData.table2Thresh = validTrens2Thresh;
    validData.histRec.timestamps = validHistRec.timestamps;
    validData.histRec.historyRecords = validHistRec.historyRecords;

    testCkpt.SetProcessData(testData);

    vector<string> embNames { testCkpt.GetEmbNames() };
    EXPECT_EQ(validData.table2Thresh.size(), embNames.size());
    EXPECT_EQ(testCkpt.GetDataTypes(), vector<CkptDataType>({ CkptDataType::TABLE_2_THRESH, CkptDataType::HIST_REC }));
    EXPECT_EQ(testCkpt.GetDirNames(), vector<string>({ "HashTable", "FEAT_INFO" }));

    InputArgs args = {embNames, validData, testCkpt, validTrens2ThreshArr, validTrens2ThreshAttrib,
                      validHistRecAttrib, validHistRecArr, testData};
    // 测试save
    TestForSave(args);

    // 测试load
    TestForLoad(args);
}

TEST_F(CkptDataHandlerTest, FeatAdmitNEvict_SetProcessData_Error)
{
    CkptData testData;
    FeatAdmitNEvictCkpt testCkpt;

    EXPECT_THROW(testCkpt.SetProcessData(testData), std::runtime_error);
}

TEST_F(CkptDataHandlerTest, KeyCountMapCkpt)
{
    CkptData data;
    KeyCountMapCkpt ckpt;
    CkptTransData tranData;
    const int testCount = 10;
    const int exceptCount = 5;
    for (int i = 0; i < testCount; ++i) {
        tranData.int64Arr.push_back(i);
    }

    ckpt.SetProcessData(data);
    ckpt.SetDataset(CkptDataType::KEY_COUNT_MAP, std::string("test"), tranData);
    ckpt.GetProcessData(data);
    absl::flat_hash_map<emb_key_t, size_t> &testmap = data.keyCountMap["test"];
    EXPECT_EQ(testmap.size(), exceptCount);
}

TEST_F(CkptDataHandlerTest, SetDatasetForLoadEmb)
{
    KeyCountMapCkpt ckpt;
    CkptTransData tranData;
    CkptData data;
    try {
        ckpt.SetDatasetForLoadEmb(CkptDataType::KEY_COUNT_MAP, std::string("test"), tranData, data);
    } catch (runtime_error& e) {
        LOG_INFO(KEY_PROCESS "success");
    }
}

TEST_F(CkptDataHandlerTest, KeyFreqMapCkpt)
{
    CkptData data;
    KeyFreqMapCkpt ckpt;
    CkptTransData tranData;
    const int testCount = 10;

    ckpt.GetProcessData(data);

    for (int i = 0; i < testCount; ++i) {
        tranData.int64Arr.push_back(i);
    }
    ckpt.SetDataset(CkptDataType::DDR_FREQ_MAP, "test", tranData);
    EXPECT_EQ(tranData.datasetSize, 0);

    for (int i = 0; i < testCount; ++i) {
        tranData.int64Arr.push_back(i);
    }
    ckpt.SetDataset(CkptDataType::EXCLUDE_FREQ_MAP, "test", tranData);
    EXPECT_EQ(tranData.datasetSize, 0);
    
    EXPECT_EQ(ckpt.GetDirNames(), vector<string>({ "HashTable", "SSD" }));
}

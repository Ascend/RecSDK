/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-12-03
 */

#include <gtest/gtest.h>

#include "ckpt_data_handler/host_emb_ckpt/host_emb_ckpt.h"
#include "ckpt_data_handler/emb_hash_ckpt/emb_hash_ckpt.h"
#include "ckpt_data_handler/nddr_offset_ckpt/nddr_offset_ckpt.h"
#include "ckpt_data_handler/nddr_feat_map_ckpt/nddr_feat_map_ckpt.h"
#include "ckpt_data_handler//feat_admit_n_evict_ckpt/feat_admit_n_evict_ckpt.h"

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

    void SetTable2Threshold(table_2_thresh_mem_t& testTable2Threshold,
                           valid_int_t& validArr,
                           valid_attrib_t& validAttrib)
    {
        int countThreshold { 20 };
        int timeThreshold { 100 };

        for (const auto& testEmbInfo : testEmbInfos) {
            ThresholdValue val;
            val.tableName = testEmbInfo.name;
            val.countThreshold = countThreshold;
            val.timeThreshold = timeThreshold;

            vector<int> valid { countThreshold, timeThreshold };

            countThreshold++;
            timeThreshold++;

            testTable2Threshold[testEmbInfo.name] = move(val);
            validArr[testEmbInfo.name] = move(valid);
            validAttrib[testEmbInfo.name].push_back(2); // 2 is element num in  one vector
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
    table_2_thresh_mem_t testTrens2Thresh;
    table_2_thresh_mem_t validTrens2Thresh;
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

    InputArgs args = {embNames, validData, testCkpt, validTrens2ThreshArr, validTrens2ThreshAttrib,
                      validHistRecAttrib, validHistRecArr, testData};
    // 测试save
    TestForSave(args);

    // 测试load
    TestForLoad(args);
}
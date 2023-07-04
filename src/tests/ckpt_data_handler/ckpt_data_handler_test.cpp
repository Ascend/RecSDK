/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-12-03
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/ranges.h>

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

class CkptDataHandlerTest : public testing::Test {
protected:
    int floatBytes { 4 };
    int int32Bytes { 4 };
    int int64Bytes { 8 };

    int64_t int64Min { static_cast<int64_t>(UINT32_MAX) };

    int maxChannelNum { MAX_CHANNEL_NUM };
    int keyProcessThread { PerfConfig::keyProcessThreadNum };

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

    void SetTens2Threshold(tensor_2_thresh_mem_t& testTens2Threshold,
                           valid_int_t& validArr,
                           valid_attrib_t& validAttrib)
    {
        int countThreshold { 20 };
        int timeThreshold { 100 };

        for (const auto& testEmbInfo : testEmbInfos) {
            ThresholdValue val;
            val.tensorName = testEmbInfo.name;
            val.countThreshold = countThreshold;
            val.timeThreshold = timeThreshold;

            vector<int> valid { countThreshold, timeThreshold };

            countThreshold++;
            timeThreshold++;

            testTens2Threshold[testEmbInfo.name] = move(val);
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
};

TEST_F(CkptDataHandlerTest, FeatAdmitNEvict)
{
    tensor_2_thresh_mem_t testTrens2Thresh;
    tensor_2_thresh_mem_t validTrens2Thresh;
    AdmitAndEvictData testHistRec;
    AdmitAndEvictData validHistRec;

    valid_int_t validTrens2ThreshArr;
    valid_int64_t validHistRecArr;
    valid_attrib_t validTrens2ThreshAttrib;
    valid_attrib_t validHistRecAttrib;

    SetEmbInfo();
    SetTens2Threshold(testTrens2Thresh, validTrens2ThreshArr, validTrens2ThreshAttrib);
    validTrens2Thresh = testTrens2Thresh;
    SetHistRec(testHistRec, validHistRecArr, validHistRecAttrib);
    validHistRec = testHistRec;

    CkptData testData;
    CkptData validData;
    FeatAdmitNEvictCkpt testCkpt;

    testData.tens2Thresh = testTrens2Thresh;
    testData.histRec.timestamps = testHistRec.timestamps;
    testData.histRec.historyRecords = testHistRec.historyRecords;
    validData.tens2Thresh = validTrens2Thresh;
    validData.histRec.timestamps = validHistRec.timestamps;
    validData.histRec.historyRecords = validHistRec.historyRecords;

    testCkpt.SetProcessData(testData);

    vector<string> embNames { testCkpt.GetEmbNames() };
    CkptTransData testSaveData;
    EXPECT_EQ(validData.tens2Thresh.size(), embNames.size());

    for (const auto& embName : embNames) {
        EXPECT_EQ(1, validData.tens2Thresh.count(embName));

        EXPECT_EQ(1, validData.histRec.timestamps.count(embName));
        EXPECT_EQ(1, validData.histRec.historyRecords.count(embName));

        testSaveData = testCkpt.GetDataset(CkptDataType::TENSOR_2_THRESH, embName);
        EXPECT_EQ(validTrens2ThreshArr.at(embName), testSaveData.int32Arr); // need other test method
        EXPECT_EQ(validTrens2ThreshAttrib.at(embName), testSaveData.attribute);
        testSaveData = testCkpt.GetDataset(CkptDataType::HIST_REC, embName);
        EXPECT_EQ(validHistRecAttrib.at(embName), testSaveData.attribute);
    }

    CkptTransData testLoadData;
    for (const auto& embName : embNames) {
        testLoadData.int32Arr = validTrens2ThreshArr.at(embName);
        testLoadData.attribute = validTrens2ThreshAttrib.at(embName);
        testCkpt.SetDataset(CkptDataType::TENSOR_2_THRESH, embName, testLoadData);

        testLoadData.int64Arr = validHistRecArr.at(embName);
        testLoadData.attribute = validHistRecAttrib.at(embName);
        testCkpt.SetDataset(CkptDataType::HIST_REC, embName, testLoadData);
    }
    testCkpt.GetProcessData(testData);

    EXPECT_EQ(validData.tens2Thresh.size(), testData.tens2Thresh.size());
    EXPECT_EQ(validData.histRec.historyRecords.size(), testData.histRec.historyRecords.size());
    for (const auto& it : validData.tens2Thresh) {
        EXPECT_EQ(1, testData.tens2Thresh.count(it.first));

        const auto& tens2Thresh = testData.tens2Thresh.at(it.first);

        EXPECT_EQ(it.second.tensorName, tens2Thresh.tensorName);
        EXPECT_EQ(it.second.countThreshold, tens2Thresh.countThreshold);
        EXPECT_EQ(it.second.timeThreshold, tens2Thresh.timeThreshold);
    }

    for (const auto& it : validData.histRec.timestamps) {
        EXPECT_EQ(1, testData.histRec.timestamps.count(it.first));
        EXPECT_EQ(1, testData.histRec.historyRecords.count(it.first));

        const auto& historyRecords = testData.histRec.historyRecords.at(it.first);
        const auto& validHistRec = validData.histRec.historyRecords.at(it.first);

        for (const auto& validHR : validHistRec) {
            const auto& testHR = historyRecords.at(validHR.first);

            EXPECT_EQ(validHR.second.count, testHR.count);
            EXPECT_EQ(validHR.second.lastTime, testHR.lastTime);
        }
    }
}
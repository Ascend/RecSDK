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

#include <ctime>
#include <cstdio>
#include <memory>
#include <thread>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "utils/common.h"
#include "key_process/feature_admit_and_evict.h"
#include "absl/container/flat_hash_map.h"

using namespace std;
using namespace testing;
using namespace MxRec;

enum SleepTime : uint32_t {
    SLEEP_SECOND_1 = 1,
    SLEEP_SECOND_2 = 2,
    SLEEP_SECOND_3 = 3,
    SLEEP_SECOND_4 = 4,
    SLEEP_SECOND_5 = 5,
    SLEEP_SECOND_6 = 6,
    SLEEP_SECOND_7 = 7,
    SLEEP_SECOND_8 = 8,
    SLEEP_SECOND_9 = 9,
    SLEEP_SECOND_10 = 10
};

using HashMapInfo = absl::flat_hash_map<int64_t, FeatureItemInfo>;
struct InputArgs {
    KeysT keys;
    vector<uint32_t> cnt;
    KeysT expectKeys;
    HashMapInfo lastHistory;
    HashMapInfo expectHistory;
};

class FeatureAdmitAndEvictTest : public testing::Test {
protected:
    HashMapInfo GetHistoryRecords(KeysT& keys, vector<uint32_t>& cnt, time_t ts, std::string embName,
        HashMapInfo& oldInfos)
    {
        HashMapInfo newInfos;
        unordered_map<int64_t, uint32_t> mergeKeys;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto it = mergeKeys.find(keys[i]);
            if (it == mergeKeys.end()) {
                mergeKeys[keys[i]] = cnt[i];
            } else {
                mergeKeys[keys[i]] += cnt[i];
            }
        }

        for (auto& ele : mergeKeys) {
            if (ele.first == -1) {
                continue ;
            }

            uint32_t oldCnt = 0;
            auto it = oldInfos.find(ele.first);
            if (it != oldInfos.end()) { // 把原有历史记录累加
                oldCnt = it->second.count;
                oldInfos.erase(ele.first);
            }

            FeatureItemInfo info = {ele.second + oldCnt, ts};
            newInfos.insert(std::pair<int64_t, FeatureItemInfo>(ele.first, info));
        }

        for (auto& ele : oldInfos) { // 把原有存在、但当前不存在的，汇总
            newInfos.insert(std::pair<int64_t, FeatureItemInfo>(ele.first, ele.second));
        }

        printf("now, expect history info: \n");
        for (auto& ele : newInfos) {
            printf("\t{featureId[%ld], count[%d], lastTime[%ld]}\n", ele.first,
                ele.second.count, ele.second.lastTime);
        }
        printf("\n");

        return newInfos;
    }
    bool IsAllTheSameMap(HashMapInfo& records1, HashMapInfo& records2)
    {
        if (records1.empty() || records1.size() != records2.size()) {
            printf("IsAllTheSameMap() 111111\n");
            return false;
        }

        for (auto& ele1 : records1) {
            FeatureItemInfo& info1 = ele1.second;
            auto it = records2.find(ele1.first);
            if (it == records2.end()) {
                printf("IsAllTheSameMap() 222222\n");
                return false;
            }

            FeatureItemInfo& info2 = records2[ele1.first];
            if (info1.count != info2.count ||
                info1.lastTime != info2.lastTime) {
                printf("IsAllTheSameMap() 333333\n");
                return false;
            }
        }

        return true;
    }
    bool IsAllTheSameVector(KeysT& keys1, KeysT& keys2)
    {
        printf("\nrun ret: keys1 ===> \n\t");
        for (auto &k1 : keys1) {
            printf("%ld  ", k1);
        }
        printf("\nexpect ret: keys2 ===> \n\t");
        for (auto &k2 : keys2) {
            printf("%ld  ", k2);
        }
        printf("\n\n");

        if (keys1.empty() || keys1.size() != keys2.size()) {
            printf("IsAllTheSameVector() AAAAAA\n");
            return false;
        }

        size_t loopTimes = keys1.size();
        for (size_t i = 0; i < loopTimes; ++i) {
            if (keys1[i] != keys2[i]) {
                printf("IsAllTheSameVector() BBBBBB\n");
                return false;
            }
        }
        return true;
    }
    void FeatureAdmitCommon(FeatureAdmitAndEvict& faae, int channel, string embName, InputArgs& args)
    {
        time_t ts = time(nullptr);
        KeysT tmpKeys = args.keys;
        std::unique_ptr<EmbBatchT> batch = make_unique<EmbBatchT>();
        batch->name = embName;
        batch->timestamp = ts;
        printf("\n");
        LOG_INFO("current admit embName[{}] at time[{}] ...", embName, ts);

        // 校验调接口不出错
        ASSERT_EQ(faae.FeatureAdmit(channel, batch, args.keys, args.cnt) !=
                  FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR, true);

        // 校验特征准入的结果
        ASSERT_EQ(IsAllTheSameVector(args.keys, args.expectKeys), true);

        // 校验历史记录表信息
        args.expectHistory = GetHistoryRecords(tmpKeys, args.cnt, ts, embName, args.lastHistory);
        std::lock_guard<std::mutex> lock(faae.m_syncMutexs); // 与 evict-thread 竞争资源
        ASSERT_EQ(IsAllTheSameMap(faae.m_recordsData.historyRecords[embName], args.expectHistory), true);
    }

    void FeatureAdmitCommonMultiThr(FeatureAdmitAndEvict& faae, int channel, string embName, InputArgs& args)
    {
        time_t ts = time(nullptr);
        KeysT tmpKeys = args.keys;
        std::unique_ptr<EmbBatchT> batch = make_unique<EmbBatchT>();
        batch->name = embName;
        batch->timestamp = ts;
        printf("\n");
        LOG_INFO("current admit embName[{}] at time[{}] ...", embName, ts);

        // 校验调接口不出错
        faae.FeatureAdmit(channel, batch, args.keys, args.cnt);
    }

    void TestCaseHelpMultiThr(std::string thrName)
    {
        printf("\t############# [%s] tid[%lu] ############# begin ...\n",
               thrName.c_str(), std::hash<std::thread::id>{}(std::this_thread::get_id()));
        /*
        {"tableAAA", 2, 5}
        keys1 = {11, 11, 33, 44, 11, 55, 88, 55}
        cnt1 =   1   2   1   3   1   1   4   1
        */
        InputArgs args1 = {keys1, cnt1, {}, initHistory, {}}; // 每个表的第一次记录，要用initHistory追加
        FeatureAdmitCommonMultiThr(faae, 0, thresholds[0].tableName, args1);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_1));

        /*
        {"tableAAA", 2, 5}
        keys2 = {11, 12, 33, 21, 11, 12}
        cnt2 =   1   2   1   1   2   3
        */
        InputArgs args2 = {keys2, cnt2, {}, args1.expectHistory, {}};
        FeatureAdmitCommonMultiThr(faae, 0, thresholds[0].tableName, args2);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_2));

        /*
        {"tableBBB", 3, 7}
        keys3 = {123, 121, 121, 212, 211}
        cnt3 =   1    2    1    1    2
        */
        InputArgs args3 = {keys3, cnt3, {}, initHistory, {}};
        FeatureAdmitCommonMultiThr(faae, 0, thresholds[1].tableName, args3);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_6));

        /*
        {"tableAAA", 2, 5}
        keys4 = {11, 11, 33, 44, 55, 88, 55}
        cnt4 =   1   2   3   2   1   2   1
        */
        InputArgs args4 = {keys4, cnt4, {}, args2.expectHistory, {}};
        FeatureAdmitCommonMultiThr(faae, 0, thresholds[0].tableName, args4);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_2));

        /*
        {"tableBBB", 3, 7}
        keys5 = {125, 121, 122, 212, 211}
        cnt5 =   1    2    1    3    1
        */
        InputArgs args5 = {keys5, cnt5, {}, args3.expectHistory, {}};
        FeatureAdmitCommonMultiThr(faae, 0, thresholds[1].tableName, args5);

        printf("\t############# [%s] tid[%lu] ############# end ...\n", thrName.c_str(),
               std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }

    void StartEvictThread()
    {
        evictThr = std::thread([&]() {
            LOG_INFO("Evict-thread start ...");

            time_t currTime = 0;
            time_t lastTime = 0;
            while (!isExitFlag) {
                std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_2));
                currTime = time(nullptr);
                if (currTime - lastTime >= SleepTime::SLEEP_SECOND_4) {
                    LOG_INFO("Evict-thread doing at currTime[{}] ...", currTime);
                    map<std::string, std::vector<emb_cache_key_t>> evictPosMap {};
                    faae.FeatureEvict(evictPosMap);
                    lastTime = currTime;
                }
            }
            LOG_INFO("Evict-thread exit ...");
        });
    }
    void WaitEvictThread()
    {
        map<std::string, std::vector<emb_cache_key_t>> evictPosMap {};
        faae.FeatureEvict(evictPosMap); // 退出前保证执行了一次“淘汰”
        isExitFlag = true;
        if (evictThr.joinable()) {
            evictThr.join();
        }
    }

    // 同时配置“准入、淘汰”阈值 && batch带时间戳，淘汰功能正常
    void TestCase1()
    {
        faae.ResetAllRecords();
        faae.ParseThresholdCfg(thresholds);
        faae.SetCombineSwitch();
        StartEvictThread();

        printf("Current test single-thread is [%lu]\n",
               std::hash<std::thread::id>{}(std::this_thread::get_id()));
        /*
        {"tableAAA", 2, 5}
        keys1 = {11, 11, 33, 44, 11, 55, 88, 55}
        cnt1 =   1   2   1   3   1   1   4   1
        */
        KeysT expectRet1 = {11, 11, -1, 44, 11, 55, 88, 55};
        InputArgs args1 = {keys1, cnt1, expectRet1, initHistory, {}}; // 每个表的第一次记录，要用initHistory追加
        FeatureAdmitCommon(faae, 0, thresholds[0].tableName, args1);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_1));

        /*
        {"tableAAA", 2, 5}
        keys2 = {11, 12, 33, 21, 11, 12}
        cnt2 =   1   2   1   1   2   3
        */
        KeysT expectRet2 = {11, 12, 33, -1, 11, 12};
        InputArgs args2 = {keys2, cnt2, expectRet2, args1.expectHistory, {}};
        FeatureAdmitCommon(faae, 0, thresholds[0].tableName, args2);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_2));

        /*
        {"tableBBB", 3, 7}
        keys3 = {123, 121, 121, 212, 211}
        cnt3 =   1    2    1    1    2
        */
        KeysT expectRet3 = {-1, 121, 121, -1, -1};
        InputArgs args3 = {keys3, cnt3, expectRet3, initHistory, {}};
        FeatureAdmitCommon(faae, 0, thresholds[1].tableName, args3);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_6));

        /*
        {"tableAAA", 2, 5}
        keys4 = {11, 11, 33, 44, 55, 88, 55}
        cnt4 =   1   2   3   2   1   2   1
        */
        KeysT expectRet4 = {11, 11, 33, 44, 55, 88, 55};
        InputArgs args4 = {keys4, cnt4, expectRet4, args2.expectHistory, {}};
        FeatureAdmitCommon(faae, 0, thresholds[0].tableName, args4);
        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_2));

        /*
        {"tableBBB", 3, 7}
        keys5 = {125, 121, 122, 212, 211}
        cnt5 =   1    2    1    3    1
        */
        KeysT expectRet5 = {-1, 121, -1, 212, 211};
        InputArgs args5 = {keys5, cnt5, expectRet5, args3.expectHistory, {}};
        FeatureAdmitCommon(faae, 0, thresholds[1].tableName, args5);

        WaitEvictThread();
        LOG_INFO("TestCase1: single thread test over ...");
    }

    // 进行“准入”逻辑时，若(splitKey.size() != keyCount.size())，则业务报错退出；（说明是前面all2all通信数据错误）
    void TestCase2()
    {
        faae.ResetAllRecords();
        faae.ParseThresholdCfg(thresholds);

        // 测试点：tmpCnt.size() != tmpKeys.size()
        KeysT tmpKeys = {11, 11, 33, 44, 11, 55, 88, 55};
        vector<uint32_t> tmpCnt = {1, 2, 1, 3, 1, 1, 4};

        std::unique_ptr<EmbBatchT> batch = make_unique<EmbBatchT>();
        batch->name = thresholds[0].tableName;
        batch->timestamp = time(nullptr);

        // 校验调接口，出错
        ASSERT_EQ(faae.FeatureAdmit(0, batch, tmpKeys, tmpCnt) ==
                  FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR, true);
        LOG_INFO("TestCase2 over ...");
    }

    // 准入、淘汰阈值可单独配置；只配置“准入”阈值、却不配置“淘汰”阈值，功能正常；
    // 但是，配置了“淘汰”阈值，就必须得有“准入”阈值，才功能正常；
    void TestCase3()
    {}

    // 传递时间戳，与不传递时，保证batch数据相同
    void TestCase4()
    {}

    // 校验多线程跑的结果
    void CheckMultiThreadRet(KeysT& expectKeys, std::vector<uint32_t>& expectCnt, const std::string& embName,
                             int threadCnt)
    {
        // 校验历史记录表信息
        unordered_map<int64_t, uint32_t> mergeKeys;
        for (size_t i = 0; i < expectKeys.size(); ++i) {
            mergeKeys.insert(std::pair<int64_t, uint32_t>(expectKeys[i], expectCnt[i] * threadCnt));
        }

        auto &history = faae.m_recordsData.historyRecords[embName];
        for (auto& ele : mergeKeys) {
            auto it = history.find(ele.first);
            ASSERT_EQ(it != history.end(), true);
            ASSERT_EQ(history[ele.first].count == ele.second, true);
        }
    }
    static void TestMultiThread(FeatureAdmitAndEvictTest* testObj, std::string& thrName)
    {
        testObj->TestCaseHelpMultiThr(thrName);
    }
    // 多线程跑，特征准入&淘汰功能正常；
    void TestCase5()
    {
        faae.ResetAllRecords();
        faae.ParseThresholdCfg(thresholds);
        faae.SetCombineSwitch();
        StartEvictThread();

        std::thread thrs[6];
        int threadNum = 1;
        // 测试多线程的
        for (int i = 0; i < threadNum; ++i) {
            std::string name("thread-");
            name += std::to_string(i);
            thrs[i] = std::thread(TestMultiThread, this, std::ref(name));
            std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_1));
        }

        for (int i = 0; i < threadNum; ++i) {
            if (thrs[i].joinable()) {
                thrs[i].join();
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(SleepTime::SLEEP_SECOND_8));
        {
            /*
            如果没有淘汰功能
            tableAAA数据将会是 {11, 12, 21, 33, 44, 55, 88}
                               10  5   1   5   5   4   6
            tableBBB数据将会是 {121, 122, 123, 125, 211, 212};
                               5    1    1    1    3    4
            */
            KeysT expectKeys1 = {11, 33, 44, 55, 88};      // 12,21被淘汰掉了
            vector<uint32_t> expectCnt1 = {10, 5, 5, 4, 6};
            KeysT expectKeys2 = {121, 122, 125, 211, 212}; // 123被淘汰掉了
            vector<uint32_t> expectCnt2 = {5, 1, 1, 3, 4};
            std::lock_guard <std::mutex> lock(faae.m_syncMutexs); // 与 evict-thread 竞争资源
            CheckMultiThreadRet(expectKeys1, expectCnt1, thresholds[0].tableName, threadNum);
            CheckMultiThreadRet(expectKeys2, expectCnt2, thresholds[1].tableName, threadNum);
        }

        WaitEvictThread();
        LOG_INFO("TestCase5: multi thread test over ...");
    }

    // 同时不配置“准入、淘汰”阈值，特征准入&淘汰功能“不支持”；
    void TestCase6()
    {
        faae.ResetAllRecords();
        faae.ParseThresholdCfg(thresholds);

        std::unique_ptr<EmbBatchT> batch = make_unique<EmbBatchT>();
        // 测试点：tableDDD表没有配置阈值，则不支持
        batch->name = std::string("tableDDD");
        batch->timestamp = time(nullptr);

        // 校验调接口，不支持
        LOG_INFO("TestCase6 over ...");
    }

    void TestAdmitAndEvictInit()
    {
        faae.ResetAllRecords();
        std::vector<ThresholdValue> emptyThresholds = {};
        bool parseRet = faae.Init(emptyThresholds);
        ASSERT_EQ(parseRet, false);

        parseRet = faae.Init(thresholds);
        ASSERT_EQ(parseRet, true);
    }

    void TestAdmitAndEvictThresholdCfgCheck()
    {
        faae.ResetAllRecords();
        std::vector<ThresholdValue> emptyThresholds = {};
        bool ret = faae.ParseThresholdCfg(emptyThresholds);
        ASSERT_EQ(ret, false);

        // when thresholds is empty, should return true.
        std::vector<std::string> embNames = {"tableCCC", "tableBBB"};
        bool checkRet = faae.IsThresholdCfgOK(emptyThresholds, embNames, false);
        ASSERT_EQ(checkRet, true);

        // when can not find table name in table list, return false.
        checkRet = faae.IsThresholdCfgOK(thresholds, embNames, false);
        ASSERT_EQ(checkRet, false);

        embNames.emplace_back("tableAAA");

        // check threshold cfg parse ret.
        faae.m_embStatus["tableAAA"] = SingleEmbTableStatus::SETS_ERROR;
        checkRet = faae.IsThresholdCfgOK(thresholds, embNames, false);
        ASSERT_EQ(checkRet, false);
        faae.m_embStatus["tableAAA"] = SingleEmbTableStatus::SETS_BOTH;
        ASSERT_EQ(checkRet, false);
        for (auto i = 0; i < embNames.size(); i++) {
            faae.m_embStatus[embNames[i]] = SingleEmbTableStatus::SETS_NONE;
        }
        checkRet = faae.IsThresholdCfgOK(thresholds, embNames, false);
        ASSERT_EQ(checkRet, true);
    }

    void TestAdmitAndEvictSetThresholdTest()
    {
        faae.ResetAllRecords();
        std::string embName = "";
        int thresholdTmp = 0;
        bool ret = faae.SetTableThresholds(thresholdTmp, embName);

        embName = "tableAAA";
        faae.m_table2Threshold[embName] = {"tableAAA", 2, 5, 1, true};
        ret = faae.SetTableThresholds(thresholdTmp, embName);
        ASSERT_EQ(ret, true);

        thresholdTmp = 1;
        ret = faae.SetTableThresholds(thresholdTmp, embName);
        ASSERT_EQ(ret, true);
    }

    void TestAdmitAndEvictGetAndLoadThresholds()
    {
        faae.ResetAllRecords();
        auto ret = faae.GetTableThresholds();
        ASSERT_EQ(ret.empty(), true);

        ThresholdValue tv = {"tableAAA", 2, 5, 1, true};
        absl::flat_hash_map<std::string, ThresholdValue> thresholdMap;
        thresholdMap.emplace("tableAAA", tv);
        faae.LoadTableThresholds(thresholdMap);
        ret = faae.GetTableThresholds();
        ASSERT_EQ(ret.size(), 1);
        ASSERT_EQ(ret.find("tableAAA") != ret.end(), true);
    }

    void TestAdmitAndEvictHistoryRecords()
    {
        faae.ResetAllRecords();
        auto ret = faae.GetHistoryRecords();
        ASSERT_EQ(ret.historyRecords.empty(), true);

        // build data
        absl::flat_hash_map<int64_t, FeatureItemInfo> keyRecords;
        FeatureItemInfo info = {1, -1};
        keyRecords.emplace(1, info);
        HistoryRecords historyRecords;
        historyRecords.emplace("tableAAA", keyRecords);

        absl::flat_hash_map<std::string, time_t> timestamps;
        timestamps.emplace("tableAAA", -1);
        AdmitAndEvictData histRec;
        histRec.historyRecords = historyRecords;
        histRec.timestamps = timestamps;
        faae.LoadHistoryRecords(histRec);
        auto record = faae.m_recordsData.historyRecords;
        ASSERT_EQ(record.size(), 1);
        ASSERT_EQ(record.find("tableAAA") != record.end(), true);
    }

    void TestAdmitAndEvictSetFunctionSwitch()
    {
        faae.ResetAllRecords();
        bool ret = faae.GetFunctionSwitch();
        ASSERT_EQ(ret, true);
        faae.SetFunctionSwitch(false);
        ret = faae.GetFunctionSwitch();
        ASSERT_EQ(ret, false);
    }

    bool isExitFlag { false };
    HashMapInfo initHistory;
    FeatureAdmitAndEvict faae;
    std::thread evictThr;
    KeysT keys1 = {11, 11, 33, 44, 11, 55, 88, 55};
    vector<uint32_t> cnt1 = {1, 2, 1, 3, 1, 1, 4, 1};
    KeysT keys2 = {11, 12, 33, 21, 11, 12};
    vector<uint32_t> cnt2 = {1, 2, 1, 1, 2, 3};
    KeysT keys3 = {123, 121, 121, 212, 211};
    vector<uint32_t> cnt3 = {1, 2, 1, 1, 2};
    KeysT keys4 = {11, 11, 33, 44, 55, 88, 55};
    vector<uint32_t> cnt4 = {1, 2, 3, 2, 1, 2, 1};
    KeysT keys5 = {125, 121, 122, 212, 211};
    vector<uint32_t> cnt5 = {1, 2, 1, 3, 1};
    std::vector<ThresholdValue> thresholds = {{"tableAAA", 2, 5, 1, true}, {"tableBBB", 3, 7, 1, true},
                                              {"tableCCC", 5, 9, 1, true}};
};

void SetEnv()
{
    const char* name = "useCombineFaae";
    const char* mode = "0";
    int overwrite = 1;

    ASSERT_EQ(setenv(name, mode, overwrite), 0);
}

TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvict1)
{
    SetEnv();
    TestCase1();
}
TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvict2)
{
    TestCase2();
}
TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvict3)
{
    TestCase3();
}
TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvict4)
{
    TestCase4();
}
TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvict5)
{
    SetEnv();
    TestCase5();
}
TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvict6)
{
    TestCase6();
}

TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvictInit)
{
    TestAdmitAndEvictInit();
}

TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvictThresholdCfgCheck)
{
    TestAdmitAndEvictThresholdCfgCheck();
}

TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvictSetThresholdTest)
{
    TestAdmitAndEvictSetThresholdTest();
}

TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvictGetAndLoadThresholds)
{
    TestAdmitAndEvictGetAndLoadThresholds();
}

TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvictHistoryRecords)
{
    TestAdmitAndEvictHistoryRecords();
}

TEST_F(FeatureAdmitAndEvictTest, TestAdmitAndEvictSetFunctionSwitch)
{
    TestAdmitAndEvictSetFunctionSwitch();
}

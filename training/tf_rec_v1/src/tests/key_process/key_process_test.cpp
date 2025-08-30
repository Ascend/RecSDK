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
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "utils/common.h"
#include "key_process/key_process.h"
#include "ock_ctr_common/include/unique.h"
#include "ock_ctr_common/include/error_code.h"
#include "emb_table/embedding_mgmt.h"
#include "emock/emock.hpp"

using namespace std;
using namespace MxRec;
using namespace testing;

static constexpr size_t BATCH_NUM_EACH_THREAD = 3;
const string EMB_TABLE_0 = "emb0";

class SimpleThreadPool {
public:
    static void SyncRun(const std::vector<std::function<void()>> &tasks)
    {
        std::vector<std::future<void>> futs;
        for (auto &task : tasks) {
            futs.push_back(std::async(task));
        }
        for (auto &fut : futs) {
            fut.wait();
        }
    }
};

class KeyProcessTest : public testing::Test {
protected:
    void SetUp()
    {
        int defaultUBSize = 196608;
        EMOCK(GetUBSize).stubs().with(emock::any()).will(returnValue(defaultUBSize));

        int claimed;
        MPI_Query_thread(&claimed);
        ASSERT_EQ(claimed, MPI_THREAD_MULTIPLE);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        LOG_INFO(KEY_PROCESS "wordRank: {}, worldSize: {}", worldRank, worldSize);
        // 初始化rank信息
        rankInfo.rankId = worldRank;
        rankInfo.rankSize = worldSize;
        rankInfo.localRankSize = worldSize;
        rankInfo.useStatic = useStatic;
        rankInfo.localRankId = rankInfo.rankId % rankInfo.localRankSize;
        rankInfo.deviceId = rankInfo.localRankId;
        rankInfo.isDDR = false;
        rankInfo.useDynamicExpansion = false;
        rankInfo.ctrlSteps = { 1, -1 };
        // 初始化emb信息
        GenEmbInfos(embNum, embInfos, fieldNums);
        splits = fieldNums;
        BuildExpect();

        Logger::SetLevel(Logger::DEBUG);
    }

    // 使用该方法构造的数据需要使用掉，否则会影响其他用例
    vector<vector<EmbBatchT>> PrepareBatch()
    {
        vector<vector<EmbBatchT>> result(KEY_PROCESS_THREAD * MAX_CHANNEL_NUM);
        // 向共享队列中写入本进程所有线程要处理的 KEY_PROCESS_THREAD * BATCH_NUM_EACH_THREAD 个batch数据
        for (size_t threadId = 0; threadId < KEY_PROCESS_THREAD; ++threadId) {
            int batchQueueId = threadId + KEY_PROCESS_THREAD * channel;
            unsigned int seed = batchQueueId * 10;
            auto queue = SingletonQueue<EmbBatchT>::GetInstances(batchQueueId);

            for (size_t batchNum = 0; batchNum < BATCH_NUM_EACH_THREAD; ++batchNum) {
                size_t batchId =
                        batchNum * KEY_PROCESS_THREAD + threadId;

                for (size_t i = 0; i < embInfos.size(); i++) { // key按照不同emb表的存储切分开
                    auto batch = queue->GetOne();
                    batch->sample.resize(batchSize * fieldNums[i]);
                    GenData(batch->sample, 0, seed++);
                    batch->name = embInfos[i].name;
                    batch->batchId = batchId;
                    batch->channel = channel;
                    LOG_DEBUG("[{}/{}]" KEY_PROCESS "PrepareBatch: batchQueueId: {}, {}[{}]{}, sampleSize:{}",
                              worldRank, worldSize,
                              batchQueueId, batch->name, batch->channel, batch->batchId, batch->sample.size()
                    );
                    EmbBatchT temp;
                    temp.sample = batch->sample;
                    temp.name = batch->name;
                    temp.batchId = batch->batchId;
                    temp.channel = batch->channel;
                    result[batchQueueId].push_back(temp);
                    queue->Pushv(std::move(batch));
                }
            }
        }
        return result;
    }

    // 生成随机数
    template<class T>
    void GenData(vector<T>& totBatchData, int start, unsigned int seed = 0)
    {
        default_random_engine generator { seed };
        uniform_int_distribution<T> distribution(start, randMax);
        for (size_t i = 0; i < totBatchData.size(); ++i) {
            totBatchData[i] = distribution(generator);
        }
    }

    // 生成emb表信息
    bool GenEmbInfos(size_t embNums, vector<EmbInfo>& allEmbInfos, vector<int>& geFieldNums)
    {
        default_random_engine generator;
        int embSizeMin = 5;
        int embSizeMax = 8;
        int base = 2;
        int vocabSize = 100;
        uniform_int_distribution<int> embSizeDistribution(embSizeMin, embSizeMax);
        stringstream ss;
        for (unsigned int i = 0; i < embNums; ++i) {
            EmbInfo temp;
            ss << i;
            temp.name = "emb" + ss.str();
            ss.str("");
            ss.clear();
            temp.sendCount = sendCount; // 10~25
            temp.extEmbeddingSize = pow(base, embSizeDistribution(generator)); // 2^5~2^8
            temp.devVocabSize = vocabSize;
            geFieldNums.push_back(sampleSize);
            allEmbInfos.push_back(move(temp));
            LOG_INFO("GenEmbInfos, emb Name: {}, sendCount:{}, extEmbeddingSize: {}, devVocabSize: {}",
                     temp.name, temp.sendCount, temp.extEmbeddingSize, temp.devVocabSize = vocabSize);
        }
        return true;
    }

    auto GetSplitAndRestore(KeysT& sample) -> tuple<vector<KeysT>, vector<int32_t>>
    {
        vector<KeysT> expectSplitKeys(worldSize);
        vector<int> expectRestore(sample.size());
        absl::flat_hash_map<emb_key_t, int> uKey;
        for (unsigned int i = 0; i < sample.size(); ++i) {
            int devId = sample[i] % worldSize;
            auto result = uKey.find(sample[i]);
            if (result == uKey.end()) {
                expectSplitKeys[devId].push_back(sample[i]);
                uKey.insert(make_pair(sample[i], expectSplitKeys[devId].size() - 1));
                expectRestore[i] = expectSplitKeys[devId].size() - 1;
            } else {
                expectRestore[i] = result->second;
            }
        }
        return { expectSplitKeys, expectRestore };
    }

    void PrintHotHashSplit(const vector<KeysT>& splitKeys,
                           const vector<int32_t>& restore,
                           const vector<int32_t>& hotPos, int rankSize)
    {
        for (int i = 0; i < rankSize; ++i) {
            std::cout << "splitKeys dev" << i << std::endl;
            LOG_INFO(VectorToString(splitKeys[i]));
        }
        std::cout << "restore" << std::endl;
        LOG_INFO(VectorToString(restore));
        std::cout << "hotPos" << std::endl;
        LOG_INFO(VectorToString(hotPos));
    }

    void GetExpectRestore(KeysT& sample, vector<int>& blockOffset, vector<int>& restoreVec)
    {
        for (unsigned int i = 0; i < sample.size(); ++i) {
            int devId = sample[i] % worldSize;
            restoreVec[i] += blockOffset[devId];
        }
    }

    unique_ptr<EmbBatchT> GenBatch(string batchName, int batchId, int channelId) // 用于端到端test
    {
        unique_ptr<EmbBatchT> batch = std::make_unique<EmbBatchT>();
        vector<KeysT> allBatchKeys = { { 11, 11, 6, 16, 14, 8, 6, 5, 8, 6, 14, 11, 4, 12, 1, 13 },
                                       { 8, 6, 2, 4, 3, 8, 13, 2, 1, 4, 2, 2, 11, 8, 14, 5 },
                                       { 16, 3, 2, 12, 4, 12, 12, 2, 6, 4, 1, 5, 9, 3, 5, 14 },
                                       {  2, 8, 2, 12, 1, 14, 9, 8, 14, 16, 11, 15, 1, 7, 5, 2 } };
        batch->sample = std::move(allBatchKeys[worldRank]);
        batch->name = batchName;
        batch->batchId = batchId;
        batch->channel = channelId;
        LOG_INFO(KEY_PROCESS "test GenExpect: rank {}, batchKeys {}",
                 worldRank, VectorToString(batch->sample));

        return batch;
    }
    void BuildExpect()
    {
        allExpectSs = { { 0, 4, 7, 9 }, { 0, 2, 5, 8 }, { 0, 3, 6, 9 }, { 0, 3, 6, 8 } };
        allExpectRestore = { { 9, 9, 7, 0, 8, 1, 7, 4, 1, 7, 8, 9, 2, 3, 5, 6 },
                             { 0, 5, 6, 1, 8, 0, 2, 6, 3, 1, 6, 6, 9, 0, 7, 4 },
                             { 0, 9, 6, 1, 2, 1, 1, 6, 7, 2, 3, 4, 5, 9, 4, 8 },
                             { 6, 0, 6, 1, 3, 7, 4, 0, 7, 2, 8, 9, 3, 10, 5, 6 } };
        // sendCount = 10， 按照10padding的结果
        allExpectRestoreStatic = { { 30, 30, 20, 0, 21, 1, 20, 10, 1, 20, 21, 30, 2, 3, 11, 12 },
                                   { 0, 20, 21, 1, 30, 0, 10, 21, 11, 1, 21, 21, 31, 0, 22, 12 },
                                   { 0, 30, 20, 1, 2, 1, 1, 20, 21, 2, 10, 11, 12, 30, 11, 22 },
                                   { 20, 0, 20, 1, 10, 21, 11, 0, 21, 2, 30, 31, 10, 32, 12, 20 } };
        allExpectLookupKeys = { { 16, 8, 4, 12, 8, 4, 16, 12, 4, 8, 12, 16 },
                                { 5, 1, 13, 13, 1, 5, 1, 5, 9, 1, 9, 5 },
                                { 6, 14, 6, 2, 14, 2, 6, 14, 2, 14 },
                                { 11, 3, 11, 3, 11, 15, 7 } };
        allExpectOffset = { { 0, 1, 2, 3, 1, 2, 0, 3, 2, 1, 3, 0 },
                            { 0, 1, 2, 2, 1, 0, 1, 0, 3, 1, 3, 0 },
                            { 0, 1, 0, 2, 1, 2, 0, 1, 2, 1  },
                            { 0, 1, 0, 1, 0, 2 } };
        allExpectCount = { { 1, 2, 1, 1, 3, 2, 1, 3, 2, 2, 1, 1 },
                           { 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1 },
                           { 3, 2, 1, 4, 1, 2, 1, 1, 3, 2  },
                           { 3, 1, 1, 2, 1, 1, 1 } };
        allExpectAll2all = { { 4, 2, 3, 3 }, { 3, 3, 3, 3 }, { 2, 3, 3, 2 }, { 1, 2, 1, 3 } };
    }

    bool CheckMatrixTensor(const vector<Tensor>& actual, const vector<vector<int64_t>>& expect) // 主要用于all2all的校验
    {
        int row = expect.size();
        int col = expect[0].size();
        auto tmpTensor = actual.at(0);
        auto tmpData = tmpTensor.matrix<int64>();
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                if (!(tmpData(i, j) == expect[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool CheckFlatTensor(const vector<Tensor>& actual, const vector<int64_t>& expect) // 主要用于lookup和restore的校验
    {
        int num = expect.size();
        auto tmpTensor = actual.at(0);
        auto tmpData = tmpTensor.flat<int32>();
        for (int i = 0; i < num; ++i) {
            if (!(tmpData(i) == expect[i])) {
                return false;
            }
        }
        return true;
    }

    bool CheckPaddingVec(const vector<int64_t>& actual, const vector<int64_t>& expect) // 主要用于lookup静态下padding校验
    {
        for (int i = 0, j = 0; i < actual.size(); i++) {
            if (actual[i] == -1) {
                continue;
            }
            if (!(actual[i] == expect[j])) {
                return false;
            }
            j++;
        }
        return true;
    }

    EmbBaseInfo GetEmbBaseInfo()
    {
        EmbBaseInfo embBaseInfo;
        embBaseInfo.batchId = 0;
        embBaseInfo.channelId = 0;
        embBaseInfo.name = EMB_TABLE_0;
        return embBaseInfo;
    }

    RankInfo rankInfo;
    vector<EmbInfo> embInfos;
    int worldRank {};
    int worldSize {};
    vector<int> splits;
    int sampleSize = 20; // dim维度
    int channel = 0;
    int sendCount = 10;
    int randMax = 25; // GenData生成数据0~max范围

    int batchSize = 5;
    bool useStatic = true;

    // vector<EmbInfo> embInfos
    int embNum = 1;
    vector<int> fieldNums; // 多个表的dim维度

    vector<KeysT> splitKeys;
    vector<int32_t> restore;
    vector<vector<uint32_t>> keyCount;
    KeyProcess process;
    vector<vector<int64_t>> allExpectSs;
    vector<vector<int64_t>> allExpectAll2all;
    vector<vector<int64_t>> allExpectRestore;
    vector<vector<int64_t>> allExpectRestoreStatic;
    vector<vector<emb_key_t>> allExpectLookupKeys;
    vector<vector<emb_key_t>> allExpectOffset;
    vector<vector<int64_t>> allExpectCount;
    int originalLogLevel = Logger::GetLevel();

    void TearDown()
    {
        Logger::SetLevel(originalLogLevel);
        // delete
        GlobalMockObject::reset();
    }
};

TEST_F(KeyProcessTest, Initialize)
{
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    ASSERT_EQ(process.rankInfo.rankId, rankInfo.rankId);
    ASSERT_EQ(process.rankInfo.rankSize, rankInfo.rankSize);
    ASSERT_EQ(process.rankInfo.localRankSize, rankInfo.localRankSize);
    ASSERT_EQ(process.rankInfo.useStatic, rankInfo.useStatic);
    ASSERT_EQ(process.rankInfo.localRankId, rankInfo.localRankId);
    ASSERT_EQ(process.embInfos.size(), embInfos.size());
    for (const EmbInfo& info: embInfos) {
        ASSERT_NE(process.embInfos.find(info.name), process.embInfos.end());
    }

    ock::ctr::Factory::Create(factory);
}

TEST_F(KeyProcessTest, AttributeGetAndSetTest)
{
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    auto keyCountMap = process.GetKeyCountMap();
    ASSERT_EQ(keyCountMap.empty(), true);

    absl::flat_hash_map<emb_key_t, size_t> keyCountMapTmp;
    keyCountMapTmp.emplace(1, 1);
    KeyCountMemT keyCountMemT;
    keyCountMemT.emplace("table", keyCountMapTmp);
    process.LoadKeyCountMap(keyCountMemT);
    keyCountMap = process.GetKeyCountMap();
    ASSERT_EQ(keyCountMap.size(), 1);

    // test LoadMaxOffset method
    OffsetMemT offsetMemT;
    offsetMemT.emplace("table", 1);
    process.LoadMaxOffset(offsetMemT);
    auto& maxOffsetTmp = process.maxOffset;
    ASSERT_EQ(maxOffsetTmp.find("table") != maxOffsetTmp.end(), true);

    // test LoadKeyOffsetMap method
    KeyOffsetMemT keyOffsetMemT;
    absl::flat_hash_map<emb_key_t, int64_t> koMap;
    koMap.emplace(1, 1);
    keyOffsetMemT.emplace("table", koMap);
    process.LoadKeyOffsetMap(keyOffsetMemT);
    auto& keyOffsetMap = process.keyOffsetMap;
    ASSERT_EQ(keyOffsetMap.find("table") != keyOffsetMap.end(), true);

    process.Destroy();
}

TEST_F(KeyProcessTest, KeyProcessTaskWithFastUniqueTest)
{
    LOG_INFO("start KeyProcessTaskWithFastUniqueTest");
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);

    // only test invoke process, detail methods has been tested by other test function.
    auto fn = [this](int channel, int threadId) {
        process.KeyProcessTaskWithFastUnique(channel, threadId);
        LOG_INFO("rankid :{},threadId: {}", rankInfo.rankId, threadId);
    };
    for (int channel = 0; channel < 1; ++channel) {
        for (int id = 0; id < 1; ++id) {
            process.procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }

    this_thread::sleep_for(10s);
    process.Destroy();
    LOG_INFO("end KeyProcessTaskWithFastUniqueTest");
}

TEST_F(KeyProcessTest, HashSplitHelperTest)
{
    int rankSize = 4;
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();
    KeysT batchKeys = { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 };
    vector<int> expectRestore = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 2 };
    vector<vector<long int>> expectSplitKeys = { { 4, 16 }, { 1, 21, 29 }, { 14, 2 }, { 23, 7 } };
    batch->sample = std::move(batchKeys);
    LOG_INFO(KEY_PROCESS "test batch sample: {}", VectorToString(batch->sample));
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    process.rankInfo.rankSize = rankSize;

    vector<KeysT> splitKeysTmp;
    vector<int32_t> restoreTmp;
    vector<int32_t> hotPosTmp;
    vector<vector<uint32_t>> keyCountTmp;
    process.HashSplitHelper(batch, splitKeysTmp, restoreTmp, hotPosTmp, keyCountTmp);
    LOG_INFO("restoreTmp:{}, hotPosTmp:{}, ",
             VectorToString(restoreTmp), VectorToString(hotPosTmp));
    ASSERT_EQ(restoreTmp.empty(), false);
    EXPECT_EQ(restoreTmp, expectRestore);
    for (ssize_t i = 0; i < splitKeysTmp.size(); i++) {
        LOG_INFO("sub splitKeysTmp:{} ", VectorToString(splitKeysTmp[i]));
        EXPECT_EQ(splitKeysTmp[i], expectSplitKeys[i]);
    }
}

TEST_F(KeyProcessTest, KeyProcessTaskHelperForDpTest)
{
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);

    LOG_INFO("embInfos[0].name:{}", embInfos[0].name);
    int rankSize = 4;
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();
    KeysT batchKeys = { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 };
    vector<int> expectRestore = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 2 };
    vector<vector<int>> expectSplitKeys = { { 4, 16 }, { 1, 21, 29 }, { 14, 2 }, { 23, 7 } };
    batch->sample = std::move(batchKeys);
    batch->name = EMB_TABLE_0;

    // Test EOS for device memery mode.
    batch->isEos = true;
    bool ret = process.KeyProcessTaskHelperForDp(batch, 0, 0);
    ASSERT_EQ(ret, true);
    LOG_INFO("infoList.size():{}", process.infoList.size());
    ASSERT_EQ(process.infoList.size() == 1, true);
    ProcessedInfo typeTmp = ProcessedInfo::RESTORE;
    EmbBaseInfo embBaseInfo = GetEmbBaseInfo();
    bool isEos = false;
    process.GetInfoVec(embBaseInfo, typeTmp, isEos);
    ASSERT_EQ(isEos, true);

    // test EOS for ddr
    isEos = false;
    process.rankInfo.isDDR = true;
    ret = process.KeyProcessTaskHelperForDp(batch, 0, 0);
    ASSERT_EQ(ret, true);
    LOG_INFO("uniqueKeysList.size():{}", process.uniqueKeysList.size());
    ASSERT_EQ(process.uniqueKeysList.size() == 1, true);
    process.GetUniqueKeys(embBaseInfo, isEos);
    ASSERT_EQ(isEos, true);
    process.Destroy();
}

TEST_F(KeyProcessTest, TestKeyProcessTaskHelperForDp)
{
    auto& embInfoTmp = embInfos[0];
    embInfoTmp.isDp = true;
    embInfoTmp.embeddingSize = 16;

    GlobalEnv::recordKeyCount = true;

    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);

    // build batch
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();
    KeysT batchKeys = {1, 4, 23, 14, 16, 7, 2, 21, 21, 29};
    vector<uint64_t> expectUniqueKeys = {1, 4, 23, 14, 16, 7, 2, 21, 29};
    batch->sample = std::move(batchKeys);
    batch->name = EMB_TABLE_0;
    batch->batchId = 0;
    batch->channel = 0;
    FeatureAdmitAndEvict::m_embStatus[EMB_TABLE_0] = SingleEmbTableStatus::SETS_NONE;
    process.hotEmbTotCount[EMB_TABLE_0] = 10;
    LOG_INFO("test batch sample: {}", VectorToString(batch->sample));
    auto ret = process.KeyProcessTaskHelperForDp(batch, 0, 0);
    ASSERT_EQ(ret, true);

    // check keyCountMap
    auto kcMap = process.keyCountMap[EMB_TABLE_0];
    ASSERT_EQ(kcMap[4], 1);
    ASSERT_EQ(kcMap[23], 1);
    ASSERT_EQ(kcMap[21], 2);
    ASSERT_EQ(kcMap.find(55) == kcMap.end(), true);

    // Device memery mode
    ProcessedInfo typeTmp = ProcessedInfo::RESTORE;
    EmbBaseInfo embBaseInfo = GetEmbBaseInfo();
    bool isEos = false;
    unique_ptr<vector<Tensor>> storage = process.GetInfoVec(embBaseInfo, typeTmp, isEos);
    ASSERT_EQ(storage != nullptr, true);
    auto detailData = storage->back().flat<tensorflow::int32>();
    for (int i = 0; i < detailData.size(); i++) {
        ASSERT_EQ(detailData(i), i);
    }

    // DDR: check list[info.name][info.channelId].top()
    process.rankInfo.isDDR = true;
    ret = process.KeyProcessTaskHelperForDp(batch, 0, 0);
    ASSERT_EQ(ret, true);
    int batchIdTmp;
    std::string tableNameTmp;
    bool isEosTmp;
    vector<uint64_t> uniqueKeysTmp1;
    tie(batchIdTmp, tableNameTmp, isEosTmp, uniqueKeysTmp1) = process.uniqueKeysList[EMB_TABLE_0][0].top();
    vector<uint64_t> uniqueKeysTmp2 = process.GetUniqueKeys(embBaseInfo, isEos);
    LOG_INFO("uniqueKeysTmp is:{}", VectorToString(uniqueKeysTmp2));
    process.SendEosTensor(EMB_TABLE_0, 0);
    ASSERT_EQ(uniqueKeysTmp1, uniqueKeysTmp2);
    ASSERT_EQ(uniqueKeysTmp2, expectUniqueKeys);

    GlobalEnv::recordKeyCount = false;
    process.Destroy();
}

TEST_F(KeyProcessTest, TestStart4KeyProcessTaskHelperForDp)
{
    auto& embInfoTmp = embInfos[0];
    embInfoTmp.isDp = true;
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    setenv("keyProcessThreadNum", "2", 1);
    ASSERT_EQ(process.Start(), 0);
    setenv("keyProcessThreadNum", "abc", 1);
    ASSERT_EQ(process.Start(), 0);
    LOG_INFO("key process start successful");
    process.Destroy();
}

TEST_F(KeyProcessTest, TestGetFeatAdmitAndEvict)
{
    vector<ThresholdValue> thresholdValues;
    ThresholdValue tv = ThresholdValue(EMB_TABLE_0, 2, 1, 1, false);
    thresholdValues.emplace_back(tv);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos, thresholdValues), true);
    auto threadValueTmp = process.GetFeatAdmitAndEvict().m_table2Threshold[EMB_TABLE_0];
    ASSERT_EQ(threadValueTmp.tableName, EMB_TABLE_0);
    ASSERT_EQ(threadValueTmp.countThreshold, 2);
    ASSERT_EQ(threadValueTmp.timeThreshold, 1);
    process.Destroy();
}

TEST_F(KeyProcessTest, TestFeatureAdmitForDp)
{
    std::vector<emb_key_t> lookupKeys = {2, 3, 12, 22, 21, 24, 5, 31};
    std::vector<emb_key_t> globalDpIdVec = {3, 12, 22, 21, 3, 12};
    // deduplicated for globalDpIdVec as globalUniqueIdVec, and
    // update lookupKeys element to INVALID_KEY_VALUE which not in globalUniqueIdVec.
    std::vector<emb_key_t> expectRet = {3, 12, 22, 21};
    auto globalUniqueIdVec = process.FeatureAdmitForDp(lookupKeys, globalDpIdVec);
    ASSERT_EQ(globalUniqueIdVec, expectRet);
    ASSERT_EQ(lookupKeys[0], -1);
    ASSERT_EQ(lookupKeys[1], 3);
}

TEST_F(KeyProcessTest, TestGetCountRecvForDp)
{
    // every communicator send 1 element.
    embInfos[0].sendCount = 1;
    process.Initialize(rankInfo, embInfos);

    auto fn = [this](int threadId) {
        // build batch
        auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
        auto batch = queue->GetOne();
        batch->name = EMB_TABLE_0;
        batch->batchId = 0;
        batch->channel = 0;
        // init send data vector element by rankId.
        vector<uint32_t> sendData(1, rankInfo.rankId);
        vector<int> sendCountAll = {1, 1, 1, 1};
        LOG_INFO("in test, rankInfo.rankId:{}", rankInfo.rankId);
        vector<uint32_t> countRecv = process.GetCountRecvForDp(batch, threadId, sendData, sendCountAll);
        vector<uint32_t> expectRet = {0, 1, 2, 3};
        ASSERT_EQ(countRecv, expectRet);
    };
    for (int channel = 0; channel < 1; ++channel) {
        for (int id = 0; id < KEY_PROCESS_THREAD; ++id) {
            process.procThreads.emplace_back(std::make_unique<std::thread>(fn, id));
        }
    }
    process.Destroy();
}

TEST_F(KeyProcessTest, KeyProcessTaskHelperTest)
{
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    int rankSize = 4;
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();

    KeysT batchKeys = {1, 4, 23, 14, 16, 7, 2, 21, 21, 29 };
    vector<vector<int>> expectInfoVec = {{0, 1}, {0, 1, 2}, {0, 1}, {0, 1}};
    batch->sample = std::move(batchKeys);
    batch->name = EMB_TABLE_0;
    batch->channel = 0;
    batch->isEos = false;

    // test data
    process.hotEmbTotCount[EMB_TABLE_0] = 10;
    auto ret = process.KeyProcessTaskHelper(batch, 0, 0);
    ASSERT_EQ(ret, true);
    ProcessedInfo typeTmp = ProcessedInfo::RESTORE;
    EmbBaseInfo embBaseInfo = GetEmbBaseInfo();
    bool isEos = false;
    unique_ptr<vector<Tensor>> storage = process.GetInfoVec(embBaseInfo, typeTmp, isEos);
    ASSERT_EQ(storage != nullptr, true);
    auto detailData = storage->back().flat<tensorflow::int32>();
    for (int i = 0; i < expectInfoVec[rankInfo.rankId].size(); i++) {
        ASSERT_EQ(detailData(i), expectInfoVec[rankInfo.rankId][i]);
    }

    // Test EOS for device memery mode.
    batch->isEos = true;
    ret = process.KeyProcessTaskHelper(batch, 0, 0);
    ASSERT_EQ(ret, true);
    LOG_INFO("infoList.size():{}", process.infoList.size());
    ASSERT_EQ(process.infoList.size() == 1, true);
    process.GetInfoVec(embBaseInfo, typeTmp, isEos);
    ASSERT_EQ(isEos, true);

    // test EOS for ddr
    isEos = false;
    process.rankInfo.isDDR = true;
    ASSERT_EQ(process.KeyProcessTaskHelper(batch, 0, 0), true);
    LOG_INFO("uniqueKeysList.size():{}", process.uniqueKeysList.size());
    ASSERT_EQ(process.uniqueKeysList.size() == 1, true);
    process.GetUniqueKeys(embBaseInfo, isEos);
    ASSERT_EQ(isEos, true);

    // check incrementally save info
    process.rankInfo.isDDR = true;
    process.rankInfo.useStatic = false;
    process.isIncrementalCheckpoint = true;
    batch->isEos = false;
    process.KeyProcessTaskHelper(batch, 0, 0);
    unique_ptr<vector<Tensor>> kcInfoVec = process.GetKCInfoVec(embBaseInfo);
    LOG_INFO("keyCountInfoList.size():{}", process.keyCountInfoList.size());
    ASSERT_EQ(process.keyCountInfoList.size() == 1, true);
    ASSERT_EQ(kcInfoVec != nullptr, true);
    auto kcDetailData = kcInfoVec->back().flat<tensorflow::int64>();
    vector<vector<int>> expectKcVec = {{4, 4, 16, 4}, {1, 4, 21, 8, 29, 4}, {2, 4, 14, 4}, {7, 4, 23, 4}};
    for (int i = 0; i < expectKcVec.size(); i++) {
        ASSERT_EQ(kcDetailData(i), expectKcVec[rankInfo.rankId][i]);
    }

    vector<int32_t> restoreVecSec = process.GetRestoreVecSec(embBaseInfo);
    vector<vector<int32_t>> expectRestoreVecSec = {
        {0, 1, 0, 1, 0, 1, 0, 1}, {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}, {0, 1, 0, 1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0, 1, 0, 1}
    };
    for (int i = 0; i < restoreVecSec.size(); i++) {
        ASSERT_EQ(restoreVecSec[i], expectRestoreVecSec[rankInfo.rankId][i]);
    }
    process.Destroy();
}

TEST_F(KeyProcessTest, GetMaxStepTest)
{
    vector<int> ctrlStep = {200, 100, 200, 400};
    rankInfo.ctrlSteps = ctrlStep;
    process.Initialize(rankInfo, embInfos);
    int step = process.GetMaxStep(0);
    for (size_t i = 0; i < ctrlStep.size(); ++i) {
        ASSERT_EQ(process.GetMaxStep(i), ctrlStep[i]);
    }
    process.Destroy();
}

TEST_F(KeyProcessTest, EnqueueEosBatchTest)
{
    process.Initialize(rankInfo, embInfos);
    int batchId = 0;
    int channelId = 0;
    process.EnqueueEosBatch(batchId, channelId);
    int threadNum = GetThreadNumEnv();
    int batchQueueId = int(batchId % threadNum) + (MAX_KEY_PROCESS_THREAD * channelId);
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(batchQueueId);
    auto data = queue->TryPop();
    ASSERT_EQ(data != nullptr, true);
    vector<int64_t> expectRet = {0, 0, 0, 0, 0, 0, 0, 0};
    ASSERT_EQ(data->sample, expectRet);
    process.Destroy();
}

TEST_F(KeyProcessTest, EnqueueEosBatch_ThrowWhenThreadNumIsZero)
{
    int threadNum = GetThreadNumEnv();
    EMOCK(GetThreadNumEnv).stubs().will(returnValue(0));

    // 预期抛出 std::runtime_error
    EXPECT_THROW({
        process.EnqueueEosBatch(100, 1);
    }, std::runtime_error);

    EMOCK(GetThreadNumEnv).stubs().will(returnValue(threadNum));
}

TEST_F(KeyProcessTest, DumpSplitKeysTest)
{
    vector<vector<emb_key_t>> keys = {{1, 4}, {23}, {14, 16, 7}, {2, 21}};
    process.Initialize(rankInfo, embInfos);
    auto ret = process.DumpSplitKeys(keys);
    string expectRet = "|0:1,4,||1:23,||2:14,16,7,||3:2,21,|";
    ASSERT_EQ(ret, expectRet);
    process.Destroy();
}

TEST_F(KeyProcessTest, Start)
{
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    setenv("keyProcessThreadNum", "2", 1);
    ASSERT_EQ(process.Start(), 0);
    setenv("keyProcessThreadNum", "abc", 1);
    ASSERT_EQ(process.Start(), 0);
    LOG_INFO("key process start successful");
    process.Destroy();
}

TEST_F(KeyProcessTest, HashSplit)
{
    int rankSize = 4;
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();
    KeysT batchKeys = { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 };
    vector<int> expectRestore = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 2 };
    vector<vector<int>> expectSplitKeys = { { 4, 16 }, { 1, 21, 29 }, { 14, 2 }, { 23, 7 } };
    batch->sample = std::move(batchKeys);
    LOG_DEBUG(KEY_PROCESS "batch sample: {}", VectorToString(batch->sample));
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    process.rankInfo.rankSize = rankSize;
    auto [splitKeys, restore] = process.HashSplit(batch);
    for (unsigned int i = 0; i < splitKeys.size(); ++i) {
        ASSERT_THAT(splitKeys[i], ElementsAreArray(expectSplitKeys[i]));
    }
    ASSERT_THAT(restore, ElementsAreArray(expectRestore));
}

TEST_F(KeyProcessTest, HashSplitWithFAAE)
{
    int rankSize = 4;
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();
    KeysT batchKeys = { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 };
    vector<int> expectRestore = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 2 };
    vector<vector<int>> expectSplitKeys = { { 4, 16 }, { 1, 21, 29 }, { 14, 2 }, { 23, 7 } };
    vector <vector<uint32_t>> expectCount = {{1, 1}, {1, 2, 1}, {1, 1}, {1, 1}};
    batch->sample = std::move(batchKeys);
    LOG_DEBUG(KEY_PROCESS "batch sample: {}", VectorToString(batch->sample));
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    process.rankInfo.rankSize = rankSize;
    auto [splitKeys, restore, keyCount] = process.HashSplitWithFAAE(batch, false);
    LOG_INFO(KEY_PROCESS "HashSplitWithFAAE， batch splitKeys: {}, keyCount: {}", VectorToString(splitKeys[0]),
             VectorToString(keyCount[0]));

    for (unsigned int i = 0; i < splitKeys.size(); ++i) {
        ASSERT_THAT(splitKeys[i], ElementsAreArray(expectSplitKeys[i]));
        ASSERT_THAT(keyCount[i], ElementsAreArray(expectCount[i]));
    }
    ASSERT_THAT(restore, ElementsAreArray(expectRestore));
}

// 准入+动态shape下，有padding
TEST_F(KeyProcessTest, PaddingHashSplitWithFAAE)
{
    int rankSize = 4;
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();
    KeysT batchKeys = { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 };
    vector<int> expectRestore = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 2 };
    vector<vector<int>> expectSplitKeys = { { 4, 16 }, { 1, 21, 29 }, { 14, 2 }, { 23, 7 } };
    vector <vector<uint32_t>> expectCount = {{1, 1}, {1, 2, 1}, {1, 1}, {1, 1}};
    batch->sample = std::move(batchKeys);
    LOG_DEBUG(KEY_PROCESS "batch sample: {}", VectorToString(batch->sample));

    rankInfo.useStatic = false;
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    process.rankInfo.rankSize = rankSize;
    auto [splitKeys, restore, keyCount] = process.HashSplitWithFAAE(batch, false);
    LOG_INFO(KEY_PROCESS "HashSplitWithFAAE Padding， batch splitKeys: {}, keyCount: {}", VectorToString(splitKeys[0]),
             VectorToString(keyCount[0]));

    for (unsigned int i = 0; i < splitKeys.size(); ++i) {
        ASSERT_EQ(splitKeys[i].size(), ALLTOALLVC_ALIGN);
        ASSERT_EQ(keyCount[i].size(), ALLTOALLVC_ALIGN);
    }
}

TEST_F(KeyProcessTest, GetScAll)
{
    vector<int> keyScLocal(worldSize, worldRank + 1); // 用worldRank+1初始化发送数据量
    LOG_DEBUG(KEY_PROCESS "rank {} keyScLocal: {}", worldRank, VectorToString(keyScLocal));
    vector<int> expectScAll(worldSize * worldSize);
    for (unsigned int i = 0; i < expectScAll.size(); ++i) {
        expectScAll[i] = floor(i / worldSize) + 1;
    }
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    // 仅用于集合通信获取sendCount信息，构造EmbBatchT对象即可，通道传0，不用构造batch数据
    EmbBatchT tempBatch;
    tempBatch.channel = 0;
    unique_ptr<EmbBatchT> batch = std::make_unique<EmbBatchT>(tempBatch);
    vector<int> scAll = process.GetScAll(keyScLocal, 0, batch);
    ASSERT_THAT(scAll, ElementsAreArray(expectScAll));
}

TEST_F(KeyProcessTest, GetScAllForUnique)
{
    vector<int> keyScLocal(worldSize, worldRank + 1); // 用worldRank+1初始化发送数据量
    LOG_INFO(KEY_PROCESS "rank {} keyScLocal: {}", worldRank, VectorToString(keyScLocal));
    vector<int> expectScAll(worldSize * worldSize);
    for (unsigned int i = 0; i < expectScAll.size(); ++i) {
        expectScAll[i] = floor(i / worldSize) + 1;
    }
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    vector<int> scAll;
    // 仅用于集合通信获取sendCount信息，构造EmbBatchT对象即可，通道传0，不用构造batch数据
    EmbBatchT tempBatch;
    tempBatch.channel = 0;
    unique_ptr<EmbBatchT> batch = std::make_unique<EmbBatchT>(tempBatch);
    process.GetScAllForUnique(keyScLocal, 0, batch, scAll);
    LOG_INFO("scAll:{}", VectorToString(scAll));
    ASSERT_THAT(scAll, ElementsAreArray(expectScAll));
}

// 非hot、非准入模式，固定batch输入，校验restore
TEST_F(KeyProcessTest, BuildRestoreVec_4cpu)
{
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(0);
    auto batch = queue->GetOne();
    vector<KeysT> allBatchKeys = { { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 },
                                   { 5, 17, 26, 9, 27, 22, 27, 28, 15, 3 },
                                   { 10, 4, 22, 17, 24, 13, 24, 26, 29, 11 },
                                   { 14, 21, 18, 25, 21, 4, 20, 24, 13, 19 } };
    vector<vector<int>> allExpectSs = { { 0, 2, 5, 7, 9 }, { 0, 1, 4, 6 }, { 0, 2, 5, 8 }, { 0, 3, 6, 8 } };
    vector<vector<int>> allExpectRestore = { { 2, 0, 7, 5, 1, 8, 6, 3, 3, 4 },
                                             { 1, 2, 4, 3, 6, 5, 6, 0, 7, 8 },
                                             { 5, 0, 6, 2, 1, 3, 1, 7, 4, 8 },
                                             { 6, 3, 7, 4, 3, 0, 1, 2, 5, 8 } };
    batch->sample = std::move(allBatchKeys[worldRank]);
    LOG_INFO(KEY_PROCESS "test BuildRestoreVec: rank {}, batchKeys {}",
             worldRank, VectorToString(batch->sample));
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    auto [splitKeys, restore] = process.HashSplit(batch);

    LOG_DEBUG("rank: {} splitKeys: {}", worldRank, [&] {
        vector<string> tmp;
        for (const auto& i : splitKeys) {
            tmp.emplace_back(VectorToString(i));
        }
        return VectorToString(tmp);
    }());

    process.BuildRestoreVec(batch, allExpectSs[worldRank], restore);
    ASSERT_THAT(restore, ElementsAreArray(allExpectRestore[worldRank]));
}

// 准入模式，batch随机数，ProcessSplitKeys后人为校验lookupKeys、scAll、count
TEST_F(KeyProcessTest, GetCountRecv)
{
    PrepareBatch();
    process.m_featureAdmitAndEvict.m_isEnableFunction = true;
    for (size_t i = 0; i < embInfos.size(); i++) {
        FeatureAdmitAndEvict::m_embStatus[embInfos[i].name] = SingleEmbTableStatus::SETS_BOTH;
    }

    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    LOG_INFO("CPU Core Num: {}", sysconf(_SC_NPROCESSORS_CONF)); // 查看CPU核数

    auto fn = [this](int channel, int id) {
        auto embName = embInfos[0].name;
        vector<KeysT> splitKeys;
        vector<int32_t> restore;
        vector<vector<uint32_t>> count;
        unique_ptr<EmbBatchT> batch;
        batch = process.GetBatchData(channel, id); // get batch data from SingletonQueue<EmbBatchT>
        LOG_INFO("rankid :{}, batchid: {}", rankInfo.rankId, batch->batchId);
        tie(splitKeys, restore, count) = process.HashSplitWithFAAE(batch, false);
        auto [lookupKeys, scAll, ss] = process.ProcessSplitKeys(batch, id, splitKeys);
        vector<uint32_t> countRecv = process.GetCountRecv(batch, id, count, scAll, ss);

        LOG_INFO("rankid :{}, batchid: {}, lookupKeys: {}, scAll: {}, count after build {}",
                 rankInfo.rankId, batch->batchId, VectorToString(lookupKeys),
                 VectorToString(scAll), VectorToString(countRecv));
    }; // for clean code
    for (int channel = 0; channel < 1; ++channel) {
        for (int id = 0; id < KEY_PROCESS_THREAD; ++id) {
        // use lambda expression initialize thread
            process.procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }
    this_thread::sleep_for(10s);
    process.Destroy();
}

TEST_F(KeyProcessTest, Key2Offset)
{
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos);
    KeysT lookupKeys = { 4, 16, 28, 4, 24, 4, 20, 24 };
    KeysT expectOffset = { 0, 1, 2, 0, 3, 0, 4, 3 };
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    EmbeddingMgmt::Instance()->Key2Offset("emb0", lookupKeys, TRAIN_CHANNEL_ID);
    map<EmbNameT, string> tmp;
    MxRec::KeyOffsetMemT kom = EmbeddingMgmt::Instance()->GetKeyOffsetMap();
    for (auto it = kom.begin(); it != kom.end(); ++it) {
        tmp.insert(pair<EmbNameT, string>(it->first, MapToString(it->second).c_str()));
    }

    LOG_DEBUG(KEY_PROCESS "test Key2Offset: lookupKeys: {}, keyOffsetMap: {}",
              VectorToString(lookupKeys), MapToString(tmp));
    ASSERT_THAT(lookupKeys, ElementsAreArray(expectOffset));

    KeysT lookupKeys2 = { 5, 17, 29, 5, 25, 5, 21, 25 };
    KeysT expectOffset2 = { -1, -1, -1, -1, -1, -1, -1, -1 };
    EmbeddingMgmt::Instance()->Key2Offset("emb0", lookupKeys2, EVAL_CHANNEL_ID);
    map<EmbNameT, string> tmp2;
    MxRec::KeyOffsetMemT kom2 = EmbeddingMgmt::Instance()->GetKeyOffsetMap();
    for (auto it = kom2.begin(); it != kom2.end(); ++it) {
        tmp.insert(pair<EmbNameT, string>(it->first, MapToString(it->second).c_str()));
    }
    LOG_DEBUG(KEY_PROCESS "test Key2Offset: lookupKeys: {}, keyOffsetMap: {}",
              VectorToString(lookupKeys2), MapToString(tmp2).c_str());
    ASSERT_THAT(lookupKeys2, ElementsAreArray(expectOffset2));
}

TEST_F(KeyProcessTest, GetUniqueConfig)
{
    ock::ctr::UniqueConf uniqueConf;
    process.rankInfo.rankSize = worldSize;
    process.rankInfo.useStatic = true;
    process.GetUniqueConfig(uniqueConf);
    process.rankInfo.useStatic = false;
    process.GetUniqueConfig(uniqueConf);
}

TEST_F(KeyProcessTest, InitializeUnique)
{
    ASSERT_EQ(ock::ctr::Factory::Create(factory), -1);
    ock::ctr::UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    PrepareBatch();

    unique_ptr <EmbBatchT> batch;
    batch = process.GetBatchData(0, 0);
    ock::ctr::UniqueConf uniqueConf;
    process.rankInfo.
    rankSize = worldSize;
    process.rankInfo.
    useStatic = true;
    bool uniqueInitialize = false;
    size_t preBatchSize = 0;
    process.InitializeUnique(uniqueConf, preBatchSize, uniqueInitialize, batch, unique);
    unique->UnInitialize();
}

TEST_F(KeyProcessTest, GetKeySize)
{
    PrepareBatch();
    unique_ptr<EmbBatchT> batch;
    batch = process.GetBatchData(0, 0);
    process.rankInfo.rankSize = worldSize;
    process.rankInfo.useStatic = true;
    process.GetKeySize(batch);
}

TEST_F(KeyProcessTest, ProcessBatchWithFastUnique)
{
    PrepareBatch();

    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    LOG_INFO("CPU Core Num: {}", sysconf(_SC_NPROCESSORS_CONF)); // 查看CPU核数

    auto fn = [this](int channel, int id) {
        ock::ctr::UniquePtr unique;

        auto embName = embInfos[0].name;
        process.hotEmbTotCount[embName] = 10;
        vector<KeysT> splitKeys;
        vector<int32_t> restore;
        vector<int32_t> hotPos;
        unique_ptr<EmbBatchT> batch;
        UniqueInfo uniqueInfo;
        batch = process.GetBatchData(channel, id); // get batch data from SingletonQueue<EmbBatchT>

        ASSERT_EQ(factory->CreateUnique(unique), ock::ctr::H_OK);
        ock::ctr::UniqueConf uniqueConf;
        size_t preBatchSize = 0;
        bool uniqueInitialize = false;
        process.GetUniqueConfig(uniqueConf);
        process.InitializeUnique(uniqueConf, preBatchSize, uniqueInitialize, batch, unique);

        LOG_INFO("rankid :{},batchid: {}", rankInfo.rankId, batch->batchId);
        process.KeyProcessTaskHelperWithFastUnique(batch, unique, channel, id);
        LOG_INFO("rankid :{},batchid: {}, hotPos {}", rankInfo.rankId, batch->batchId, VectorToString(hotPos));
        unique->UnInitialize();
    }; // for clean code
    for (int channel = 0; channel < 1; ++channel) {
        for (int id = 0; id < 1; ++id) {
            // use lambda expression initialize thread
            process.procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }
    this_thread::sleep_for(10s);
    process.Destroy();
}

TEST_F(KeyProcessTest, LoadSaveLock)
{
    process.LoadSaveLock();
    process.LoadSaveUnlock();
}

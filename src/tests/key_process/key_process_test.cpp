/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: key process test
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include <random>
#include <limits>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <easy/profiler.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/ranges.h>

#include "utils/common.h"
#include "host_emb/host_emb.h"
#include "key_process/key_process.h"
#include "hybrid_mgmt/hybrid_mgmt.h"
#include "ock_ctr_common/include/unique.h"

using namespace std;
using namespace MxRec;
using namespace testing;

static constexpr size_t BATCH_NUM_EACH_THREAD = 3;
static constexpr int DESIRED_SIZE = 1;
FactoryPtr factory;

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

static void CTRLog(int level, const char *msg)
{
    switch (level) {
        case 0:
            spdlog::debug("{}", msg);
            break;
        default:
            break;
    }
}

class KeyProcessTest : public testing::Test {
protected:
    void SetUp()
    {
        spdlog::set_level(spdlog::level::debug);
        int claimed;
        MPI_Query_thread(&claimed);
        ASSERT_EQ(claimed, MPI_THREAD_MULTIPLE);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        spdlog::info(KEY_PROCESS "wordRank: {}, worldSize: {}", worldRank, worldSize);
        // 初始化rank信息
        rankInfo.rankId = worldRank;
        rankInfo.rankSize = worldSize;
        rankInfo.localRankSize = worldSize;
        rankInfo.useStatic = useStatic;
        rankInfo.localRankId = rankInfo.rankId % rankInfo.localRankSize;
        rankInfo.noDDR = false;
        rankInfo.maxStep = { 1, -1 };
        // 初始化emb信息
        GenEmbInfos(embNum, embInfos, fieldNums);
        splits = fieldNums;
    }

    vector<vector<emb_batch_t>> PrepareBatch()
    {
        vector<vector<emb_batch_t>> result(KEY_PROCESS_THREAD * MAX_CHANNEL_NUM);
        // 向共享队列中写入本进程所有线程要处理的 KEY_PROCESS_THREAD * BATCH_NUM_EACH_THREAD 个batch数据
        for (size_t threadId = 0; threadId < KEY_PROCESS_THREAD; ++threadId) {
            int batchQueueId = threadId + KEY_PROCESS_THREAD * channel;
            unsigned int seed = batchQueueId * 10;
            auto queue = SingletonQueue<emb_batch_t>::getInstances(batchQueueId);

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
                    spdlog::debug("[{}/{}]"
                    KEY_PROCESS "PrepareBatch: batchQueueId: {}, {}[{}]{}, sampleSize:{}", worldRank, worldSize,
                        batchQueueId, batch->name, batch->channel, batch->batchId, batch->sample.size());
                    emb_batch_t temp;
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

    template<class T>
    inline vector<T> Count2Start(const vector<T>& count)
    {
        vector<T> start = { 0 };
        for (size_t i = 0; i < count.size() - 1; ++i) {
            start.push_back(count[i] + start.back());
        }
        return start;
    }

    // 生成emb表信息
    bool GenEmbInfos(size_t embNums, vector<EmbInfo>& allEmbInfos, vector<int>& geFieldNums)
    {
        default_random_engine generator;
        uniform_int_distribution<int> distribution(randMin, randMax);
        int embSizeMin = 5, embSizeMax = 8, base = 2, vocabSize = 100;
        uniform_int_distribution<int> embSizeDistribution(embSizeMin, embSizeMax);
        stringstream ss;
        for (unsigned int i = 0; i < embNums; ++i) {
            EmbInfo temp;
            ss << i;
            temp.name = "emb" + ss.str();
            ss.str("");
            ss.clear();
            temp.sendCount = distribution(generator);
            temp.extEmbeddingSize = pow(base, embSizeDistribution(generator));
            temp.devVocabSize = vocabSize;
            geFieldNums.push_back(sampleSize);
            allEmbInfos.push_back(move(temp));
        }
        return true;
    }

    auto GetSplitAndRestore(keys_t& sample) -> tuple<vector<keys_t>, vector<int32_t>>
    {
        vector<keys_t> expectSplitKeys(worldSize);
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

    void PrintHotHashSplit(const vector<keys_t>& splitKeys,
                           const vector<int32_t>& restore,
                           const vector<int32_t>& hotPos, int rankSize)
    {
        for (int i = 0; i < rankSize; ++i) {
            std::cout << "splitKeys dev" << i << std::endl;
            spdlog::info("{}", splitKeys[i]);
        }
        std::cout << "restore" << std::endl;
        spdlog::info("{}", restore);
        std::cout << "hotPos" << std::endl;
        spdlog::info("{}", hotPos);
    }

    void GetExpectRestore(keys_t& sample, vector<int>& blockOffset, vector<int>& restoreVec)
    {
        for (unsigned int i = 0; i < sample.size(); ++i) {
            int devId = sample[i] % worldSize;
            restoreVec[i] += blockOffset[devId];
        }
    }

    enum class A2A {
        SC, SS, RC, RS, INVALID
    };

    RankInfo rankInfo;
    int worldRank {};
    int worldSize {};
    vector<int> splits;
    int sampleSize = 20;
    int channel = 0;
    int randMin = 10;
    int randMax = 25; // 最大随机数范围
    // RankInfo rankInfo
    int batchSize = 5;
    int localRankSize = 2;
    bool useStatic = true;
    int staticSendCount = 65536;

    int maxRankSize = 8;

    // vector<EmbInfo> embInfos
    int embNum = 1;
    vector<int> fieldNums;

    vector<int64_t> src;
    vector<RankInfo> allRankInfo;
    vector<EmbInfo> embInfos;
    unique_ptr<emb_batch_t> batchData;
    vector<keys_t> splitKeys;
    vector<int32_t> restore;
    KeyProcess process;

    void TearDown()
    {
        // delete
    }
};

TEST_F(KeyProcessTest, Initialize)
{
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

    Factory::Create(factory);
}

TEST_F(KeyProcessTest, Start)
{
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    ASSERT_EQ(process.Start(), 0);
    process.Destroy();
}

TEST_F(KeyProcessTest, HashSplit)
{
    int rankSize = 4;
    auto queue = SingletonQueue<emb_batch_t>::getInstances(0);
    auto batch = queue->GetOne();
    keys_t batchKeys = { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 };
    vector<int> expectRestore = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 2 };
    vector<vector<int>> expectSplitKeys = { { 4, 16 }, { 1, 21, 29 }, { 14, 2 }, { 23, 7 } };
    batch->sample = std::move(batchKeys);
    spdlog::debug(KEY_PROCESS "batch sample: {}", batch->sample);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    process.rankInfo.rankSize = rankSize;
    auto [splitKeys, restore] = process.HashSplit(batch);
    for (unsigned int i = 0; i < splitKeys.size(); ++i) {
        ASSERT_THAT(splitKeys[i], ElementsAreArray(expectSplitKeys[i]));
    }
    ASSERT_THAT(restore, ElementsAreArray(expectRestore));
}

#ifndef GTEST
TEST_F(KeyProcessTest, GetScAll)
{
    vector<int> keyScLocal(worldSize, worldRank + 1); // 用worldRank+1初始化发送数据量
    spdlog::debug(KEY_PROCESS "rank {} keyScLocal: {}", worldRank, keyScLocal);
    vector<int> expectScAll(worldSize * worldSize);
    for (unsigned int i = 0; i < expectScAll.size(); ++i) {
        expectScAll[i] = floor(i / worldSize) + 1;
    }
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    vector<int> scAll;
    process.GetScAll(keyScLocal, 0, 0, scAll);
    ASSERT_THAT(scAll, ElementsAreArray(expectScAll));
}
#endif

TEST_F(KeyProcessTest, GetScAllForUnique)
{
    vector<int> keyScLocal(worldSize, worldRank + 1); // 用worldRank+1初始化发送数据量
    spdlog::debug(KEY_PROCESS "rank {} keyScLocal: {}", worldRank, keyScLocal);
    vector<int> expectScAll(worldSize * worldSize);
    for (unsigned int i = 0; i < expectScAll.size(); ++i) {
        expectScAll[i] = floor(i / worldSize) + 1;
    }
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    vector<int> scAll;
    process.GetScAllForUnique(keyScLocal, 0, 0, scAll);
    ASSERT_THAT(scAll, ElementsAreArray(expectScAll));
}

TEST_F(KeyProcessTest, BuildRestoreVec_4cpu)
{
    auto queue = SingletonQueue<emb_batch_t>::getInstances(0);
    auto batch = queue->GetOne();
    vector<keys_t> allBatchKeys = { { 1, 4, 23, 14, 16, 7, 2, 21, 21, 29 },
                                    { 5, 17, 26, 9, 27, 22, 27, 28, 15, 3 },
                                    { 10, 4, 22, 17, 24, 13, 24, 26, 29, 11 },
                                    { 14, 21, 18, 25, 21, 4, 20, 24, 13, 19 } };
    vector<vector<int>> allExpectSs = { { 0, 2, 5, 7, 9 }, { 0, 1, 4, 6 }, { 0, 2, 5, 8 }, { 0, 3, 6, 8 } };
    vector<vector<int>> allExpectRestore = { { 2, 0, 7, 5, 1, 8, 6, 3, 3, 4 },
                                             { 1, 2, 4, 3, 6, 5, 6, 0, 7, 8 },
                                             { 5, 0, 6, 2, 1, 3, 1, 7, 4, 8 },
                                             { 6, 3, 7, 4, 3, 0, 1, 2, 5, 8 } };
    batch->sample = std::move(allBatchKeys[worldRank]);
    spdlog::info(KEY_PROCESS "test BuildRestoreVec: rank {}, batchKeys {}", worldRank, batch->sample);
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    auto [splitKeys, restore] = process.HashSplit(batch);
    spdlog::debug("rank: {} splitKeys: {}", worldRank, splitKeys);
    process.BuildRestoreVec(batch, allExpectSs[worldRank], restore);
    ASSERT_THAT(restore, ElementsAreArray(allExpectRestore[worldRank]));
}

TEST_F(KeyProcessTest, ProcessKeySplit_rebuilt)
{
    PrepareBatch();
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    spdlog::info("CPU Core Num: {}", sysconf(_SC_NPROCESSORS_CONF)); // 查看CPU核数

    auto fn = [this](int channel, int id) {
        auto embName = embInfos[0].name;
        process.hotEmbTotCount[embName] = 10;
        vector<keys_t> splitKeys;
        vector<int32_t> restore;
        vector<int32_t> hotPos;
        unique_ptr<emb_batch_t> batch;
        batch = process.GetBatchData(channel, id); // get batch data from SingletonQueue<emb_batch_t>
        spdlog::info("rankid :{},batchid: {}", rankInfo.rankId, batch->batchId);
        tie(splitKeys, restore, hotPos) = process.HotHashSplit(batch);
        spdlog::info("rankid :{},batchid: {}, hotPos {}", rankInfo.rankId, batch->batchId,
                     hotPos);
    }; // for clean code
    for (int channel = 0; channel < 1; ++channel) {
        for (int id = 0; id < 1; ++id) {
            // use lambda expression initialize thread
            process.procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }
    this_thread::sleep_for(20s);
    process.Destroy();
}

TEST_F(KeyProcessTest, BuildRestoreVec_rebuilt)
{
    PrepareBatch();
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    spdlog::info("CPU Core Num: {}", sysconf(_SC_NPROCESSORS_CONF)); // 查看CPU核数

    auto fn = [this](int channel, int id) {
        auto embName = embInfos[0].name;
        vector<keys_t> splitKeys;
        vector<int32_t> restore;
        vector<int32_t> hotPos;
        unique_ptr<emb_batch_t> batch;
        batch = process.GetBatchData(channel, id); // get batch data from SingletonQueue<emb_batch_t>
        spdlog::info("rankid :{},batchid: {}", rankInfo.rankId, batch->batchId);
        tie(splitKeys, restore, hotPos) = process.HotHashSplit(batch);
        auto[lookupKeys, scAll, ss] = process.ProcessSplitKeys(batch, id, splitKeys);
        process.BuildRestoreVec(batch, ss, restore, hotPos.size());
        spdlog::info("rankid :{},batchid: {}, lookupKeys: {}, scAll: {}, restore after build {}", rankInfo.rankId,
            batch->batchId, lookupKeys, scAll, restore);
    }; // for clean code
    for (int channel = 0; channel < 1; ++channel) {
        for (int id = 0; id < KEY_PROCESS_THREAD; ++id) {
            // use lambda expression initialize thread
            process.procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }
    this_thread::sleep_for(20s);
    process.Destroy();
}

TEST_F(KeyProcessTest, Key2Offset)
{
    keys_t lookupKeys = { 4, 16, 28, 4, 24, 4, 20, 24 };
    keys_t expectOffset = { 0, 1, 2, 0, 3, 0, 4, 3 };
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    ASSERT_EQ(process.isRunning, true);
    process.Key2Offset("emb0", lookupKeys);
    spdlog::debug(KEY_PROCESS "test Key2Offset: lookupKeys: {}, keyOffsetMap: {}", lookupKeys, process.keyOffsetMap);
    ASSERT_THAT(lookupKeys, ElementsAreArray(expectOffset));
}

TEST_F(KeyProcessTest, GetUniqueConfig)
{
    UniqueConf uniqueConf;
    process.rankInfo.rankSize = worldSize;
    process.rankInfo.useStatic = true;
    process.GetUniqueConfig(uniqueConf);
    process.rankInfo.useStatic = false;
    process.GetUniqueConfig(uniqueConf);
}

// 自动化测试用例
// 边界值、重复度测试
TEST_F(KeyProcessTest, ProcessPrefetchTask)
{
    PrepareBatch();
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    process.rankInfo.rankSize = worldSize;
    process.rankInfo.localRankId = process.rankInfo.rankId % process.rankInfo.localRankSize;
    ASSERT_EQ(process.isRunning, true);
    ASSERT_EQ(process.Start(), 0);
    // 所有线程处理完（训练结束）后调用
    this_thread::sleep_for(5s);
    spdlog::info("wait 20s for thread running");
    this_thread::sleep_for(20s);
    process.Destroy();
}

TEST_F(KeyProcessTest, InitializeUnique)
{
    ASSERT_EQ(Factory::Create(factory), -1);
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    PrepareBatch();
    unique_ptr<emb_batch_t> batch;
    batch = process.GetBatchData(0, 0);
    UniqueConf uniqueConf;
    process.rankInfo.rankSize = worldSize;
    process.rankInfo.useStatic = true;
    bool uniqueInitialize = false;
    size_t preBatchSize = 0;
    process.InitializeUnique(uniqueConf, preBatchSize, uniqueInitialize, batch, unique);
}

TEST_F(KeyProcessTest, GetKeySize)
{
    PrepareBatch();
    unique_ptr<emb_batch_t> batch;
    batch = process.GetBatchData(0, 0);
    process.rankInfo.rankSize = worldSize;
    process.rankInfo.useStatic = true;
    process.GetKeySize(batch);
}

TEST_F(KeyProcessTest, ProcessBatchWithFastUnique)
{
    PrepareBatch();
    
    ASSERT_EQ(process.Initialize(rankInfo, embInfos), true);
    spdlog::info("CPU Core Num: {}", sysconf(_SC_NPROCESSORS_CONF)); // 查看CPU核数

    auto fn = [this](int channel, int id) {
        UniquePtr unique;
        
        auto embName = embInfos[0].name;
        process.hotEmbTotCount[embName] = 10;
        vector<keys_t> splitKeys;
        vector<int32_t> restore;
        vector<int32_t> hotPos;
        unique_ptr<emb_batch_t> batch;
        UniqueInfo uniqueInfo;
        batch = process.GetBatchData(channel, id); // get batch data from SingletonQueue<emb_batch_t>

        ASSERT_EQ(factory->CreateUnique(unique), 0);
        UniqueConf uniqueConf;
        process.GetUniqueConfig(uniqueConf);
        unique->Initialize(uniqueConf);
        
        spdlog::info("rankid :{},batchid: {}", rankInfo.rankId, batch->batchId);
        process.KeyProcessTaskHelperWithFastUnique(batch, unique, channel, id);
        spdlog::info("rankid :{},batchid: {}, hotPos {}", rankInfo.rankId, batch->batchId,
                     hotPos);
    }; // for clean code
    for (int channel = 0; channel < 1; ++channel) {
        for (int id = 0; id < 1; ++id) {
            // use lambda expression initialize thread
            process.procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }
    this_thread::sleep_for(20s);
    process.Destroy();
}
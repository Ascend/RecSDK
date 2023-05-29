/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#include "key_process.h"

#include <iostream>
#include <shared_mutex>

#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <mpi.h>

#include "checkpoint/checkpoint.h"
#include "hd_transfer/hd_transfer.h"
#include "utils/common.h"
#include "utils/time_cost.h"

using namespace std;
using namespace chrono;
using namespace MxRec;

static shared_mutex g_smut;

template<class T>
inline vector<T> Count2Start(const vector<T>& count)
{
    vector<T> start = { 0 };
    for (size_t i = 0; i < count.size() - 1; ++i) {
        start.push_back(count[i] + start.back());
    }
    return start;
}

KeyProcess::KeyProcess()
{
    // init class members with PerfConfig::keyProcessThreadNum
    for (size_t i = 0; i < MAX_CHANNEL_NUM; ++i) {
        comm[i].resize(PerfConfig::keyProcessThreadNum);
        for (int j = 0; j < PerfConfig::keyProcessThreadNum; ++j) {
            comm[i][j] = MPI_COMM_WORLD;
        }
    }

    for (size_t i = 0; i < MAX_CHANNEL_NUM; ++i) {
        std::vector<std::mutex> tmp(PerfConfig::keyProcessThreadNum);
        loadSaveMut[i].swap(tmp);
    }
    std::vector<std::mutex> tmp(PerfConfig::keyProcessThreadNum);
    getInfoMut.swap(tmp);

    storage.resize(PerfConfig::keyProcessThreadNum);
    lookupKeysList.resize(PerfConfig::keyProcessThreadNum);
    infoList.resize(PerfConfig::keyProcessThreadNum);
    all2AllList.resize(PerfConfig::keyProcessThreadNum);
}

int KeyProcess::Initialize(const RankInfo& rInfo, const vector<EmbInfo>& eInfos,
                           const vector<ThresholdValue>& thresholdValues,
                           bool ifLoad, int seed)
{
    this->rankInfo = rInfo;
    if (rankInfo.useHot) {
        const char* env = getenv("HOT_EMB_UPDATE_STEP");
        if (env == nullptr) {
            hotEmbUpdateStep = HOT_EMB_UPDATE_STEP_DEFAULT;
        } else {
            hotEmbUpdateStep = stoi(env);
            if (hotEmbUpdateStep == 0) {
                hotEmbUpdateStep = HOT_EMB_UPDATE_STEP_DEFAULT;
            }
        }
    }

    map<emb_name_t, int> scInfo;
    for (const auto& info: eInfos) {
        spdlog::debug(KEY_PROCESS "Init sendCountMap:{}, channelNames:{}", info.sendCountMap, info.channelNames);
        embInfos[info.name] = info;
        scInfo[info.name] = info.sendCount;
        if (rankInfo.useHot) {
            InitHotEmbTotCount(info, rInfo);
        }
        if (rankInfo.useDynamicExpansion) {
            // 动态扩容
            embeddingTableMap[info.name].Init(info, rInfo, seed);
            spdlog::info(KEY_PROCESS "EmbeddingTableMap：{} init success", info.name);
        }

        if (rankInfo.rankId == 0 && !ifLoad) {
            Key2OffsetInit(info.name);
        }
    }
    spdlog::info(KEY_PROCESS "hot emb count info:{}", hotEmbTotCount);
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    for (auto& i: comm) {
        for (auto& j: i) {
            MPI_Comm_create(MPI_COMM_WORLD, world_group, &j);
        }
    }
    isRunning = true;

    // 特征准入与特征淘汰
    if (!thresholdValues.empty()) {
        m_featureAdmitAndEvict.SetFunctionSwitch(true);
        m_featureAdmitAndEvict.Init(thresholdValues);
    } else {
        m_featureAdmitAndEvict.SetFunctionSwitch(false);
        spdlog::warn(KEY_PROCESS "Feature admit-and-evict function is unavailable ...");
    }

    spdlog::info(KEY_PROCESS "scInfo:{}, localRankSize:{}, rankSize:{}, useStatic:{}, useHot:{}", scInfo,
        rInfo.localRankSize, rInfo.rankSize, rInfo.useStatic, rInfo.useHot);
    return 0;
}

// bind and start main process
int KeyProcess::Start()
{
    // bind like:
    // 0 1 2 3 4 5 0 1 2 3 4 5
    // |  rank0  | |  rank1  |
    // each rank creates PerfConfig::keyProcessThreadNum threads, each thread process one batchdata
    spdlog::info("CPU Core Num: {}", sysconf(_SC_NPROCESSORS_CONF)); // 查看CPU核数
    auto fn = [this](int channel, int id) {
#ifndef GTEST
        auto ret = aclrtSetDevice(static_cast<int32_t>(rankInfo.deviceId));
        if (ret != ACL_ERROR_NONE) {
            spdlog::error("Set device failed, device_id:{}", rankInfo.deviceId);
            return;
        }
#endif
        KeyProcessTask(channel, id);
    }; // for clean code
    for (int channel = 0; channel < MAX_CHANNEL_NUM; ++channel) {
        for (int id = 0; id < PerfConfig::keyProcessThreadNum; ++id) {
            procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }
    return 0;
}

void KeyProcess::InitHotEmbTotCount(const EmbInfo& info, const RankInfo& rInfo)
{
    auto embeddingSize = info.extEmbeddingSize;
    if (rankInfo.useDynamicExpansion) {
        embeddingSize = info.embeddingSize;
    }
    hotEmbTotCount[info.name] = static_cast<int>(GetUBSize(rInfo.deviceId) / sizeof(float) * HOT_EMB_CACHE_PCT /
                                                 embeddingSize);
}

auto KeyProcess::GetSendCount(const string& name, const string& channelName, bool modifyGraph)
{
    auto sendCountSize = embInfos[name].sendCount;
    if (modifyGraph) {
        sendCountSize = embInfos[name].sendCountMap[channelName];
    }
    return sendCountSize;
}

auto KeyProcess::GetMaxOffset() -> offset_mem_t
{
    return maxOffset;
}

auto KeyProcess::GetKeyOffsetMap() -> key_offset_mem_t
{
    return keyOffsetMap;
}

auto KeyProcess::GetFeatAdmitAndEvict() -> FeatureAdmitAndEvict&
{
    return m_featureAdmitAndEvict;
}

void KeyProcess::LoadMaxOffset(offset_mem_t& loadData)
{
    maxOffset = std::move(loadData);
}

void KeyProcess::LoadKeyOffsetMap(key_offset_mem_t& loadData)
{
    keyOffsetMap = std::move(loadData);
}

// 只在python侧当训练结束时调用，如果出现死锁直接结束程序即可,测试时让进程等待足够长的时间再调用
void KeyProcess::Destroy()
{
    isRunning = false;
    spdlog::info(KEY_PROCESS "rank {} begin destroy.", rankInfo.rankId);
    for (auto& t: procThreads) {
        t->join();
    }
    procThreads.clear();
    spdlog::info(KEY_PROCESS "rank {} destroy success.", rankInfo.rankId);
}

void KeyProcess::LoadSaveLock()
{
    for (int channelId { 0 }; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadId { 0 }; threadId < PerfConfig::keyProcessThreadNum; ++threadId) {
            loadSaveMut[channelId][threadId].lock();
        }
    }
}

void KeyProcess::LoadSaveUnlock()
{
    for (int channelId { 0 }; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadId { 0 }; threadId < PerfConfig::keyProcessThreadNum; ++threadId) {
            loadSaveMut[channelId][threadId].unlock();
        }
    }
}

void KeyProcess::KeyProcessTask(int channel, int id)
{
    unique_ptr<emb_batch_t> batch;

    GroupMethod groupMethod;
    groupMethod.SetGroupCount(rankInfo.rankSize);

    shared_ptr<sharded_dedup> unique;
    map<int, shared_ptr<sharded_dedup>> uniquePtrMap;

    spdlog::stopwatch sw;
    try {
        while (true) {
            TimeCost getAndProcesTC;
            TimeCost getBatchTC;
            batch = GetBatchData(channel, id); // get batch data from SingletonQueue<emb_batch_t>
            TIME_PRINT("GetBatchData TimeCost(ms):{}", getBatchTC.ElapsedMS());

            if (batch == nullptr) {
                spdlog::info(KEY_PROCESS "batch is nullptr");
                break;
            }
            auto getBatchTime = TO_MS(sw);
            sw.reset();

            auto sendCountSize = GetSendCount(batch->name, batch->channelName, batch->modifyGraph);
            shared_ptr<sharded_dedup> uniquePtr;
            if (uniquePtrMap.find(sendCountSize) == uniquePtrMap.end()) {
                uniquePtr.reset(new sharded_dedup(groupMethod, batch->batchSize, sendCountSize));
                uniquePtrMap.insert(std::make_pair(sendCountSize, uniquePtr));
            }
            unique = uniquePtrMap[sendCountSize];

            if (unique != nullptr) {
                unique->StartNewRound();
            }

            auto batchQueue =
                    SingletonQueue<emb_batch_t>::getInstances(id + PerfConfig::keyProcessThreadNum * batch->channel);

            if (!KeyProcessTaskHelper(batch, unique, channel, id, sw)) {
                free(batch->tensorAddr);
                batchQueue->PutDirty(move(batch));
                break;
            }
            TIME_PRINT("getAndProcesTC TimeCost(ms):{}", getAndProcesTC.ElapsedMS());
            spdlog::info(KEY_PROCESS "key process cost:{}, get data time:{} batch {}[{}]:{} ",
                         TO_MS(sw), getBatchTime, batch->name, batch->channel, batch->batchId);
            free(batch->tensorAddr);
            batch->tensorAddr = nullptr;
            batchQueue->PutDirty(move(batch));
        }
    } catch (const EndRunError &e) {
        spdlog::debug(KEY_PROCESS "abort run: {}", e.what());
    }
    spdlog::info(KEY_PROCESS "KeyProcessTask exit. rank:{} thread:{}, channel:{}", rankInfo.rankId, id, channel);
}

bool KeyProcess::KeyProcessTaskHelper(unique_ptr<emb_batch_t> &batch, shared_ptr<sharded_dedup> unique,
                                      int channel, int id, spdlog::stopwatch &sw)
{
    // tuple for keyRec restore hotPos scAll countRecv
    std::tuple<keys_t, vector<int32_t>, vector<int32_t>, vector<int32_t>, vector<int>, vector<uint32_t>> rets;
    isWithFAAE = m_featureAdmitAndEvict.GetFunctionSwitch() &&
                  FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE;
    TimeCost tc;
    UniqueInfo uniqueInfo;
    ProcessBatchWithUniqueCompute(batch, unique, id, uniqueInfo);
    TIME_PRINT("no copy ProcessBatchWithUniqueCompute TimeCost(ms):{}", tc.ElapsedMS());

    // 特征准入&淘汰
    if (isWithFAAE &&
        (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, uniqueInfo.all2AllInfo.keyRecv,
                                             uniqueInfo.all2AllInfo.countRecv)
                                             == FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR)) {
        spdlog::error(KEY_PROCESS "rank:{} thread:{}, channel:{}, Feature-admit-and-evict error ...",
                      rankInfo.rankId, id, channel);
        return false;
    }
    int batchListId = batch->batchId % PerfConfig::keyProcessThreadNum;

    // without host, just device, all embedding vectors were stored in device
    // map key to offset directly by lookup keyOffsetMap (hashmap)
    if (rankInfo.noDDR) {
        TimeCost key2OffsetTc;
        Key2Offset(batch->name, uniqueInfo.all2AllInfo.keyRecv);
        TIME_PRINT("Key2Offset TimeCost(ms):{}", key2OffsetTc.ElapsedMS());
    }
    if (!rankInfo.useStatic) { // Static all2all，need send count
        auto embName = batch->name;
        if (batch->modifyGraph) {
            embName = batch->channelName;
        }
        SendA2A(uniqueInfo.all2AllInfo.scAll, embName, batch->channel, batch->batchId);
    }

    auto tensors = make_unique<vector<Tensor>>();
    tensors->push_back(Vec2TensorI32(uniqueInfo.restore));
    if (rankInfo.useHot) {
        uniqueInfo.hotPos.resize(hotEmbTotCount[batch->name], -1);
        tensors->push_back(Vec2TensorI32(uniqueInfo.hotPos));
    }
    if (rankInfo.noDDR) {
        if (rankInfo.useDynamicExpansion) {
            tensors->push_back(Vec2TensorI64(uniqueInfo.all2AllInfo.keyRecv));
        } else {
            tensors->push_back(Vec2TensorI32(uniqueInfo.all2AllInfo.keyRecv));
        }
    }
    TimeCost pushTensorTc;
    PushResult(batch, move(tensors), uniqueInfo.all2AllInfo.keyRecv, batchListId);
    TIME_PRINT("pushTensorToListTC TimeCost(ms):{}", pushTensorTc.ElapsedMS());
    return true;
}

vector<uint32_t> KeyProcess::GetCountRecv(const unique_ptr<emb_batch_t>& batch, int id,
                                          vector<vector<uint32_t>>& keyCount, vector<int> scAll, vector<int> ss)
{
    if (rankInfo.useStatic) {
        for (auto& cnt: keyCount) {
            cnt.resize(embInfos[batch->name].sendCount, 0);
        }
    }
    vector<uint32_t> countSend;
    for (auto& cnt: keyCount) {
        countSend.insert(countSend.end(), cnt.begin(), cnt.end());
    }
    vector<int> sc;
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        sc.push_back(scAll.at(rankInfo.rankSize * rankInfo.rankId + i));
    }
    vector<int> rc;                                // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        rc.push_back(scAll.at(i * rankInfo.rankSize + rankInfo.rankId));
    }
    auto rs = Count2Start(rc); // receive displays/offset 接受数据的起始偏移量
    vector<uint32_t> countRecv;
    countRecv.resize(rs.back() + rc.back());
    MPI_Alltoallv(countSend.data(), sc.data(), ss.data(), MPI_UINT32_T, countRecv.data(),
                  rc.data(), rs.data(), MPI_UINT32_T, comm[batch->channel][id]);
    return countRecv;
}

void KeyProcess::PushResult(unique_ptr<emb_batch_t>& batch, unique_ptr<vector<Tensor>> tensors,
                            keys_t& lookupKeys, int id)
{
    std::unique_lock<std::mutex> lockGuard(getInfoMut[id]);
    storage[id].push_front(move(tensors));
    if (batch->modifyGraph) {
        infoList[id][batch->channelName][batch->channel].push(
            make_tuple(batch->batchId, batch->channelName, storage[id].begin()));
    } else {
        infoList[id][batch->name][batch->channel].push(
            make_tuple(batch->batchId, batch->name, storage[id].begin()));
    }
    if (!rankInfo.noDDR) {
        lookupKeysList[id][batch->name][batch->channel].push(
            make_tuple(batch->batchId, batch->name, move(lookupKeys)));
    }
    lockGuard.unlock();
}

/*
 * 从共享队列SingletonQueue<emb_batch_t>中读取batch数据并返回。batch数据由 ReadEmbKeyV2 写入。
 * commID为线程标识[0, PerfConfig::keyProcessThreadNum-1]，不同线程、训练或推理数据用不同的共享队列通信
 */
unique_ptr<emb_batch_t> KeyProcess::GetBatchData(int channel, int commId)
{
    EASY_FUNCTION()
    unique_ptr<emb_batch_t> batch = nullptr;
    // train data, queue id = thread id [0, PerfConfig::keyProcessThreadNum-1]
    auto batchQueue = SingletonQueue<emb_batch_t>::getInstances(commId + PerfConfig::keyProcessThreadNum * channel);
    EASY_BLOCK("get samples")
    EASY_VALUE("run on CPU", sched_getcpu())
    spdlog::stopwatch sw;
    while (true) {
        batch = batchQueue->TryPop();
        if (batch != nullptr) {
            break;
        } else {
            this_thread::sleep_for(100us);
        }
        if (duration_cast<seconds>(sw.elapsed()).count() > GET_BATCH_TIMEOUT) {
            if (commId == 0) {
                spdlog::warn(KEY_PROCESS "getting batch timeout! 1. check last 'read batch cost' print. "
                             "channel[{}] commId[{}]", channel, commId);
            }
            this_thread::sleep_for(seconds(1));
            sw.reset();
        }
        if (!isRunning) {
            // 通信终止信号，同步退出，防止线程卡住
            int exitFlag = isRunning;
            MPI_Allreduce(&exitFlag, &exitFlag, 1, MPI_INT, MPI_SUM, comm[channel][commId]);
            throw EndRunError("GetBatchData end run.");
        }
    }
    EASY_END_BLOCK
    spdlog::info(KEY_PROCESS "GetBatchData get batchId:{}, batchSize:{}, batch.channel:{}, batch.channelName:{}, "
                 "name:{}, channel:{}, commId:{}, ",
                 batch->batchId, batch->batchSize, batch->channel, batch->channelName, batch->name, channel, commId);
#if defined(PROFILING) && defined(BUILD_WITH_EASY_PROFILER)
    if (batch->batchId == PROFILING_START_BATCH_ID) {
        EASY_PROFILER_ENABLE
    } else if (batch->batchId == PROFILING_END_BATCH_ID) {
        EASY_PROFILER_ENABLE
        ::profiler::dumpBlocksToFile(fmt::format("/home/MX_REC-profile-{}.prof", rankInfo.rankId).c_str());
    }
#endif
    return batch;
}

size_t KeyProcess::GetKeySize(const unique_ptr<emb_batch_t> &batch)
{
    size_t size = rankInfo.rankSize * embInfos[batch->name].sendCount;
    if (batch->modifyGraph) {
        size = rankInfo.rankSize * embInfos[batch->name].sendCountMap[batch->channelName];
    }
    if (!rankInfo.useStatic) {
        size = batch->batchSize;
    }
    return size;
}

void KeyProcess::ProcessBatchWithUniqueCompute(const unique_ptr<emb_batch_t> &batch, shared_ptr<sharded_dedup> unique,
                                               int id, UniqueInfo& uniqueInfoOut)
{
    EASY_FUNCTION(profiler::colors::Purple)
    EASY_VALUE("batchId", batch->batchId)

    EASY_BLOCK("ock-unique")

    TimeCost unique_tc;

    SimpleThreadPool pool_;
    KeySendInfo keySendInfo;
    size_t size = GetKeySize(batch);
    keySendInfo.keySend.resize(size);
    vector<int32_t> splitSize(rankInfo.rankSize);
    vector<int64_t> uniqueVector(batch->batchSize);
    uniqueInfoOut.restore.resize(batch->batchSize);
    vector<int32_t> idCount(batch->batchSize);
    keySendInfo.keyCount.resize(size);
    std::shared_lock<std::shared_mutex> lock(g_smut);
    auto hotMap = hotKey[batch->name];
    lock.unlock();
    int hotOffset = 0;

    if (rankInfo.useHot) {
        uniqueInfoOut.hotPos.resize(hotEmbTotCount[batch->name]);
        hotOffset = hotEmbTotCount[batch->name];
    }
    absl::flat_hash_map<emb_key_t, int> keyCountMap;

    UniqueData uniqueData = {batch->tensorAddr, batch->batchSize, uniqueInfoOut.restore.data(), uniqueVector.data(),
                             splitSize.data(), keySendInfo.keySend.data(), idCount.data(), keySendInfo.keyCount.data()};
    UniqueFlag uniqueFlag = {batch->isInt64, rankInfo.useStatic, rankInfo.useHot};
    UniqueForHot uniqueForHot = {hotOffset, uniqueInfoOut.hotPos.data(), hotMap, keyCountMap};
    UniqueThreadNum uniqueThreadNum = {MIN_UNIQUE_THREAD_NUM, MAX_UNIQUE_THREAD_NUM};

    unique->Compute<int, SimpleThreadPool>(&pool_,  uniqueData, uniqueFlag, uniqueForHot, uniqueThreadNum);
    EASY_END_BLOCK
    TIME_PRINT("UniqueCompute TimeCost(ms):{}", unique_tc.ElapsedMS());

    if (rankInfo.useHot) {
        UpdateHotMap(keyCountMap, hotEmbTotCount[batch->name], batch->batchId % hotEmbUpdateStep == 0, batch->name);
    }

    vector<int> sc; // send count
    if (rankInfo.useStatic) {
        sc.resize(rankInfo.rankSize, GetSendCount(batch->name, batch->channelName, batch->modifyGraph));
    } else {
        sc.resize(rankInfo.rankSize);
        for (int i = 0;i < rankInfo.rankSize; i++) {
            sc[i] = splitSize[i];
        }
    }
    All2All(sc, id, batch->channel, keySendInfo, uniqueInfoOut.all2AllInfo);

    spdlog::debug(KEY_PROCESS "ProcessBatchWithUniqueCompute get batchId:{}, batchSize:{}, channel:{}, "
                             "channelName:{}, name:{}, restore:{}, keyCount:{}", batch->batchId, batch->batchSize,
                             batch->channel, batch->channelName, batch->name, uniqueInfoOut.restore.size(),
                             keySendInfo.keyCount.size());
}

void KeyProcess::All2All(vector<int>& sc, int id, int channel, KeySendInfo& keySendInfo,
                         All2AllInfo& all2AllInfoOut)

{
    TimeCost get_sc_all;
    GetScAll(sc, id, channel, all2AllInfoOut.scAll); // Allgather通信获取所有（不同rank相同thread id的）
    TIME_PRINT("GetScAll TimeCost(ms):{}", get_sc_all.ElapsedMS());

    TimeCost all2allTC;
    auto ss = Count2Start(sc); // send displays/offset 发送数据的起始偏移量
    vector<int> rc(rankInfo.rankSize);            // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        // 通信量矩阵某一列的和即为本地要从其他设备接受的key数据量
        rc[i] = all2AllInfoOut.scAll.at(i * rankInfo.rankSize + rankInfo.rankId);
    }
    auto rs = Count2Start(rc); // receive displays/offset 接受数据的起始偏移量
    all2AllInfoOut.keyRecv.resize(rs.back() + rc.back());
    EASY_BLOCK("all2all")
    MPI_Alltoallv(keySendInfo.keySend.data(), sc.data(), ss.data(), MPI_INT64_T, all2AllInfoOut.keyRecv.data(),
                  rc.data(), rs.data(), MPI_INT64_T, comm[channel][id]);

    all2AllInfoOut.countRecv.resize(rs.back() + rc.back());
    if (isWithFAAE) {
        MPI_Alltoallv(keySendInfo.keyCount.data(), sc.data(), ss.data(), MPI_UINT32_T, all2AllInfoOut.countRecv.data(),
                      rc.data(), rs.data(), MPI_UINT32_T, comm[channel][id]);
    }
    TIME_PRINT("all2allTC TimeCost(ms):{}", all2allTC.ElapsedMS());
    EASY_END_BLOCK
}

auto KeyProcess::ProcessSplitKeys(const unique_ptr<emb_batch_t>& batch, int id,
                                  vector<keys_t>& splitKeys) -> tuple<keys_t, vector<int>, vector<int>>
{
    EASY_FUNCTION(profiler::colors::Purple)
    EASY_VALUE("batchId", batch->batchId)
    spdlog::info(KEY_PROCESS "ProcessSplitKeys start batchId:{}, channel:{}", batch->batchId, batch->channel);

    // 使用静态all2all通信：发送或接受量为预置固定值 scInfo[batch->name] = 65536 / rankSize 经验值
    if (rankInfo.useStatic) { // maybe move after all2all
        for (auto& i: splitKeys) {
            if (static_cast<int>(i.size()) > embInfos[batch->name].sendCount) {
                spdlog::error("{}[{}]:{} overflow! set send count bigger than {}",
                              batch->name, batch->channel, batch->batchId, i.size());
                throw runtime_error(fmt::format("{}[{}]:{} overflow! set send count bigger than {}",
                                                batch->name, batch->channel, batch->batchId, i.size()).c_str());
            }
            i.resize(embInfos[batch->name].sendCount, -1);
        }
    }
    keys_t keySend;
    vector<int> sc; // send count
    for (const auto& i: splitKeys) {
        sc.push_back(static_cast<int>(i.size()));
        keySend.insert(keySend.end(), i.begin(), i.end());
    }
    keys_t keyRecv;
    vector<int> scAll;
    GetScAll(sc, id, batch->channel, scAll);     // Allgather通信获取所有（不同rank相同thread id的）线程间通信量矩阵
    auto ss = Count2Start(sc);  // send displays/offset 发送数据的起始偏移量
    vector<int> rc; // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        // 通信量矩阵某一列的和即为本地要从其他设备接受的key数据量
        rc.push_back(scAll.at(i * rankInfo.rankSize + rankInfo.rankId));
    }
    auto rs = Count2Start(rc); // receive displays/offset 接受数据的起始偏移量
    keyRecv.resize(rs.back() + rc.back());
    spdlog::trace(KEY_PROCESS "MPI_Alltoallv begin. rank {} thread {} batch {} {}",
                  rankInfo.rankId, id, batch->batchId, batch->name);
    EASY_BLOCK("all2all")
    MPI_Alltoallv(keySend.data(), sc.data(), ss.data(), MPI_INT64_T,
                  keyRecv.data(), rc.data(), rs.data(), MPI_INT64_T,
                  comm[batch->channel][id]);
    EASY_END_BLOCK
    spdlog::trace(KEY_PROCESS "MPI_Alltoallv finish. rank {} thread {} batch {} {}",
                  rankInfo.rankId, id, batch->batchId, batch->name);

    return { keyRecv, scAll, ss };
}

/*
 * 将batch内的key按照所存储的dev id哈希切分并去重，哈希函数为模运算
 * splitKeys返回：将数据的key切分到其所在dev id对应的桶中，并去重。
 * restore返回：去重后key在桶内偏移量（用于计算恢复向量）
 */
auto KeyProcess::HashSplit(const unique_ptr<emb_batch_t>& batch) const -> tuple<vector<keys_t>, vector<int32_t>>
{
    EASY_FUNCTION(profiler::colors::Gold)
    auto* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    ASSERT(batchData != nullptr);
    vector<keys_t> splitKeys(rankInfo.rankSize);
    vector<int32_t> restore(batch->Size());
    vector<int> hashSplitLens(rankInfo.rankSize); // 初始化全0，记录每个桶的长度
    absl::flat_hash_map<emb_key_t, int> uKey;     // 用于去重查询
    EASY_BLOCK("split push back")
    for (size_t i = 0; i < miniBs; i++) {
        const emb_key_t& key = batchData[i];
        int devId = static_cast<int>(key) & (rankInfo.rankSize - 1); // 数据所在的设备devID = key % dev总数 support -1
        auto result = uKey.find(key);
        if (result == uKey.end()) {
            splitKeys[devId].push_back(key);
            restore[i] = hashSplitLens[devId]++; // restore记录去重后key在桶内偏移量（用于计算恢复向量）
            uKey[key] = restore[i];
        } else { // 去重
            restore[i] = result->second;
        }
    }
    EASY_END_BLOCK
    if (spdlog::get_level() == spdlog::level::trace) {
        stringstream ssTrace;
        for (int devId = 0; devId < rankInfo.rankSize; ++devId) {
            ssTrace << '|' << devId << ":";
            for (auto x: splitKeys[devId]) {
                ssTrace << x << ',';
            }
            ssTrace << '|';
        }
        spdlog::trace("dump splitKeys\n{}", ssTrace.str());
    }
    return { splitKeys, restore };
}

auto KeyProcess::HashSplit_withFAAE(const unique_ptr<emb_batch_t>& batch) const
    -> tuple<vector<keys_t>, vector<int32_t>, vector<vector<uint32_t>>>
{
    EASY_FUNCTION(profiler::colors::Gold)
    auto* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    ASSERT(batchData != nullptr);
    vector<keys_t> splitKeys(rankInfo.rankSize);
    vector<vector<uint32_t>> keyCount(rankInfo.rankSize); // splitKeys在原始batch中对应的频次
    vector<int32_t> restore(batch->Size());
    vector<int> hashSplitLens(rankInfo.rankSize);                  // 初始化全0，记录每个桶的长度
    absl::flat_hash_map<emb_key_t, std::pair<int, uint32_t>> uKey; // 用于去重查询
    EASY_BLOCK("split push back")
    for (size_t i = 0; i < miniBs; i++) {
        const emb_key_t& key = batchData[i];
        int devId = static_cast<int>(key) & (rankInfo.rankSize - 1); // 数据所在的设备devID = key % dev总数 support -1
        auto result = uKey.find(key);
        if (result == uKey.end()) {
            splitKeys[devId].push_back(key);
            restore[i] = hashSplitLens[devId]++; // restore记录去重后key在桶内偏移量（用于计算恢复向量）
            uKey[key].first = restore[i];
            uKey[key].second = 1;
        } else { // 去重
            restore[i] = result->second.first;
            uKey[key].second++;
        }
    }

    // 处理splitKeys对应的count
    for (int j = 0; j < rankInfo.rankSize; ++j) {
        vector<uint32_t> count;
        for (size_t k = 0; k < splitKeys[j].size(); ++k) {
            count.emplace_back(uKey[splitKeys[j][k]].second);
        }
        keyCount[j] = count;
    }

    EASY_END_BLOCK
    if (spdlog::get_level() == spdlog::level::trace) {
        stringstream ssTrace;
        for (int devId = 0; devId < rankInfo.rankSize; ++devId) {
            ssTrace << '|' << devId << ":";
            for (auto x : splitKeys[devId]) {
                ssTrace << x << ',';
            }
            ssTrace << '|';
        }
        spdlog::trace("dump splitKeys\n{}", ssTrace.str());
    }

    return { splitKeys, restore, keyCount };
}

auto KeyProcess::HotHashSplit(const unique_ptr<emb_batch_t>& batch) ->
tuple<vector<keys_t>, vector<int32_t>, vector<int>>
{
    EASY_FUNCTION(profiler::colors::Gold)
    auto* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    vector<keys_t> splitKeys(rankInfo.rankSize);
    vector<int32_t> restore(batch->Size());
    absl::flat_hash_map<emb_key_t, int> uKey;   // 用于去重查询
    absl::flat_hash_map<emb_key_t, int> keyCountMap;
    std::shared_lock<std::shared_mutex> lock(g_smut);
    auto hotMap = hotKey[batch->name];
    lock.unlock();
    vector<int> hotPos(hotEmbTotCount[batch->name]);
    vector<int> hotPosDev(hotEmbTotCount[batch->name]);

    int hotCount = 0;
    int hotOffset = hotEmbTotCount[batch->name];
    for (size_t i = 0; i < miniBs; i++) { // for mini batch
        const emb_key_t& key = batchData[i];
        if (batch->batchId % hotEmbUpdateStep == 0) {
            keyCountMap[key]++;
        }
        int devId = static_cast<int>(key) & (rankInfo.rankSize - 1);   // 数据所在的设备devID = key % dev总数 support -1
        auto result = uKey.find(key);
        if (result != uKey.end()) { // // already in splitKeys
            restore[i] = result->second;
            continue;
        }
        // new key in current batch
        splitKeys[devId].push_back(key); // push to bucket
        auto hot = hotMap.find(key);
        if (hot != hotMap.end()) { // is hot key
            if (hot->second == -1) { // is new hot key in this batch
                hotPos[hotCount] = splitKeys[devId].size() - 1;  // pos in lookup vec (need add ss) for hot-gather
                hotPosDev[hotCount] = devId; // which dev, for get ss
                hot->second = hotCount;
                restore[i] = hotCount++; // get pos of hot emb
            } else {
                restore[i] = hot->second;
            }
        } else { // is not hot key
            restore[i] = splitKeys[devId].size() + hotOffset - 1;    // restore记录去重后key在桶内偏移量（用于计算恢复向量）
        }
        uKey[key] = restore[i];
    }

    UpdateHotMap(keyCountMap, hotEmbTotCount[batch->name], batch->batchId % hotEmbUpdateStep == 0, batch->name);
    AddCountStartToHotPos(splitKeys, hotPos, hotPosDev, batch->name);

    return { splitKeys, restore, hotPos };
}

void KeyProcess::AddCountStartToHotPos(vector<keys_t>& splitKeys, vector<int>& hotPos, const vector<int>& hotPosDev,
                                       const string& embName) const
{
    vector<int> splitKeysSize {};
    if (rankInfo.useStatic) {
        for (size_t i = 0; i < splitKeys.size(); i++) {
            splitKeysSize.push_back(embInfos.at(embName).sendCount);
        }
    } else {
        for (auto& splitKey: splitKeys) {
            splitKeysSize.push_back(splitKey.size());
        }
    }
    auto cs = Count2Start(splitKeysSize);
    for (size_t i = 0; i < hotPos.size(); ++i) {
        hotPos[i] += cs[hotPosDev[i]];
    }
}

void KeyProcess::UpdateHotMap(absl::flat_hash_map<emb_key_t, int>& keyCountMap, uint32_t count, bool refresh,
                              const string& embName)
{
    auto& hotMap = hotKey[embName];
    if (refresh) {
        priority_queue<pair<int, emb_key_t>> pq; // top k key
        for (auto& p: keyCountMap) {
            pq.push(pair<int, emb_key_t>(-p.second, p.first));
            if (pq.size() > count) {
                pq.pop();
            }
        }
        // gen new hot map
        std::unique_lock<std::shared_mutex> lock(g_smut);
        hotMap.clear();
        while (!pq.empty()) {
            hotMap.insert(make_pair(pq.top().second, -1));
            pq.pop();
        }
    }
}

/*
 * 将本地（rank）batch要发送的key数据量进行Allgather通信，获取所有（不同rank相同thread id的）线程间的通信量矩阵
 * scAll返回：所有线程间的通信量矩阵（按行平铺的一维向量）
 */
void KeyProcess::GetScAll(const vector<int>& keyScLocal, int commId, int channel, vector<int> &scAllOut) const
{
    EASY_FUNCTION()
    scAllOut.resize(rankInfo.rankSize * rankInfo.rankSize);
    EASY_BLOCK("barrier");
    // 通信终止信号，同步退出，防止线程卡住
    spdlog::stopwatch sw;
    int exitFlag = isRunning;
    MPI_Allreduce(&exitFlag, &exitFlag, 1, MPI_INT, MPI_SUM, comm[channel][commId]);
    if (exitFlag < rankInfo.rankSize) {
        throw EndRunError("GetScAll end run.");
    }
    EASY_END_BLOCK;
    spdlog::debug(KEY_PROCESS "barrier time:{}", TO_MS(sw));
    // allgather keyScLocal(key all2all keyScLocal = device all2all rc)
    MPI_Allgather(keyScLocal.data(), rankInfo.rankSize, MPI_INT,
                  scAllOut.data(), rankInfo.rankSize, MPI_INT, comm[channel][commId]);
    spdlog::debug("rank {} key scAllOut matrix:\n{}", rankInfo.rankId, scAllOut);
}

void KeyProcess::Key2Offset(const emb_name_t& embName, keys_t& splitKey)
{
    EASY_FUNCTION(profiler::colors::Blue600)
    std::lock_guard<std::mutex> lk(key2OffsetMut); // lock for PROCESS_THREAD
    auto& key2Offset = keyOffsetMap[embName];
    auto& maxOffsetTmp  = maxOffset[embName];
    auto& evictPos = evictPosMap[embName];
    auto& curEmbTable = embeddingTableMap[embName]; // empty when not use dynamic expansion
    for (long& key : splitKey) {
        if (key == -1) {
            if (rankInfo.useDynamicExpansion) {
                key = 0;
            }
            continue;
        }
        const auto& iter = key2Offset.find(key);
        if (iter != key2Offset.end()) {
            // 老值
            key = iter->second;
        } else if (evictPos.size() != 0) {
            size_t offset;
            // 新值, emb有pos可复用
            offset = evictPos.back();
            spdlog::trace("HBM mode, evictPos is not null, name[{}] key [{}] reuse offset [{}], evictSize [{}]!!!",
                          embName, key, offset, evictPos.size());
            key2Offset[key] = offset;
            key = offset;
            evictPos.pop_back();
        } else {
            // 新值
            if (rankInfo.useDynamicExpansion) {
                auto addr = curEmbTable.GetEmbAddress();
                key2Offset[key] = addr;
                key = addr;
                maxOffsetTmp++;
            } else {
                key2Offset[key] = maxOffsetTmp;
                key = maxOffsetTmp++;
            }
        }
    }
    if (!rankInfo.useDynamicExpansion && maxOffsetTmp > embInfos[embName].devVocabSize) {
        spdlog::error("dev cache overflow {}>{}", maxOffsetTmp, embInfos[embName].devVocabSize);
    }
    spdlog::debug("current dev emb usage:{}/{}", maxOffsetTmp, embInfos[embName].devVocabSize);
}

void KeyProcess::Key2OffsetInit(const emb_name_t& embName)
{
    auto& key2Offset = keyOffsetMap[embName];
    auto& offset = maxOffset[embName];
    key2Offset[rankInfo.rankId] = offset; // 0 rank init feature id 0 to offset 0
    offset++;
}

/*
 * 构建恢复向量，以便从去重后的emb向量/key恢复回batch对应的emb向量
 * 输入接收到emb块的偏移blockOffset，batch内每个key在块内的偏移restoreVec
 * 输出恢复向量restoreVec，即batch到keySend（平铺的splitKeys）的映射
 * 实现方案2：用map记录keySend中key和表内index/offset的映射，在恢复emb时直接根据batch的key查询该map即可找到receive
 * emb中的 位置，时间复杂度：O(map构建keySend.size + map查询)，空间复杂度：O(map)
 */
void KeyProcess::BuildRestoreVec(const unique_ptr<emb_batch_t>& batch, const vector<int>& blockOffset,
                                 vector<int>& restoreVec, int hotPosSize) const
{
    EASY_FUNCTION()
    int hotNum = 0;
    bool spdDebug = (spdlog::get_level() == spdlog::level::debug);
    for (size_t i = 0; i < batch->Size(); ++i) {
        const emb_key_t d = batch->sample[i];
        int devId = static_cast<int>(d) & (rankInfo.rankSize - 1);
        if (restoreVec[i] >= hotPosSize) {
            restoreVec[i] += blockOffset[devId];
        } else if (spdDebug) {
            hotNum += 1;
        }
    }
    spdlog::debug("hot num in all:{}/{}", hotNum, batch->Size());
}

class EmptyList : public std::exception {
};

class WrongListTop : public std::exception {
};

template<class T>
T KeyProcess::GetInfo(std::vector<info_list_t<T>>& list, int batch, const string& embName, int channel)
{
    int batchListId = batch % PerfConfig::keyProcessThreadNum;
    std::lock_guard<std::mutex> lockGuard(getInfoMut[batchListId]);
    if (list[batchListId][embName][channel].empty()) {
        spdlog::trace("get info list is empty.");
        throw EmptyList();
    }
    auto topBatch = get<int>(list[batchListId][embName][channel].top());
    if (topBatch < batch) {
        spdlog::warn("wrong batch id, top:{} expect:{}, channel:{}, embName: {}, queue_size:{}, may not clear channel",
                     topBatch, batch, channel, embName, list[batchListId][embName][channel].size());
        this_thread::sleep_for(1s);
    }
    if (topBatch != batch) {
        spdlog::trace("topBatch({}) is not equal batch({}).", topBatch, batch);
        throw WrongListTop();
    }
    auto t = list[batchListId][embName][channel].top();
    list[batchListId][embName][channel].pop();
    return move(t);
}

keys_t KeyProcess::GetLookupKeys(int batch, const string& embName, int channel)
{
    spdlog::stopwatch sw;
    while (true) {
        if (!isRunning) {
            return {};
        }
        if (batch != 0 && channel != 0 && duration_cast<seconds>(sw.elapsed()).count() > KEY_PROCESS_TIMEOUT) {
            spdlog::warn(KEY_PROCESS "getting lookup keys timeout! {}[{}]:{}", embName, channel, batch);
            return {};
        }
        try {
            auto ret = GetInfo(lookupKeysList, batch, embName, channel);
            return get<keys_t>(ret);
        } catch (EmptyList&) {
            spdlog::trace("GetLookupKeys GetInfo failed {}[{}]:{} no input, wait and retry",
                embName, channel, batch);
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            spdlog::trace("GetLookupKeys GetInfo failed {}[{}]:{} wrong top",
                embName, channel, batch);
            this_thread::sleep_for(1ms);
        }
    }
}

unique_ptr<vector<Tensor>> KeyProcess::GetInfoVec(int batch, const string& embName, int channel, ProcessedInfo type)
{
    spdlog::stopwatch sw;
    std::vector<info_list_t<tensor_info_t>>* list;
    switch (type) {
        case ProcessedInfo::ALL2ALL:
            list = &all2AllList;
            break;
        case ProcessedInfo::RESTORE:
            list = &infoList;
            break;
        default:
            throw runtime_error("ERROR list type");
    }
    while (true) {
        if (!isRunning) {
            return nullptr;
        }
        if (batch != 0 && channel != 0 && duration_cast<seconds>(sw.elapsed()).count() > KEY_PROCESS_TIMEOUT) {
            spdlog::warn(KEY_PROCESS "getting lookup keys timeout! {}[{}]:{}", embName, channel, batch);
            return nullptr;
        }
        try {
            auto ret = GetInfo(*list, batch, embName, channel);
            auto it = get<std::list<unique_ptr<vector<Tensor>>>::iterator>(ret);
            auto uTensor = move(*it);
            int batchListId = batch % PerfConfig::keyProcessThreadNum;
            unique_lock<mutex> lockGuard(getInfoMut[batchListId]);
            storage[batchListId].erase(it);
            return uTensor;
        } catch (EmptyList&) {
            spdlog::trace("GetInfoVec GetInfo failed {}[{}]:{} type: {} no input and retry",
                embName, channel, batch, type);
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            spdlog::trace("GetInfoVec GetInfo failed {}[{}]:{} type: {} wrong top",
                embName, channel, batch, type);
            this_thread::sleep_for(1ms);
        }
    }
}

void KeyProcess::SendA2A(const vector<int>& a2aInfo, const string& embName, int channel, int batchId)
{
    // 数据放到队列里，在mgmt里面发送（检查发送数据量）
    auto tensors = make_unique<vector<Tensor>>();
    Tensor tmpTensor(tensorflow::DT_INT64, { rankInfo.rankSize, rankInfo.rankSize });
    auto tmpData = tmpTensor.matrix<int64>();
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        for (int j = 0; j < rankInfo.rankSize; ++j) {
            tmpData(i, j) = a2aInfo[j * rankInfo.rankSize + i];
        }
    }
    tensors->emplace_back(move(tmpTensor));

    int batchListId = batchId % PerfConfig::keyProcessThreadNum;
    std::unique_lock<std::mutex> lockGuard(getInfoMut[batchListId]);
    storage[batchListId].push_front(move(tensors));
    all2AllList[batchListId][embName][channel].push(make_tuple(batchId, embName, storage[batchListId].begin()));
    lockGuard.unlock();
}

int KeyProcess::GetMaxStep(int channelId) const
{
    return rankInfo.maxStep.at(channelId);
}

void KeyProcess::EvictKeys(const string& embName, const vector<emb_key_t>& keys) // hbm
{
    spdlog::info(KEY_PROCESS "hbm funEvictCall: [{}]! keySize:{}", embName, keys.size());

    // 删除映射关系
    if (keys.size() != 0) {
        EvictDeleteDeviceEmb(embName, keys);
    }

    // 初始化 dev
    EvictInitDeviceEmb(embName, evictPosMap.at(embName));
}

void KeyProcess::EvictDeleteDeviceEmb(const string& embName, const vector<emb_key_t>& keys)
{
    EASY_FUNCTION(profiler::colors::Blue600)
    std::lock_guard<std::mutex> lk(key2OffsetMut); // lock for PROCESS_THREAD

    size_t keySize = keys.size();
    auto& devHashMap = keyOffsetMap.at(embName);
    auto& evictPos = evictPosMap.at(embName);

    for (size_t i = 0; i < keySize; i++) {
        size_t offset;
        auto key = keys[i];
        if (key == -1) {
            spdlog::warn("evict key equal -1!");
            continue;
        }
        const auto& iter = devHashMap.find(key);
        if (iter == devHashMap.end()) { // not found
            continue;
        }
        offset = iter->second;
        devHashMap.erase(iter);
        evictPos.emplace_back(offset);
        spdlog::trace("evict embName {} , offset , {}", embName, offset);
    }
    spdlog::info(KEY_PROCESS "hbm EvictDeleteDeviceEmb: [{}]! evict size on dev:{}", embName, evictPos.size());
}

void KeyProcess::EvictInitDeviceEmb(const string& embName, vector<size_t> offset)
{
    if (offset.size() > embInfos[embName].devVocabSize) {
        spdlog::error("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}",
                      embName, offset.size(), embInfos[embName].devVocabSize);
        throw runtime_error(fmt::format("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}",
                                        embName, offset.size(), embInfos[embName].devVocabSize).c_str());
    }
    if (rankInfo.useStatic) {
        offset.resize(embInfos[embName].devVocabSize, -1);
    }

    auto trans = Singleton<HDTransfer>::GetInstance();
    // evict key发送给dev侧，dev侧初始化emb
    auto tmpData = Vec2TensorI32(offset);
    trans->Send(EVICT, { tmpData }, TRAIN_CHANNEL_ID, embName);

    spdlog::info(KEY_PROCESS "hbm EvictInitDeviceEmb: [{}]! send offsetSize:{}", embName, offset.size());
}

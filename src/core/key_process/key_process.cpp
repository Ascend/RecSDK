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
#include <climits>

#include "checkpoint/checkpoint.h"
#include "hd_transfer/hd_transfer.h"
#include "utils/common.h"
#include "utils/time_cost.h"

using namespace std;
using namespace chrono;
using namespace MxRec;
using namespace ock::ctr;

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

bool KeyProcess::Initialize(const RankInfo& rInfo, const vector<EmbInfo>& eInfos,
                            const vector<ThresholdValue>& thresholdValues,
                            int seed)
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
    }
    spdlog::info(KEY_PROCESS "hot emb count info:{}", hotEmbTotCount);
    MPI_Group worldGroup;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    for (auto& i: comm) {
        for (auto& j: i) {
            MPI_Comm_create(MPI_COMM_WORLD, worldGroup, &j);
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

    if (PerfConfig::fastUnique) {
        Factory::Create(factory);
    }

    spdlog::info(KEY_PROCESS "scInfo:{}, localRankSize:{}, rankSize:{}, useStatic:{}, useHot:{}", scInfo,
        rInfo.localRankSize, rInfo.rankSize, rInfo.useStatic, rInfo.useHot);
    return true;
}

// bind and start main process
int KeyProcess::Start()
{
    // bind like:
    // 0 1 2 3 4 5 0 1 2 3 4 5
    // |  rank0  | |  rank1  |
    // each rank creates KEY_PROCESS_THREAD threads, each thread process one batchdata
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
    int threadNum;
    for (int channel = 0; channel < MAX_CHANNEL_NUM; ++channel) {
        const char* threadNumEnv = getenv("KEY_PROCESS_THREAD_NUM");
        if (threadNumEnv != nullptr) {
            threadNum = static_cast<int>(*threadNumEnv) - static_cast<int>('0');
            if (threadNum > KEY_PROCESS_THREAD || threadNum < 0) {
                throw runtime_error(fmt::format("{} is not valid", threadNum));
            }
        } else {
            threadNum = KEY_PROCESS_THREAD;
        }
        spdlog::info(KEY_PROCESS "key process thread num: {}", threadNum);
        for (int id = 0; id < threadNum; ++id) {
            procThreads.emplace_back(
                std::make_unique<std::thread>(fn, channel, id)); // use lambda expression initialize thread
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
    hotEmbTotCount[info.name] = static_cast<int>(static_cast<float>(GetUBSize(rInfo.deviceId) / sizeof(float)) *
                                                 HOT_EMB_CACHE_PCT / static_cast<float>(embeddingSize));
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
    for (auto& i: procThreads) {
        i->join();
    }
    procThreads.clear();
    spdlog::info(KEY_PROCESS "rank {} destroy success.", rankInfo.rankId);
}

void KeyProcess::LoadSaveLock()
{
    for (int channelId { 0 }; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadId { 0 }; threadId < KEY_PROCESS_THREAD; ++threadId) {
            loadSaveMut[channelId][threadId].lock();
        }
    }
}

void KeyProcess::LoadSaveUnlock()
{
    for (int channelId { 0 }; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadId { 0 }; threadId < KEY_PROCESS_THREAD; ++threadId) {
            loadSaveMut[channelId][threadId].unlock();
        }
    }
}

void KeyProcess::GetUniqueConfig(UniqueConf& uniqueConf)
{
    if (rankInfo.rankSize > 0) {
        uniqueConf.useSharding = true;
        uniqueConf.shardingNum = rankInfo.rankSize;
    }

    if (rankInfo.useStatic) {
        uniqueConf.usePadding = true;
        uniqueConf.paddingVal = -1;
    } else {
        uniqueConf.usePadding = false;
    }

    uniqueConf.useIdCount = true;
    uniqueConf.outputType = OutputType::ENHANCED;
    uniqueConf.minThreadNum = MIN_UNIQUE_THREAD_NUM;
    uniqueConf.maxThreadNum = PerfConfig::maxUniqueThreadNum;
}

void KeyProcess::InitializeUnique(UniqueConf& uniqueConf, size_t& preBatchSize, bool& uniqueInitialize,
                                  const unique_ptr <emb_batch_t>& batch, UniquePtr& unique)
{
    uniqueConf.desiredSize = (uint32_t)batch->Size();
    if (preBatchSize != batch->Size()) {
        uniqueInitialize = false;
        preBatchSize = batch->Size();
    }

    if (!uniqueInitialize) {
        if (rankInfo.useStatic) {
            uniqueConf.paddingSize = embInfos[batch->name].sendCount;
        }

        uniqueConf.maxIdVal = INT64_MAX;
        uniqueConf.dataType = ock::ctr::DataType::INT64;

        unique->Initialize(uniqueConf);
        uniqueInitialize = true;
    }
}

void KeyProcess::KeyProcessTask(int channel, int id) // thread id [0, KEY_PROCESS_THREAD-1]
{
    unique_ptr<emb_batch_t> batch;
    UniquePtr unique = nullptr;
    UniqueConf uniqueConf;
    size_t preBatchSize = 0;
    bool uniqueInitialize = false;

    if (PerfConfig::fastUnique) {
        factory->CreateUnique(unique);
        GetUniqueConfig(uniqueConf);
    }

    spdlog::stopwatch sw;
    try {
        while (true) {
            TimeCost getAndProcessTC;
            TimeCost getBatchDataTC;
            batch = GetBatchData(channel, id); // get batch data from SingletonQueue<emb_batch_t>
            TIME_PRINT("getBatchDataTC(ms):{}", getBatchDataTC.ElapsedMS());
            if (batch == nullptr) {
                break;
            }
            auto getBatchTime = Format2Ms(sw);
            sw.reset();

            bool ret = false;
            if (PerfConfig::fastUnique) {
                InitializeUnique(uniqueConf, preBatchSize, uniqueInitialize, batch, unique);
                ret = KeyProcessTaskHelperWithFastUnique(batch, unique, channel, id);
            } else {
                ret = KeyProcessTaskHelper(batch, channel, id);
            }

            if (!ret) {
                break;
            }
            spdlog::info(KEY_PROCESS "getAndProcessTC(ms):{}, key process cost:{}, get data time:{} batch {}[{}]:{} ",
                getAndProcessTC.ElapsedMS(), Format2Ms(sw), getBatchTime, batch->name, batch->channel, batch->batchId);
            auto batchQueue = SingletonQueue<emb_batch_t>::getInstances(id + KEY_PROCESS_THREAD * batch->channel);
            batchQueue->PutDirty(move(batch));
        }
        if (PerfConfig::fastUnique) {
            unique->UnInitialize();
        }
    } catch (const EndRunError &e) {
        spdlog::debug(KEY_PROCESS "abort run: {}", e.what());
    }
    spdlog::info(KEY_PROCESS "KeyProcessTask exit. rank:{} thread:{}, channel:{}", rankInfo.rankId, id, channel);
}

void KeyProcess::HashSplitHelper(const unique_ptr <emb_batch_t>& batch, vector <keys_t>& splitKeys,
                                 vector <int32_t>& restore, vector <int32_t>& hotPos,
                                 vector <vector<uint32_t>>& keyCount)
{
    TimeCost UniqueTC;
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE) {
        tie(splitKeys, restore, keyCount) = HashSplit_withFAAE(batch); // 按存储dev id切分并去重
    } else {
        if (rankInfo.useHot) {
            tie(splitKeys, restore, hotPos) = HotHashSplit(batch);   // 按存储dev id切分并去重
        } else {
            tie(splitKeys, restore) = HashSplit(batch);   // 按存储dev id切分并去重
        }
    }
    TIME_PRINT("UniqueTC(ms):{}", UniqueTC.ElapsedMS());
}

bool KeyProcess::KeyProcessTaskHelperWithFastUnique(unique_ptr<emb_batch_t>& batch, UniquePtr& unique,
                                                    int channel, int id)
{
    // tuple for keyRec restore hotPos scAll countRecv
    isWithFAAE = m_featureAdmitAndEvict.GetFunctionSwitch() &&
                  FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE;
    TimeCost fastUniqueTC;
    UniqueInfo uniqueInfo;
    ProcessBatchWithFastUnique(batch, unique, id, uniqueInfo);
    TIME_PRINT("ProcessBatchWithFastUnique(ms):{}", fastUniqueTC.ElapsedMS());

    // 特征准入&淘汰
    if (isWithFAAE &&
        (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, uniqueInfo.all2AllInfo.keyRecv,
                                             uniqueInfo.all2AllInfo.countRecv)
                                             == FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR)) {
        spdlog::error(KEY_PROCESS "rank:{} thread:{}, channel:{}, Feature-admit-and-evict error ...",
                      rankInfo.rankId, id, channel);
        return false;
    }

    // without host, just device, all embedding vectors were stored in device
    // map key to offset directly by lookup keyOffsetMap (hashmap)
    if (rankInfo.noDDR) {
        TimeCost key2OffsetTC;
        Key2Offset(batch->name, uniqueInfo.all2AllInfo.keyRecv, channel);
        TIME_PRINT("key2OffsetTC(ms):{}", key2OffsetTC.ElapsedMS());
    }
    if (!rankInfo.useStatic) { // Static all2all，need send count
        SendA2A(uniqueInfo.all2AllInfo.scAll, batch->name, batch->channel, batch->batchId);
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
    TimeCost pushResultTC;
    PushResult(batch, move(tensors), uniqueInfo.all2AllInfo.keyRecv);
    TIME_PRINT("pushResultTC(ms):{}", pushResultTC.ElapsedMS());
    return true;
}

bool KeyProcess::KeyProcessTaskHelper(unique_ptr<emb_batch_t>& batch, int channel, int id)
{
    vector<keys_t> splitKeys;
    vector<int32_t> restore;
    vector<int32_t> hotPos;
    vector<vector<uint32_t>> keyCount;

    HashSplitHelper(batch, splitKeys, restore, hotPos, keyCount);
    auto [lookupKeys, scAll, ss] = ProcessSplitKeys(batch, id, splitKeys);

    vector<uint32_t> countRecv;
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE) {
        countRecv = GetCountRecv(batch, id, keyCount, scAll, ss);
    }

    BuildRestoreVec(batch, ss, restore, static_cast<int>(hotPos.size()));

    // 特征准入&淘汰
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE &&
        (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, lookupKeys,
                                             countRecv) == FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR)) {
        spdlog::error(KEY_PROCESS "rank:{} thread:{}, channel:{}, Feature-admit-and-evict error ...",
            rankInfo.rankId, id, channel);
        return false;
    }

    // without host, just device, all embedding vectors were stored in device
    // map key to offset directly by lookup keyOffsetMap (hashmap)
    if (rankInfo.noDDR) {
        if (rankInfo.useDynamicExpansion) {
            Key2OffsetDynamicExpansion(batch->name, lookupKeys, channel);
        } else {
            Key2Offset(batch->name, lookupKeys, channel);
        }
    }

    if (!rankInfo.useStatic) { // Static all2all，need send count
        SendA2A(scAll, batch->name, batch->channel, batch->batchId);
    }

    TimeCost pushResultTC;
    auto tensors = make_unique<vector<Tensor>>();
    tensors->push_back(Vec2TensorI32(restore));
    if (rankInfo.useHot) {
        hotPos.resize(hotEmbTotCount[batch->name], 0);
        tensors->push_back(Vec2TensorI32(hotPos));
    }
    if (rankInfo.noDDR) {
        if (rankInfo.useDynamicExpansion) {
            tensors->push_back(Vec2TensorI64(lookupKeys));
        } else {
            tensors->push_back(Vec2TensorI32(lookupKeys));
        }
    }
    PushResult(batch, move(tensors), lookupKeys);
    TIME_PRINT("pushResultTC(ms):{}", pushResultTC.ElapsedMS());
    return true;
}

vector<uint32_t> KeyProcess::GetCountRecv(const unique_ptr<emb_batch_t>& batch, int id,
                                          vector<vector<uint32_t>>& keyCount, vector<int> scAll, vector<int> ss)
{
    TimeCost getCountRecvTC;
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
    TIME_PRINT("getCountRecvTC(ms)(with-all2all):{}", getCountRecvTC.ElapsedMS());
    return countRecv;
}

void KeyProcess::PushResult(unique_ptr<emb_batch_t>& batch, unique_ptr<vector<Tensor>> tensors,
                            keys_t& lookupKeys)
{
    std::unique_lock<std::mutex> lockGuard(mut);
    storage.push_front(move(tensors));
    infoList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, storage.begin()));
    if (!rankInfo.noDDR) {
        lookupKeysList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, move(lookupKeys)));
    }
    lockGuard.unlock();
}

/*
 * 从共享队列SingletonQueue<emb_batch_t>中读取batch数据并返回。batch数据由 ReadEmbKeyV2 写入。
 * commID为线程标识[0, KEY_PROCESS_THREAD-1]，不同线程、训练或推理数据用不同的共享队列通信
 */
unique_ptr<emb_batch_t> KeyProcess::GetBatchData(int channel, int commId)
{
    EASY_FUNCTION()
    unique_ptr<emb_batch_t> batch = nullptr;

    // train data, queue id = thread id [0, KEY_PROCESS_THREAD-1]
    auto batchQueue = SingletonQueue<emb_batch_t>::getInstances(commId + KEY_PROCESS_THREAD * channel);
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
    spdlog::debug(KEY_PROCESS "rank {} thread {} get batch {}[{}]:{} done. bs:{} sample:[{}]",
        rankInfo.rankId, commId, batch->name, batch->channel, batch->batchId, batch->Size(),
        batch->UnParse());
#if defined(PROFILING) && defined(BUILD_WITH_EASY_PROFILER)
    if (batch->batchId == PROFILING_START_BATCH_ID) {
        EASY_PROFILER_ENABLE
    } else if (batch->batchId == PROFILING_END_BATCH_ID) {
        ::profiler::dumpBlocksToFile(fmt::format("/home/MX_REC-profile-{}.prof", rankInfo.rankId).c_str());
    }
#endif
    return batch;
}

size_t KeyProcess::GetKeySize(const unique_ptr<emb_batch_t> &batch)
{
    size_t size = rankInfo.rankSize * embInfos[batch->name].sendCount;
    if (!rankInfo.useStatic) {
        size = batch->Size();
    }
    return size;
}

void KeyProcess::ProcessBatchWithFastUnique(const unique_ptr<emb_batch_t> &batch, UniquePtr& unique,
                                            int id, UniqueInfo& uniqueInfoOut)
{
    EASY_FUNCTION(profiler::colors::Purple)
    EASY_VALUE("batchId", batch->batchId)

    EASY_BLOCK("ock-unique")
    TimeCost uniqueTC;

    KeySendInfo keySendInfo;
    size_t size = GetKeySize(batch);
    keySendInfo.keySend.resize(size);
    vector<int> splitSize(rankInfo.rankSize);
    vector<int64_t> uniqueVector(batch->Size());
    uniqueInfoOut.restore.resize(batch->Size());
    vector<int32_t> idCount(batch->Size());
    keySendInfo.keyCount.resize(size);

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = (uint32_t)batch->Size();
    uniqueIn.inputId = reinterpret_cast<void *>(batch->sample.data());

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(keySendInfo.keySend.data());
    uniqueOut.index = (uint32_t*)uniqueInfoOut.restore.data();
    if (rankInfo.useStatic) {
        uniqueOut.idCnt = idCount.data();
        uniqueOut.idCntFill = keySendInfo.keyCount.data();
    } else {
        uniqueOut.idCnt = keySendInfo.keyCount.data();
    }
    uniqueOut.uniqueIdCntInBucket = splitSize.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueVector.data());
    uniqueOut.uniqueIdCnt = 0;

    int ret = unique->DoEnhancedUnique(uniqueIn, uniqueOut);
    EASY_END_BLOCK
    TIME_PRINT("FastUniqueCompute(ms):{}, ret:{}", uniqueTC.ElapsedMS(), ret);

    vector<int> sc;
    HandleHotAndSendCount(batch, uniqueInfoOut, keySendInfo, sc, splitSize);

    All2All(sc, id, batch->channel, keySendInfo, uniqueInfoOut.all2AllInfo);

    spdlog::debug(KEY_PROCESS "ProcessBatchWithFastUnique get batchId:{}, batchSize:{}, channel:{}, "
                             "name:{}, restore:{}, keyCount:{}", batch->batchId, batch->Size(),
                             batch->channel, batch->name, uniqueInfoOut.restore.size(), keySendInfo.keyCount.size());
}

void KeyProcess::HandleHotAndSendCount(const unique_ptr<emb_batch_t> &batch, UniqueInfo& uniqueInfoOut,
                                       KeySendInfo& keySendInfo, vector<int>& sc, vector<int>& splitSize)
{
    std::shared_lock<std::shared_mutex> lock(g_smut);
    auto hotMap = hotKey[batch->name];
    lock.unlock();

    if (rankInfo.useHot) {
        int hotOffset = 0;
        uniqueInfoOut.hotPos.resize(hotEmbTotCount[batch->name]);
        hotOffset = hotEmbTotCount[batch->name];
    
        TimeCost ComputeHotTc;
        ComputeHotPos(batch, hotMap, uniqueInfoOut.hotPos, uniqueInfoOut.restore, hotOffset);
        TIME_PRINT("ComputeHot TimeCost(ms):{}", ComputeHotTc.ElapsedMS());
        UpdateHotMapForUnique(keySendInfo.keySend, keySendInfo.keyCount,
                              hotOffset, batch->batchId % hotEmbUpdateStep == 0, batch->name);
    }

    if (rankInfo.useStatic) {
        sc.resize(rankInfo.rankSize, embInfos[batch->name].sendCount);
    } else {
        sc.resize(rankInfo.rankSize);
        for (int i = 0;i < rankInfo.rankSize; i++) {
            sc[i] = splitSize[i];
        }
    }
}

void KeyProcess::ComputeHotPos(const unique_ptr<emb_batch_t> &batch, absl::flat_hash_map<emb_key_t, int> &hotMap,
                               vector<int> &hotPos, vector<int32_t> &restore, const int hotOffset)
{
    auto* inputData = batch->sample.data();
    size_t miniBs = batch->Size();

    int hotCount = 0;
    for (size_t i = 0;i < miniBs; i++) {
        const emb_key_t& key = inputData[i];
        auto hot = hotMap.find(key);
        if (hot != hotMap.end()) {
            if (hot->second == -1) {
                hotPos[hotCount] = restore[i];
                hot->second = hotCount;
                restore[i] = hotCount++;
            } else {
                restore[i] = hot->second;
            }
        } else {
            restore[i] += hotOffset;
        }
    }
}

void KeyProcess::All2All(vector<int>& sc, int id, int channel, KeySendInfo& keySendInfo,
                         All2AllInfo& all2AllInfoOut)
{
    TimeCost getScAllTC;
    GetScAllForUnique(sc, id, channel, all2AllInfoOut.scAll); // Allgather通信获取所有（不同rank相同thread id的）
    TIME_PRINT("GetScAll TimeCost(ms):{}", getScAllTC.ElapsedMS());

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
    TimeCost processSplitKeysTC;
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

    TimeCost getScAllTC;
    auto scAll = GetScAll(sc, id, batch->channel);    // Allgather通信获取所有（不同rank相同thread id的）线程间通信量矩阵
    TIME_PRINT("getScAllTC(ms)(AllReduce-AllGather):{}", getScAllTC.ElapsedMS());

    auto ss = Count2Start(sc);  // send displays/offset 发送数据的起始偏移量
    vector<int> rc; // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        // 通信量矩阵某一列的和即为本地要从其他设备接受的key数据量
        rc.push_back(scAll.at(i * rankInfo.rankSize + rankInfo.rankId));
    }
    auto rs = Count2Start(rc); // receive displays/offset 接受数据的起始偏移量
    keyRecv.resize(rs.back() + rc.back());
    spdlog::trace(KEY_PROCESS "MPI_Alltoallv begin. rank {} thread {} batch {} {}", rankInfo.rankId, id, batch->batchId,
        batch->name);
    EASY_BLOCK("all2all")

    TimeCost uniqueAll2AllTC;
    MPI_Alltoallv(keySend.data(), sc.data(), ss.data(), MPI_INT64_T,
                  keyRecv.data(), rc.data(), rs.data(), MPI_INT64_T,
                  comm[batch->channel][id]);
    TIME_PRINT("uniqueAll2AllTC(ms):{}", uniqueAll2AllTC.ElapsedMS());

    EASY_END_BLOCK
    spdlog::trace(KEY_PROCESS "MPI_Alltoallv finish. rank {} thread {} batch {} {}",
        rankInfo.rankId, id, batch->batchId, batch->name);
    TIME_PRINT("processSplitKeysTC(ms):{}", processSplitKeysTC.ElapsedMS());
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
                // pos in lookup vec (need add ss) for hot-gather
                hotPos[hotCount] = static_cast<int>(splitKeys[devId].size()) - 1;
                hotPosDev[hotCount] = devId; // which dev, for get ss
                hot->second = hotCount;
                restore[i] = hotCount++; // get pos of hot emb
            } else {
                restore[i] = hot->second;
            }
        } else { // is not hot key
            // restore记录去重后key在桶内偏移量（用于计算恢复向量）
            restore[i] = static_cast<int32_t>(splitKeys[devId].size() + (hotOffset - 1));
        }
        uKey[key] = restore[i];
    }

    UpdateHotMap(keyCountMap, hotEmbTotCount[batch->name], batch->batchId % hotEmbUpdateStep == 0, batch->name);
    AddCountStartToHotPos(splitKeys, hotPos, hotPosDev, batch);

    return { splitKeys, restore, hotPos };
}

void KeyProcess::AddCountStartToHotPos(vector<keys_t>& splitKeys, vector<int>& hotPos, const vector<int>& hotPosDev,
                                       const unique_ptr<emb_batch_t>& batch)
{
    vector<int> splitKeysSize {};
    if (rankInfo.useStatic) {
        for (size_t i = 0; i < splitKeys.size(); i++) {
            splitKeysSize.push_back(embInfos[batch->name].sendCount);
        }
    } else {
        for (auto& splitKey: splitKeys) {
            splitKeysSize.push_back(static_cast<int>(splitKey.size()));
        }
    }
    auto cs = Count2Start(splitKeysSize);
    for (size_t i = 0; i < hotPos.size(); ++i) {
        hotPos[i] += cs[hotPosDev[i]];
    }
}

void KeyProcess::UpdateHotMapForUnique(const keys_t &keySend, const vector<int32_t> &keyCount,
                                       uint32_t count, bool refresh, const string& embName)
{
    auto& hotMap = hotKey[embName];
    if (refresh) {
        priority_queue<pair<int, emb_key_t>> pq;
        for (size_t i = 0;i < keySend.size(); ++i) {
            if (keySend[i] == -1) {
                continue;
            }
            pq.push(pair<int, emb_key_t>(-keyCount[i], keySend[i]));
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
vector<int> KeyProcess::GetScAll(const vector<int>& keyScLocal, int commId, int channel) const
{
    EASY_FUNCTION()
    vector<int> scAll;
    scAll.resize(rankInfo.rankSize * rankInfo.rankSize);
    EASY_BLOCK("barrier");
    // 通信终止信号，同步退出，防止线程卡住
    spdlog::stopwatch sw;
    int exitFlag = isRunning;
    MPI_Allreduce(&exitFlag, &exitFlag, 1, MPI_INT, MPI_SUM, comm[channel][commId]);
    if (exitFlag < rankInfo.rankSize) {
        throw EndRunError("GetScAll end run.");
    }
    EASY_END_BLOCK;
    spdlog::debug(KEY_PROCESS "barrier time:{}", Format2Ms(sw));
    // allgather keyScLocal(key all2all keyScLocal = device all2all rc)
    MPI_Allgather(keyScLocal.data(), rankInfo.rankSize, MPI_INT,
                  scAll.data(), rankInfo.rankSize, MPI_INT, comm[channel][commId]);
    spdlog::debug("rank {} key scAll matrix:\n{}", rankInfo.rankId, scAll);
    return scAll;
}

void KeyProcess::GetScAllForUnique(const vector<int>& keyScLocal, int commId, int channel, vector<int> &scAllOut) const
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
    spdlog::debug(KEY_PROCESS "barrier time:{}", duration_cast<milliseconds>((sw).elapsed()));
    // allgather keyScLocal(key all2all keyScLocal = device all2all rc)
    MPI_Allgather(keyScLocal.data(), rankInfo.rankSize, MPI_INT,
                  scAllOut.data(), rankInfo.rankSize, MPI_INT, comm[channel][commId]);
    spdlog::debug("rank {} key scAllOut matrix:\n{}", rankInfo.rankId, scAllOut);
}

void KeyProcess::Key2Offset(const emb_name_t& embName, keys_t& splitKey, int channel)
{
    TimeCost key2OffsetTC;
    EASY_FUNCTION(profiler::colors::Blue600)
    std::lock_guard<std::mutex> lk(mut); // lock for PROCESS_THREAD
    auto& key2Offset = keyOffsetMap[embName];
    auto& maxOffsetTmp  = maxOffset[embName];
    auto& evictPos = evictPosMap[embName];
    for (long& key : splitKey) {
        if (key == -1) {
            continue;
        }
        const auto& iter = key2Offset.find(key);
        if (iter != key2Offset.end()) {
            key = iter->second;
        } else if (evictPos.size() != 0 && channel == TRAIN_CHANNEL_ID) {
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
            if (channel == TRAIN_CHANNEL_ID) {
                key2Offset[key] = maxOffsetTmp;
                key = maxOffsetTmp++;
            } else {
                key = INVALID_KEY_VALUE;
            }
        }
    }
    if (maxOffsetTmp > embInfos[embName].devVocabSize) {
        spdlog::error("dev cache overflow {}>{}", maxOffsetTmp, embInfos[embName].devVocabSize);
        throw std::runtime_error("dev cache overflow!");
    }
    spdlog::debug("current dev emb usage:{}/{}", maxOffsetTmp, embInfos[embName].devVocabSize);
    TIME_PRINT("key2OffsetTC(ms):{}", key2OffsetTC.ElapsedMS());
}

void KeyProcess::Key2OffsetDynamicExpansion(const emb_name_t& embName, keys_t& splitKey, int channel)
{
    TimeCost key2OffsetTC;
    EASY_FUNCTION(profiler::colors::Blue600)
    std::lock_guard<std::mutex> lk(mut); // lock for PROCESS_THREAD
    auto& key2Offset = keyOffsetMap[embName];
    auto& maxOffsetTmp  = maxOffset[embName];
    auto& curEmbTable = embeddingTableMap[embName]; // empty when not use dynamic expansion
    for (long& key : splitKey) {
        if (key == -1) {
            key = 0;
            continue;
        }
        const auto& iter = key2Offset.find(key);
        if (iter != key2Offset.end()) {
            key = iter->second;
        } else {
            // 新值
            if (channel == TRAIN_CHANNEL_ID) {
#ifndef GTEST
                auto addr = curEmbTable.GetEmbAddress();
                key2Offset[key] = addr;
                key = addr;
#endif
                maxOffsetTmp++;
                continue;
            }
            key = 0;
        }
    }
    spdlog::debug("current dev emb usage:{}/{}", maxOffsetTmp, embInfos[embName].devVocabSize);
    TIME_PRINT("key2OffsetTC(ms):{}", key2OffsetTC.ElapsedMS());
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
    TimeCost buildRestoreVecTC;
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
    TIME_PRINT("buildRestoreVecTC(ms):{}", buildRestoreVecTC.ElapsedMS());
}

class EmptyList : public std::exception {
};

class WrongListTop : public std::exception {
};

template<class T>
T KeyProcess::GetInfo(info_list_t<T>& list, int batch, const string& embName, int channel)
{
    std::lock_guard<std::mutex> lockGuard(mut);
    if (list[embName][channel].empty()) {
        spdlog::trace("get info list is empty.");
        throw EmptyList();
    }
    auto topBatch = get<int>(list[embName][channel].top());
    if (topBatch < batch) {
        spdlog::error("wrong batch id, top:{} getting:{}, channel:{}, may not clear channel", topBatch,
                      batch, channel);
        this_thread::sleep_for(1s);
    }
    if (topBatch != batch) {
        spdlog::trace("topBatch({}) is not equal batch({}).", topBatch, batch);
        throw WrongListTop();
    }
    auto t = list[embName][channel].top();
    list[embName][channel].pop();
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
            spdlog::trace("getting info failed {}[{}]:{}", embName, channel, batch);
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            spdlog::trace("getting info failed {}[{}]:{} wrong top", embName, channel, batch);
            this_thread::sleep_for(1ms);
        }
    }
}

unique_ptr<vector<Tensor>> KeyProcess::GetInfoVec(int batch, const string& embName, int channel, ProcessedInfo type)
{
    spdlog::stopwatch sw;
    info_list_t<tensor_info_t>* list;
    switch (type) {
        case ProcessedInfo::ALL2ALL:
            list = &all2AllList;
            break;
        case ProcessedInfo::RESTORE:
            list = &infoList;
            break;
        default:
            throw std::invalid_argument("Invalid ProcessedInfo Type.");
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
            std::unique_lock<std::mutex> lockGuard(mut);
            storage.erase(it);
            return uTensor;
        } catch (EmptyList&) {
            spdlog::trace("getting info failed {}[{}]:{}", embName, channel, batch);
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            spdlog::trace("getting info failed {}[{}]:{} wrong top", embName, channel, batch);
            this_thread::sleep_for(1ms);
        }
    }
}

void KeyProcess::SendA2A(const vector<int>& a2aInfo, const string& embName, int channel, int batch)
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

    std::unique_lock<std::mutex> lockGuard(mut);
    storage.push_front(move(tensors));
    all2AllList[embName][channel].push(make_tuple(batch, embName, storage.begin()));
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
    std::lock_guard<std::mutex> lk(mut); // lock for PROCESS_THREAD

    size_t keySize = keys.size();
    auto& devHashMap = keyOffsetMap.at(embName);
    auto& evictPos = evictPosMap.at(embName);

    for (size_t i = 0; i < keySize; i++) {
        size_t offset;
        auto key = keys[i];
        if (key == -1) {
            spdlog::error("evict key equal -1!");
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

    vector<Tensor> tmpDataOut;
    Tensor tmpData = Vec2TensorI32(offset);
    tmpDataOut.emplace_back(tmpData);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));

    auto evictLen = tmpDataOut.back().flat<int32>();
    auto evictSize = static_cast<int>(offset.size());
    evictLen(0) = evictSize;

    // evict key发送给dev侧，dev侧初始化emb
    auto trans = Singleton<HDTransfer>::GetInstance();
    trans->Send(TransferChannel::EVICT, tmpDataOut, TRAIN_CHANNEL_ID, embName);

    spdlog::info(KEY_PROCESS "hbm EvictInitDeviceEmb: [{}]! send offsetSize:{}", embName, offset.size());
}

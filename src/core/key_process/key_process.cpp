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
#include "key_process.h"

#include <cstddef>
#include <iostream>

#include <mpi.h>

#include "emb_table/embedding_mgmt.h"
#include "hd_transfer/hd_transfer.h"
#include "ock_ctr_common/include/error_code.h"
#include "utils/common.h"
#include "utils/config.h"
#include "utils/logger.h"
#include "utils/safe_queue.h"
#include "utils/singleton.h"
#include "utils/time_cost.h"

using namespace std;
using namespace chrono;
using namespace MxRec;

static shared_mutex g_smut;

void KeyProcess::SetupHotEmbUpdateStep()
{
    this->hotEmbUpdateStep = GlobalEnv::hotEmbUpdateStep;
}

bool KeyProcess::Initialize(const RankInfo& rInfo, const vector<EmbInfo>& eInfos,
                            const vector<ThresholdValue>& thresholdValues,
                            int seed, bool isIncrementalCkpt)
{
    readySendEosCnt[TRAIN_CHANNEL_ID].store(0);
    readySendEosCnt[EVAL_CHANNEL_ID].store(0);
    finishSendEosCnt[TRAIN_CHANNEL_ID].store(0);
    finishSendEosCnt[EVAL_CHANNEL_ID].store(0);

    this->rankInfo = rInfo;

    SetupHotEmbUpdateStep();

    map<EmbNameT, int> scInfo;
    for (const auto& info : eInfos) {
        embInfos[info.name] = info;
        scInfo[info.name] = info.sendCount;
        InitHotEmbTotCount(info, rInfo);
    }

    LOG_INFO(KEY_PROCESS "hot emb count info:{}", MapToString(hotEmbTotCount));
    MPI_Group worldGroup;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    for (auto& i : comm) {
        for (auto& j : i) {
            MPI_Comm_create(MPI_COMM_WORLD, worldGroup, &j);
        }
    }
    isRunning = true;
    isIncrementalCheckpoint = isIncrementalCkpt;

    // 特征准入与特征淘汰
    if (!thresholdValues.empty()) {
        m_featureAdmitAndEvict.SetFunctionSwitch(true);
        m_featureAdmitAndEvict.Init(thresholdValues);
    } else {
        m_featureAdmitAndEvict.SetFunctionSwitch(false);
        LOG_WARN(KEY_PROCESS "Feature admit-and-evict function is unavailable ...");
    }

    LOG_INFO(KEY_PROCESS "scInfo:{}, localRankSize:{}, rankSize:{}, useStatic:{}", MapToString(scInfo),
             rInfo.localRankSize, rInfo.rankSize, rInfo.useStatic);
#ifndef GTEST
    Start();
#endif
    return true;
}

// bind and start main process
int KeyProcess::Start()
{
    // bind like:
    // 0 1 2 3 4 5 0 1 2 3 4 5
    // |  rank0  | |  rank1  |
    // each rank creates KEY_PROCESS_THREAD threads, each thread process one batchdata
    LOG_INFO("CPU Core Num: {}", sysconf(_SC_NPROCESSORS_CONF));  // 查看CPU核数
    auto fn = [this](int channel, int threadId) {
#ifndef GTEST
        auto ret = aclrtSetDevice(static_cast<int32_t>(rankInfo.deviceId));
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("Set device failed, device_id:{}", rankInfo.deviceId);
            return;
        }
#endif
        if (GlobalEnv::fastUnique) {
            KeyProcessTaskWithFastUnique(channel, threadId);
        } else {
            KeyProcessTask(channel, threadId);
        }
    };  // for clean code
    int threadNum = GetThreadNumEnv();
    for (int channel = 0; channel < MAX_CHANNEL_NUM; ++channel) {
        LOG_INFO(KEY_PROCESS "key process thread num: {}", threadNum);
        for (int id = 0; id < threadNum; ++id) {
            // use lambda expression initialize thread
            procThreads.emplace_back(std::make_unique<std::thread>(fn, channel, id));
        }
    }
    return 0;
}

void KeyProcess::InitHotEmbTotCount(const EmbInfo& info, const RankInfo& rInfo)
{
    hotEmbTotCount[info.name] = static_cast<int>(static_cast<float>(GetUBSize(rInfo.deviceId) / sizeof(float)) *
                                                 HOT_EMB_CACHE_PCT / static_cast<float>(info.embeddingSize));
}

OffsetMemT KeyProcess::GetMaxOffset()
{
    return EmbeddingMgmt::Instance()->GetMaxOffset();
}

KeyOffsetMemT KeyProcess::GetKeyOffsetMap()
{
    return keyOffsetMap;
}

KeyCountMemT KeyProcess::GetKeyCountMap()
{
    return keyCountMap;
}

FeatureAdmitAndEvict& KeyProcess::GetFeatAdmitAndEvict()
{
    return m_featureAdmitAndEvict;
}

void KeyProcess::LoadMaxOffset(OffsetMemT& loadData)
{
    maxOffset = std::move(loadData);
}

/// 加载每张表key到offset的映射
/// \param loadData
void KeyProcess::LoadKeyOffsetMap(KeyOffsetMemT& loadData)
{
    keyOffsetMap = std::move(loadData);
}

void KeyProcess::LoadKeyCountMap(KeyCountMemT& loadData)
{
    keyCountMap = std::move(loadData);
}

// 只在python侧当训练结束时调用，如果出现死锁直接结束程序即可,测试时让进程等待足够长的时间再调用
void KeyProcess::Destroy()
{
    isRunning = false;
    LOG_INFO(KEY_PROCESS "rankId:{} KeyProcess begin destroy.", rankInfo.rankId);
    for (auto& i : procThreads) {
        i->join();
    }
    procThreads.clear();
    LOG_INFO(KEY_PROCESS "rankId:{} KeyProcess destroy success.", rankInfo.rankId);
}

/// 每个数据通道的所有数据处理线程上锁
void KeyProcess::LoadSaveLock()
{
    for (int channelId{0}; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadId{0}; threadId < MAX_KEY_PROCESS_THREAD; ++threadId) {
            loadSaveMut[channelId][threadId].lock();
        }
    }
}

/// 每个数据通道的所有数据处理线程释放锁
void KeyProcess::LoadSaveUnlock()
{
    for (int channelId{0}; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadId{0}; threadId < MAX_KEY_PROCESS_THREAD; ++threadId) {
            loadSaveMut[channelId][threadId].unlock();
        }
    }
}

void KeyProcess::GetUniqueConfig(ock::ctr::UniqueConf& uniqueConf)
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
    uniqueConf.outputType = ock::ctr::OutputType::ENHANCED;
    uniqueConf.minThreadNum = MIN_UNIQUE_THREAD_NUM;
    uniqueConf.maxThreadNum = GlobalEnv::maxUniqueThreadNum;
}

void KeyProcess::InitializeUnique(ock::ctr::UniqueConf& uniqueConf, size_t& preBatchSize, bool& uniqueInitialize,
                                  const unique_ptr<EmbBatchT>& batch, ock::ctr::UniquePtr& unique)
{
    uniqueConf.desiredSize = static_cast<uint32_t>(batch->Size());
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

        int ret = unique->Initialize(uniqueConf);
        if (ret != ock::ctr::H_OK) {
            throw runtime_error(Logger::Format("fast unique init failed, code:{}", ret));
        }
        uniqueInitialize = true;
    }
}

void KeyProcess::KeyProcessTaskWithFastUnique(int channel, int threadId)
{
    unique_ptr<EmbBatchT> batch;
    ock::ctr::UniquePtr unique = nullptr;
    ock::ctr::UniqueConf uniqueConf;
    size_t preBatchSize = 0;
    bool uniqueInitialize = false;

    int ret = factory->CreateUnique(unique);
    if (ret != ock::ctr::H_OK) {
        throw runtime_error(Logger::Format("create fast unique failed, error code:{}", ret));
    }
    GetUniqueConfig(uniqueConf);

    try {
        while (true) {
            TimeCost getAndProcessTC;
            TimeCost getBatchDataTC;
            batch = GetBatchData(channel, threadId); // get batch data from SingletonQueue<EmbBatchT>
            LOG_DEBUG("getBatchDataTC(ms):{}", getBatchDataTC.ElapsedMS());
            if (batch == nullptr) {
                break;
            }
            size_t getBatchTime = getBatchDataTC.ElapsedMS();
            TimeCost processDataTime = TimeCost();

            InitializeUnique(uniqueConf, preBatchSize, uniqueInitialize, batch, unique);
            if (!KeyProcessTaskHelperWithFastUnique(batch, unique, channel, threadId)) {
                break;
            }
            LOG_INFO(KEY_PROCESS "getAndProcessTC(ms):{}, key process with fast unique cost:{},"
                                 " get data time(ms):{}, batch name:{}, channelId:{}, threadId:{}, batchId:{}",
                     getAndProcessTC.ElapsedMS(), processDataTime.ElapsedMS(), getBatchTime, batch->name,
                     batch->channel, threadId, batch->batchId);
            int queueIndex = threadId + (MAX_KEY_PROCESS_THREAD * batch->channel);
            auto batchQueue = SingletonQueue<EmbBatchT>::GetInstances(queueIndex);
            batchQueue->PutDirty(move(batch));
        }
        unique->UnInitialize();
    } catch (const EndRunExit& e) {
        LOG_INFO(KEY_PROCESS "channel: {}, thread: {}, abort run: {}", channel, threadId, e.what());
    }
    LOG_INFO(KEY_PROCESS "KeyProcessTaskWithFastUnique exit. rank:{} channelId:{}, threadId:{}", rankInfo.rankId,
             channel, threadId);
}

void KeyProcess::KeyProcessTask(int channel, int threadId)
{
    unique_ptr<EmbBatchT> batch;
    try {
        while (true) {
            TimeCost getAndProcessTC;
            TimeCost getBatchDataTC;
            batch = GetBatchData(channel, threadId);  // get batch data from SingletonQueue<EmbBatchT>
            LOG_DEBUG("getBatchDataTC(ms):{}", getBatchDataTC.ElapsedMS());
            if (batch == nullptr) {
                break;
            }
            size_t getBatchTime = getBatchDataTC.ElapsedMS();
            TimeCost processDataTime = TimeCost();

            if (!KeyProcessTaskHelper(batch, channel, threadId)) {
                break;
            }
            LOG_INFO(KEY_PROCESS "getAndProcessTC(ms):{}, key process cost:{},"
                                 " get data time(ms):{}, batch name:{}, channelId:{}, threadId:{}, batchId:{}",
                     getAndProcessTC.ElapsedMS(), processDataTime.ElapsedMS(), getBatchTime, batch->name,
                     batch->channel, threadId, batch->batchId);
            int queueIndex = threadId + (MAX_KEY_PROCESS_THREAD * batch->channel);
            auto batchQueue = SingletonQueue<EmbBatchT>::GetInstances(queueIndex);
            batchQueue->PutDirty(move(batch));
        }
    } catch (const EndRunExit& e) {
        LOG_INFO(KEY_PROCESS "channel: {}, thread: {}, abort run: {}", channel, threadId, e.what());
    }
    LOG_INFO(KEY_PROCESS "KeyProcessTask exit. rank:{} channelId:{}, threadId:{}", rankInfo.rankId, channel, threadId);
}

void KeyProcess::HashSplitHelper(const unique_ptr <EmbBatchT>& batch, vector <KeysT>& splitKeys,
                                 vector <int32_t>& restore, vector <int32_t>& hotPos,
                                 vector <vector<uint32_t>>& keyCount, vector<emb_key_t>& keyCountVec)
{
    TimeCost uniqueTc;
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE) {
        tie(splitKeys, restore, keyCount) = HashSplitWithFAAE(batch);  // 按存储dev id切分并去重
    } else {
        tie(splitKeys, restore, hotPos, keyCountVec) = HotHashSplit(batch);   // 按存储dev id切分并去重
    }
    LOG_DEBUG("uniqueTc(ms):{}", uniqueTc.ElapsedMS());
}

bool KeyProcess::KeyProcessTaskHelperWithFastUnique(unique_ptr<EmbBatchT>& batch, ock::ctr::UniquePtr& unique,
                                                    int channel, int threadId)
{
    // tuple for keyRec restore hotPos scAll countRecv
    isWithFAAE = m_featureAdmitAndEvict.GetFunctionSwitch() &&
                 FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE;
    TimeCost totalTimeCost = TimeCost();
    TimeCost fastUniqueTC;
    UniqueInfo uniqueInfo;
    ProcessBatchWithFastUnique(batch, unique, threadId, uniqueInfo);
    LOG_DEBUG("ProcessBatchWithFastUnique(ms):{}", fastUniqueTC.ElapsedMS());

    // 特征准入&淘汰
    if (isWithFAAE && (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, uniqueInfo.all2AllInfo.keyRecv,
                                                           uniqueInfo.all2AllInfo.countRecv) ==
                       FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR)) {
        LOG_ERROR(KEY_PROCESS "rank:{} thread:{}, channel:{}, Feature-admit-and-evict error ...", rankInfo.rankId,
                  threadId, channel);
        return false;
    }
    std::lock_guard<std::mutex> lock(loadSaveMut[channel][threadId]);
    // without host, just device, all embedding vectors were stored in device
    // map key to offset directly by lookup keyOffsetMap (hashmap)

    RecordKeyCountMap(batch);
    if (!rankInfo.isDDR) {
        TimeCost key2OffsetTC;
        EmbeddingMgmt::Instance()->Key2Offset(batch->name, uniqueInfo.all2AllInfo.keyRecv, channel);
        LOG_DEBUG("key2OffsetTC(ms):{}", key2OffsetTC.ElapsedMS());
    }
    // Static all2all，need send count
    if (!rankInfo.useStatic) {
        SendA2A(uniqueInfo.all2AllInfo.scAll, batch->name, batch->channel, batch->batchId);
    }

    TimeCost pushResultTC;
    auto tensors = make_unique<vector<Tensor>>();
    tensors->push_back(Vec2TensorI32(uniqueInfo.restore));

    uniqueInfo.hotPos.resize(hotEmbTotCount[batch->name], -1);
    tensors->push_back(Vec2TensorI32(uniqueInfo.hotPos));

    if (!rankInfo.isDDR) {
        PushGlobalUniqueTensors(move(tensors), uniqueInfo.all2AllInfo.keyRecv, channel);
        tensors->push_back(rankInfo.useDynamicExpansion ? Vec2TensorI64(uniqueInfo.all2AllInfo.keyRecv)
                                                        : Vec2TensorI32(uniqueInfo.all2AllInfo.keyRecv));
        PushResultHBM(batch, move(tensors));
    } else {
        std::vector<uint64_t> lookupKeysUint(uniqueInfo.all2AllInfo.keyRecv.begin(),
                                             uniqueInfo.all2AllInfo.keyRecv.end());
        vector<uint64_t> uniqueKeys;
        vector<int32_t> restoreVecSec;
        GlobalUnique(lookupKeysUint, uniqueKeys, restoreVecSec);
        PushResultDDR(batch, move(tensors), uniqueKeys, restoreVecSec);
    }

    LOG_DEBUG("pushResultTC(ms):{}", pushResultTC.ElapsedMS());
    if (GlogConfig::gStatOn) {
        LOG_INFO(STAT_INFO "channel_id {} batch_id {} rank_id {} key_process_time_cost_with_fast_unique {}", channel,
                 batch->batchId, rankInfo.rankId, totalTimeCost.ElapsedMS());
    }
    return true;
}

bool KeyProcess::KeyProcessTaskHelper(unique_ptr<EmbBatchT>& batch, int channel, int threadId)
{
    vector<KeysT> splitKeys;
    vector<int32_t> restore;
    vector<int32_t> hotPos;
    vector<vector<uint32_t>> keyCount;
    vector<emb_key_t> keyCountVec;
    TimeCost totalTimeCost = TimeCost();
    HashSplitHelper(batch, splitKeys, restore, hotPos, keyCount, keyCountVec);
    auto [lookupKeys, scAll, ss] = ProcessSplitKeys(batch, threadId, splitKeys);

    vector<uint32_t> countRecv;
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE) {
        countRecv = GetCountRecv(batch, threadId, keyCount, scAll, ss);
    }
    std::lock_guard<std::mutex> lock(loadSaveMut[channel][threadId]);
    RecordKeyCountMap(batch);
    BuildRestoreVec(batch, ss, restore, static_cast<int>(hotPos.size()));

    // 特征准入&淘汰
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE &&
        (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, lookupKeys, countRecv) ==
         FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR)) {
        LOG_ERROR(KEY_PROCESS "rank:{} thread:{}, channel:{}, Feature-admit-and-evict error ...", rankInfo.rankId,
                  threadId, channel);
        return false;
    }

    // without host, just device, all embedding vectors were stored in device
    // map key to offset directly by lookup keyOffsetMap (hashmap)
    if (!rankInfo.isDDR) { EmbeddingMgmt::Instance()->Key2Offset(batch->name, lookupKeys, channel); }

    // Static all2all，need send count
    if (!rankInfo.useStatic) { SendA2A(scAll, batch->name, batch->channel, batch->batchId); }

    TimeCost pushResultTC;
    auto tensors = make_unique<vector<Tensor>>();
    tensors->push_back(Vec2TensorI32(restore));

    // 将keyCountVec放进tensor里并推到一个队列里
    auto keyCountTensors = make_unique<vector<Tensor>>();
    if (isIncrementalCheckpoint) {
        keyCountTensors->push_back(Vec2TensorI64(keyCountVec));
    }

    hotPos.resize(hotEmbTotCount[batch->name], 0);
    tensors->push_back(Vec2TensorI32(hotPos));

    if (!rankInfo.isDDR) {
        PushGlobalUniqueTensors(tensors, lookupKeys, channel);
        tensors->push_back(rankInfo.useDynamicExpansion ? Vec2TensorI64(lookupKeys) : Vec2TensorI32(lookupKeys));
        PushResultHBM(batch, move(tensors));
        if (isIncrementalCheckpoint) {
            PushKeyCountHBM(batch, move(keyCountTensors));
        }
    } else {
        std::vector<uint64_t> lookupKeysUint(lookupKeys.begin(), lookupKeys.end());
        vector<uint64_t> uniqueKeys;
        vector<int32_t> restoreVecSec;
        GlobalUnique(lookupKeysUint, uniqueKeys, restoreVecSec);
        PushResultDDR(batch, move(tensors), uniqueKeys, restoreVecSec);
    }

    LOG_DEBUG("pushResultTC(ms):{}", pushResultTC.ElapsedMS());
    if (GlogConfig::gStatOn) {
        LOG_INFO(STAT_INFO "channel_id {} batch_id {} rank_id {} key_process_time_cost {}", channel, batch->batchId,
                 rankInfo.rankId, totalTimeCost.ElapsedMS());
    }
    return true;
}

void KeyProcess::PushGlobalUniqueTensors(const unique_ptr<vector<Tensor>>& tensors, KeysT& lookupKeys, int channel)
{
    LOG_INFO(KEY_PROCESS "rank:{}, channel:{}, useSumSameIdGradients:{} ...", rankInfo.rankId, channel,
             rankInfo.useSumSameIdGradients);
    if (rankInfo.useSumSameIdGradients && channel == TRAIN_CHANNEL_ID) {
        KeysT uniqueKeys;
        vector<int32_t> restoreVecSec;

        TimeCost globalUniqueSyncTC;
        GlobalUnique(lookupKeys, uniqueKeys, restoreVecSec);
        LOG_DEBUG("globalUniqueSyncTC(ms):{}", globalUniqueSyncTC.ElapsedMS());
        tensors->push_back(Vec2TensorI32(restoreVecSec));
        tensors->push_back(rankInfo.useDynamicExpansion ? Vec2TensorI64(uniqueKeys) : Vec2TensorI32(uniqueKeys));
    }
}

vector<uint32_t> KeyProcess::GetCountRecv(const unique_ptr<EmbBatchT>& batch, int id,
                                          vector<vector<uint32_t>>& keyCount, vector<int> scAll, vector<int> ss)
{
    TimeCost getCountRecvTC;
    if (rankInfo.useStatic) {
        for (auto& cnt : keyCount) {
            cnt.resize(embInfos[batch->name].sendCount, 0);
        }
    }
    vector<uint32_t> countSend;
    for (auto& cnt : keyCount) {
        countSend.insert(countSend.cend(), cnt.cbegin(), cnt.cend());
    }
    vector<int> sc;
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        sc.push_back(scAll.at(rankInfo.rankSize * rankInfo.rankId + i));
    }
    vector<int> rc;  // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        rc.push_back(scAll.at(i * rankInfo.rankSize + rankInfo.rankId));
    }
    vector<int> rs = Count2Start(rc);  // receive displays/offset 接受数据的起始偏移量
    vector<uint32_t> countRecv;
    countRecv.resize(rs.back() + rc.back());
    int retCode = MPI_Alltoallv(countSend.data(), sc.data(), ss.data(), MPI_UINT32_T, countRecv.data(), rc.data(),
                                rs.data(), MPI_UINT32_T, comm[batch->channel][id]);
    if (retCode != MPI_SUCCESS) {
        LOG_ERROR("rank {}, MPI_Alltoallv failed:{}", rankInfo.rankId, retCode);
    }
    LOG_DEBUG("getCountRecvTC(ms)(with-all2all):{}", getCountRecvTC.ElapsedMS());
    return countRecv;
}

void KeyProcess::PushResultHBM(unique_ptr<EmbBatchT>& batch, unique_ptr<vector<Tensor>> tensors)
{
    std::unique_lock<std::mutex> lockGuard(mut);
    storage.push_front(move(tensors));
    infoList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, storage.begin()));
    lockGuard.unlock();
}

void KeyProcess::PushResultDDR(unique_ptr<EmbBatchT>& batch, unique_ptr<vector<Tensor>> tensors,
                               std::vector<uint64_t>& uniqueKeys, std::vector<int32_t>& restoreVecSec)
{
    std::unique_lock<std::mutex> lockGuard(mut);
    storage.push_front(move(tensors));
    infoList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, storage.begin()));
    uniqueKeysList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, move(uniqueKeys)));
    restoreVecSecList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, move(restoreVecSec)));
    lockGuard.unlock();
}

void KeyProcess::PushKeyCountHBM(unique_ptr<EmbBatchT>& batch, unique_ptr<vector<Tensor>> tensors)
{
    std::unique_lock<std::mutex> lockGuard(mut);
    keyCountStorage.push_front(move(tensors));
    keyCountInfoList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name,
                                                                  keyCountStorage.begin()));
    lockGuard.unlock();
    LOG_INFO("Push key count to list success.");
}

/*
 * 从共享队列SingletonQueue<EmbBatchT>中读取batch数据并返回。batch数据由 ReadEmbKeyV2 写入。
 * commID为线程标识[0, KEY_PROCESS_THREAD-1]，不同线程、训练或推理数据用不同的共享队列通信
 */
unique_ptr<EmbBatchT> KeyProcess::GetBatchData(int channel, int commId) const
{
    EASY_FUNCTION()
    unique_ptr<EmbBatchT> batch = nullptr;

    // train data, queue id = thread id [0, KEY_PROCESS_THREAD-1]
    int queueIndex = commId + (MAX_KEY_PROCESS_THREAD * channel);
    auto batchQueue = SingletonQueue<EmbBatchT>::GetInstances(queueIndex);
    EASY_BLOCK("get samples")
    EASY_VALUE("run on CPU", sched_getcpu())
    TimeCost tc = TimeCost();
    while (true) {
        batch = batchQueue->TryPop();
        if (batch != nullptr) {
            break;
        }
        this_thread::sleep_for(100us);
        if (tc.ElapsedSec() > GET_BATCH_TIMEOUT) {
            if (commId == 0) {
                LOG_WARN(KEY_PROCESS "getting batch timeout! 1. check last 'read batch cost' print. "
                                     "channel[{}] commId[{}]",
                         channel, commId);
            }
            this_thread::sleep_for(seconds(1));
            tc = TimeCost();
        }

        if (!isRunning) {
            LOG_WARN("channelId:{} threadId:{}, isRunning is false when GetBatchData", channel, commId);
            throw EndRunExit("GetBatchData end run.");
        }
    }
    EASY_END_BLOCK
    LOG_DEBUG(KEY_PROCESS "channelId:{} threadId:{} batchId:{}, get batch data done, batchName:{}. bs:{} sample:[{}]",
              batch->channel, commId, batch->batchId, batch->name, batch->Size(), batch->UnParse());
#if defined(PROFILING) && defined(BUILD_WITH_EASY_PROFILER)
    if (batch->batchId == PROFILING_START_BATCH_ID) {
        EASY_PROFILER_ENABLE
    } else if (batch->batchId == PROFILING_END_BATCH_ID) {
        ::profiler::dumpBlocksToFile(StringFormat("/home/MX_REC-profile-%d.prof", rankInfo.rankId).c_str());
    }
#endif
    return batch;
}

size_t KeyProcess::GetKeySize(const unique_ptr<EmbBatchT>& batch)
{
    size_t size = rankInfo.rankSize * embInfos[batch->name].sendCount;
    if (!rankInfo.useStatic) {
        size = batch->Size();
    }
    return size;
}

void KeyProcess::ProcessBatchWithFastUnique(const unique_ptr<EmbBatchT>& batch, ock::ctr::UniquePtr& unique, int id,
                                            UniqueInfo& uniqueInfoOut)
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

    ock::ctr::UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = static_cast<uint32_t>(batch->Size());
    uniqueIn.inputId = reinterpret_cast<void*>(batch->sample.data());

    ock::ctr::EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void*>(keySendInfo.keySend.data());
    uniqueOut.index = reinterpret_cast<uint32_t*>(uniqueInfoOut.restore.data());
    if (rankInfo.useStatic) {
        uniqueOut.idCnt = idCount.data();
        uniqueOut.idCntFill = keySendInfo.keyCount.data();
    } else {
        uniqueOut.idCnt = keySendInfo.keyCount.data();
    }
    uniqueOut.uniqueIdCntInBucket = splitSize.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void*>(uniqueVector.data());
    uniqueOut.uniqueIdCnt = 0;

    int ret = unique->DoEnhancedUnique(uniqueIn, uniqueOut);
    if (ret != ock::ctr::H_OK) {
        throw runtime_error(StringFormat("fast unique DoEnhancedUnique failed, code:%d", ret));
    }
    EASY_END_BLOCK
    LOG_DEBUG("FastUniqueCompute(ms):{}, ret:{}", uniqueTC.ElapsedMS(), ret);

    vector<int> sc;
    HandleHotAndSendCount(batch, uniqueInfoOut, keySendInfo, sc, splitSize);

    All2All(sc, id, batch, keySendInfo, uniqueInfoOut.all2AllInfo);

    LOG_DEBUG(KEY_PROCESS "ProcessBatchWithFastUnique get batchId:{}, batchSize:{},"
                          " channel:{}, name:{}, restore:{}, keyCount:{}",
              batch->batchId, batch->Size(), batch->channel, batch->name, uniqueInfoOut.restore.size(),
              keySendInfo.keyCount.size());

    if (GlogConfig::gStatOn) {
        LOG_INFO(STAT_INFO "channel_id {} batch_id {} rank_id {} "
                           "batch_key_num_with_fast_unique {} unique_key_num_with_fast_unique {}",
                 batch->channel, batch->batchId, rankInfo.rankId, batch->Size(), uniqueOut.uniqueIdCnt);
    }
}

void KeyProcess::HandleHotAndSendCount(const unique_ptr<EmbBatchT>& batch, UniqueInfo& uniqueInfoOut,
                                       KeySendInfo& keySendInfo, vector<int>& sc, vector<int>& splitSize)
{
    std::shared_lock<std::shared_mutex> lock(g_smut);
    absl::flat_hash_map<emb_key_t, int> hotMap = hotKey[batch->name];
    lock.unlock();

    int hotOffset = 0;
    uniqueInfoOut.hotPos.resize(hotEmbTotCount[batch->name]);
    hotOffset = hotEmbTotCount[batch->name];

    TimeCost computeHotTc;
    ComputeHotPos(batch, hotMap, uniqueInfoOut.hotPos, uniqueInfoOut.restore, hotOffset);
    LOG_DEBUG("ComputeHot TimeCost(ms):{}", computeHotTc.ElapsedMS());
    UpdateHotMapForUnique(keySendInfo.keySend, keySendInfo.keyCount, hotOffset, batch->batchId % hotEmbUpdateStep == 0,
                          batch->name);

    if (rankInfo.useStatic) {
        sc.resize(rankInfo.rankSize, embInfos[batch->name].sendCount);
    } else {
        sc.resize(rankInfo.rankSize);
        for (int i = 0; i < rankInfo.rankSize; i++) {
            sc[i] = splitSize[i];
        }
    }
}

void KeyProcess::ComputeHotPos(const unique_ptr<EmbBatchT>& batch, absl::flat_hash_map<emb_key_t, int>& hotMap,
                               vector<int>& hotPos, vector<int32_t>& restore, const int hotOffset) const
{
    emb_key_t* inputData = batch->sample.data();
    size_t miniBs = batch->Size();

    int hotCount = 0;
    for (size_t i = 0; i < miniBs; i++) {
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

void KeyProcess::All2All(vector<int>& sc, int id, const unique_ptr<EmbBatchT>& batch, KeySendInfo& keySendInfo,
                         All2AllInfo& all2AllInfoOut)
{
    TimeCost getScAllTC;
    int channel = batch->channel;
    GetScAllForUnique(sc, id, batch, all2AllInfoOut.scAll);  // Allgather通信获取所有（不同rank相同thread id的）
    LOG_DEBUG("GetScAll TimeCost(ms):{}", getScAllTC.ElapsedMS());

    TimeCost all2allTC;
    vector<int> ss = Count2Start(sc);   // send displays/offset 发送数据的起始偏移量
    vector<int> rc(rankInfo.rankSize);  // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        // 通信量矩阵某一列的和即为本地要从其他设备接受的key数据量
        rc[i] = all2AllInfoOut.scAll.at(i * rankInfo.rankSize + rankInfo.rankId);
    }
    vector<int> rs = Count2Start(rc);  // receive displays/offset 接受数据的起始偏移量
    all2AllInfoOut.keyRecv.resize(rs.back() + rc.back());
    EASY_BLOCK("all2all")
    int retCode = MPI_Alltoallv(keySendInfo.keySend.data(), sc.data(), ss.data(), MPI_INT64_T,
                                all2AllInfoOut.keyRecv.data(), rc.data(), rs.data(), MPI_INT64_T, comm[channel][id]);
    if (retCode != MPI_SUCCESS) {
        LOG_ERROR("rank {}, MPI_Alltoallv failed:{}", rankInfo.rankId, retCode);
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, All2All MPI_Alltoallv end.", channel, id, batch->batchId);
    all2AllInfoOut.countRecv.resize(rs.back() + rc.back());
    if (isWithFAAE) {
        retCode = MPI_Alltoallv(keySendInfo.keyCount.data(), sc.data(), ss.data(), MPI_UINT32_T,
                                all2AllInfoOut.countRecv.data(), rc.data(), rs.data(), MPI_UINT32_T, comm[channel][id]);
        if (retCode != MPI_SUCCESS) {
            LOG_ERROR("channelId:{} threadId:{} batchId:{}, MPI_Alltoallv failed:{}", channel, id, batch->batchId,
                      retCode);
        }
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, All2All end, all2allTC TimeCost(ms):{}", channel, id,
              batch->batchId, all2allTC.ElapsedMS());
    EASY_END_BLOCK
}

auto KeyProcess::ProcessSplitKeys(const unique_ptr<EmbBatchT>& batch, int id, vector<KeysT>& splitKeys)
    -> tuple<KeysT, vector<int>, vector<int>>
{
    TimeCost processSplitKeysTC;
    EASY_FUNCTION(profiler::colors::Purple)
    EASY_VALUE("batchId", batch->batchId)
    LOG_INFO(KEY_PROCESS "channelId:{} threadId:{} batchId:{}, ProcessSplitKeys start.", batch->channel, id,
             batch->batchId);

    // 使用静态all2all通信：发送或接受量为预置固定值 scInfo[batch->name] = 65536 / rankSize 经验值
    if (rankInfo.useStatic) {  // maybe move after all2all
        for (KeysT& i : splitKeys) {
            if (static_cast<int>(i.size()) > embInfos[batch->name].sendCount) {
                LOG_ERROR("{}[{}]:{} overflow! set send count bigger than {}", batch->name, batch->channel,
                          batch->batchId, i.size());
                throw runtime_error(StringFormat("%s[%d]:%d overflow! set send count bigger than %d",
                                                 batch->name.c_str(), batch->channel, batch->batchId, i.size())
                                        .c_str());
            }
            i.resize(embInfos[batch->name].sendCount, -1);
        }
    }
    KeysT keySend;
    vector<int> sc;  // send count
    for (const auto& i : splitKeys) {
        sc.push_back(static_cast<int>(i.size()));
        keySend.insert(keySend.cend(), i.cbegin(), i.cend());
    }
    KeysT keyRecv;

    TimeCost getScAllTC;
    vector<int> scAll = GetScAll(sc, id, batch);  // Allgather通信获取所有（不同rank相同thread id的）线程间通信量矩阵
    LOG_DEBUG("getScAllTC(ms)(AllReduce-AllGather):{}", getScAllTC.ElapsedMS());

    vector<int> ss = Count2Start(sc);  // send displays/offset 发送数据的起始偏移量
    vector<int> rc;                    // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        // 通信量矩阵某一列的和即为本地要从其他设备接受的key数据量
        rc.push_back(scAll.at(i * rankInfo.rankSize + rankInfo.rankId));
    }
    vector<int> rs = Count2Start(rc);  // receive displays/offset 接受数据的起始偏移量
    keyRecv.resize(rs.back() + rc.back());
    EASY_BLOCK("all2all")

    TimeCost uniqueAll2AllTC;
    int retCode = MPI_Alltoallv(keySend.data(), sc.data(), ss.data(), MPI_INT64_T, keyRecv.data(), rc.data(), rs.data(),
                                MPI_INT64_T, comm[batch->channel][id]);
    if (retCode != MPI_SUCCESS) {
        LOG_ERROR("rank {}, MPI_Alltoallv failed:{}", rankInfo.rankId, retCode);
    }
    LOG_DEBUG("uniqueAll2AllTC(ms):{}", uniqueAll2AllTC.ElapsedMS());

    EASY_END_BLOCK
    LOG_DEBUG(KEY_PROCESS "channelId:{} threadId:{} batchId:{}, batchName:{}, MPI_Alltoallv finish."
                          " processSplitKeysTC(ms):{}",
              batch->channel, id, batch->batchId, batch->name, processSplitKeysTC.ElapsedMS());
    return {keyRecv, scAll, ss};
}

/*
 * 将batch内的key按照所存储的dev id哈希切分并去重，哈希函数为模运算
 * splitKeys返回：将数据的key切分到其所在dev id对应的桶中，并去重。
 * restore返回：去重后key在桶内偏移量（用于计算恢复向量）
 */
tuple<vector<KeysT>, vector<int32_t>> KeyProcess::HashSplit(const unique_ptr<EmbBatchT>& batch) const
{
    EASY_FUNCTION(profiler::colors::Gold)
    emb_key_t* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    vector<KeysT> splitKeys(rankInfo.rankSize);
    vector<int32_t> restore(batch->Size());
    vector<int> hashSplitLens(rankInfo.rankSize);  // 初始化全0，记录每个桶的长度
    absl::flat_hash_map<emb_key_t, int> uKey;      // 用于去重查询
    EASY_BLOCK("split push back")
    for (size_t i = 0; i < miniBs; i++) {
        const emb_key_t& key = batchData[i];
        emb_key_t devId = abs(key % static_cast<emb_key_t>(rankInfo.rankSize));
        auto result = uKey.find(key);
        if (result == uKey.end()) {
            splitKeys[devId].push_back(key);
            restore[i] = hashSplitLens[devId]++;  // restore记录去重后key在桶内偏移量（用于计算恢复向量）
            uKey[key] = restore[i];
        } else {  // 去重
            restore[i] = result->second;
        }
    }
    EASY_END_BLOCK

    LOG_TRACE("dump splitKeys {}", DumpSplitKeys(splitKeys));

    if (GlogConfig::gStatOn) {
        size_t uniqueKeyNum = 0;
        for (int devId = 0; devId < rankInfo.rankSize; ++devId) {
            uniqueKeyNum += splitKeys[devId].size();
        }
        LOG_INFO(STAT_INFO "channel_id {} batch_id {} rank_id {} batch_key_num {} unique_key_num {}", batch->channel,
                 batch->batchId, rankInfo.rankId, batch->Size(), uniqueKeyNum);
    }
    return {splitKeys, restore};
}

void KeyProcess::PaddingAlltoallVC(vector<KeysT>& splitKeys) const
{
    for (auto& keys : splitKeys) {
        if (keys.size() % ALLTOALLVC_ALIGN == 0) {
            continue;
        }
        int paddingSize = ALLTOALLVC_ALIGN - (keys.size() % ALLTOALLVC_ALIGN);
        std::fill_n(std::back_inserter(keys), paddingSize, INVALID_KEY_VALUE);
    }
    return;
}

tuple<vector<KeysT>, vector<int32_t>, vector<vector<uint32_t>>> KeyProcess::HashSplitWithFAAE(
    const unique_ptr<EmbBatchT>& batch) const
{
    EASY_FUNCTION(profiler::colors::Gold)
    emb_key_t* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    vector<KeysT> splitKeys(rankInfo.rankSize);
    vector<vector<uint32_t>> keyCount(rankInfo.rankSize);  // splitKeys在原始batch中对应的频次
    vector<int32_t> restore(batch->Size());
    vector<int> hashSplitLens(rankInfo.rankSize);                   // 初始化全0，记录每个桶的长度
    absl::flat_hash_map<emb_key_t, std::pair<int, uint32_t>> uKey;  // 用于去重查询
    EASY_BLOCK("split push back")
    for (size_t i = 0; i < miniBs; i++) {
        const emb_key_t& key = batchData[i];
        emb_key_t devId = abs(key % static_cast<emb_key_t>(rankInfo.rankSize));
        auto result = uKey.find(key);
        if (result == uKey.end()) {
            splitKeys[devId].push_back(key);
            restore[i] = hashSplitLens[devId]++;  // restore记录去重后key在桶内偏移量（用于计算恢复向量）
            uKey[key].first = restore[i];
            uKey[key].second = 1;
        } else {  // 去重
            restore[i] = result->second.first;
            uKey[key].second++;
        }
    }

    if (!rankInfo.useStatic) {
        PaddingAlltoallVC(splitKeys);
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
    LOG_TRACE("dump splitKeys {}", DumpSplitKeys(splitKeys));

    if (GlogConfig::gStatOn) {
        size_t uniqueKeyNum = 0;
        for (int devId = 0; devId < rankInfo.rankSize; ++devId) {
            uniqueKeyNum += splitKeys[devId].size();
        }
        LOG_INFO(STAT_INFO "channel_id {} batch_id {} rank_id {} batch_key_num {} faae_unique_key_num {}",
                 batch->channel, batch->batchId, rankInfo.rankId, batch->Size(), uniqueKeyNum);
    }
    return {splitKeys, restore, keyCount};
}

tuple<vector<KeysT>, vector<int32_t>, vector<int>, vector<emb_key_t>> KeyProcess::HotHashSplit(const
unique_ptr<EmbBatchT>& batch)
{
    EASY_FUNCTION(profiler::colors::Gold)
    emb_key_t* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    vector<KeysT> splitKeys(rankInfo.rankSize);
    vector<int32_t> restore(batch->Size());
    absl::flat_hash_map<emb_key_t, int> uKey;  // 用于去重查询
    absl::flat_hash_map<emb_key_t, int> keyCountMapByEmbName;
    absl::flat_hash_map<emb_key_t, emb_key_t> keyCountOneBatch;
    vector<emb_key_t> keyCountVec;
    std::shared_lock<std::shared_mutex> lock(g_smut);
    auto hotMap = hotKey[batch->name];
    lock.unlock();
    vector<int> hotPos(hotEmbTotCount[batch->name]);
    vector<int> hotPosDev(hotEmbTotCount[batch->name]);
    int hotCount = 0;
    int hotOffset = hotEmbTotCount[batch->name];
    for (size_t i = 0; i < miniBs; i++) {  // for mini batch
        const emb_key_t& key = batchData[i];
        if (batch->batchId % hotEmbUpdateStep == 0) {
            keyCountMapByEmbName[key]++;
        }
        if (isIncrementalCheckpoint) {
            keyCountOneBatch[key]++;
        }
        emb_key_t devId = abs(key % static_cast<emb_key_t>(rankInfo.rankSize));
        auto result = uKey.find(key);
        if (result != uKey.end()) {  // // already in splitKeys
            restore[i] = result->second;
            continue;
        }
        // new key in current batch
        splitKeys[devId].push_back(key);  // push to bucket
        auto hot = hotMap.find(key);
        if (hot != hotMap.end()) {    // is hot key
            if (hot->second == -1) {  // is new hot key in this batch
                // pos in lookup vec (need add ss) for hot-gather
                hotPos[hotCount] = static_cast<int>(splitKeys[devId].size()) - 1;
                hotPosDev[hotCount] = devId;  // which dev, for get ss
                hot->second = hotCount;
                restore[i] = hotCount++;  // get pos of hot emb
            } else {
                restore[i] = hot->second;
            }
        } else {  // is not hot key
            // restore记录去重后key在桶内偏移量（用于计算恢复向量）
            restore[i] = static_cast<int32_t>(splitKeys[devId].size() + (hotOffset - 1));
        }
        uKey[key] = restore[i];
    }
    if (isIncrementalCheckpoint) {
        for (auto& it : keyCountOneBatch) {
            keyCountVec.emplace_back(it.first);
            keyCountVec.emplace_back(it.second);
        }
    }
    LOG_INFO(KEY_PROCESS "Hot hash split, batch id: {}, batch name: {}, channel: {}, kc size: {}, data: {}",
             batch->batchId, batch->name, batch->channel, keyCountVec.size(), VectorToString(keyCountVec));
    if (GlogConfig::gStatOn) {
        size_t uniqueKeyNum = 0;
        for (int devId = 0; devId < rankInfo.rankSize; ++devId) {
            uniqueKeyNum += splitKeys[devId].size();
        }
        LOG_INFO(STAT_INFO "channel_id {} batch_id {} rank_id {} batch_key_num {} hot_unique_key_num {}",
                 batch->channel, batch->batchId, rankInfo.rankId, batch->Size(), uniqueKeyNum);
    }

    UpdateHotMap(keyCountMapByEmbName, hotEmbTotCount[batch->name], batch->batchId % hotEmbUpdateStep == 0,
                 batch->name);
    AddCountStartToHotPos(splitKeys, hotPos, hotPosDev, batch);
    return { splitKeys, restore, hotPos, keyCountVec };
}

void KeyProcess::AddCountStartToHotPos(vector<KeysT>& splitKeys, vector<int>& hotPos, const vector<int>& hotPosDev,
                                       const unique_ptr<EmbBatchT>& batch)
{
    vector<int> splitKeysSize;
    for (auto& splitKey : splitKeys) {
        int tmp = rankInfo.useStatic ? embInfos[batch->name].sendCount : static_cast<int>(splitKey.size());
        splitKeysSize.push_back(tmp);
    }

    vector<int> cs = Count2Start(splitKeysSize);
    for (size_t i = 0; i < hotPos.size(); ++i) {
        hotPos[i] += cs[hotPosDev[i]];
    }
}

void KeyProcess::UpdateHotMapForUnique(const KeysT& keySend, const vector<int32_t>& keyCount, uint32_t count,
                                       bool refresh, const string& embName)
{
    auto& hotMap = hotKey[embName];
    if (refresh) {
        priority_queue<pair<int, emb_key_t>> pq;
        for (size_t i = 0; i < keySend.size(); ++i) {
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

void KeyProcess::UpdateHotMap(absl::flat_hash_map<emb_key_t, int>& keyCountMapByEmbName, uint32_t count, bool refresh,
                              const string& embName)
{
    if (!refresh) {
        return;
    }
    auto& hotMap = hotKey[embName];
    priority_queue<pair<int, emb_key_t>> pq;  // top k key
    for (auto& p : keyCountMapByEmbName) {
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

/*
 * 将本地（rank）batch要发送的key数据量进行Allgather通信，获取所有（不同rank相同thread id的）线程间的通信量矩阵
 * scAll返回：所有线程间的通信量矩阵（按行平铺的一维向量）
 */
vector<int> KeyProcess::GetScAll(const vector<int>& keyScLocal, int commId, const unique_ptr<EmbBatchT>& batch)
{
    EASY_FUNCTION()
    vector<int> scAll;
    scAll.resize(rankInfo.rankSize * rankInfo.rankSize);
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, GetScAll start.", batch->channel, commId, batch->batchId);

    // allgather keyScLocal(key all2all keyScLocal = device all2all rc)
    auto retCode = MPI_Allgather(keyScLocal.data(), rankInfo.rankSize, MPI_INT, scAll.data(), rankInfo.rankSize,
                                 MPI_INT, comm[batch->channel][commId]);
    if (retCode != MPI_SUCCESS) {
        LOG_ERROR("rank {} commId {}, MPI_Allgather failed:{}", rankInfo.rankId, commId, retCode);
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, GetScAll MPI_Allgather end, key scAll matrix:\n{}", batch->channel,
              commId, batch->batchId, VectorToString(scAll));
    return scAll;
}

void KeyProcess::GetScAllForUnique(const vector<int>& keyScLocal, int commId, const unique_ptr<EmbBatchT>& batch,
                                   vector<int>& scAllOut)
{
    EASY_FUNCTION()
    int channel = batch->channel;
    scAllOut.resize(rankInfo.rankSize * rankInfo.rankSize);

    // allgather keyScLocal(key all2all keyScLocal = device all2all rc)
    auto retCode = MPI_Allgather(keyScLocal.data(), rankInfo.rankSize, MPI_INT, scAllOut.data(), rankInfo.rankSize,
                                 MPI_INT, comm[channel][commId]);
    if (retCode != MPI_SUCCESS) {
        LOG_ERROR("rank {}, MPI_Allgather failed:{}", rankInfo.rankId, retCode);
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, GetScAllForUnique end, key scAllOut matrix:\n{}", channel, commId,
              batch->batchId, VectorToString(scAllOut));
}

void KeyProcess::Key2Offset(const EmbNameT& embName, KeysT& splitKey, int channel)
{
    TimeCost key2OffsetTC;
    EASY_FUNCTION(profiler::colors::Blue600)
    std::lock_guard<std::mutex> lk(mut);  // lock for PROCESS_THREAD
    auto& key2Offset = keyOffsetMap[embName];
    auto& maxOffsetTmp = maxOffset[embName];
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
            LOG_TRACE("HBM mode, evictPos is not null, name[{}] key [{}] reuse offset [{}], evictSize [{}]!!!", embName,
                      key, offset, evictPos.size());
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
        LOG_ERROR("dev cache overflow {} > {}", maxOffsetTmp, embInfos[embName].devVocabSize);
        throw std::runtime_error("dev cache overflow!");
    }
    LOG_DEBUG("current hbm emb:{}, usage:{}/{} key2OffsetTC({} ms)", embName, maxOffsetTmp,
              embInfos[embName].devVocabSize, key2OffsetTC.ElapsedMS());
}

/*
 * 构建恢复向量，以便从去重后的emb向量/key恢复回batch对应的emb向量
 * 输入接收到emb块的偏移blockOffset，batch内每个key在块内的偏移restoreVec
 * 输出恢复向量restoreVec，即batch到keySend（平铺的splitKeys）的映射
 * 实现方案2：用map记录keySend中key和表内index/offset的映射，在恢复emb时直接根据batch的key查询该map即可找到receive
 * emb中的 位置，时间复杂度：O(map构建keySend.size + map查询)，空间复杂度：O(map)
 */
void KeyProcess::BuildRestoreVec(const unique_ptr<EmbBatchT>& batch, const vector<int>& blockOffset,
                                 vector<int>& restoreVec, int hotPosSize) const
{
    TimeCost buildRestoreVecTC;
    EASY_FUNCTION()
    int hotNum = 0;
    for (size_t i = 0; i < batch->Size(); ++i) {
        const emb_key_t key = batch->sample[i];
        emb_key_t devId = abs(key % static_cast<emb_key_t>(rankInfo.rankSize));
        if (restoreVec[i] >= hotPosSize) {
            restoreVec[i] += blockOffset[devId];
        } else if (Logger::GetLevel() >= Logger::DEBUG) {
            hotNum += 1;
        }
    }
    LOG_DEBUG("hot num in all:{}/{} buildRestoreVecTC(ms):{}", hotNum, batch->Size(), buildRestoreVecTC.ElapsedMS());
}

template <class T>
T KeyProcess::GetInfo(info_list_t<T>& list, const EmbBaseInfo& info)
{
    std::lock_guard<std::mutex> lockGuard(mut);
    if (list[info.name][info.channelId].empty()) {
        LOG_TRACE("get info list is empty.");
        throw EmptyList();
    }
    auto topBatch = get<int>(list[info.name][info.channelId].top());
    if (topBatch < info.batchId) {
        LOG_ERROR("wrong batch id, top:{} getting:{}, channel:{}, may not clear channel", topBatch, info.batchId,
                  info.channelId);
        this_thread::sleep_for(1s);
    }
    if (topBatch != info.batchId) {
        LOG_TRACE("topBatch({}) is not equal batch({}).", topBatch, info.batchId);
        throw WrongListTop();
    }
    auto t = list[info.name][info.channelId].top();
    list[info.name][info.channelId].pop();
    return move(t);
}

template<class T>
T KeyProcess::GetKeyCountVec(info_list_t<T>& list, const EmbBaseInfo& info)
{
    std::lock_guard<std::mutex> lockGuard(mut);
    if (list[info.name][info.channelId].empty()) {
        LOG_ERROR("get info list is empty.");
        throw EmptyList();
    }
    auto t = list[info.name][info.channelId].top();
    if (list.empty()) {
        LOG_INFO("Get data t is null.");
    }
    list[info.name][info.channelId].pop();
    LOG_INFO("Get key count vector from list success.");
    return move(t);
}

vector<uint64_t> KeyProcess::GetUniqueKeys(const EmbBaseInfo& info, bool& isEos)
{
    TimeCost tc = TimeCost();

    HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
    vector<uint64_t> ret;
    auto startTime = std::chrono::system_clock::now();
    while (true) {
        if (!isRunning) {
            break;
        }
        auto endTime = std::chrono::system_clock::now();
        // 判断此时的info.batchId id是否已经过期，即通道已经刷新
        if (info.batchId != hybridMgmtBlock->hybridBatchId[info.channelId]) {
            LOG_DEBUG(KEY_PROCESS "Detected that the batch has expired at this time, exiting the loop! {}[{}]:{}",
                      info.name, info.channelId, info.batchId);
            break;
        }
        if (info.batchId != 0 && info.channelId != 0 && tc.ElapsedSec() > KEY_PROCESS_TIMEOUT) {
            LOG_WARN(KEY_PROCESS "getting lookup keys timeout! {}[{}]:{}", info.name, info.channelId, info.batchId);
            break;
        }
        try {
            auto infoVec = GetInfo(uniqueKeysList, info);
            ret = get<std::vector<uint64_t>>(infoVec);
            break;
        } catch (EmptyList&) {
            unique_lock<mutex> lockEosGuard(eosMutex);
            isEos = IsGetUniqueKeysEos(info, startTime);
            if (isEos) {
                break;
            }
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            LOG_TRACE("getting info failed table:{}, channel:{}, mgmt batchId:{}, wrong top", info.name, info.channelId,
                      info.channelId);
            this_thread::sleep_for(1ms);
        }
    }
    return ret;
}

bool KeyProcess::IsGetUniqueKeysEos(const EmbBaseInfo& info, std::chrono::_V2::system_clock::time_point& startTime)
{
    HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
    auto endTime = std::chrono::system_clock::now();

    // readEmbKey start with 0
    int readEmbKeyBatchId = hybridMgmtBlock->readEmbedBatchId[info.channelId] - 1;
    // 避免eos在keyProcess还未处理完数据时插队到通道前面
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    if (info.batchId != 0 && elapsedTime.count() >= timeoutGetUniqueKeysEmpty) {
        LOG_DEBUG("table:{}, channelId:{}, isNeedSendEos:{}, current batchId:{}, L1 pipeline readEmbKeyBatchId:{}, "
                  "L2 pipeline lookUpSwapAddrsPushId:{}, L3 pipeline h2dNextBatchId:{}",
                  info.name, info.channelId, isNeedSendEos[info.channelId], info.batchId, readEmbKeyBatchId,
                  hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId],
                  hybridMgmtBlock->h2dNextBatchId[info.name][info.channelId]);
        startTime = std::chrono::system_clock::now();
    }
    // Check '>= readEmbedBatchIdAll' condition to avoid send eos before handle all batch data from readEmbKey Op.
    if (isNeedSendEos[info.channelId] && readEmbKeyBatchId < info.batchId &&
        hybridMgmtBlock->h2dNextBatchId[info.name][info.channelId] ==
        hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId] &&
        hybridMgmtBlock->h2dNextBatchId[info.name][info.channelId] >=
        hybridMgmtBlock->readEmbedBatchId[info.channelId]) {
        LOG_INFO("table:{}, channelId:{} current batchId:{}, GetUniqueKeys eos, L1 pipeline readEmbKeyBatchId:{}, "
                 "L2 pipeline hybridBatchId:{}, L3 pipeline lookUpSwapAddrsPushId:{}, L4 pipeline h2dNextBatchId:{}",
                 info.name, info.channelId, info.batchId, readEmbKeyBatchId,
                 hybridMgmtBlock->hybridBatchId[info.channelId],
                 hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId],
                 hybridMgmtBlock->h2dNextBatchId[info.name][info.channelId]);
        return true;
    }
    LOG_TRACE("getting uniqueKeys failed, table:{}, channel:{}, mgmt batchId:{}, readEmbKey batchId:{}, list is empty",
              info.name, info.channelId, info.batchId, readEmbKeyBatchId);
    return false;
}

std::vector<int32_t> KeyProcess::GetRestoreVecSec(const EmbBaseInfo& info)
{
    TimeCost tc = TimeCost();
    // 循环尝试获取list中的数据；如果key process线程退出或者处理数据超时，返回空vector
    while (true) {
        if (!isRunning) {
            return {};
        }
        // 判断此时的batch id是否已经过期，即通道已经刷新
        HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
        if (info.batchId != hybridMgmtBlock->hybridBatchId[info.channelId]) {
            LOG_DEBUG(KEY_PROCESS "Detected that the batch has expired at this time, exiting the loop! {}[{}]:{}",
                      info.name, info.channelId, info.batchId);
            return {};
        }
        if (info.batchId != 0 && info.channelId != 0 && tc.ElapsedSec() > KEY_PROCESS_TIMEOUT) {
            LOG_WARN(KEY_PROCESS "getting lookup keys timeout! {}[{}]:{}", info.name, info.channelId, info.batchId);
            return {};
        }
        try {
            auto ret = GetInfo(restoreVecSecList, info);
            return get<std::vector<int32_t>>(ret);
        } catch (EmptyList&) {
            unique_lock<mutex> lockEosGuard(eosMutex);
            // readEmbKey真实的次数是readEmbedBatchId减1
            int readEmbKeyBatchId = hybridMgmtBlock->readEmbedBatchId[info.channelId] - 1;
            // 避免eos在keyProcess还未处理完数据时插队到通道前面
            if (isNeedSendEos[info.channelId] && readEmbKeyBatchId < info.batchId &&
                hybridMgmtBlock->h2dNextBatchId[info.name][info.channelId] == info.batchId) {
                LOG_ERROR("channelId:{} batchId:{}, GetRestoreVecSec eos, code should not reach here", info.channelId,
                          info.batchId);
                throw runtime_error("GetRestoreVecSec eos, code should not reach here");
            }
            LOG_TRACE("getting info failed {}[{}], list is empty, and mgmt batchId: {}, readEmbKey batchId: {}.",
                      info.name, info.channelId, info.batchId, readEmbKeyBatchId);
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            LOG_TRACE("getting info failed {}[{}]:{} wrong top", info.name, info.channelId, info.batchId);
            this_thread::sleep_for(1ms);
        }
    }
}

/// 当数据列表为空，且eos标志位为true时，主动发送eos
/// \param embName 表名
/// \param batchId 已处理的batch数
/// \param channel 通道索引（训练/推理）
void KeyProcess::SendEos(const std::string& embName, int batchId, int channel)
{
#ifndef GTEST
    finishSendEosCnt[channel].store(0);
    ++readySendEosCnt[channel];
    LOG_INFO("table:{}, channelId:{} batchId:{}, readySendEosCnt:{}, ready to SendEos", embName, channel, batchId,
             readySendEosCnt[channel]);
    while (readySendEosCnt[channel] != static_cast<int>(embInfos.size())) {
        LOG_DEBUG("table:{}, readySendEosCnt:{}, waiting other table enter SendEos", embName, readySendEosCnt[channel]);
        this_thread::sleep_for(1000ms);
    }
    LOG_INFO("table:{}, channelId:{} batchId:{}, SendEos start, acquiring destroyMutex", embName, channel, batchId);
    destroyMutex.lock();

    LOG_INFO("table:{}, channelId:{} batchId:{}, SendEos start", embName, channel, batchId);
    if (!isRunning) {
        LOG_INFO("other table trigger eos ahead, keyProcess already destroyed. skip sending eos for table:{}", embName);
        ++finishSendEosCnt[channel];
        destroyMutex.unlock();
        return;
    }
    SendEosTensor(embName, channel);
    destroyMutex.unlock();
    LOG_INFO("channelId:{} batchId:{}, the embName:{} SendEos end, release destroyMutex", channel, batchId, embName);

    ++finishSendEosCnt[channel];
    LOG_INFO("table:{}, channelId:{} batchId:{}, finishSendEosCnt:{}, finish SendEos", embName, channel, batchId,
             finishSendEosCnt[channel]);
    while (finishSendEosCnt[channel] != static_cast<int>(embInfos.size())) {
        LOG_DEBUG("table:{}, channelId:{} batchId:{}, finishSendEosCnt:{}, waiting other table finish SendEos", embName,
                  channel, batchId, finishSendEosCnt[channel]);
        this_thread::sleep_for(1000ms);
    }
    readySendEosCnt[channel].store(0);
    isNeedSendEos[channel] = false;
    LOG_DEBUG("isNeedSendEos set to false, table:{}, channelId:{} batchId:{}", embName, channel, batchId);
#endif
}

/// HBM模式下，从list中获取指定类型的tensor向量
/// \param batch 已处理的batch数
/// \param embName 表名
/// \param channel 通道索引（训练/推理）
/// \param type 数据类型
/// \return
unique_ptr<vector<Tensor>> KeyProcess::GetInfoVec(const EmbBaseInfo& info, ProcessedInfo type, bool& isEos)
{
    TimeCost tc = TimeCost();
    info_list_t<TensorInfoT>* list;

    // 根据数据类型，选择对应的list
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

    unique_ptr<vector<Tensor>> ret = nullptr;
    // 循环尝试获取list中的数据；如果key process线程退出或者处理数据超时，返回空指针
    while (true) {
        if (!isRunning) {
            break;
        }
        // 判断此时的batch id是否已经过期，即通道已经刷新
        HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
        if (info.batchId != hybridMgmtBlock->hybridBatchId[info.channelId]) {
            LOG_DEBUG(KEY_PROCESS "Detected that the batch has expired at this time, exiting the loop! {}[{}]:{}",
                      info.name, info.channelId, info.batchId);
            break;
        }
        if (info.batchId != 0 && info.channelId != 0 && tc.ElapsedSec() > KEY_PROCESS_TIMEOUT) {
            LOG_WARN(KEY_PROCESS "getting lookup keys timeout! {}[{}]:{}", info.name, info.channelId, info.batchId);
            break;
        }

        try {
            auto infoVec = GetInfo(*list, info);
            auto it = get<std::list<unique_ptr<vector<Tensor>>>::iterator>(infoVec);
            ret = std::move(*it);
            std::unique_lock<std::mutex> lockGuard(mut);
            storage.erase(it);
            break;
        } catch (EmptyList&) {
            unique_lock<mutex> lockEosGuard(eosMutex);
            isEos = IsGetInfoVecEos(info.batchId, info.name, info.channelId);
            if (isEos) {
                break;
            }
            LOG_TRACE("getting info failed {}[{}], list is empty, and mgmt batchId: {}, readEmbKey batchId: {}.",
                      info.name, info.channelId, info.batchId, (hybridMgmtBlock->readEmbedBatchId[info.channelId] - 1));
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            LOG_TRACE("getting info failed {}[{}]:{} wrong top", info.name, info.channelId, info.batchId);
            this_thread::sleep_for(1ms);
        }
    }
    return ret;
}

unique_ptr<vector<Tensor>> KeyProcess::GetKCInfoVec(const EmbBaseInfo& info)
{
    info_list_t<TensorInfoT>* list;
    list = &keyCountInfoList;
    unique_ptr<vector<Tensor>> ret = nullptr;
    // 循环尝试获取list中的数据
    while (true) {
        if (!isRunning) {
            break;
        }

        try {
            auto infoVec = GetKeyCountVec(*list, info);
            auto it = get<std::list<unique_ptr<vector<Tensor>>>::iterator>(infoVec);
            ret = std::move(*it);
            std::unique_lock<std::mutex> lockGuard(mut);
            keyCountStorage.erase(it);
            break;
        } catch (EmptyList&) {
            unique_lock<mutex> lockEosGuard(eosMutex);
            LOG_TRACE("getting info failed, list is empty.");
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            LOG_TRACE("getting info failed {}[{}]:{} wrong top", info.name, info.channelId, info.batchId);
            this_thread::sleep_for(1ms);
        }
    }
    return ret;
}

void KeyProcess::SendA2A(const vector<int>& a2aInfo, const string& embName, int channel, int batch)
{
    // 数据放到队列里，在mgmt里面发送（检查发送数据量）
    auto tensors = make_unique<vector<Tensor>>();
    Tensor tmpTensor(tensorflow::DT_INT64, {rankInfo.rankSize, rankInfo.rankSize});
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
    return rankInfo.ctrlSteps.at(channelId);
}

void KeyProcess::EvictKeys(const string& embName, const vector<emb_cache_key_t>& keys)  // hbm
{
    LOG_INFO(KEY_PROCESS "hbm funEvictCall: [{}]! keySize:{}", embName, keys.size());
    EmbeddingMgmt::Instance()->EvictKeys(embName, keys);
}

void KeyProcess::EvictKeysCombine(const vector<emb_cache_key_t>& keys)  // hbm
{
    LOG_INFO(KEY_PROCESS "hbm combine funEvictCall, keySize:{}", keys.size());
    EmbeddingMgmt::Instance()->EvictKeysCombine(keys);
}

void KeyProcess::EvictDeleteDeviceEmb(const string& embName, const vector<emb_key_t>& keys)
{
    EASY_FUNCTION(profiler::colors::Blue600)
    std::lock_guard<std::mutex> lk(mut);  // lock for PROCESS_THREAD

    size_t keySize = keys.size();
    auto& devHashMap = keyOffsetMap.at(embName);
    auto& evictPos = evictPosMap.at(embName);

    for (size_t i = 0; i < keySize; i++) {
        size_t offset;
        emb_key_t key = keys[i];
        if (key == -1) {
            LOG_ERROR("evict key equal -1!");
            continue;
        }
        const auto& iter = devHashMap.find(key);
        if (iter == devHashMap.end()) {  // not found
            continue;
        }
        offset = iter->second;
        devHashMap.erase(iter);
        evictPos.emplace_back(offset);
        LOG_TRACE("evict embName:{}, offset:{}", embName, offset);
    }
    LOG_INFO(KEY_PROCESS "hbm EvictDeleteDeviceEmb: [{}]! evict size on dev:{}", embName, evictPos.size());
}

void KeyProcess::EvictInitDeviceEmb(const string& embName, vector<size_t> offset)
{
    if (offset.size() > embInfos[embName].devVocabSize) {
        LOG_ERROR("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}", embName,
                  offset.size(), embInfos[embName].devVocabSize);
        throw runtime_error(
            Logger::Format("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}", embName,
                           offset.size(), embInfos[embName].devVocabSize)
                .c_str());
    }

    vector<Tensor> tmpDataOut;
    Tensor tmpData = Vec2TensorI32(offset);
    tmpDataOut.emplace_back(tmpData);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, {1}));

    auto evictLen = tmpDataOut.back().flat<int32>();
    int evictSize = static_cast<int>(offset.size());
    evictLen(0) = evictSize;

    // evict key发送给dev侧，dev侧初始化emb
    auto trans = Singleton<HDTransfer>::GetInstance();
    trans->Send(TransferChannel::EVICT, tmpDataOut, TRAIN_CHANNEL_ID, embName);

    LOG_INFO(KEY_PROCESS "hbm EvictInitDeviceEmb: [{}]! send offsetSize:{}", embName, offset.size());
}

string KeyProcess::DumpSplitKeys(vector<vector<emb_key_t>>& splitKeys) const
{
    stringstream ssTrace;
    for (int devId = 0; devId < rankInfo.rankSize; ++devId) {
        ssTrace << '|' << devId << ":";
        for (auto key : splitKeys[devId]) {
            ssTrace << key << ',';
        }
        ssTrace << '|';
    }
    return ssTrace.str();
}

int64_t KeyProcess::GetExpansionTableSize(const string& embName)
{
    return EmbeddingMgmt::Instance()->GetSize(embName);
}

int64_t KeyProcess::GetExpansionTableCapacity(const string& embName)
{
    return EmbeddingMgmt::Instance()->GetCapacity(embName);
}

void KeyProcess::RecordKeyCountMap(const unique_ptr<EmbBatchT>& batch)
{
    if (!GlobalEnv::recordKeyCount) {
        return;
    }
    std::lock_guard<std::mutex> lk(mut);
    size_t miniBs = batch->Size();
    auto* batchData = batch->sample.data();
    auto& singleKeyCountMap = keyCountMap[batch->name];
    for (size_t i = 0; i < miniBs; i++) {
        const emb_key_t& key = batchData[i];
        if (singleKeyCountMap.find(key) == singleKeyCountMap.end()) {
            singleKeyCountMap[key] = 1;
        }
        singleKeyCountMap[key]++;
    }
}

void KeyProcess::SetEos(int status, int channelId)
{
    unique_lock<mutex> lockGuard(eosMutex);
    LOG_INFO("isNeedSendEos status is changed, channel:{}, before status:{}, input status:{}", channelId,
             isNeedSendEos[channelId], status);
    isNeedSendEos[channelId] = (status == 1);
}

bool KeyProcess::IsGetInfoVecEos(int batch, const string& embName, int channel)
{
    HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();

    // 避免eos在keyProcess还未处理完数据时插队到通道前面, readEmbKey真实的次数是readEmbedBatchId减1
    int readEmbKeyBatchId = hybridMgmtBlock->readEmbedBatchId[channel] - 1;
    if (rankInfo.isDDR) {
        if (isNeedSendEos[channel] && readEmbKeyBatchId < batch &&
            hybridMgmtBlock->h2dNextBatchId[embName][channel] == batch) {
            LOG_ERROR("channelId:{} batchId:{}, GetInfoVec eos, code should not reach here", channel, batch);
            throw runtime_error("GetInfoVec eos, code should not reach here");
        }
    } else {
        LOG_TRACE("table:{}, channelId:{}, readEmbKeyBatchId:{}, batchId:{}, isNeedSendEos:{}", embName, channel,
                  readEmbKeyBatchId, batch, isNeedSendEos[channel]);
        if (isNeedSendEos[channel] && readEmbKeyBatchId < batch) {
            LOG_INFO("table:{}, channelId:{} batchId:{}, GetInfoVec eos", embName, channel, batch);
            return true;
        }
    }
    return false;
}

void KeyProcess::SendEosTensor(const std::string& embName, int channel)
{
#ifndef GTEST
    auto trans = Singleton<HDTransfer>::GetInstance();
    unordered_map<std::string, acltdtChannelHandle*> transChannels = trans->GetTransChannel();
    std::set<std::string> usedChannelNames = trans->GetUsedTransChannel()[channel];

    vector<Tensor> tensors;
    bool isNeedResend = true;
    string sendName;
    for (const string& transName : usedChannelNames) {
        if (transName == TransferChannel2Str(TransferChannel::SAVE_D2H) ||
            transName == TransferChannel2Str(TransferChannel::SAVE_H2D)) {
            // do nothing on save channel, it's independent to train, eval and predict channel;
            continue;
        }

        sendName = StringFormat("%s_%s_%d", embName.c_str(), transName.c_str(), channel);
        size_t channelSize = 0;
        acltdtQueryChannelSize(transChannels[sendName], &channelSize);
        LOG_INFO("[EOS] Before send eos, channel:{}, size:{}.", sendName, channelSize);
        SendTensorsByAcl(transChannels[sendName], ACL_TENSOR_DATA_END_OF_SEQUENCE, tensors, isNeedResend);
        acltdtQueryChannelSize(transChannels[sendName], &channelSize);
        LOG_INFO("[EOS] After send eos, channel:{}, size:{}.", sendName, channelSize);
    }
#endif
}

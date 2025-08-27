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

#include <absl/container/flat_hash_set.h>
#include <mpi.h>

#include "emb_table/embedding_mgmt.h"
#include "hd_transfer/hd_transfer.h"
#include "ock_ctr_common/include/error_code.h"
#include "utils/common.h"
#include "utils/config.h"
#include "log/logger.h"
#include "utils/safe_queue.h"
#include "utils/singleton.h"
#include "utils/time_cost.h"
#include "error/error.h"

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
                            bool isIncrementalCkpt, bool useLccl)
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
    this->enableLccl = useLccl;

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
            auto error = Error(ModuleName::M_ACL, ErrorType::UNKNOWN,
                               StringFormat("Set device failed, device_id:%d, please check plog.", rankInfo.deviceId));
            LOG_ERROR(error.ToString());
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
            batch = GetBatchData(channel, threadId);  // Get batch data from SingletonQueue<EmbBatchT>.
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

            bool isKeyProcessTaskSuccess;
            if (embInfos[batch->name].isDp) {
                // Data parallel key processing.
                isKeyProcessTaskSuccess = KeyProcessTaskHelperForDp(batch, channel, threadId);
            } else {
                // Model parallel key processing.
                isKeyProcessTaskSuccess = KeyProcessTaskHelper(batch, channel, threadId);
            }
            if (!isKeyProcessTaskSuccess) {
                break;
            }
            LOG_INFO(KEY_PROCESS
                     "getAndProcessTC(ms):{}, key process cost:{},"
                     " get data time(ms):{}, batch name:{}, channelId:{}, threadId:{}, batchId:{}, isEos:{}",
                     getAndProcessTC.ElapsedMS(), processDataTime.ElapsedMS(), getBatchTime, batch->name,
                     batch->channel, threadId, batch->batchId, batch->isEos);
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
                                 vector <vector<uint32_t>>& keyCount)
{
    TimeCost uniqueTc;
    // Deduplicate the Key, and model parallel requires bucketing, data parallel does not.
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE) {
        tie(splitKeys, restore, keyCount) = HashSplitWithFAAE(batch, embInfos[batch->name].isDp);
    } else {
        tie(splitKeys, restore, hotPos, keyCount) = HotHashSplit(batch);
    }
    LOG_DEBUG("uniqueTc(ms):{}", uniqueTc.ElapsedMS());
}

bool KeyProcess::KeyProcessTaskHelperWithFastUnique(unique_ptr<EmbBatchT>& batch, ock::ctr::UniquePtr& unique,
                                                    int channel, int threadId)
{
    // tuple for keyRec restore hotPos scAll countRecv
    isWithFAAE = m_featureAdmitAndEvict.GetFunctionSwitch() &&
                 FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE;
    TimeCost fastUniqueTC;
    UniqueInfo uniqueInfo;
    ProcessBatchWithFastUnique(batch, unique, threadId, uniqueInfo);
    LOG_DEBUG("ProcessBatchWithFastUnique(ms):{}", fastUniqueTC.ElapsedMS());

    // 特征准入&淘汰
    if (isWithFAAE && (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, uniqueInfo.all2AllInfo.keyRecv,
                                                           uniqueInfo.all2AllInfo.countRecv) ==
                       FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR)) {
        auto error = Error(ModuleName::M_FEATURE_ADMIT_AND_EVICT, ErrorType::UNKNOWN,
                           StringFormat("Feature-admit-and-evict error, check previous log for detail."));
        LOG_ERROR(error.ToString());
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
        PushGlobalUniqueTensors(move(tensors), uniqueInfo.all2AllInfo.keyRecv, channel, batch->name);
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
    return true;
}

KeysT KeyProcess::BroadcastGlobalDpIdUnique(const unique_ptr<EmbBatchT>& batch, const KeysT& globalDpIdVec,
                                            int threadId)
{
    int globalDpIdUniqueSize;
    KeysT globalDpIdUniqueVec;
    TimeCost broadcastGlobalDpIdUniqueCalTC;
    // Each thread of each card processes different batch data.
    int masterId = abs(threadId % rankInfo.rankSize);
    if (masterId == rankInfo.rankId) {
        // Duplicate removal.
        absl::flat_hash_set<emb_key_t> globalDpIdSet;
        for (emb_key_t key : globalDpIdVec) {
            auto result = globalDpIdSet.find(key);
            if (result == globalDpIdSet.end()) {
                globalDpIdUniqueVec.push_back(key);
                globalDpIdSet.insert(key);
            }
        }
        globalDpIdUniqueSize = globalDpIdUniqueVec.size();
    }
    LOG_DEBUG("Rank:{}, thread:{}, table name:{}, broadcastGlobalDpIdUniqueCalTC(ms):{}", rankInfo.rankId, threadId,
              batch->name, broadcastGlobalDpIdUniqueCalTC.ElapsedMS());

    // Broadcast globalDpIdUniqueSize.
    TimeCost globalDpIdUniqueBcastSizeCommTC;
    int retCode = MPI_Bcast(&globalDpIdUniqueSize, 1, MPI_INT, masterId, comm[batch->channel][threadId]);
    if (retCode != MPI_SUCCESS) {
        auto error =
            Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                  StringFormat("Rank %d, globalDpIdUniqueSize MPI_Bcast failed: %d.", rankInfo.rankId, retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("Rank:{}, thread:{}, table name:{}, globalDpIdUniqueBcastSizeCommTC(ms):{}", rankInfo.rankId, threadId,
              batch->name, globalDpIdUniqueBcastSizeCommTC.ElapsedMS());

    // Broadcast globalDpIdUniqueVec.
    globalDpIdUniqueVec.resize(globalDpIdUniqueSize);
    TimeCost globalDpIdUniqueBcastVecCommTC;
    retCode = MPI_Bcast(globalDpIdUniqueVec.data(), globalDpIdUniqueSize, MPI_INT64_T, masterId,
                        comm[batch->channel][threadId]);
    if (retCode != MPI_SUCCESS) {
        auto error =
            Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                  StringFormat("Rank %d, globalDpIdUniqueVec MPI_Bcast failed: %d.", rankInfo.rankId, retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("Rank:{}, thread:{}, table name:{}, globalDpIdUniqueSize:{}, globalDpIdUniqueBcastVecCommTC(ms):{}",
              rankInfo.rankId, threadId, batch->name, globalDpIdUniqueSize, globalDpIdUniqueBcastVecCommTC.ElapsedMS());
    return globalDpIdUniqueVec;
}

bool KeyProcess::KeyProcessTaskHelperForDp(unique_ptr<EmbBatchT>& batch, int channel, int threadId)
{
    if (batch->isEos) {
        HandleEos(batch, channel, threadId);
        return true;
    }
    vector<KeysT> splitKeys;
    vector<int32_t> restore;
    vector<int32_t> hotPos;
    vector<vector<uint32_t>> keyCount;
    vector<emb_key_t> keyCountVec;
    TimeCost totalTimeCost = TimeCost();
    HashSplitHelper(batch, splitKeys, restore, hotPos, keyCount);

    vector<int> scAll;
    vector<int> ss;
    KeysT lookupKeys;
    KeysT globalDpIdVec;
    lookupKeys = splitKeys[rankInfo.rankId];
    tie(globalDpIdVec, scAll, ss) = ProcessSplitKeys(batch, threadId, splitKeys);

    KeysT globalDpIdUniqueVec;
    vector<uint32_t> countRecv;
    bool featureAdmitAndEvictSwitch = m_featureAdmitAndEvict.GetFunctionSwitch() &&
                                      FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE;
    if (!featureAdmitAndEvictSwitch) {
        TimeCost broadcastGlobalDpIdUniqueTC;
        globalDpIdUniqueVec = BroadcastGlobalDpIdUnique(batch, globalDpIdVec, threadId);
        LOG_DEBUG(KEY_PROCESS "Rank:{}, thread:{}, table name:{}, broadcastGlobalDpIdUniqueTC(ms):{}", rankInfo.rankId,
                  threadId, batch->name, broadcastGlobalDpIdUniqueTC.ElapsedMS());
    } else {
        // No need to lock.
        countRecv = GetCountRecvForDp(batch, threadId, keyCount[rankInfo.rankId], scAll);
    }

    std::lock_guard<std::mutex> lock(loadSaveMut[channel][threadId]);
    RecordKeyCountMap(batch);

    // Feature admit and evict.
    if (featureAdmitAndEvictSwitch) {
        // Use global ids.
        if (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, globalDpIdVec, countRecv) ==
            FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR) {
            auto error = Error(ModuleName::M_FEATURE_ADMIT_AND_EVICT, ErrorType::UNKNOWN,
                               StringFormat("Feature-admit-and-evict error, check previous log for detail."));
            LOG_ERROR(error.ToString());
            return false;
        }

        // Process the local key once and synchronize non-access ids from the global ids to the local ids,
        // and get the globalDpIdUniqueVec.
        globalDpIdUniqueVec = FeatureAdmitForDp(lookupKeys, globalDpIdVec);
    }

    // Without host, just device, all embedding vectors were stored in device map key to offset directly by
    // lookup keyOffsetMap (hashmap).
    if (!rankInfo.isDDR) {
        EmbeddingMgmt::Instance()->Key2Offset(batch->name, globalDpIdUniqueVec, channel);
        EmbeddingMgmt::Instance()->Key2OffsetForDp(batch->name, lookupKeys, channel);
    }

    TimeCost pushResultTC;
    auto tensors = make_unique<vector<Tensor>>();
    tensors->push_back(Vec2TensorI32(restore));

    hotPos.resize(hotEmbTotCount[batch->name], 0);
    tensors->push_back(Vec2TensorI32(hotPos));

    if (!rankInfo.isDDR) {
        PushGlobalUniqueTensorsForDp(tensors, lookupKeys, channel, globalDpIdUniqueVec, batch->name);
        tensors->push_back(rankInfo.useDynamicExpansion ? Vec2TensorI64(lookupKeys) : Vec2TensorI32(lookupKeys));
        PushResultHBM(batch, move(tensors));
    } else {
        std::vector<uint64_t> lookupKeysUint(lookupKeys.begin(), lookupKeys.end());
        vector<uint64_t> uniqueKeys;
        vector<int32_t> restoreVecSec;
        GlobalUniqueForDp(lookupKeysUint, uniqueKeys, restoreVecSec, globalDpIdUniqueVec);
        PushResultDDR(batch, move(tensors), uniqueKeys, restoreVecSec);
    }
    LOG_DEBUG("Rank:{}, pushResultTC(ms):{}", rankInfo.rankId, pushResultTC.ElapsedMS());
    return true;
}

void KeyProcess::PushResultBasedOnMemoryMode(unique_ptr <EmbBatchT>& batch, unique_ptr<vector<Tensor>> tensors,
                                             int channel, unique_ptr<vector<Tensor>> keyCountTensors,
                                             std::vector<emb_key_t>& lookupKeys)
{
    if (!rankInfo.isDDR) {
        PushGlobalUniqueTensors(tensors, lookupKeys, channel, batch->name);
        // The first-order optimizer does not have a second USS, which needs to mask the lookupKeys.
        if (!rankInfo.useSumSameIdGradients) {
            PushPaddingKeysTensors(tensors, batch->name, channel, lookupKeys);
        }
        tensors->push_back(rankInfo.useDynamicExpansion ? Vec2TensorI64(lookupKeys) : Vec2TensorI32(lookupKeys));
        PushResultHBM(batch, move(tensors));
        if (isIncrementalCheckpoint && channel == TRAIN_CHANNEL_ID) {
            PushKeyCount(batch, move(keyCountTensors));
        }
        return;
    }
    vector<uint64_t> lookupKeysUint(lookupKeys.begin(), lookupKeys.end());
    vector<uint64_t> uniqueKeys;
    vector<int32_t> restoreVecSec;
    GlobalUnique(lookupKeysUint, uniqueKeys, restoreVecSec);
    PushResultDDR(batch, move(tensors), uniqueKeys, restoreVecSec);
    if (isIncrementalCheckpoint && channel == TRAIN_CHANNEL_ID) {
        PushKeyCount(batch, move(keyCountTensors));
    }
}

bool KeyProcess::KeyProcessTaskHelper(unique_ptr<EmbBatchT>& batch, int channel, int threadId)
{
    if (batch->isEos) {
        HandleEos(batch, channel, threadId);
        return true;
    }
    vector<KeysT> splitKeys;
    vector<int32_t> restore;
    vector<int32_t> hotPos;
    vector<vector<uint32_t>> keyCount;
    vector<emb_key_t> keyCountVec;
    HashSplitHelper(batch, splitKeys, restore, hotPos, keyCount);
    size_t uniqueKeyNum = 0;
    if (enableLccl && !rankInfo.useStatic) {
        for (int devId = 0; devId < rankInfo.rankSize; ++devId) {
            uniqueKeyNum += splitKeys[devId].size();
        }
    }
    auto [lookupKeys, scAll, ss] = ProcessSplitKeys(batch, threadId, splitKeys);

    vector<uint32_t> countRecv;
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE) {
        countRecv = GetCountRecv(batch, threadId, keyCount, scAll, ss);
    }
    if (isIncrementalCheckpoint && channel == TRAIN_CHANNEL_ID) {
        countRecv = GetCountRecv(batch, threadId, keyCount, scAll, ss);
        map<emb_key_t, emb_key_t> tmpKeyCountMap;
        auto keySize = lookupKeys.size();
        for (int i = 0; i < keySize; ++i) {
            tmpKeyCountMap[lookupKeys[i]] += countRecv[i];
        }
        for (const auto& it : tmpKeyCountMap) {
            keyCountVec.push_back(it.first);
            keyCountVec.push_back(it.second);
        }
        LOG_INFO("Current batch: {}, emb table:{} , key count size is: {}, key count: {}.", batch->batchId+1,
                 batch->name, tmpKeyCountMap.size(), VectorToString(keyCountVec));
    }
    std::lock_guard<std::mutex> lock(loadSaveMut[channel][threadId]);
    RecordKeyCountMap(batch);
    BuildRestoreVec(batch, ss, restore, static_cast<int>(hotPos.size()));

    // 特征准入&淘汰
    if (m_featureAdmitAndEvict.GetFunctionSwitch() &&
        FeatureAdmitAndEvict::m_embStatus[batch->name] != SingleEmbTableStatus::SETS_NONE &&
        (m_featureAdmitAndEvict.FeatureAdmit(channel, batch, lookupKeys, countRecv) ==
         FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR)) {
        auto error = Error(ModuleName::M_FEATURE_ADMIT_AND_EVICT, ErrorType::UNKNOWN,
                           StringFormat("Feature-admit-and-evict error, check previous log for detail."));
        LOG_ERROR(error.ToString());
        return false;
    }

    // without host, just device, all embedding vectors were stored in device
    // map key to offset directly by lookup keyOffsetMap (hashmap)
    if (!rankInfo.isDDR) {
        TimeCost key2OffsetTC;
        EmbeddingMgmt::Instance()->Key2Offset(batch->name, lookupKeys, channel);
        LOG_DEBUG("key2OffsetTC(ms):{}, batchId:{}, emb table:{}, key size:{}", key2OffsetTC.ElapsedMS(),
                  batch->batchId, batch->name, lookupKeys.size());
    }

    // Static all2all，need send count
    if (!rankInfo.useStatic) {
        SendA2A(scAll, batch->name, batch->channel, batch->batchId);
    }

    TimeCost pushResultTC;
    auto tensors = make_unique<vector<Tensor>>();
    tensors->push_back(Vec2TensorI32(restore));

    // 将keyCountVec放进tensor里并推到一个队列里
    auto keyCountTensors = make_unique<vector<Tensor>>();
    if (isIncrementalCheckpoint && channel == TRAIN_CHANNEL_ID) {
        keyCountTensors->push_back(Vec2TensorI64(keyCountVec));
    }

    hotPos.resize(hotEmbTotCount[batch->name], 0);
    tensors->push_back(Vec2TensorI32(hotPos));

    if (enableLccl && !rankInfo.useStatic) {
        vector<float> all2AllRecvShape {};
        LOG_INFO("Create all2AllRecvShape, uniqueKeyNum:{}.", uniqueKeyNum);
        all2AllRecvShape.resize(uniqueKeyNum, 0);
        tensors->push_back(Vec2TensorI32(all2AllRecvShape));
    }

    // Tensors contains restore、hotPos、restoreSec&unique、idOffset in order when HBM mode, and is pushed in infolist.
    PushResultBasedOnMemoryMode(batch, move(tensors), channel, move(keyCountTensors), lookupKeys);

    LOG_DEBUG("pushResultTC(ms):{}", pushResultTC.ElapsedMS());
    return true;
}

void KeyProcess::HandleEos(unique_ptr<EmbBatchT>& batch, int channel, int threadId)
{
    if (!rankInfo.isDDR) {  // HBM
        std::unique_lock<std::mutex> lockGuard(mut);
        infoList[batch->name][batch->channel].push(
            make_tuple(batch->batchId, batch->name, batch->isEos, storage.begin()));
        lockGuard.unlock();
        LOG_INFO("KeyProcessTaskHelper hbm eos, batch name:{}, batch id: {}, channelId:{} threadId:{}", batch->name,
                 batch->batchId, batch->channel, threadId);
        return;
    }
    // DDR
    vector<uint64_t> uniqueKeys;
    std::unique_lock<std::mutex> lockGuard(mut);
    uniqueKeysList[batch->name][batch->channel].push(
        make_tuple(batch->batchId, batch->name, batch->isEos, move(uniqueKeys)));
    lockGuard.unlock();
    LOG_INFO("KeyProcessTaskHelper ddr eos, batch name:{}, batch id: {}, channelId:{} threadId:{}", batch->name,
             batch->batchId, batch->channel, threadId);
}

KeysT KeyProcess::FeatureAdmitForDp(KeysT& lookupKeys, KeysT& globalDpIdVec)
{
    KeysT globalDpIdUniqueVec;
    TimeCost featureAdmitForDpTC;
    // Duplicate removal, and get globalDpIdUniqueVec.
    absl::flat_hash_set<emb_key_t> globalDpIdIdxSet;
    for (emb_key_t key : globalDpIdVec) {
        auto result = globalDpIdIdxSet.find(key);
        if (result == globalDpIdIdxSet.end()) {
            globalDpIdIdxSet.insert(key);
            globalDpIdUniqueVec.push_back(key);
        }
    }
    // The global ids is a superset of the local ids. If the local ids has an id that the global ids does not have,
    // it is non-access.
    for (auto& key : lookupKeys) {
        auto result = globalDpIdIdxSet.find(key);
        if (result == globalDpIdIdxSet.end()) {
            key = INVALID_KEY_VALUE;
        }
    }
    LOG_DEBUG("Rank:{}, featureAdmitForDpTC(ms):{}", rankInfo.rankId, featureAdmitForDpTC.ElapsedMS());
    return globalDpIdUniqueVec;
}

void KeyProcess::PushGlobalUniqueTensors(const unique_ptr<vector<Tensor>>& tensors, KeysT& lookupKeys, int channel,
                                         const string& embName)
{
    LOG_INFO(KEY_PROCESS "rank:{}, channel:{}, useSumSameIdGradients:{} ...", rankInfo.rankId, channel,
             rankInfo.useSumSameIdGradients);
    if (rankInfo.useSumSameIdGradients && channel == TRAIN_CHANNEL_ID) {
        KeysT uniqueKeys;
        vector<int32_t> restoreVecSec;

        TimeCost globalUniqueSyncTC;
        GlobalUnique(lookupKeys, uniqueKeys, restoreVecSec);
        LOG_DEBUG("globalUniqueSyncTC(ms):{}", globalUniqueSyncTC.ElapsedMS());

        // The second-order optimizer has a second USS, which needs to mask the uniqueKeys.
        PushPaddingKeysTensors(tensors, embName, channel, uniqueKeys);

        tensors->push_back(Vec2TensorI32(restoreVecSec));
        tensors->push_back(rankInfo.useDynamicExpansion ? Vec2TensorI64(uniqueKeys) : Vec2TensorI32(uniqueKeys));
    }
}

void KeyProcess::PushGlobalUniqueTensorsForDp(const unique_ptr<vector<Tensor>>& tensors, KeysT& lookupKeys, int channel,
                                              KeysT& globalDpIdUniqueVec, const string& embName)
{
    LOG_INFO(KEY_PROCESS "Rank:{}, channel:{}, table name:{}, useSumSameIdGradients:{}.", rankInfo.rankId, channel,
             embName, rankInfo.useSumSameIdGradients);
    // In the DP mode, the second USS is used to align the length of the grad in the allreduce.
    if (channel == TRAIN_CHANNEL_ID) {
        KeysT uniqueKeys;
        vector<int32_t> restoreVecSec;
        TimeCost globalUniqueSyncTC;
        GlobalUniqueForDp(lookupKeys, uniqueKeys, restoreVecSec, globalDpIdUniqueVec);
        LOG_DEBUG("Rank:{}, globalUniqueSyncTC(ms):{}", rankInfo.rankId, globalUniqueSyncTC.ElapsedMS());
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
        auto error = Error(ModuleName::M_KEY_PROCESS, ErrorType::UNKNOWN,
                           StringFormat("MPI_Alltoallv failed, error:%d.", retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }
    LOG_DEBUG("getCountRecvTC(ms)(with-all2all):{}", getCountRecvTC.ElapsedMS());
    return countRecv;
}

vector<uint32_t> KeyProcess::GetCountRecvForDp(const unique_ptr<MxRec::EmbBatchT>& batch, const int id,
                                               vector<uint32_t>& keyCount, vector<int> scAll)
{
    TimeCost getCountRecvDpTC;
    if (rankInfo.useStatic) {
        keyCount.resize(embInfos[batch->name].sendCount, 0);
    }
    size_t sc = keyCount.size();
    vector<int> rc;
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        rc.push_back(scAll.at(i));
    }
    vector<int> rs = Count2Start(rc);
    vector<uint32_t> countRecv;
    countRecv.resize(rs.back() + rc.back());
    int retCode = MPI_Allgatherv(keyCount.data(), sc, MPI_UINT32_T, countRecv.data(), rc.data(), rs.data(),
                                 MPI_UINT32_T, comm[batch->channel][id]);
    if (retCode != MPI_SUCCESS) {
        auto error =
            Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                  StringFormat("Rank %d, MPI_Allgatherv for count receive failed: %d.", rankInfo.rankId, retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("Rank {}, getCountRecvDpTC(ms)(with-allgather):{}", rankInfo.rankId, getCountRecvDpTC.ElapsedMS());
    return countRecv;
}

void KeyProcess::PushResultHBM(unique_ptr<EmbBatchT>& batch, unique_ptr<vector<Tensor>> tensors)
{
    std::unique_lock<std::mutex> lockGuard(mut);
    storage.push_front(move(tensors));
    infoList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, batch->isEos, storage.begin()));
    lockGuard.unlock();
}

void KeyProcess::PushResultDDR(unique_ptr<EmbBatchT>& batch, unique_ptr<vector<Tensor>> tensors,
                               std::vector<uint64_t>& uniqueKeys, std::vector<int32_t>& restoreVecSec)
{
    std::unique_lock<std::mutex> lockGuard(mut);
    storage.push_front(move(tensors));
    infoList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, batch->isEos, storage.begin()));
    uniqueKeysList[batch->name][batch->channel].push(
        make_tuple(batch->batchId, batch->name, batch->isEos, move(uniqueKeys)));
    restoreVecSecList[batch->name][batch->channel].push(make_tuple(batch->batchId, batch->name, move(restoreVecSec)));
    lockGuard.unlock();
}

void KeyProcess::PushKeyCount(unique_ptr<EmbBatchT>& batch, unique_ptr<vector<Tensor>> tensors)
{
    std::unique_lock<std::mutex> lockGuard(mut);
    keyCountStorage.push_front(move(tensors));
    keyCountInfoList[batch->name][batch->channel].push(
        make_tuple(batch->batchId, batch->name, batch->isEos, keyCountStorage.begin()));
    lockGuard.unlock();
    LOG_INFO("Push key count to list success.");
}

/*
 * 从共享队列SingletonQueue<EmbBatchT>中读取batch数据并返回。batch数据由 ReadEmbKeyV2 写入。
 * commID为线程标识[0, KEY_PROCESS_THREAD-1]，不同线程、训练或推理数据用不同的共享队列通信
 */
unique_ptr<EmbBatchT> KeyProcess::GetBatchData(int channel, int commId) const
{
    unique_ptr<EmbBatchT> batch = nullptr;

    // train data, queue id = thread id [0, KEY_PROCESS_THREAD-1]
    int queueIndex = commId + (MAX_KEY_PROCESS_THREAD * channel);
    auto batchQueue = SingletonQueue<EmbBatchT>::GetInstances(queueIndex);
    TimeCost tc = TimeCost();
    while (true) {
        batch = batchQueue->TryPop();
        if (batch != nullptr) {
            if (batch->isEos) {
                LOG_INFO("GetBatchData eos, table name:{}, batchId:{}, channelId:{} threadId:{}", batch->name,
                         batch->batchId, channel, commId);
            }
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
    LOG_DEBUG(KEY_PROCESS "channelId:{} threadId:{} batchId:{}, get batch data done, batchName:{}. bs:{} sample:[{}]",
              batch->channel, commId, batch->batchId, batch->name, batch->Size(), batch->UnParse());
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
    LOG_DEBUG("FastUniqueCompute(ms):{}, ret:{}", uniqueTC.ElapsedMS(), ret);

    vector<int> sc;
    HandleHotAndSendCount(batch, uniqueInfoOut, keySendInfo, sc, splitSize);

    All2All(sc, id, batch, keySendInfo, uniqueInfoOut.all2AllInfo);

    LOG_DEBUG(KEY_PROCESS "ProcessBatchWithFastUnique get batchId:{}, batchSize:{},"
                          " channel:{}, name:{}, restore:{}, keyCount:{}",
              batch->batchId, batch->Size(), batch->channel, batch->name, uniqueInfoOut.restore.size(),
              keySendInfo.keyCount.size());
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
    int retCode = MPI_Alltoallv(keySendInfo.keySend.data(), sc.data(), ss.data(), MPI_INT64_T,
                                all2AllInfoOut.keyRecv.data(), rc.data(), rs.data(), MPI_INT64_T, comm[channel][id]);
    if (retCode != MPI_SUCCESS) {
        auto error = Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                           StringFormat("MPI_Allgatherv for count receive failed, error:%d.", retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, All2All MPI_Alltoallv end.", channel, id, batch->batchId);
    all2AllInfoOut.countRecv.resize(rs.back() + rc.back());
    if (isWithFAAE) {
        retCode = MPI_Alltoallv(keySendInfo.keyCount.data(), sc.data(), ss.data(), MPI_UINT32_T,
                                all2AllInfoOut.countRecv.data(), rc.data(), rs.data(), MPI_UINT32_T, comm[channel][id]);
        if (retCode != MPI_SUCCESS) {
            auto error = Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                               StringFormat("MPI_Alltoallv failed, error:%d.", retCode));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, All2All end, all2allTC TimeCost(ms):{}", channel, id,
              batch->batchId, all2allTC.ElapsedMS());
}

void KeyProcess::ProcessKeysWithStatic(const unique_ptr<EmbBatchT>& batch, vector<MxRec::KeysT>& splitKeys)
{
    for (KeysT& i : splitKeys) {
        if (i.size() > embInfos[batch->name].sendCount) {
            auto error = Error(ModuleName::M_KEY_PROCESS, ErrorType::INVALID_ARGUMENT,
                               StringFormat("%s[%d]:%d overflow! set send count bigger than %d.", batch->name.c_str(),
                                            batch->channel, batch->batchId, i.size()));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
        i.resize(embInfos[batch->name].sendCount, -1);
    }
}

auto KeyProcess::ProcessSplitKeys(const unique_ptr<EmbBatchT>& batch, int id, vector<KeysT>& splitKeys)
    -> tuple<KeysT, vector<int>, vector<int>>
{
    TimeCost processSplitKeysTC;
    LOG_INFO(KEY_PROCESS "channelId:{} threadId:{} batchId:{}, ProcessSplitKeys start.", batch->channel, id,
             batch->batchId);

    // 使用静态all2all通信：发送或接受量为预置固定值 scInfo[batch->name] = 65536 / rankSize 经验值
    if (rankInfo.useStatic) {  // maybe move after all2all
        ProcessKeysWithStatic(batch, splitKeys);
    }

    // Data parallel mode requires allgather on the local key to obtain the global key.
    if (embInfos[batch->name].isDp) {
        auto [keyRecv, scAll, ss] = ProcessGlobalDpId(batch, id, splitKeys[rankInfo.rankId]);
        LOG_DEBUG(KEY_PROCESS "channelId:{} threadId:{} batchId:{}, batchName:{}, MPI_Allgatherv finish."
                              " processSplitKeysTC(ms):{}",
                  batch->channel, id, batch->batchId, batch->name, processSplitKeysTC.ElapsedMS());
        return {keyRecv, scAll, ss};
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

    TimeCost uniqueAll2AllTC;
    int retCode = MPI_Alltoallv(keySend.data(), sc.data(), ss.data(), MPI_INT64_T, keyRecv.data(), rc.data(), rs.data(),
                                MPI_INT64_T, comm[batch->channel][id]);
    if (retCode != MPI_SUCCESS) {
        auto error = Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                           StringFormat("Rank %d, MPI_Alltoallv failed: %d.", rankInfo.rankId, retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("uniqueAll2AllTC(ms):{}", uniqueAll2AllTC.ElapsedMS());

    LOG_DEBUG(KEY_PROCESS "channelId:{} threadId:{} batchId:{}, batchName:{}, MPI_Alltoallv finish."
                          " processSplitKeysTC(ms):{}",
              batch->channel, id, batch->batchId, batch->name, processSplitKeysTC.ElapsedMS());
    return {keyRecv, scAll, ss};
}

tuple<KeysT, vector<int>, vector<int>> KeyProcess::ProcessGlobalDpId(const unique_ptr<EmbBatchT>& batch, int id,
                                                                     KeysT& lookupKeys)
{
    vector<int> sc{static_cast<int>(lookupKeys.size())};
    vector<int> scAll = GetScAll(sc, id, batch);

    vector<int> rc;  // receive count
    for (int i = 0; i < rankInfo.rankSize; ++i) {
        rc.push_back(scAll.at(i));
    }
    vector<int> rs = Count2Start(rc);  // receive displays/offset

    KeysT keyRecv;
    keyRecv.resize(rs.back() + rc.back());

    TimeCost uniqueAllGatherTC;
    int retCode = MPI_Allgatherv(lookupKeys.data(), sc.at(0), MPI_INT64_T, keyRecv.data(), rc.data(), rs.data(),
                                 MPI_INT64_T, comm[batch->channel][id]);
    if (retCode != MPI_SUCCESS) {
        auto error = Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                           StringFormat("Rank %d, MPI_Allgatherv failed: %d.", rankInfo.rankId, retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("uniqueAllGatherTC(ms):{}", uniqueAllGatherTC.ElapsedMS());
    return {keyRecv, scAll, Count2Start(sc)};
}

/*
 * 将batch内的key按照所存储的dev id哈希切分并去重，哈希函数为模运算
 * splitKeys返回：将数据的key切分到其所在dev id对应的桶中，并去重。
 * restore返回：去重后key在桶内偏移量（用于计算恢复向量）
 */
tuple<vector<KeysT>, vector<int32_t>> KeyProcess::HashSplit(const unique_ptr<EmbBatchT>& batch) const
{
    emb_key_t* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    vector<KeysT> splitKeys(rankInfo.rankSize);
    vector<int32_t> restore(batch->Size());
    vector<int> hashSplitLens(rankInfo.rankSize);  // 初始化全0，记录每个桶的长度
    absl::flat_hash_map<emb_key_t, int> uKey;      // 用于去重查询
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
    LOG_TRACE("dump splitKeys {}", DumpSplitKeys(splitKeys));
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
    const unique_ptr<EmbBatchT>& batch, bool isDp) const
{
    emb_key_t* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    vector<KeysT> splitKeys(rankInfo.rankSize);
    vector<vector<uint32_t>> keyCount(rankInfo.rankSize);  // splitKeys在原始batch中对应的频次
    vector<int32_t> restore(batch->Size());
    vector<int> hashSplitLens(rankInfo.rankSize);                   // 初始化全0，记录每个桶的长度
    absl::flat_hash_map<emb_key_t, std::pair<int, uint32_t>> uKey;  // 用于去重查询
    for (size_t i = 0; i < miniBs; i++) {
        const emb_key_t& key = batchData[i];
        auto result = uKey.find(key);
        if (result == uKey.end()) {
            // Model parallel requires bucketing, data parallel does not.
            emb_key_t devId;
            if (isDp) {
                devId = static_cast<emb_key_t>(rankInfo.rankId);
            } else {
                devId = abs(key % static_cast<emb_key_t>(rankInfo.rankSize));
            }

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

    LOG_TRACE("dump splitKeys {}", DumpSplitKeys(splitKeys));
    return {splitKeys, restore, keyCount};
}

tuple<vector<KeysT>, vector<int32_t>, vector<int>, vector<vector<uint32_t>>> KeyProcess::HotHashSplit(const
unique_ptr<EmbBatchT>& batch)
{
    emb_key_t* batchData = batch->sample.data();
    size_t miniBs = batch->Size();
    vector<KeysT> splitKeys(rankInfo.rankSize);
    vector<int32_t> restore(batch->Size());
    absl::flat_hash_map<emb_key_t, std::pair<int, uint32_t>> uKey;  // 用于去重查询
    absl::flat_hash_map<emb_key_t, int> keyCountMapByEmbName;
    vector<vector<uint32_t>> keyCount(rankInfo.rankSize);
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
        auto result = uKey.find(key);
        if (result != uKey.end()) {  // // already in splitKeys
            restore[i] = result->second.first;
            uKey[key].second++;
            continue;
        }

        // Model parallel requires bucketing, data parallel does not.
        emb_key_t devId;
        if (embInfos[batch->name].isDp) {
            devId = static_cast<emb_key_t>(rankInfo.rankId);
        } else {
            devId = abs(key % static_cast<emb_key_t>(rankInfo.rankSize));
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
        uKey[key].first = restore[i];
        uKey[key].second = 1;
    }
    // Process key count in splitKeys
    if (isIncrementalCheckpoint) {
        for (int j = 0; j < rankInfo.rankSize; ++j) {
            vector<uint32_t> count;
            for (size_t k = 0; k < splitKeys[j].size(); ++k) {
                count.emplace_back(uKey[splitKeys[j][k]].second);
            }
            keyCount[j] = count;
        }
    }

    UpdateHotMap(keyCountMapByEmbName, hotEmbTotCount[batch->name], batch->batchId % hotEmbUpdateStep == 0,
                 batch->name);
    // DP mode does not need to accumulate device offsets.
    if (!embInfos[batch->name].isDp) {
        AddCountStartToHotPos(splitKeys, hotPos, hotPosDev, batch);
    }
    return {splitKeys, restore, hotPos, keyCount};
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
    vector<int> scAll;
    int sendAndRecvCount;
    if (embInfos[batch->name].isDp) {
        scAll.resize(rankInfo.rankSize);
        sendAndRecvCount = 1;
    } else {
        scAll.resize(rankInfo.rankSize * rankInfo.rankSize);
        sendAndRecvCount = rankInfo.rankSize;
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, GetScAll start.", batch->channel, commId, batch->batchId);

    // allgather keyScLocal(key all2all keyScLocal = device all2all rc)
    auto retCode = MPI_Allgather(keyScLocal.data(), sendAndRecvCount, MPI_INT, scAll.data(), sendAndRecvCount, MPI_INT,
                                 comm[batch->channel][commId]);
    if (retCode != MPI_SUCCESS) {
        auto error =
            Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                  StringFormat("Rank %d commId %d, MPI_Allgather failed: %d.", rankInfo.rankId, commId, retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, GetScAll MPI_Allgather end, key scAll matrix:\n{}", batch->channel,
              commId, batch->batchId, VectorToString(scAll));
    return scAll;
}

void KeyProcess::GetScAllForUnique(const vector<int>& keyScLocal, int commId, const unique_ptr<EmbBatchT>& batch,
                                   vector<int>& scAllOut)
{
    int channel = batch->channel;
    scAllOut.resize(rankInfo.rankSize * rankInfo.rankSize);

    // allgather keyScLocal(key all2all keyScLocal = device all2all rc)
    auto retCode = MPI_Allgather(keyScLocal.data(), rankInfo.rankSize, MPI_INT, scAllOut.data(), rankInfo.rankSize,
                                 MPI_INT, comm[channel][commId]);
    if (retCode != MPI_SUCCESS) {
        auto error = Error(ModuleName::M_KEY_PROCESS, ErrorType::MPI_ERROR,
                           StringFormat("MPI_Allgather failed, error:%d.", retCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("channelId:{} threadId:{} batchId:{}, GetScAllForUnique end, key scAllOut matrix:\n{}", channel, commId,
              batch->batchId, VectorToString(scAllOut));
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
        LOG_WARN("Wrong batch id, top:{} getting:{}, channel:{}, may not clear channel.",
                 topBatch, info.batchId, info.channelId);
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

template <class T>
T KeyProcess::GetKeyCountVec(info_list_t<T>& list, const EmbBaseInfo& info)
{
    std::lock_guard<std::mutex> lockGuard(mut);
    if (list[info.name][info.channelId].empty()) {
        auto error = MxRec::Error(ModuleName::M_KEY_PROCESS, ErrorType::LIST_EMPTY,
                                  StringFormat("Get info list is empty, please check if the channel id and"
                                               " info name is correct, or check if the list is correct."));
        LOG_ERROR(error.ToString());
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
            isEos = get<bool>(infoVec);
            if (isEos) {
                LOG_INFO(KEY_PROCESS "GetUniqueKeys eos! {}[{}]:{}", info.name, info.channelId, info.batchId);
                break;
            }
            ret = get<std::vector<uint64_t>>(infoVec);
            break;
        } catch (EmptyList&) {
            LOG_TRACE("getting unique info failed {}[{}], list is empty, and mgmt batchId: {}, readEmbKey batchId: {}.",
                      info.name, info.channelId, info.batchId, hybridMgmtBlock->readEmbedBatchId[info.channelId]);
            this_thread::sleep_for(1ms);
        } catch (WrongListTop&) {
            LOG_TRACE("getting info failed table:{}, channel:{}, mgmt batchId:{}, wrong top", info.name, info.channelId,
                      info.channelId);
            this_thread::sleep_for(1ms);
        }
    }
    return ret;
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
            LOG_TRACE("getting info failed {}[{}], list is empty, and mgmt batchId: {}, readEmbKey batchId: {}.",
                      info.name, info.channelId, info.batchId, hybridMgmtBlock->readEmbedBatchId[info.channelId]);
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
        if (!isRunning) {
            LOG_INFO("isRunning in KeyProcess is false, table:{} no need to wait other table enter SendEos", embName);
            break;
        }
    }
    LOG_INFO("table:{}, channelId:{} batchId:{}, SendEos start, acquiring destroyMutex", embName, channel, batchId);
    destroyMutex.lock();

    LOG_INFO("table:{}, channelId:{} batchId:{}, SendEos start", embName, channel, batchId);
    if (!isRunning) {
        LOG_INFO("other table trigger eos ahead, KeyProcess already destroyed. skip sending eos for table:{}", embName);
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

    LOG_DEBUG("sendEos finish all, table:{}, channelId:{} batchId:{}", embName, channel, batchId);
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
            isEos = get<bool>(infoVec);
            if (isEos) {
                LOG_INFO(KEY_PROCESS "GetInfoVec eos! {}[{}]:{}", info.name, info.channelId, info.batchId);
                break;
            }
            auto it = get<std::list<unique_ptr<vector<Tensor>>>::iterator>(infoVec);
            ret = std::move(*it);
            std::unique_lock<std::mutex> lockGuard(mut);
            storage.erase(it);
            break;
        } catch (EmptyList&) {
            LOG_TRACE("getting info failed {}[{}], list is empty, and mgmt batchId: {}, readEmbKey batchId: {}.",
                      info.name, info.channelId, info.batchId, hybridMgmtBlock->readEmbedBatchId[info.channelId]);
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
    all2AllList[embName][channel].push(make_tuple(batch, embName, false, storage.begin()));
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
        } else {
            singleKeyCountMap[key]++;
        }
    }
}

void KeyProcess::EnqueueEosBatch(int64_t batchNum, int channelId)
{
    LOG_INFO("Enqueue dataSet eos on batch queue, channel:{}, eos number:{}", channelId, batchNum);
    int threadNum = GetThreadNumEnv();
    int batchQueueId = int(batchNum % threadNum) + (MAX_KEY_PROCESS_THREAD * channelId);
    auto queue = SingletonQueue<EmbBatchT>::GetInstances(batchQueueId);
    for (auto& emb : embInfos) {
        auto batchData = queue->GetOne();  // get dirty or empty data block
        batchData->name = emb.first;
        batchData->channel = channelId;
        batchData->batchId = batchNum;
        batchData->sample = {0, 0, 0, 0, 0, 0, 0, 0};  // fake data
        batchData->isEos = true;
        queue->Pushv(move(batchData));
    }
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

void KeyProcess::PushPaddingKeysTensors(const unique_ptr<vector<Tensor>>& tensors, const string& embName, int channel,
                                        vector<emb_key_t>& offsetKeys)
{
    if (channel != TRAIN_CHANNEL_ID || !embInfos[embName].paddingKeysMask) {
        return;
    }

    TimeCost pushMaskSyncTC;
    auto paddingKeysOffset = EmbeddingMgmt::Instance()->GetPaddingKeysOffset(embName);
    vector<int64_t> paddingKeysMask(offsetKeys.size(), 1);
    for (size_t i = 0; i < offsetKeys.size(); i++) {
        int64_t key = offsetKeys[i];
        if (paddingKeysOffset.find(key) != paddingKeysOffset.end()) {
            paddingKeysMask[i] = 0;
        }
    }

    tensors->push_back(Vec2TensorI32(paddingKeysMask));
    LOG_DEBUG("In PushPaddingKeysTensors, pushMaskSyncTC(ms):{}.", pushMaskSyncTC.ElapsedMS());
}

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */
#include "hybrid_mgmt.h"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/ranges.h>

#include "checkpoint/checkpoint.h"
#include "utils/time_cost.h"

using namespace MxRec;
using namespace std;

bool HybridMgmt::InitKeyProcess(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos,
                                const vector<ThresholdValue>& thresholdValues, bool ifLoad, int seed)
{
    if (getenv("KEY_PROCESS_THREAD_NUM") != nullptr) {
        int num = std::atoi(getenv("KEY_PROCESS_THREAD_NUM"));
        if (num < 1 || num > MAX_KEY_PROCESS_THREAD) {
            spdlog::error("[HybridMgmt::InitKeyProcess] KEY_PROCESS_THREAD_NUM:{}, should in range [1, {}]",
                          num, MAX_KEY_PROCESS_THREAD);
            return false;
        }
        PerfConfig::keyProcessThreadNum = num;
        spdlog::info("config KEY_PROCESS_THREAD_NUM:{}", num);
    }

    if (getenv("MAX_UNIQUE_THREAD_NUM") != nullptr) {
        int num = std::atoi(getenv("MAX_UNIQUE_THREAD_NUM"));
        if (num < 1 || num > DEFAULT_MAX_UNIQUE_THREAD_NUM) {
            spdlog::error("[HybridMgmt::InitKeyProcess] MAX_UNIQUE_THREAD_NUM:{}, should in range [1, {}]",
                          num, DEFAULT_MAX_UNIQUE_THREAD_NUM);
            return false;
        }
        PerfConfig::maxUniqueThreadNum = num;
        spdlog::info("config MAX_UNIQUE_THREAD_NUM:{}", num);
    }

    preprocess = Singleton<KeyProcess>::GetInstance();
    preprocess->Initialize(rankInfo, embInfos, thresholdValues, ifLoad, seed);
    preprocess->Start();
    return true;
}

void HybridMgmt::InitRankInfo(RankInfo& rankInfo, const vector<EmbInfo>& embInfos)
{
    MPI_Comm_size(MPI_COMM_WORLD, &rankInfo.rankSize);
    rankInfo.localRankId = rankInfo.deviceId;

    size_t totHostVocabSize = 0;
    for (const auto& emb : embInfos) {
        totHostVocabSize += emb.hostVocabSize;
    }
    if (totHostVocabSize == 0) {
        rankInfo.noDDR = true;
    }
    rankInfo.useDataset = getenv("DATASET") != nullptr;
}

bool HybridMgmt::Initialize(RankInfo rankInfo, const vector<EmbInfo>& embInfos, int seed,
                            const vector<ThresholdValue>& thresholdValues, bool ifLoad)
{
    if (isRunning) {
        return true;
    }
    SetLog(rankInfo.rankId);
    InitRankInfo(rankInfo, embInfos);

    spdlog::info(MGMT + "begin initialize, localRankSize:{}, localRankId {}, rank {}",
                 rankInfo.localRankSize, rankInfo.localRankId, rankInfo.rankId);

    mgmtRankInfo = rankInfo;
    mgmtEmbInfo = embInfos;
    skipUpdate = getenv("SKIP_UPDATE") != nullptr;

    hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    hdTransfer->Init(embInfos, rankInfo.deviceId);

    bool rc = InitKeyProcess(rankInfo, embInfos, thresholdValues, ifLoad, seed);
    if (!rc) {
        return false;
    }

    lookUpKeysQueue = make_unique<Common::TaskQueue<vector<Tensor>>>();
    restoreQueue = make_unique<Common::TaskQueue<vector<Tensor>>>();
    isRunning = true;

    if (!rankInfo.noDDR) {
        hostEmbs = make_unique<HostEmb>();
        hostHashMaps = make_unique<EmbHashMap>();
        hostEmbs->Initialize(embInfos, seed);
        hostHashMaps->Init(rankInfo, embInfos, ifLoad);
    }
    isLoad = ifLoad;
    if (!rankInfo.useDataset && !isLoad) {
        Start();
    }

    for (const auto& info: embInfos) {
        spdlog::info(MGMT + "emb[{}] vocab size {}+{} sc:{}", info.name, info.devVocabSize, info.hostVocabSize,
            info.sendCount);
    }
    spdlog::info(MGMT + "end initialize, useDataset:{}, noDDR:{}, maxStep:{}, rank:{}",
        rankInfo.useDataset, rankInfo.noDDR, rankInfo.maxStep, rankInfo.rankId);
    return true;
}

bool HybridMgmt::Save(const string savePath)
{
    preprocess->LoadSaveLock();

    CkptData saveData;
    Checkpoint saveCkpt;
    if (!mgmtRankInfo.noDDR) {
        spdlog::debug(MGMT + "Start host side save: ddr mode hashmap");
        saveData.hostEmbs = hostEmbs->GetHostEmbs();
        saveData.embHashMaps = hostHashMaps->GetHashMaps();
    } else {
        spdlog::debug(MGMT + "Start host side save: no ddr mode hashmap");
        saveData.maxOffset = preprocess->GetMaxOffset();
        saveData.keyOffsetMap = preprocess->GetKeyOffsetMap();
    }

    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        spdlog::debug(MGMT + "Start host side save: feature admit and evict");
        saveData.tens2Thresh = featAdmitNEvict.GetTensorThresholds();
        saveData.histRec.timestamps = featAdmitNEvict.GetHistoryRecords().timestamps;
        saveData.histRec.historyRecords = featAdmitNEvict.GetHistoryRecords().historyRecords;
    }

    saveCkpt.SaveModel(savePath, saveData, mgmtRankInfo, mgmtEmbInfo);

    preprocess->LoadSaveUnlock();

    return true;
}

bool HybridMgmt::Load(const string& loadPath)
{
    preprocess->LoadSaveLock();

    spdlog::debug(MGMT + "Start host side load process");

    CkptData loadData;
    Checkpoint loadCkpt;
    vector<CkptFeatureType> loadFeatures;
    if (!mgmtRankInfo.noDDR) {
        loadFeatures.push_back(CkptFeatureType::HOST_EMB);
        loadFeatures.push_back(CkptFeatureType::EMB_HASHMAP);
    } else {
        loadFeatures.push_back(CkptFeatureType::MAX_OFFSET);
        loadFeatures.push_back(CkptFeatureType::KEY_OFFSET_MAP);
    }

    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        loadFeatures.push_back(CkptFeatureType::FEAT_ADMIT_N_EVICT);
    }

    loadData.hostEmbs = hostEmbs->GetHostEmbs();
    loadCkpt.LoadModel(loadPath, loadData, mgmtRankInfo, mgmtEmbInfo, loadFeatures);
    if (!mgmtRankInfo.noDDR && !LoadMatchesDDRSetup(loadData)) {
        return false;
    }

    if (!mgmtRankInfo.noDDR) {
        spdlog::debug(MGMT + "Start host side load: ddr mode hashmap");
        hostHashMaps->LoadHashMap(loadData.embHashMaps);
    } else {
        spdlog::debug(MGMT + "Start host side load: no ddr mode hashmap");
        preprocess->LoadMaxOffset(loadData.maxOffset);
        preprocess->LoadKeyOffsetMap(loadData.keyOffsetMap);
    }
    if (featAdmitNEvict.GetFunctionSwitch()) {
        spdlog::debug(MGMT + "Start host side load: feature admit and evict");
        featAdmitNEvict.LoadTensorThresholds(loadData.tens2Thresh);
        featAdmitNEvict.LoadHistoryRecords(loadData.histRec);
    }

    spdlog::debug(MGMT + "Finish host side load process");

    preprocess->LoadSaveUnlock();

    if (!mgmtRankInfo.useDataset && isLoad) {
        Start();
    }

    return true;
}

bool HybridMgmt::LoadMatchesDDRSetup(const CkptData& loadData)
{
    bool loadDataMatches { true };
    size_t embTableCount { 0 };
    auto loadHostEmbs { loadData.hostEmbs };
    for (const auto& setupHostEmbs : mgmtEmbInfo) {
        const auto& loadEmbTable { loadHostEmbs->find(setupHostEmbs.name) };
        if (loadEmbTable != loadHostEmbs->end()) {
            embTableCount++;

            const auto& loadEmbInfo { loadEmbTable->second.hostEmbInfo };
            if (setupHostEmbs.sendCount != loadEmbInfo.sendCount) {
                spdlog::error(MGMT + "Load data sendCount {} for table {} does not match setup sendCount {}",
                    setupHostEmbs.sendCount, setupHostEmbs.name, loadEmbInfo.sendCount);
                loadDataMatches = false;
            }
            if (setupHostEmbs.extEmbeddingSize != loadEmbInfo.extEmbeddingSize) {
                spdlog::error(MGMT + "Load data extEmbeddingSize {} for table {} does not match "
                                     "setup extEmbeddingSize {}",
                              setupHostEmbs.extEmbeddingSize, setupHostEmbs.name, loadEmbInfo.extEmbeddingSize);
                loadDataMatches = false;
            }
            if (setupHostEmbs.devVocabSize != loadEmbInfo.devVocabSize) {
                spdlog::error(MGMT + "Load data devVocabSize {} for table {} does not match setup devVocabSize {}",
                    setupHostEmbs.devVocabSize, setupHostEmbs.name, loadEmbInfo.devVocabSize);
                loadDataMatches = false;
            }
            if (setupHostEmbs.hostVocabSize != loadEmbInfo.hostVocabSize) {
                spdlog::error(MGMT + "Load data hostVocabSize {} for table {} does not match setup hostVocabSize {}",
                    setupHostEmbs.hostVocabSize, setupHostEmbs.name, loadEmbInfo.hostVocabSize);
                loadDataMatches = false;
            }
            if (!loadDataMatches) {
                return loadDataMatches;
            }
        } else {
            spdlog::error(MGMT + "Load data does not contain table with table name: {}", setupHostEmbs.name);
            return false;
        }
    }

    if (embTableCount < loadHostEmbs->size()) {
        spdlog::error(MGMT + "Load data has {} tables more than setup table num {}",
                      loadHostEmbs->size(), embTableCount);
        return false;
    }
    return true;
}

void HybridMgmt::Start()
{
    if (mgmtRankInfo.noDDR) {
        auto getInfoTask = [this]() {
            auto ret = GetInfoTask();
            spdlog::info("getInfoTask done");
            return ret;
        };
        procThreads.emplace_back(std::make_unique<std::thread>(getInfoTask));

        auto sendInfoTask = [this]() {
            auto ret = SendTask();
            spdlog::info("sendInfoTask done");
            return ret;
        };
        procThreads.emplace_back(std::make_unique<std::thread>(sendInfoTask));
    }

    if (!mgmtRankInfo.noDDR) {
        auto parseKeysTask = [this]() {
            auto ret = ParseKeysTask();
            spdlog::info("parseKeysTask done");
            return ret;
        };
        procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTask));
    }
}

bool HybridMgmt::TrainParseKeys()
{
    do {
        if (!isRunning) {
            return false;
        }
        ParseKeys(TRAIN_CHANNEL_ID, getInfoBatchId);
        spdlog::info(MGMT + "parseKeysBatchId = {}", getInfoBatchId);
    } while (getInfoBatchId % mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] != 0 ||
             mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] == -1);

    return true;
}

bool HybridMgmt::EvalParseKeys()
{
    int evalGetInfoBatchId = 0; // 0-99, 0-99
    do {
        if (!isRunning) {
            return false;
        }
        bool status = ParseKeys(EVAL_CHANNEL_ID, evalGetInfoBatchId);
        if (!status) {
            mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] = evalGetInfoBatchId;
            break;
        }
    } while (evalGetInfoBatchId % mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] != 0 ||
             mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] == -1);

    return true;
}

// 腾讯需要DDR开启特性
bool HybridMgmt::ParseKeysTask()
{
    while (isRunning) {
        spdlog::info(MGMT + "Start Mgmt ParseKeysTask");
        if (mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] == -1 || mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] > 0) {
            if (!TrainParseKeys()) {
                return false;
            }
        }

        if (mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] == -1 || mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] > 0) {
            if (!EvalParseKeys()) {
                return false;
            }
        }
    }

    return false;
}

bool HybridMgmt::GetInfoTask()
{
    while (isRunning) {
        spdlog::info(MGMT + "Start Mgmt GetInfoTask");
        do {
            if (!isRunning) {
                return false;
            }
            GetLookupAndRestore(TRAIN_CHANNEL_ID, getInfoBatchId);
            spdlog::info(MGMT + "getInfoBatchId = {}", getInfoBatchId);
        } while (getInfoBatchId % mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] != 0 ||
                mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] == -1);

        int evalGetInfoBatchId = 0; // 0-99, 0-99
        do {
            if (!isRunning) {
                return false;
            }
            bool status = GetLookupAndRestore(EVAL_CHANNEL_ID, evalGetInfoBatchId);
            if (!status) {
                mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] = evalGetInfoBatchId;
                break;
            }
        } while (evalGetInfoBatchId % mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] != 0 ||
                mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] == -1);
    }
    return false;
}

bool HybridMgmt::SendTask()
{
    while (isRunning) {
        spdlog::info(MGMT + "Start Mgmt SendTask");
        do {
            if (!isRunning) {
                return false;
            }
            SendLookupAndRestore(TRAIN_CHANNEL_ID, sendBatchId);
#if defined(PROFILING) && defined(BUILD_WITH_EASY_PROFILER)
            spdlog::info(MGMT + "sendBatchId = {}", sendBatchId);
            if (trainBatchId == PROFILING_START_BATCH_ID) {
                EASY_PROFILER_ENABLE
            } else if (trainBatchId == PROFILING_END_BATCH_ID) {
                EASY_PROFILER_DISABLE
                ::profiler::dumpBlocksToFile(
                    fmt::format("/home/MX_REC-mgmt-profile-{}.prof", mgmtRankInfo.rankId).c_str());
            }
#endif
        } while (sendBatchId % mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] != 0 ||
                 mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] == -1);

        int evalSendBatchId = 0;
        do {
            if (!isRunning) {
                return false;
            }
            bool status = SendLookupAndRestore(EVAL_CHANNEL_ID, evalSendBatchId);
            if (!status) {
                mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] = evalSendBatchId;
                break;
            }
        } while (evalSendBatchId % mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] != 0 ||
                 mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] == -1);
    }
    return false;
}

bool HybridMgmt::GetLookupAndRestore(const int channelId, int &batchId)
{
    spdlog::info(MGMT + "start parse keys, nBatch:{} , [{}]:{}", mgmtRankInfo.nBatch, channelId, batchId);
    for (const auto& embInfo: mgmtEmbInfo) {
        TimeCost getAllTensorTC;
        vector<string> names = {embInfo.name};
        if (embInfo.modifyGraph) {
            names = embInfo.channelNames;
        }
        spdlog::debug(MGMT + "GetLookupAndRestore embInfoName:{}, modifyGraph:{}, names:{}",
                      embInfo.name, embInfo.modifyGraph, names);
        for (const string& name: names) {
            auto infoVecs = preprocess->GetInfoVec(batchId, name, channelId, ProcessedInfo::RESTORE);
            if (infoVecs == nullptr) {
                spdlog::info(MGMT + "ParseKeys infoVecs empty ! batchId:{}, channelId:{}", batchId, channelId);
                return false;
            }
            lookUpKeysQueue->Pushv({ infoVecs->back() });
            infoVecs->pop_back();
            restoreQueue->Pushv(*infoVecs);
        }
        TIME_PRINT("getAllTensorTC TimeCost(ms):{}", getAllTensorTC.ElapsedMS());
    }
    batchId++;
    return true;
}

bool HybridMgmt::SendLookupAndRestore(const int channelId, int &batchId)
{
    for (const auto& embInfo: mgmtEmbInfo) {
        vector<string> names = {embInfo.name};
        if (embInfo.modifyGraph) {
            names = embInfo.channelNames;
        }
        spdlog::debug(MGMT + "SendLookupAndRestore embInfoName:{}, modifyGraph:{}, names:{}",
                      embInfo.name, embInfo.modifyGraph, names);
        if (!mgmtRankInfo.useStatic) {
            for (const string& name: names) {
                auto all2all = preprocess->GetInfoVec(batchId, name, channelId, ProcessedInfo::ALL2ALL);
                hdTransfer->Send(TransferChannel::ALL2ALL, { *all2all }, channelId, name);
            }
        }
        spdlog::info("SendLookupAndRestore batchId: {}, name: {}, channelId: {}",
                     batchId, embInfo.name, channelId);

        TimeCost sendTensorTC;
        omp_set_num_threads(SEND_TENSOR_TYPE_NUM);
#pragma omp parallel sections
        {
#pragma omp section
            {
                TimeCost sendLookupTC;
                for (const string& name: names) {
                    auto lookUpKeys = lookUpKeysQueue->WaitAndPop();
                    hdTransfer->Send(TransferChannel::LOOKUP, lookUpKeys, channelId, name);
                }
                TIME_PRINT("LOOKUP Send TimeCost(ms):{}", sendLookupTC.ElapsedMS());
            }
#pragma omp section
            {
                TimeCost sendRestoreTC;
                for (const string& name: names) {
                    auto restore = restoreQueue->WaitAndPop();
                    hdTransfer->Send(TransferChannel::RESTORE, restore, channelId, name);
                }
                TIME_PRINT("RESTORE Send TimeCost(ms):{}", sendRestoreTC.ElapsedMS());
            }
        }
        TIME_PRINT("sendTensorTC TimeCost(ms):{}", sendTensorTC.ElapsedMS());
    }
    batchId++;
    return true;
}

bool HybridMgmt::EndBatch(int batchId, int channelId) const
{
    return (batchId % mgmtRankInfo.maxStep[channelId] == 0 && mgmtRankInfo.maxStep[channelId] != -1);
}

bool HybridMgmt::ParseKeys(int channelId, int& batchId)
{
    spdlog::info(MGMT + "DDR mode, start parse keys, nBatch:{} , [{}]:{}", mgmtRankInfo.nBatch, channelId, batchId);
    TimeCost parseKeyTC;
    int start = batchId;
    int iBatch = 0;
    bool ifHashmapFree = true;
    bool remainBatch = true;
    while (true) {
        spdlog::info(MGMT + "parse keys, [{}]:{}", channelId, batchId);
        for (const auto& embInfo : mgmtEmbInfo) {
            ifHashmapFree = ProcessEmbInfo(embInfo.name, batchId, channelId, iBatch, remainBatch);
            if (!remainBatch) {
                EmbHDTransWrap(channelId, batchId, start, iBatch);
                return false;
            }
        }
        batchId++;
        iBatch++;
        if (EndBatch(batchId, channelId) || iBatch == mgmtRankInfo.nBatch || !ifHashmapFree || !isRunning) {
            break;
        }
    }
    if (!isRunning) {
        return false;
    }
    EmbHDTransWrap(channelId, batchId - 1, start, iBatch);
    TIME_PRINT("[{}]-{}, parseKeyTC TimeCost(ms):{}", channelId, batchId, parseKeyTC.ElapsedMS());
    return true;
}

bool HybridMgmt::ProcessEmbInfo(const std::string& embName, int batchId,
                                int channelId, int iBatch, bool& remainBatchOut)
{
    auto& embHashMap = hostHashMaps->embHashMaps.at(embName);
    if (iBatch == 0) {
        embHashMap.SetStartCount();
    }
    auto lookupKeys = preprocess->GetLookupKeys(batchId, embName, channelId);
    if (lookupKeys.empty()) {
        remainBatchOut = false;
    }
    auto restore = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::RESTORE);
    hdTransfer->Send(TransferChannel::RESTORE, *restore, channelId, embName);
    vector<Tensor> tmpData;
    hostHashMaps->Process(embName, lookupKeys, iBatch, tmpData);
    hdTransfer->Send(TransferChannel::LOOKUP, { tmpData.front() }, channelId, embName);
    tmpData.erase(tmpData.begin());
    hdTransfer->Send(TransferChannel::SWAP, tmpData, channelId, embName);
    if (!mgmtRankInfo.useStatic) {
        auto all2all = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::ALL2ALL);
        hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embName);
    }
    if (embHashMap.HasFree(lookupKeys.size())) { // check free > next one batch
        spdlog::warn(MGMT + "embName {}[{}]{},iBatch:{} freeSize not enough, {}", embName, channelId,
                     batchId, iBatch, lookupKeys.size());
        return false;
    }
    return true;
}

// send h2d & recv d2h emb
void HybridMgmt::EmbHDTransWrap(int channelId, const int& batchId, int start, int iBatch)
{
    if (iBatch == 0) {
        return;
    }
    spdlog::info(MGMT + "trans emb, batchId:[{}-{}]", start, batchId);
    hostEmbs->Join();
    EmbHDTrans(channelId, batchId);

    for (int i = 0; i < iBatch - 1; ++i) {
        // need send empty
        spdlog::info(MGMT + "trans emb dummy, batchId:{}, ", start + 1 + i);
        EmbHDTrans(channelId, batchId);
    }
}

void HybridMgmt::EmbHDTrans(const int channelId, const int batchId)
{
    EASY_FUNCTION(profiler::colors::Blue)
    EASY_VALUE("mgmtProcess", batchId)
    spdlog::debug(MGMT + "trans emb, batchId:{}, channelId:{}", batchId, channelId);
    TimeCost tr;
    for (const auto& embInfo: mgmtEmbInfo) {
        auto& missingKeys = hostHashMaps->embHashMaps.at(embInfo.name).missingKeysHostPos;
        vector<Tensor> h2dEmb;
        hostEmbs->GetH2DEmb(missingKeys, embInfo.name, h2dEmb); // order!
        hdTransfer->Send(TransferChannel::H2D, h2dEmb, channelId, embInfo.name, batchId);
    }
    for (const auto& embInfo: mgmtEmbInfo) {
        const auto& missingKeys = hostHashMaps->GetMissingKeys(embInfo.name);
        if (!(skipUpdate && missingKeys.empty())) {
            hostEmbs->UpdateEmbV2(missingKeys, channelId, embInfo.name); // order!
        } // skip when skip update and empty missing keys
        hostHashMaps->ClearMissingKeys(embInfo.name);
    }
    TIME_PRINT("EmbHDTrans TimeCost(ms):{} batchId: {} ", tr.ElapsedMS(), batchId);
}

void HybridMgmt::EmbHDTransDummy(int channelId, int batchId, const EmbInfo& embInfo)
{
    EASY_FUNCTION(profiler::colors::Blue)
    EASY_VALUE("mgmtProcess", batchId)
    spdlog::info(MGMT + "trans emb dummy, batchId:{}, channelId:{}", batchId, channelId);
    auto transferName = TransferChannel::D2H;
    auto d2hEmb = hdTransfer->Recv(transferName, channelId, embInfo.name)[0];
    hdTransfer->Send(TransferChannel::H2D, {}, channelId, embInfo.name);
}

/*
* hook通过时间或者step数触发淘汰
*/
void HybridMgmt::Evict()
{
    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        featAdmitNEvict.FeatureEvict(evictKeyMap);
    } else {
        spdlog::warn(MGMT + "Hook can not trigger evict, cause AdmitNEvict is not open");
        return;
    }
    spdlog::debug(MGMT + "evict triggered by hook, evict TableNum {} ", evictKeyMap.size());

    if (mgmtRankInfo.noDDR) {
        for (auto evict : evictKeyMap) {
            preprocess->EvictKeys(evict.first, evict.second);
        }
    } else {
        for (auto evict : evictKeyMap) {
            EvictKeys(evict.first, evict.second);
        }
    }
}

// ddr模式淘汰->删除映射表、初始化host表、发送dev淘汰位置
void HybridMgmt::EvictKeys(const string& embName, const vector<emb_key_t>& keys)
{
    spdlog::debug(MGMT + "ddr mode, delete emb: [{}]! evict keySize:{}", embName, keys.size());
    // 删除映射关系
    if (keys.size() != 0) {
        hostHashMaps->EvictDeleteEmb(embName, keys);
    }

    // 初始化host侧的emb
    auto& evictOffset = hostHashMaps->embHashMaps.at(embName).evictPos;
    if (evictOffset.size() != 0) {
        spdlog::debug(MGMT + "ddr mode, delete emb: [{}]! evict size on host:{}", embName, evictOffset.size());
        hostEmbs->EvictInitEmb(embName, evictOffset);
    } else {
        spdlog::info(MGMT + "ddr mode, evict size on host is empty");
    }

    // 发送dev侧的淘汰pos，以便dev侧初始化emb
    auto evictDevOffset = hostHashMaps->embHashMaps.at(embName).evictDevPos;
    spdlog::debug(MGMT + "ddr mode, init dev emb: [{}]! evict size on dev :{}", embName, evictDevOffset.size());

    for (const auto& embInfo : mgmtEmbInfo) {
        if (embInfo.name != embName) {
            continue;
        }
        if (evictDevOffset.size() > embInfo.devVocabSize) {
            spdlog::error(MGMT + "{} overflow! evict pos on dev {} bigger than dev vocabSize {}",
                          embName, evictDevOffset.size(), embInfo.devVocabSize);
            throw runtime_error(fmt::format(MGMT + "{} overflow! evict pos on dev {} bigger than dev vocabSize {}",
                                            embName, evictDevOffset.size(), embInfo.devVocabSize).c_str());
        }
        if (mgmtRankInfo.useStatic) {
            evictDevOffset.resize(embInfo.devVocabSize, -1);
        }
        break;
    }

    auto tmpData = Vec2TensorI32(evictDevOffset);
    hdTransfer->Send(TransferChannel::EVICT, { tmpData }, TRAIN_CHANNEL_ID, embName);
}

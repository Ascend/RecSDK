/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */
#include "hybrid_mgmt.h"

#include "utils/time_cost.h"
#include "checkpoint/checkpoint.h"

using namespace MxRec;
using namespace std;

bool HybridMgmt::InitKeyProcess(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos,
                                const vector<ThresholdValue>& thresholdValues, int seed)
{
#ifndef GTEST
    if (getenv("APPLY_GRADIENTS_STRATEGY") != nullptr) {
        bool strategy = (!strcmp(getenv("APPLY_GRADIENTS_STRATEGY"), SUM_SAME_ID));
        PerfConfig::gradientStrategy = strategy;
        LOG(INFO) << StringFormat("config GRADIENTS_STRATEGY:%d", strategy);
    }

    if (getenv("KEY_PROCESS_THREAD_NUM") != nullptr) {
        int num = std::atoi(getenv("KEY_PROCESS_THREAD_NUM"));
        if (num < 1 || num > MAX_KEY_PROCESS_THREAD) {
            LOG(ERROR) << StringFormat(
                "[HybridMgmt::InitKeyProcess] KEY_PROCESS_THREAD_NUM:%d, should in range [1, %d]",
                num, MAX_KEY_PROCESS_THREAD);
            return false;
        }
        PerfConfig::keyProcessThreadNum = num;
        LOG(INFO) << StringFormat("config KEY_PROCESS_THREAD_NUM:%d", num);
    }

    if (getenv("MAX_UNIQUE_THREAD_NUM") != nullptr) {
        int num = std::atoi(getenv("MAX_UNIQUE_THREAD_NUM"));
        if (num < 1 || num > DEFAULT_MAX_UNIQUE_THREAD_NUM) {
            LOG(ERROR) << StringFormat(
                "[HybridMgmt::InitKeyProcess] MAX_UNIQUE_THREAD_NUM:%d, should in range [1, %d]",
                num, DEFAULT_MAX_UNIQUE_THREAD_NUM);
            return false;
        }
        PerfConfig::maxUniqueThreadNum = num;
        LOG(INFO) << StringFormat("config MAX_UNIQUE_THREAD_NUM:%d", num);
    }

    const int defaultFastUnique = false;
    PerfConfig::fastUnique = defaultFastUnique;
    const char* envFastUnique = getenv("FAST_UNIQUE");
    HybridMgmt::CheckFastUnique(envFastUnique);

    preprocess = Singleton<KeyProcess>::GetInstance();
    preprocess->Initialize(rankInfo, embInfos, thresholdValues, seed);
    preprocess->Start();
#endif
    return true;
}

void HybridMgmt::CheckFastUnique(const char *envFastUnique)
{
    if (envFastUnique != nullptr) {
        try {
            int tmp = std::stoi(envFastUnique);
            if (tmp == 0 || tmp == 1) {
                PerfConfig::fastUnique = (tmp == 1) ? true : false;
                LOG(INFO) << StringFormat("Succeed to parse ${env:FAST_UNIQUE}: %d.", PerfConfig::fastUnique);
            } else {
                LOG(ERROR) << StringFormat("Invalid ${env:FAST_UNIQUE}: %s, which should be an 0 or 1.", envFastUnique);
            }
        } catch (const std::invalid_argument &e) {
            LOG(ERROR) <<
                StringFormat("Failed to parse ${env:FAST_UNIQUE}: %s, which should be an integer.", envFastUnique);
        }
    }
}

void HybridMgmt::InitRankInfo(RankInfo& rankInfo, const vector<EmbInfo>& embInfos)
{
#ifndef GTEST
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
#endif
}

bool HybridMgmt::Initialize(RankInfo rankInfo, const vector<EmbInfo>& embInfos, int seed,
                            const vector<ThresholdValue>& thresholdValues, bool ifLoad)
{
#ifndef GTEST
    if (isRunning) {
        return true;
    }

    SetLog(rankInfo.rankId);
    InitRankInfo(rankInfo, embInfos);

    LOG(INFO) << StringFormat(
        MGMT + "begin initialize, localRankSize:%d, localRankId:%d, rank:%d",
        rankInfo.localRankSize, rankInfo.localRankId, rankInfo.rankId);

    mgmtRankInfo = rankInfo;
    mgmtEmbInfo = embInfos;
    skipUpdate = getenv("SKIP_UPDATE") != nullptr;

    hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    hdTransfer->Init(embInfos, rankInfo.deviceId);

    bool rc = InitKeyProcess(rankInfo, embInfos, thresholdValues, seed);
    if (!rc) {
        return false;
    }

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
        LOG(INFO) << StringFormat(
            MGMT + "emb[%s] vocab size %d+%d sc:%d",
            info.name.c_str(), info.devVocabSize, info.hostVocabSize, info.sendCount);
    }
    LOG(INFO) << StringFormat(
        MGMT + "end initialize, useDataset:%d, noDDR:%d, maxStep:[%d, %d], rank:%d",
        rankInfo.useDataset, rankInfo.noDDR,
        rankInfo.maxStep.at(TRAIN_CHANNEL_ID), rankInfo.maxStep.at(EVAL_CHANNEL_ID), rankInfo.rankId);
#endif
    return true;
}

bool HybridMgmt::Save(const string savePath)
{
#ifndef GTEST
    preprocess->LoadSaveLock();

    CkptData saveData;
    Checkpoint saveCkpt;
    if (!mgmtRankInfo.noDDR) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side save: ddr mode hashmap");
        saveData.hostEmbs = hostEmbs->GetHostEmbs();
        saveData.embHashMaps = hostHashMaps->GetHashMaps();
    } else {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side save: no ddr mode hashmap");
        saveData.maxOffset = preprocess->GetMaxOffset();
        saveData.keyOffsetMap = preprocess->GetKeyOffsetMap();
    }

    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side save: feature admit and evict");
        saveData.table2Thresh = featAdmitNEvict.GetTableThresholds();
        saveData.histRec.timestamps = featAdmitNEvict.GetHistoryRecords().timestamps;
        saveData.histRec.historyRecords = featAdmitNEvict.GetHistoryRecords().historyRecords;
    }

    saveCkpt.SaveModel(savePath, saveData, mgmtRankInfo, mgmtEmbInfo);

    preprocess->LoadSaveUnlock();
#endif
    return true;
}

bool HybridMgmt::Load(const string& loadPath)
{
#ifndef GTEST
    preprocess->LoadSaveLock();

    VLOG(GLOG_DEBUG) << (MGMT + "Start host side load process");

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
        preprocess->LoadSaveUnlock();
        return false;
    }

    if (!mgmtRankInfo.noDDR) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side load: ddr mode hashmap");
        hostHashMaps->LoadHashMap(loadData.embHashMaps);
    } else {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side load: no ddr mode hashmap");
        preprocess->LoadMaxOffset(loadData.maxOffset);
        preprocess->LoadKeyOffsetMap(loadData.keyOffsetMap);
    }
    if (featAdmitNEvict.GetFunctionSwitch()) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side load: feature admit and evict");
        featAdmitNEvict.LoadTableThresholds(loadData.table2Thresh);
        featAdmitNEvict.LoadHistoryRecords(loadData.histRec);
    }

    VLOG(GLOG_DEBUG) << (MGMT + "Finish host side load process");

    preprocess->LoadSaveUnlock();

    if (!mgmtRankInfo.useDataset && isLoad) {
        Start();
    }
#endif
    return true;
}

key_offset_map_t HybridMgmt::SendHostMap(const string tableName)
{
#ifndef GTEST
    preprocess->LoadSaveLock();
    key_offset_mem_t keyOffsetMap;
    key_offset_map_t sendKeyOffsetMap;

    if (!mgmtRankInfo.noDDR) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start send sparse data: ddr mode hashmap");
    } else {
        VLOG(GLOG_DEBUG) << (MGMT + "Start send sparse data: no ddr mode hashmap");
        keyOffsetMap = preprocess->GetKeyOffsetMap();
    }

    if ((!keyOffsetMap.empty()) && keyOffsetMap.count(tableName)) {
        for (const auto& it : keyOffsetMap.at(tableName)) {
            sendKeyOffsetMap[it.first] = it.second;
        }
    }

    preprocess->LoadSaveUnlock();
    return sendKeyOffsetMap;
#endif
}

void HybridMgmt::ReceiveHostMap(all_key_offset_map_t ReceiveKeyOffsetMap)
{
#ifndef GTEST
    preprocess->LoadSaveLock();
    key_offset_mem_t loadKeyOffsetMap;
    offset_mem_t loadMaxOffset;
    if (!ReceiveKeyOffsetMap.empty()) {
        for (const auto& KeyOffsetMap : ReceiveKeyOffsetMap) {
            auto& SingleHashMap = loadKeyOffsetMap[KeyOffsetMap.first];
            auto& MaxOffset = loadMaxOffset[KeyOffsetMap.first];
            for (const auto& it : KeyOffsetMap.second) {
                SingleHashMap[it.first] = it.second;
            }
            MaxOffset = KeyOffsetMap.second.size();
        }
    }
    if (!mgmtRankInfo.noDDR) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start receive sparse data: ddr mode hashmap");
    } else {
        VLOG(GLOG_DEBUG) << (MGMT + "Start receive sparse data: no ddr mode hashmap");
        preprocess->LoadKeyOffsetMap(loadKeyOffsetMap);
        preprocess->LoadMaxOffset(loadMaxOffset);
    }

    preprocess->LoadSaveUnlock();
    if (!mgmtRankInfo.useDataset && isLoad) {
        Start();
    }
#endif
}

bool HybridMgmt::IsLoadDataMatches(emb_mem_t* loadHostEmbs, EmbInfo* setupHostEmbs, size_t& embTableCount)
{
    bool loadDataMatches = { true };
    const auto& loadEmbTable { loadHostEmbs->find(setupHostEmbs->name) };
    if (loadEmbTable != loadHostEmbs->end()) {
        embTableCount++;

        const auto& loadEmbInfo { loadEmbTable->second.hostEmbInfo };
        if (setupHostEmbs->sendCount != loadEmbInfo.sendCount) {
            LOG(ERROR) << StringFormat(
                MGMT + "Load data sendCount %d for table %s does not match setup sendCount %d",
                setupHostEmbs->sendCount, setupHostEmbs->name.c_str(), loadEmbInfo.sendCount);
            loadDataMatches = false;
        }
        if (setupHostEmbs->extEmbeddingSize != loadEmbInfo.extEmbeddingSize) {
            LOG(ERROR) << StringFormat(
                MGMT + "Load data extEmbeddingSize %d for table %s does not match setup extEmbeddingSize %d",
                setupHostEmbs->extEmbeddingSize, setupHostEmbs->name.c_str(), loadEmbInfo.extEmbeddingSize);
            loadDataMatches = false;
        }
        if (setupHostEmbs->devVocabSize != loadEmbInfo.devVocabSize) {
            LOG(ERROR) << StringFormat(
                MGMT + "Load data devVocabSize %d for table %s does not match setup devVocabSize %d",
                setupHostEmbs->devVocabSize, setupHostEmbs->name.c_str(), loadEmbInfo.devVocabSize);
            loadDataMatches = false;
        }
        if (setupHostEmbs->hostVocabSize != loadEmbInfo.hostVocabSize) {
            LOG(ERROR) << StringFormat(
                MGMT + "Load data hostVocabSize %d for table %s does not match setup hostVocabSize %d",
                setupHostEmbs->hostVocabSize, setupHostEmbs->name.c_str(), loadEmbInfo.hostVocabSize);
            loadDataMatches = false;
        }
        if (!loadDataMatches) {
            return false;
        }
    } else {
        LOG(ERROR) << StringFormat(
            MGMT + "Load data does not contain table with table name: %s", setupHostEmbs->name.c_str()
        );
        return false;
    }
    return true;
}

bool HybridMgmt::LoadMatchesDDRSetup(const CkptData& loadData)
{
    size_t embTableCount { 0 };
    auto loadHostEmbs { loadData.hostEmbs };
    for (EmbInfo setupHostEmbs : mgmtEmbInfo) {
        if (!IsLoadDataMatches(loadHostEmbs, &setupHostEmbs, embTableCount)) {
            return false;
        }
    }

    if (embTableCount < loadHostEmbs->size()) {
        LOG(ERROR) << StringFormat(MGMT + "Load data has %d tables more than setup table num %d",
            loadHostEmbs->size(), embTableCount);
        return false;
    }
    return true;
}

void HybridMgmt::Start()
{
#ifndef GTEST
    if (mgmtRankInfo.noDDR) {
        InsertThreadForHBM();
    }

    if (!mgmtRankInfo.noDDR) {
        auto parseKeysTaskForTrain = [this]() {
            TaskForTrain(TaskType::DDR);
            LOG(INFO) << StringFormat("parseKeysTaskForTrain done");
        };
        procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForTrain));

        auto parseKeysTaskForEval = [this]() {
            TaskForEval(TaskType::DDR);
            LOG(INFO) << StringFormat("parseKeysTaskForEval done");
        };
        procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForEval));
    }
#endif
}

void HybridMgmt::InsertThreadForHBM()
{
#ifndef GTEST
        auto parseKeysTaskForHBMTrain = [this]() {
            TaskForTrain(TaskType::HBM);
            LOG(INFO) << "parseKeysTaskForHBMTrain done";
        };
        procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForHBMTrain));

        auto parseKeysTaskForHBMEval = [this]() {
            TaskForEval(TaskType::HBM);
            LOG(INFO) << "parseKeysTaskForHBMEval done";
        };
        procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForHBMEval));
#endif
}

#ifndef GTEST
void HybridMgmt::TaskForTrain(TaskType type)
{
    bool isFirstIn = true;
    while (isRunning) {
        if (isFirstIn) {
            LOG(INFO) << StringFormat(MGMT + "Start Train Task: %d", type);
            isFirstIn = false;
        }
        if (mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] == -1 || mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] > 0) {
            if (!TrainTask(type)) {
                return;
            }
        }
        this_thread::sleep_for(1ms);
    }
}

void HybridMgmt::TaskForEval(TaskType type)
{
    bool isFirstIn = true;
    while (isRunning) {
        if (isFirstIn) {
            LOG(INFO) << StringFormat(MGMT + "Start Eval Task: %d", type);
            isFirstIn = false;
        }
        if (mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] == -1 || mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] > 0) {
            if (!EvalTask(type)) {
                return;
            }
        }
        this_thread::sleep_for(1ms);
    }
}

bool HybridMgmt::TrainTask(TaskType type)
{
    bool isContinue = false;
    do {
        if (!isRunning) {
            return false;
        }
        bool status = false;

        switch (type) {
            case TaskType::HBM:
                status = ParseKeysHBM(TRAIN_CHANNEL_ID, trainBatchId);
                isContinue = trainBatchId % mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] != 0 ||
                             mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] == -1;
                LOG(INFO) << StringFormat(MGMT + "ParseKeysHBMBatchId = %d", trainBatchId);
                break;
            case TaskType::DDR:
                status =  ParseKeys(TRAIN_CHANNEL_ID, trainBatchId);
                isContinue = trainBatchId % mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] != 0 ||
                        mgmtRankInfo.maxStep[TRAIN_CHANNEL_ID] == -1;
                LOG(INFO) << StringFormat(MGMT + "parseKeysBatchId = %d", trainBatchId);
                break;
            default:
                throw std::invalid_argument("Invalid TaskType Type.");
        }

        if (!status) {
            return false;
        }
    } while (isContinue);

    return true;
}

bool HybridMgmt::EvalTask(TaskType type)
{
    int evalBatchId = 0; // 0-99, 0-99
    do {
        if (!isRunning) {
            return false;
        }
        bool status = false;

        switch (type) {
            case TaskType::HBM:
                status = ParseKeysHBM(EVAL_CHANNEL_ID, evalBatchId);
                LOG(INFO) << StringFormat(MGMT + "HBM evalBatchId = %d", evalBatchId);
                break;
            case TaskType::DDR:
                status = ParseKeys(EVAL_CHANNEL_ID, evalBatchId);
                LOG(INFO) << StringFormat(MGMT + "DDR evalBatchId = %d", evalBatchId);
                break;
            default:
                throw std::invalid_argument("Invalid TaskType Type.");
        }

        if (!status) {
            return false;
        }
    } while (evalBatchId % mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] != 0 ||
             mgmtRankInfo.maxStep[EVAL_CHANNEL_ID] == -1);

    return true;
}

bool HybridMgmt::ParseKeysHBM(int channelId, int& batchId)
{
    LOG(INFO) << StringFormat(
        MGMT + "start parse keys HBM, nBatch:%d , [%d]:%d", mgmtRankInfo.nBatch, channelId, batchId);
    for (const auto& embInfo: mgmtEmbInfo) {
        TimeCost ParseKeysTC;
        // get
        TimeCost getTensorsSyncTC;
        auto infoVecs = preprocess->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::RESTORE);
        if (infoVecs == nullptr) {
            LOG(INFO) << StringFormat(
                MGMT + "ParseKeys infoVecs empty ! batchId:%d, channelId:%d", batchId, channelId);
            return false;
        }
        unique_ptr<vector<Tensor>> all2all = nullptr;
        if (!mgmtRankInfo.useStatic) {
            all2all = preprocess->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::ALL2ALL);
        }
        VLOG(GLOG_DEBUG) << StringFormat("getTensorsSyncTC(ms):%d", getTensorsSyncTC.ElapsedMS());

        // send
        TimeCost sendTensorsSyncTC;
        if (!mgmtRankInfo.useStatic) {
            TimeCost sendAll2AllScSyncTC;
            hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embInfo.name);
            VLOG(GLOG_DEBUG) << StringFormat("sendAll2AllScSyncTC(ms):%d", sendAll2AllScSyncTC.ElapsedMS());
        }

        TimeCost sendLookupSyncTC;
        hdTransfer->Send(TransferChannel::LOOKUP, { infoVecs->back() }, channelId, embInfo.name);
        infoVecs->pop_back();
        VLOG(GLOG_DEBUG) << StringFormat("sendLookupSyncTC(ms):%d", sendLookupSyncTC.ElapsedMS());

        if (PerfConfig::gradientStrategy && channelId == TRAIN_CHANNEL_ID) {
            TimeCost sendUnikeysSyncTC;
            hdTransfer->Send(TransferChannel::UNIQKEYS, { infoVecs->back() }, channelId, embInfo.name);
            infoVecs->pop_back();
            VLOG(GLOG_DEBUG) << StringFormat("sendUnikeysSyncTC(ms):%d", sendUnikeysSyncTC.ElapsedMS());

            TimeCost sendRestoreVecSecSyncTC;
            hdTransfer->Send(TransferChannel::RESTORE_SECOND, { infoVecs->back() }, channelId, embInfo.name);
            infoVecs->pop_back();
            VLOG(GLOG_DEBUG) << StringFormat("sendRestoreVecSecSyncTC(ms):%d", sendRestoreVecSecSyncTC.ElapsedMS());
        }

        TimeCost sendRestoreSyncTC;
        hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, channelId, embInfo.name);
        VLOG(GLOG_DEBUG) << StringFormat("sendRestoreSyncTC(ms):%d", sendRestoreSyncTC.ElapsedMS());

        VLOG(GLOG_DEBUG) << StringFormat("sendTensorsSyncTC(ms):%d", sendTensorsSyncTC.ElapsedMS());

        VLOG(GLOG_DEBUG) << StringFormat("ParseKeysTC HBM mode (ms):%d", ParseKeysTC.ElapsedMS());
    }
    batchId++;
    return true;
}
#endif

bool HybridMgmt::EndBatch(int batchId, int channelId) const
{
    return (batchId % mgmtRankInfo.maxStep[channelId] == 0 && mgmtRankInfo.maxStep[channelId] != -1);
}

bool HybridMgmt::ParseKeys(int channelId, int& batchId)
{
#ifndef GTEST
    LOG(INFO) << StringFormat(
        MGMT + "DDR mode, start parse keys, nBatch:%d , [%d]:%d",
        mgmtRankInfo.nBatch, channelId, batchId);
    TimeCost parseKeyTC;
    int start = batchId;
    int iBatch = 0;
    bool ifHashmapFree = true;
    bool remainBatch = true;
    while (true) {
        LOG(INFO) << StringFormat(MGMT + "parse keys, [%d]:%d", channelId, batchId);
        for (const auto& embInfo : mgmtEmbInfo) {
            ifHashmapFree = ProcessEmbInfo(embInfo.name, batchId, channelId, iBatch, remainBatch);
            if (!remainBatch) {
                TimeCost embHdTrans1;
                EmbHDTransWrap(channelId, batchId, start, iBatch);
                VLOG(GLOG_DEBUG) << StringFormat("embHdTrans1TC TimeCost(ms):%d", embHdTrans1.ElapsedMS());
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
    TimeCost embHdTrans2TC;
    EmbHDTransWrap(channelId, batchId - 1, start, iBatch);
    VLOG(GLOG_DEBUG) << StringFormat("embHdTrans2TC TimeCost(ms):%d", embHdTrans2TC.ElapsedMS());
    VLOG(GLOG_DEBUG) << StringFormat("[%d]-%d, parseKeyTC TimeCost(ms):%d", channelId, batchId, parseKeyTC.ElapsedMS());
#endif
    return true;
}

#ifndef GTEST
bool HybridMgmt::ProcessEmbInfo(const std::string& embName, int batchId,
                                int channelId, int iBatch, bool& remainBatchOut)
{
    TimeCost getAndSendTensorsTC;
    TimeCost getTensorsTC;
    auto& embHashMap = hostHashMaps->embHashMaps.at(embName);
    if (iBatch == 0) {
        embHashMap.SetStartCount();
    }
    auto lookupKeys = preprocess->GetLookupKeys(batchId, embName, channelId);
    if (lookupKeys.empty()) {
        remainBatchOut = false;
    }

    auto infoVecs = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::RESTORE);
    VLOG(GLOG_DEBUG) << StringFormat("getTensorsTC(ms):%d", getTensorsTC.ElapsedMS());

    if (PerfConfig::gradientStrategy && channelId == TRAIN_CHANNEL_ID && remainBatchOut) {
        TimeCost sendUnikeysSyncTC;
        hdTransfer->Send(TransferChannel::UNIQKEYS, { infoVecs->back() }, channelId, embName);
        infoVecs->pop_back();
        VLOG(GLOG_DEBUG) << StringFormat("sendUnikeysSyncTC(ms):%d", sendUnikeysSyncTC.ElapsedMS());

        TimeCost sendRestoreVecSecSyncTC;
        hdTransfer->Send(TransferChannel::RESTORE_SECOND, { infoVecs->back() }, channelId, embName);
        infoVecs->pop_back();
        VLOG(GLOG_DEBUG) << StringFormat("sendRestoreVecSecSyncTC(ms):%d", sendRestoreVecSecSyncTC.ElapsedMS());
    }

    TimeCost sendRestoreSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, channelId, embName);
    VLOG(GLOG_DEBUG) << StringFormat("sendRestoreSyncTC(ms):%d", sendRestoreSyncTC.ElapsedMS());

    vector<Tensor> tmpData;
    TimeCost hostHashMapProcessTC;
    hostHashMaps->Process(embName, lookupKeys, iBatch, tmpData, channelId);
    VLOG(GLOG_DEBUG) << StringFormat("hostHashMapProcessTC(ms):%d", hostHashMapProcessTC.ElapsedMS());

    TimeCost sendTensorsTC;
    hdTransfer->Send(TransferChannel::LOOKUP, { tmpData.front() }, channelId, embName);
    tmpData.erase(tmpData.begin());
    hdTransfer->Send(TransferChannel::SWAP, tmpData, channelId, embName);
    if (!mgmtRankInfo.useStatic) {
        auto all2all = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::ALL2ALL);
        hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embName);
    }
    VLOG(GLOG_DEBUG) << StringFormat("sendTensorsTC(ms):%d", sendTensorsTC.ElapsedMS());

    VLOG(GLOG_DEBUG) << StringFormat(
        "getAndSendTensorsTC(ms):%d, channelId:%d", getAndSendTensorsTC.ElapsedMS(), channelId);

    if (embHashMap.HasFree(lookupKeys.size())) { // check free > next one batch
        LOG(WARNING) << StringFormat(
            MGMT + "embName %s[%d]%d,iBatch:%d freeSize not enough, %d", embName.c_str(), channelId,
            batchId, iBatch, lookupKeys.size()
        );
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
    LOG(INFO) << StringFormat(MGMT + "trans emb, batchId:[%d-%d], channelId:%d", start, batchId, channelId);
    TimeCost hostEmbsTC;
    hostEmbs->Join(channelId);
    VLOG(GLOG_DEBUG) << StringFormat("hostEmbsTC(ms):%d", hostEmbsTC.ElapsedMS());

    EmbHDTrans(channelId, batchId);

    for (int i = 0; i < iBatch - 1; ++i) {
        // need send empty
        LOG(INFO) << StringFormat(MGMT + "trans emb dummy, batchId:%d, ", start + 1 + i);
        EmbHDTrans(channelId, batchId);
    }
}

void HybridMgmt::EmbHDTrans(const int channelId, const int batchId)
{
    EASY_FUNCTION(profiler::colors::Blue)
    EASY_VALUE("mgmtProcess", batchId)
    VLOG(GLOG_DEBUG) << StringFormat(MGMT + "trans emb, batchId:%d, channelId:%d", batchId, channelId);
    TimeCost tr;
    TimeCost h2dTC;
    for (const auto& embInfo: mgmtEmbInfo) {
        auto& missingKeys = hostHashMaps->embHashMaps.at(embInfo.name).missingKeysHostPos;
        vector<Tensor> h2dEmb;
        hostEmbs->GetH2DEmb(missingKeys, embInfo.name, h2dEmb); // order!
        hdTransfer->Send(TransferChannel::H2D, h2dEmb, channelId, embInfo.name, batchId);
    }
    VLOG(GLOG_DEBUG) << StringFormat("h2dTC(ms):%d", h2dTC.ElapsedMS());

    TimeCost d2hTC;
    for (const auto& embInfo: mgmtEmbInfo) {
        const auto& missingKeys = hostHashMaps->GetMissingKeys(embInfo.name);
        if (!(skipUpdate && missingKeys.empty())) {
            auto updateEmbV2 = getenv("UpdateEmb_V2");
            if (updateEmbV2 != nullptr and atoi(updateEmbV2) == 1) {
                hostEmbs->UpdateEmbV2(missingKeys, channelId, embInfo.name); // order!
            } else {
                hostEmbs->UpdateEmb(missingKeys, channelId, embInfo.name); // order!
            }
        } // skip when skip update and empty missing keys
        hostHashMaps->ClearMissingKeys(embInfo.name);
    }
    VLOG(GLOG_DEBUG) << StringFormat("d2hTC(ms):%d", d2hTC.ElapsedMS());

    VLOG(GLOG_DEBUG) << StringFormat(
        "EmbHDTrans TimeCost(ms):%d batchId: %d channelId:%d", tr.ElapsedMS(), batchId, channelId
    );
}

void HybridMgmt::EmbHDTransDummy(int channelId, int batchId, const EmbInfo& embInfo)
{
    EASY_FUNCTION(profiler::colors::Blue)
    EASY_VALUE("mgmtProcess", batchId)
    LOG(INFO) << StringFormat(MGMT + "trans emb dummy, batchId:%d, channelId:%d", batchId, channelId);
    auto transferName = TransferChannel::D2H;
    auto d2hEmb = hdTransfer->Recv(transferName, channelId, embInfo.name)[0];
    hdTransfer->Send(TransferChannel::H2D, {}, channelId, embInfo.name);
}
#endif
/*
* hook通过时间或者step数触发淘汰
*/
bool HybridMgmt::Evict()
{
#ifndef GTEST
    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        featAdmitNEvict.FeatureEvict(evictKeyMap);
    } else {
        LOG(WARNING) << (MGMT + "Hook can not trigger evict, cause AdmitNEvict is not open");
        return false;
    }
    VLOG(GLOG_DEBUG) << StringFormat(MGMT + "evict triggered by hook, evict TableNum %d ", evictKeyMap.size());
    if (evictKeyMap.size() == 0) {
        LOG(WARNING) << (MGMT + "evict triggered by hook before dataset in injected");
        return false;
    }

    if (mgmtRankInfo.noDDR) {
        for (auto evict : evictKeyMap) {
            preprocess->EvictKeys(evict.first, evict.second);
        }
    } else {
        for (auto evict : evictKeyMap) {
            EvictKeys(evict.first, evict.second);
        }
    }
    return true;
#endif
}

// ddr模式淘汰->删除映射表、初始化host表、发送dev淘汰位置
void HybridMgmt::EvictKeys(const string& embName, const vector<emb_key_t>& keys)
{
#ifndef GTEST
    VLOG(GLOG_DEBUG) << StringFormat(
        MGMT + "ddr mode, delete emb: [%s]! evict keySize:%d", embName.c_str(), keys.size()
    );
    // 删除映射关系
    if (keys.size() != 0) {
        hostHashMaps->EvictDeleteEmb(embName, keys);
    }

    // 初始化host侧的emb
    auto& evictOffset = hostHashMaps->embHashMaps.at(embName).evictPos;
    if (evictOffset.size() != 0) {
        VLOG(GLOG_DEBUG) << StringFormat(
            MGMT + "ddr mode, delete emb: [%s]! evict size on host:%d", embName.c_str(), evictOffset.size()
        );
        hostEmbs->EvictInitEmb(embName, evictOffset);
    } else {
        LOG(INFO) << StringFormat(MGMT + "ddr mode, evict size on host is empty");
    }

    // 发送dev侧的淘汰pos，以便dev侧初始化emb
    auto evictDevOffset = hostHashMaps->embHashMaps.at(embName).evictDevPos;
    VLOG(GLOG_DEBUG) << StringFormat(
        MGMT + "ddr mode, init dev emb: [%s]! evict size on dev :%d", embName.c_str(), evictDevOffset.size()
    );

    vector<Tensor> tmpDataOut;
    Tensor tmpData = Vec2TensorI32(evictDevOffset);
    tmpDataOut.emplace_back(tmpData);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));

    auto evictLen = tmpDataOut.back().flat<int32>();
    auto evictSize = static_cast<int>(evictDevOffset.size());
    evictLen(0) = evictSize;

    hdTransfer->Send(TransferChannel::EVICT, tmpDataOut, TRAIN_CHANNEL_ID, embName);
#endif
}

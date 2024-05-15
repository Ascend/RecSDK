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

#include "hybrid_mgmt.h"

#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <string>
#include <thread>

#include "hd_transfer/hd_transfer.h"
#include "hybrid_mgmt/hybrid_mgmt_block.h"
#include "utils/time_cost.h"
#include "utils/logger.h"
#include "utils/common.h"
#include "checkpoint/checkpoint.h"
#include "key_process/key_process.h"
#include "key_process/feature_admit_and_evict.h"
#include "emb_table/embedding_mgmt.h"
#include "emb_table/embedding_ddr.h"


using namespace MxRec;
using namespace std;


/// Openmpi通信域进程数设置、计算所有表host特征数量总数、设置训练模式（HBM/DDR）
/// \param rankInfo
/// \param embInfos
void HybridMgmt::InitRankInfo(RankInfo& rankInfo, const vector<EmbInfo>& embInfos) const
{
#ifndef GTEST
    MPI_Comm_size(MPI_COMM_WORLD, &rankInfo.rankSize);
    rankInfo.localRankId = rankInfo.deviceId;

    // 计算训练任务涉及的所有表在DDR中需要分配的key数量
    size_t totHostVocabSize = 0;
    size_t totalSsdVocabSize = 0;
    for (const auto& emb : embInfos) {
        totHostVocabSize += emb.hostVocabSize;
        totalSsdVocabSize += emb.ssdVocabSize;
    }

    // 根据DDR的key数量，配置存储模式HBM/DDR
    if (totHostVocabSize != 0) {
        rankInfo.isDDR = true;
    }
    if (totalSsdVocabSize != 0) {
        rankInfo.isSSDEnabled = true;
    }
#endif
}

/// 处理进程初始化入口，由python侧调用
/// \param rankInfo 当前rank基本配置信息
/// \param embInfos 表信息list
/// \param seed 随机种子
/// \param thresholdValues 准入淘汰相关配置
/// \param ifLoad 是否断点续训
/// \return
bool HybridMgmt::Initialize(RankInfo rankInfo, const vector<EmbInfo>& embInfos, int seed,
                            const vector<ThresholdValue>& thresholdValues, bool ifLoad)
{
#ifndef GTEST
    // 环境变量初始化
    ConfigGlobalEnv();

    // 设置日志的级别，对日志格式进行配置
    SetLog(rankInfo.rankId);

    // 打印环境变量
    LogGlobalEnv();

    // 判断是否已经拉起特征处理线程（key process）
    if (isRunning) {
        return true;
    }

    InitRankInfo(rankInfo, embInfos);
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos, thresholdValues, seed);
    GlogConfig::gStatOn = GlobalEnv::statOn;

    LOG_INFO(MGMT + "begin initialize, localRankSize:{}, localRankId:{}, rank:{}",
             rankInfo.localRankSize, rankInfo.localRankId, rankInfo.rankId);

    mgmtRankInfo = rankInfo;
    mgmtEmbInfo = embInfos;

    // 进行acl资源初始化，设置当前训练进程的device，为每张表创建数据传输通道
    hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    hdTransfer->Init(embInfos, rankInfo.deviceId);

    hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
    hybridMgmtBlock->SetRankInfo(rankInfo);

    // 启动数据处理线程
    KEY_PROCESS_INSTANCE->Initialize(rankInfo, embInfos, thresholdValues, seed);

    isRunning = true;

    // DDR模式，初始化hashmap和host emb
    if (rankInfo.isDDR) {
        hostEmbs = Singleton<MxRec::HostEmb>::GetInstance();
        hostHashMaps = make_unique<EmbHashMap>();
        hostEmbs->Initialize(embInfos, seed);
        hostHashMaps->Init(rankInfo, embInfos, ifLoad);
    }

    // 非断点续训模式，启动数据传输
    isSSDEnabled = rankInfo.isSSDEnabled;
    if (isSSDEnabled) {
        cacheManager = Singleton<MxRec::CacheManager>::GetInstance();
        cacheManager->Init(hostEmbs, mgmtEmbInfo);
        hostHashMaps->isSSDEnabled = this->isSSDEnabled;
        hostHashMaps->cacheManager = this->cacheManager;
        // 启用SSD时，EmbeddingDDR依赖cacheManager
        EmbeddingMgmt::Instance()->EnableSSD();
        EmbeddingMgmt::Instance()->SetCacheManagerForEmbTable(this->cacheManager);
    }
    isLoad = ifLoad;
    if (!isLoad) {
        Start();
    }

    for (const auto& info: embInfos) {
        LOG_INFO(MGMT + "emb[{}] vocab size {}+{} sc:{}",
                 info.name, info.devVocabSize, info.hostVocabSize, info.sendCount);
    }
    LOG_INFO(MGMT + "end initialize, isDDR:{}, maxStep:[{}, {}], rank:{}", rankInfo.isDDR,
             rankInfo.ctrlSteps.at(TRAIN_CHANNEL_ID), rankInfo.ctrlSteps.at(EVAL_CHANNEL_ID), rankInfo.rankId);
#endif
    isInitialized = true;

    return true;
}

// 比较hostHashMap和cacheManager的数据是否一致
void HybridMgmt::AddCacheManagerTraceLog(CkptData& saveData)
{
    if (Logger::GetLevel() != Logger::TRACE) {
        return;
    }
    auto& embHashMaps = saveData.embHashMaps;
    auto& ddrKeyFreqMap = saveData.ddrKeyFreqMaps;
    for (auto& it : embHashMaps) {
        string embTableName = it.first;
        auto& hostMap = EmbeddingMgmt::Instance()->GetTable(embTableName)->keyOffsetMap;
        auto& devSize = it.second.devVocabSize;
        auto& lfu = ddrKeyFreqMap[embTableName];
        size_t tableKeyInDdr = 0;
        for (const auto& item : hostMap) {
            if (item.second < devSize) {
                continue;
            }
            ++tableKeyInDdr;
            auto cuKey = item.first;
            if (lfu.find(cuKey) == lfu.end()) {
                LOG_ERROR("save step error, ddr key:{}, not exist in lfu, hostHashMap offset:",
                          cuKey, item.second);
            }
        }
        LOG_INFO("save step end, table:{}, tableKeyInDdr:{}, tableKeyInLfu:{}",
                 embTableName, tableKeyInDdr, lfu.size());
    }
}

/// 保存CacheManager时恢复数据(与恢复hostHashMap类似，仅恢复保存数据,不修改源数据)
/// \param saveData 保存数据
void HybridMgmt::RestoreFreq4Save(CkptData& saveData) const
{
    // 仅在差异1步时执行恢复操作
    int checkResult = hybridMgmtBlock->CheckSaveEmbMapValid();
    if (checkResult != 1) {
        return;
    }
    auto& ddrKeyFreqMaps = saveData.ddrKeyFreqMaps;
    auto& excludeDDRKeyFreqMaps = saveData.excludeDDRKeyFreqMaps;

    for (const auto& it : saveData.embHashMaps) {
        auto& embTableName = it.first;
        auto& embHashMap = it.second;
        vector<emb_key_t> hbm2DdrKeys;
        vector<emb_key_t> ddr2HbmKeys;
        LOG_INFO("restore freq info for save step, table:{}, embHashMap.oldSwap size:{}",
                 embTableName, embHashMap.oldSwap.size());
        LOG_INFO("before, ddr key table size:{}, exclude ddr key table size:{}",
                 ddrKeyFreqMaps[embTableName].size(), excludeDDRKeyFreqMaps[embTableName].size());
        for (const auto& swapKeys : embHashMap.oldSwap) {
            hbm2DdrKeys.emplace_back(swapKeys.second);
            ddr2HbmKeys.emplace_back(swapKeys.first);
        }
        int hbm2DdrKeysNotInExcludeMapCount = 0;
        int ddr2HbmKeysNotInDDRMapCount = 0;
        for (auto& key : hbm2DdrKeys) {
            if (excludeDDRKeyFreqMaps[embTableName].find(key) == excludeDDRKeyFreqMaps[embTableName].end()) {
                ++hbm2DdrKeysNotInExcludeMapCount;
            }
            ddrKeyFreqMaps[embTableName][key] = excludeDDRKeyFreqMaps[embTableName][key];
            excludeDDRKeyFreqMaps[embTableName].erase(key);
        }
        for (auto& key : ddr2HbmKeys) {
            if (ddrKeyFreqMaps[embTableName].find(key) == ddrKeyFreqMaps[embTableName].end()) {
                ++ddr2HbmKeysNotInDDRMapCount;
            }
            excludeDDRKeyFreqMaps[embTableName][key] = ddrKeyFreqMaps[embTableName][key];
            ddrKeyFreqMaps[embTableName].erase(key);
        }
        LOG_INFO("hbm2DdrKeysNotInExcludeMapCount:{}, ddr2HbmKeysNotInDDRMapCount:{}",
                 hbm2DdrKeysNotInExcludeMapCount, ddr2HbmKeysNotInDDRMapCount);
        LOG_INFO("after, ddr key table size:{}, exclude ddr key table size:{}",
                 ddrKeyFreqMaps[embTableName].size(), excludeDDRKeyFreqMaps[embTableName].size());
    }
}

/// 保存模型
/// \param savePath 保存路径
/// \return
bool HybridMgmt::Save(const string savePath)
{
#ifndef GTEST
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    // 数据处理线程上锁
    KEY_PROCESS_INSTANCE->LoadSaveLock();

    CkptData saveData;
    Checkpoint saveCkpt;
    saveData.keyCountMap = KEY_PROCESS_INSTANCE->GetKeyCountMap();

    EmbeddingMgmt::Instance()->LockSave();  // acquire lock here to prevent HybridMgmt modify keyOffsetMap
    EmbeddingMgmt::Instance()->Save(savePath);
    offsetMapToSend = EmbeddingMgmt::Instance()->GetDeviceOffsets();

    if (isSSDEnabled) {
        LOG_DEBUG(MGMT + "Start host side save: ssd mode hashmap");
        for (auto& it : cacheManager->ddrKeyFreqMap) {
            saveData.ddrKeyFreqMaps[it.first] = it.second.GetFreqTable();
        }
        saveData.excludeDDRKeyFreqMaps = cacheManager->excludeDDRKeyCountMap;
        RestoreFreq4Save(saveData);
        AddCacheManagerTraceLog(saveData);
        auto step = GetStepFromPath(savePath);
        cacheManager->SaveSSDEngine(step);
    }
    EmbeddingMgmt::Instance()->UnLockSave();

    // 保存特征准入淘汰相关的数据
    FeatureAdmitAndEvict& featAdmitNEvict = KEY_PROCESS_INSTANCE->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        LOG_DEBUG(MGMT + "Start host side save: feature admit and evict");
        saveData.table2Thresh = featAdmitNEvict.GetTableThresholds();
        saveData.histRec.timestamps = featAdmitNEvict.GetHistoryRecords().timestamps;
        saveData.histRec.historyRecords = featAdmitNEvict.GetHistoryRecords().historyRecords;
    }

    // 执行保存操作
    saveCkpt.SaveModel(savePath, saveData, mgmtRankInfo, mgmtEmbInfo);
    // 数据处理线程释放锁
    KEY_PROCESS_INSTANCE->LoadSaveUnlock();
#endif
    return true;
}

/// 加载模型
/// \param loadPath
/// \return
bool HybridMgmt::Load(const string& loadPath, vector<string> warmStartTables)
{
#ifndef GTEST
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    // 数据处理线程上锁
    KEY_PROCESS_INSTANCE->LoadSaveLock();

    LOG_DEBUG(MGMT + "Start host side load process");

    CkptData loadData;
    Checkpoint loadCkpt;
    vector<CkptFeatureType> loadFeatures;
    SetFeatureTypeForLoad(loadFeatures);

    if(warmStartTables.size() == 0) {
        EmbeddingMgmt::Instance()->Load(loadPath);
    } else {
        for (auto& tableName: warmStartTables) {
            EmbeddingMgmt::Instance()->Load(tableName, loadPath);
        }
    }

    loadOffsetToSend = EmbeddingMgmt::Instance()->GetLoadOffsets();

    // 执行加载操作
    loadCkpt.LoadModel(loadPath, loadData, mgmtRankInfo, mgmtEmbInfo, loadFeatures);

    KEY_PROCESS_INSTANCE->LoadKeyCountMap(loadData.keyCountMap);
    if (mgmtRankInfo.isDDR) {
        // DDR模式 将加载的hash map进行赋值
        LOG_DEBUG(MGMT + "Start host side load: ddr mode hashmap");
        auto GetEmbHashMaps = EmbeddingMgmt::Instance()->GetEmbHashMaps();
        LOG_DEBUG(MGMT + "over over Start host side load: ddr mode hashmap");
        hostHashMaps->LoadHashMap(GetEmbHashMaps);
    } else {
        // HBM模式 将加载的最大偏移（真正使用了多少vocab容量）、特征到偏移的映射，进行赋值
        LOG_DEBUG(MGMT + "Start host side load: no ddr mode hashmap");
        auto keyOffsetMap = EmbeddingMgmt::Instance()->GetKeyOffsetMap();
        auto maxOffset = EmbeddingMgmt::Instance()->GetMaxOffset();
        KEY_PROCESS_INSTANCE->LoadKeyOffsetMap(keyOffsetMap);
        KEY_PROCESS_INSTANCE->LoadMaxOffset(maxOffset);
    }

    // 将加载的特征准入淘汰记录进行赋值
    FeatureAdmitAndEvict& featAdmitNEvict = KEY_PROCESS_INSTANCE->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        LOG_DEBUG(MGMT + "Start host side load: feature admit and evict");
        featAdmitNEvict.LoadTableThresholds(loadData.table2Thresh);
        featAdmitNEvict.LoadHistoryRecords(loadData.histRec);
    }

    if (isSSDEnabled) {
        LOG_DEBUG(MGMT + "Start host side load: ssd key freq map");
        auto step = GetStepFromPath(loadPath);
        cacheManager->Load(loadData.ddrKeyFreqMaps, loadData.excludeDDRKeyFreqMaps,
                           step, mgmtRankInfo.rankSize, mgmtRankInfo.rankId);
        for (auto info: mgmtEmbInfo) {
            auto tb = EmbeddingMgmt::Instance()->GetTable(info.name);
            auto tbCast = reinterpret_pointer_cast<EmbeddingDDR>(tb);
            tbCast->RefreshFreqInfoAfterLoad();
        }
    }

    LOG_DEBUG(MGMT + "Finish host side load process");

    KEY_PROCESS_INSTANCE->LoadSaveUnlock();

    // 执行训练
    if (isLoad) {
        Start();
    }
#endif
    return true;
}

void HybridMgmt::SetFeatureTypeForLoad(vector<CkptFeatureType>& loadFeatures)
{
    if (GlobalEnv::recordKeyCount) {
        loadFeatures.push_back(CkptFeatureType::KEY_COUNT_MAP);
    }

    // 添加特征准入淘汰相关的数据类型的加载
    FeatureAdmitAndEvict& featAdmitNEvict = KEY_PROCESS_INSTANCE->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        loadFeatures.push_back(CkptFeatureType::FEAT_ADMIT_N_EVICT);
    }

    if (isSSDEnabled) {
        loadFeatures.push_back(CkptFeatureType::DDR_KEY_FREQ_MAP);
    }
}

/// 获取key对应的offset，python侧调用
/// \param tableName 表名
/// \return
OffsetT HybridMgmt::SendHostMap(const string tableName)
{
#ifndef GTEST
    OffsetT offsetMap;
    // 先校验这个map是不是空的
    if ((!offsetMapToSend.empty()) && offsetMapToSend.count(tableName) > 0) {
        for (auto& it : offsetMapToSend.at(tableName)) {
            offsetMap.push_back(it);
        }
    }
    return offsetMap;
#endif
}

/// 获取加载embedding文件时，本卡对应的文件行偏移offset，python侧调用
/// \param tableName 表名
/// \return 加载embedding文件的行偏移
OffsetT HybridMgmt::SendLoadMap(const string tableName)
{
#ifndef GTEST
    OffsetT offsetMap;
    if ((!loadOffsetToSend.empty()) && loadOffsetToSend.count(tableName) > 0) {
        for (auto& it : loadOffsetToSend.at(tableName)) {
            offsetMap.push_back(it);
        }
    }
    return offsetMap;
#endif
}

/// 加载key对应的offset，python侧调用；启动数据处理线程
/// \param ReceiveKeyOffsetMap
void HybridMgmt::ReceiveHostMap(AllKeyOffsetMapT receiveKeyOffsetMap)
{
#ifndef GTEST
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    KEY_PROCESS_INSTANCE->LoadSaveLock();
    KeyOffsetMemT loadKeyOffsetMap;
    OffsetMemT loadMaxOffset;
    if (!receiveKeyOffsetMap.empty()) {
        for (const auto& keyOffsetMap : as_const(receiveKeyOffsetMap)) {
            auto& singleHashMap = loadKeyOffsetMap[keyOffsetMap.first];
            auto& maxOffset = loadMaxOffset[keyOffsetMap.first];
            for (const auto& it : keyOffsetMap.second) {
                singleHashMap[it.first] = it.second;
            }
            maxOffset = keyOffsetMap.second.size();
        }
    }
    if (mgmtRankInfo.isDDR) {
        LOG_DEBUG(MGMT + "Start receive sparse data: ddr mode hashmap");
    } else {
        LOG_DEBUG(MGMT + "Start receive sparse data: no ddr mode hashmap");
        KEY_PROCESS_INSTANCE->LoadKeyOffsetMap(loadKeyOffsetMap);
        KEY_PROCESS_INSTANCE->LoadMaxOffset(loadMaxOffset);
    }

    KEY_PROCESS_INSTANCE->LoadSaveUnlock();
    if (isLoad) {
        Start();
    }
#endif
}

/// 对加载的数据和训练配置进行一致性校验
/// \param loadHostEmbs
/// \param setupHostEmbs
/// \param embTableCount
/// \return
bool HybridMgmt::IsLoadDataMatches(const EmbMemT& loadHostEmbs,
                                   const EmbInfo& setupHostEmbs,
                                   size_t& embTableCount) const
{
    bool loadDataMatches = { true };
    const auto& loadEmbTable { loadHostEmbs.find(setupHostEmbs.name) };
    if (loadEmbTable != loadHostEmbs.end()) {
        embTableCount++;

        const auto& loadEmbInfo { loadEmbTable->second.hostEmbInfo };
        if (setupHostEmbs.sendCount != loadEmbInfo.sendCount) {
            LOG_ERROR(MGMT + "Load data sendCount {} for table {} does not match setup sendCount {}",
                      setupHostEmbs.sendCount, setupHostEmbs.name, loadEmbInfo.sendCount);
            loadDataMatches = false;
        }
        if (setupHostEmbs.extEmbeddingSize != loadEmbInfo.extEmbeddingSize) {
            LOG_ERROR(MGMT + "Load data extEmbeddingSize {} for table {} does not match setup extEmbeddingSize {}",
                      setupHostEmbs.extEmbeddingSize, setupHostEmbs.name, loadEmbInfo.extEmbeddingSize);
            loadDataMatches = false;
        }
        if (setupHostEmbs.devVocabSize != loadEmbInfo.devVocabSize) {
            LOG_ERROR(MGMT + "Load data devVocabSize {} for table {} does not match setup devVocabSize {}",
                      setupHostEmbs.devVocabSize, setupHostEmbs.name, loadEmbInfo.devVocabSize);
            loadDataMatches = false;
        }
        if (setupHostEmbs.hostVocabSize != loadEmbInfo.hostVocabSize) {
            LOG_ERROR(MGMT + "Load data hostVocabSize {} for table {} does not match setup hostVocabSize {}",
                      setupHostEmbs.hostVocabSize, setupHostEmbs.name, loadEmbInfo.hostVocabSize);
            loadDataMatches = false;
        }
        if (!loadDataMatches) {
            return false;
        }
    } else {
        LOG_ERROR(MGMT + "Load data does not contain table with table name: {}", setupHostEmbs.name);
        return false;
    }
    return true;
}

/// 对DDR模式保存的模型和训练配置进行一致性校验
/// \param loadData
/// \return 是否一致
bool HybridMgmt::LoadMatchesDDRSetup(const CkptData& loadData)
{
    size_t embTableCount { 0 };
    auto loadHostEmbs { loadData.hostEmbs };
    if (loadHostEmbs == nullptr) {
        LOG_ERROR(MGMT + "Host Embedding of load checkpoint data is nullptr!");
        return false;
    }
    for (EmbInfo setupHostEmbs : mgmtEmbInfo) {
        if (!IsLoadDataMatches(*loadHostEmbs, setupHostEmbs, embTableCount)) {
            return false;
        }
    }

    if (embTableCount < loadHostEmbs->size()) {
        LOG_ERROR(MGMT + "Load data has {} tables more than setup table num {}",
                  loadHostEmbs->size(), embTableCount);
        return false;
    }
    return true;
}

/// 根据HBM/DDR模式，启动数据处理线程
void HybridMgmt::Start()
{
#ifndef GTEST
    if (mgmtRankInfo.isDDR) {
        StartThreadForDDR();
    } else {
        StartThreadForHBM();
    }
#endif
}

/// 启动HBM模式数据处理线程
void HybridMgmt::StartThreadForHBM()
{
#ifndef GTEST
        auto parseKeysTaskForHBMTrain = [this]() {
            TrainTask(TaskType::HBM);
            LOG_INFO("parseKeysTaskForHBMTrain done");
        };
        procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForHBMTrain));

        auto parseKeysTaskForHBMEval = [this]() {
            EvalTask(TaskType::HBM);
            LOG_INFO("parseKeysTaskForHBMEval done");
        };
        procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForHBMEval));
#endif
}

void HybridMgmt::StartThreadForDDR()
{
#ifndef GTEST
    auto parseKeysTaskForTrain = [this]() {
        TrainTask(TaskType::DDR);
        LOG_INFO("parseKeysTaskForTrain done");
    };
    procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForTrain));

    auto parseKeysTaskForEval = [this]() {
        EvalTask(TaskType::DDR);
        LOG_INFO("parseKeysTaskForEval done");
    };
    procThreads.emplace_back(std::make_unique<std::thread>(parseKeysTaskForEval));
#endif
}

void HybridMgmt::Destroy()
{
    LOG_DEBUG(MGMT + "start Destroy hybrid_mgmt module");
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    if (!isRunning) {
        return;
    }
    // 先发送停止信号mgmt，先停止新lookup查询, 解除queue的限制防止卡住
    isRunning = false;

    {
        // 获取锁 避免KeyProcess中手动发送结束信息时通道关闭
        std::unique_lock<std::mutex> lockGuard(KEY_PROCESS_INSTANCE->destroyMutex);
        // 先发送停止信号给KEY_PROCESS_INSTANCE，用于停止查询中lookup卡住状态
        KEY_PROCESS_INSTANCE->isRunning = false;
        // 停止hdTransfer，用于停止mgmt的recv中卡住状态
        hdTransfer->Destroy();
        LOG_DEBUG(MGMT + "destroy hdTransfer end.");
    }

    hybridMgmtBlock->Destroy();
    for (auto& t : procThreads) {
        t->join();
    }
    if (cacheManager != nullptr) {
        cacheManager = nullptr;
    }
    if (hostEmbs != nullptr) {
        hostEmbs->Join(TRAIN_CHANNEL_ID);
        hostEmbs->Join(EVAL_CHANNEL_ID);
        hostEmbs = nullptr;
    }
    procThreads.clear();
    // 停止预处理
    KEY_PROCESS_INSTANCE->Destroy();
    LOG_DEBUG(MGMT + "Destroy hybrid_mgmt module end.");
};

#ifndef GTEST
/// 启动hybrid处理任务
/// \param type
void HybridMgmt::TrainTask(TaskType type)
{
    int channelId = TRAIN_CHANNEL_ID;
    int& theTrainBatchId = hybridMgmtBlock->hybridBatchId[channelId];
    do {
        hybridMgmtBlock->CheckAndSetBlock(channelId);
        if (hybridMgmtBlock->GetBlockStatus(channelId)) {
            hybridMgmtBlock->DoBlock(channelId);
        }
        if (!isRunning) {
            return;
        }
        LOG_INFO(HYBRID_BLOCKING + "hybrid start task channel {} batch {}", channelId, theTrainBatchId);

        switch (type) {
            case TaskType::HBM:
                ParseKeysHBM(TRAIN_CHANNEL_ID, theTrainBatchId);
                LOG_INFO(MGMT + "ParseKeysHBMBatchId = {}", theTrainBatchId);
                break;
            case TaskType::DDR:
                ParseKeys(TRAIN_CHANNEL_ID, theTrainBatchId);
                LOG_INFO(MGMT + "parseKeysBatchId = {}", theTrainBatchId);
                break;
            default:
                throw std::invalid_argument("Invalid TaskType Type.");
        }
    } while (true);
}

/// 推理数据处理：数据处理状态正常，处理的batch数小于用户预设值或者设为-1时，会循环处理；
/// \param type 存储模式
/// \return
void HybridMgmt::EvalTask(TaskType type)
{
    int channelId = EVAL_CHANNEL_ID;
    int& evalBatchId = hybridMgmtBlock->hybridBatchId[channelId];
    do {
        hybridMgmtBlock->CheckAndSetBlock(channelId);
        if (hybridMgmtBlock->GetBlockStatus(channelId)) {
            hybridMgmtBlock->DoBlock(channelId);
        }
        if (!isRunning) {
            return;
        }
        LOG_INFO(HYBRID_BLOCKING + "hybrid start task channel {} batch {}", channelId, evalBatchId);

        switch (type) {
            case TaskType::HBM:
                ParseKeysHBM(EVAL_CHANNEL_ID, evalBatchId);
                LOG_INFO(MGMT + "HBM evalBatchId = {}", evalBatchId);
                break;
            case TaskType::DDR:
                ParseKeys(EVAL_CHANNEL_ID, evalBatchId);
                LOG_INFO(MGMT + "DDR evalBatchId = {}", evalBatchId);
                break;
            default:
                throw std::invalid_argument("Invalid TaskType Type.");
        }
    } while (true);
}

/// HBM模式下，发送key process线程已处理好的各类型向量到指定通道中
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \return
bool HybridMgmt::ParseKeysHBM(int channelId, int& batchId)
{
    LOG_INFO(MGMT + "nBatch:{} channelId:{} batchId:{}, ParseKeys with HBM mode start.",
             mgmtRankInfo.nBatch, channelId, batchId);

    // 循环处理每个表的数据
    for (const auto& embInfo: mgmtEmbInfo) {
        TimeCost parseKeysTc;
        // 获取各类向量，如果为空指针，退出当前函数
        auto infoVecs = KEY_PROCESS_INSTANCE->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::RESTORE);
        if (infoVecs == nullptr) {
            LOG_INFO(MGMT + "channelId:{} batchId:{}, ParseKeys infoVecs empty !", channelId, batchId);
            return false;
        }
        LOG_DEBUG("channelId:{} batchId:{}, ParseKeysHBM GetInfoVec end.", channelId, batchId);
        // 动态shape场景下，获取all2all向量（通信量矩阵）
        TimeCost sendTensorsSyncTC;
        unique_ptr<vector<Tensor>> all2all = nullptr;
        if (!mgmtRankInfo.useStatic) {
            TimeCost getTensorsSyncTC;
            all2all = KEY_PROCESS_INSTANCE->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::ALL2ALL);
            LOG_DEBUG("channelId:{} batchId:{}, getTensorsSyncTC(ms):{}",
                      channelId, batchId, getTensorsSyncTC.ElapsedMS());
            if (all2all == nullptr) {
                LOG_ERROR("Information vector is nullptr!");
                return false;
            }
            sendTensorsSyncTC = TimeCost(); // 重新初始化，不计算getTensors耗时
            TimeCost sendAll2AllScSyncTC;
            hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embInfo.name);
            LOG_DEBUG("channelId:{} batchId:{}, sendAll2AllScSyncTC(ms):{}",
                      channelId, batchId, sendAll2AllScSyncTC.ElapsedMS());
        }

        // 发送查询向量
        TimeCost sendLookupSyncTC;
        hdTransfer->Send(TransferChannel::LOOKUP, { infoVecs->back() }, channelId, embInfo.name);
        infoVecs->pop_back();
        LOG_DEBUG("channelId:{} batchId:{}, sendLookupSyncTC(ms):{}", channelId, batchId, sendLookupSyncTC.ElapsedMS());

        // 训练时，使用全局去重聚合梯度，发送全局去重的key和对应的恢复向量
        if (GlobalEnv::applyGradientsStrategy == ApplyGradientsStrategyOptions::SUM_SAME_ID_GRADIENTS_AND_APPLY &&
            channelId == TRAIN_CHANNEL_ID) {
            SendUniqKeysAndRestoreVecHBM(channelId, batchId, embInfo, infoVecs);
        }

        // 发送恢复向量
        TimeCost sendRestoreSyncTC;
        hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, channelId, embInfo.name);
        LOG_DEBUG("sendRestoreSyncTC(ms):{}, sendTensorsSyncTC(ms):{}, parseKeysTc HBM mode (ms):{}",
                  sendRestoreSyncTC.ElapsedMS(), sendTensorsSyncTC.ElapsedMS(), parseKeysTc.ElapsedMS());
        LOG_INFO(MGMT + "channelId:{} batchId:{}, embName:{}, ParseKeys with HBM mode end.",
                 channelId, batchId, embInfo.name);
    }
    batchId++;
    return true;
}

void HybridMgmt::SendUniqKeysAndRestoreVecHBM(int channelId, int &batchId, const EmbInfo &embInfo,
                                              const unique_ptr<vector<Tensor>> &infoVecs) const
{
    TimeCost sendUniqueKeysSyncTC;
    LOG_DEBUG("channelId:{} batchId:{}, global unique, table name: {}, is grad: {}",
              channelId, batchId, embInfo.name, embInfo.isGrad);
    if (embInfo.isGrad) {
        hdTransfer->Send(TransferChannel::UNIQKEYS, {infoVecs->back()}, channelId, embInfo.name);
    }
    infoVecs->pop_back();
    LOG_DEBUG("channelId:{} batchId:{}, sendUniqueKeysSyncTC(ms):{}",
              channelId, batchId, sendUniqueKeysSyncTC.ElapsedMS());

    TimeCost sendUniqueRestoreVecSyncTC;
    if (embInfo.isGrad) {
        hdTransfer->Send(TransferChannel::RESTORE_SECOND, {infoVecs->back()}, channelId, embInfo.name);
    }
    infoVecs->pop_back();
    LOG_DEBUG("channelId:{} batchId:{}, sendUniqueRestoreVecSyncTC(ms):{}",
              channelId, batchId, sendUniqueRestoreVecSyncTC.ElapsedMS());
}

#endif

/// 当前处理的batch是否是最后一个batch
/// \param batchId 已处理的batch数
/// \param channelId 通道索引（训练/推理）
/// \return
bool HybridMgmt::EndBatch(int batchId, int channelId) const
{
    return (batchId % mgmtRankInfo.ctrlSteps[channelId] == 0 && mgmtRankInfo.ctrlSteps[channelId] != -1);
}

/// DDR模式下，发送key process线程已处理好的各类型向量到指定通道中
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \return
bool HybridMgmt::ParseKeys(int channelId, int& batchId)
{
#ifndef GTEST
    LOG_INFO(MGMT + "channelId:{} batchId:{}, DDR mode, ParseKeys start.", channelId, batchId);
    TimeCost parseKeyTC;
    int start = batchId;
    bool remainBatch = true; // 是否从通道获取了数据

    for (const auto& embInfo : mgmtEmbInfo) {
        ProcessEmbInfo(embInfo.name, batchId, channelId, remainBatch);
        // 通道数据已空
        if (!remainBatch) {
            LOG_DEBUG("last batch ending");
            return false;
        }
    }
    batchId++;

    if (!isRunning) {
        return false;
    }
    EmbHDTransWrap(channelId, batchId - 1, start);
    LOG_DEBUG(MGMT + "channelId:{} batchId:{}, ParseKeys end, parseKeyTC(ms):{}",
              channelId, batchId, parseKeyTC.ElapsedMS());
#endif
    return true;
}

void HybridMgmt::HandlePrepareDDRDataRet(TransferRet prepareSSDRet) const
{
    LOG_ERROR("Transfer embedding with DDR and SSD error.");
    if (prepareSSDRet == TransferRet::SSD_SPACE_NOT_ENOUGH) {
        LOG_ERROR("PrepareDDRData: SSD available space is not enough.");
        throw runtime_error("ssdVocabSize too small");
    }
    if (prepareSSDRet == TransferRet::DDR_SPACE_NOT_ENOUGH) {
        LOG_ERROR("PrepareDDRData: DDR available space is not enough.");
        throw runtime_error("ddrVocabSize too small");
    }
    throw runtime_error("Transfer embedding with DDR and SSD error.");
}

#ifndef GTEST

/// 构造训练所需的各种向量数据
/// \param embName 表名
/// \param batchId 已处理的batch数
/// \param channelId 通道索引（训练/推理）
/// \param remainBatchOut 是否从通道获取了数据
/// \return HBM是否还有剩余空间
bool HybridMgmt::ProcessEmbInfo(const std::string& embName, int batchId, int channelId, bool& remainBatchOut)
{
    TimeCost getAndSendTensorsTC;
    TimeCost getTensorsTC;

    if (hostHashMaps->embHashMaps.find(embName) == hostHashMaps->embHashMaps.end()) {
        LOG_ERROR("Failed to get embedding hash map with given name: {}", embName);
        return false;
    }

    auto& embHashMap = hostHashMaps->embHashMaps.at(embName);
    // 计数初始化
    std::shared_ptr<EmbeddingTable> table = EmbeddingMgmt::Instance()->GetTable(embName);
    table->SetStartCount();

    // 获取查询向量
    auto lookupKeys = KEY_PROCESS_INSTANCE->GetLookupKeys(batchId, embName, channelId);
    if (lookupKeys.empty()) {
        remainBatchOut = false;
        LOG_WARN("channelId:{} batchId:{}, embName:{}, GetLookupKeys result is empty.", channelId, batchId, embName);
        return false;
    }
    LOG_DEBUG("channelId:{} batchId:{}, embName:{}, GetLookupKeys end.", channelId, batchId, embName);
    // 获取各类向量，如果为空指针，退出当前函数
    unique_ptr<vector<Tensor>> infoVecs = KEY_PROCESS_INSTANCE->GetInfoVec(batchId, embName, channelId,
                                                                           ProcessedInfo::RESTORE);
    if (infoVecs == nullptr) {
        LOG_ERROR("Information vector is nullptr!");
        return false;
    }
    LOG_DEBUG("channelId:{} batchId:{}, GetInfoVec end, getTensorsTC(ms):{}",
              channelId, batchId, getTensorsTC.ElapsedMS());

    TimeCost sendRestoreSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, channelId, embName);
    LOG_DEBUG("channelId:{} batchId:{}, send restore end, sendRestoreSyncTC(ms):{}",
              channelId, batchId, sendRestoreSyncTC.ElapsedMS());

    // 调用SSD cache缓存处理流程，获取锁避免保存时修改keyOffsetMap
    table->mutSave_.lock();
    LOG_DEBUG("acquire save lock, table:{}", table->name);
    PrepareDDRData(table, lookupKeys, channelId, batchId);

    // 计算查询向量；记录需要被换出的HBM偏移
    vector<Tensor> tmpData;
    vector<int32_t> offsetsOut;
    DDRParam ddrParam(tmpData, offsetsOut);
    TimeCost hostHashMapProcessTC;

    hostHashMaps->Process(embName, lookupKeys, ddrParam, channelId);
    table->mutSave_.unlock();
    LOG_DEBUG("release save lock, table:{}", table->name);

    LOG_DEBUG("channelId:{} batchId:{}, hostHashMapProcessTC(ms):{}",
              channelId, batchId, hostHashMapProcessTC.ElapsedMS());

    if (GlobalEnv::applyGradientsStrategy == ApplyGradientsStrategyOptions::SUM_SAME_ID_GRADIENTS_AND_APPLY &&
        channelId == TRAIN_CHANNEL_ID && remainBatchOut) {
        SendUniqKeysAndRestoreVecDDR(embName, batchId, channelId, ddrParam);
    }

    TimeCost sendTensorsTC;
    hdTransfer->Send(TransferChannel::LOOKUP, { ddrParam.tmpDataOut.front() }, channelId, embName);
    ddrParam.tmpDataOut.erase(ddrParam.tmpDataOut.cbegin());
    hdTransfer->Send(TransferChannel::SWAP, ddrParam.tmpDataOut, channelId, embName);
    if (!mgmtRankInfo.useStatic) {
        unique_ptr<vector<Tensor>> all2all = KEY_PROCESS_INSTANCE->GetInfoVec(batchId, embName,
                                                                              channelId, ProcessedInfo::ALL2ALL);
        if (all2all == nullptr) {
            LOG_ERROR("Information vector is nullptr!");
            return false;
        }
        hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embName);
    }
    LOG_DEBUG("channelId:{} batchId:{}, ProcessEmbInfo end, sendTensorsTC(ms):{}, getAndSendTensorsTC(ms):{}",
              channelId, batchId, sendTensorsTC.ElapsedMS(), getAndSendTensorsTC.ElapsedMS());

    if (!isSSDEnabled && embHashMap.HasFree(lookupKeys.size())) { // check free > next one batch
        LOG_WARN(MGMT + "channelId:{} batchId:{}, embName:{}, freeSize not enough:{}",
                 channelId, batchId, embName, lookupKeys.size());
        return false;
    }
    return true;
}

void HybridMgmt::SendUniqKeysAndRestoreVecDDR(const string &embName, int &batchId, int &channelId, DDRParam &ddrParam)
{
    LOG_DEBUG("channelId:{} batchId:{}, embName:{}, SendUniqKeysAndRestoreVecDDR start.", channelId, batchId, embName);
    vector<int32_t> uniqueKeys;
    vector<int32_t> restoreVecSec;
    KEY_PROCESS_INSTANCE->GlobalUnique(ddrParam.offsetsOut, uniqueKeys, restoreVecSec);

    TimeCost sendUniqueKeysSyncTC;
    hdTransfer->Send(TransferChannel::UNIQKEYS, {mgmtRankInfo.useDynamicExpansion ? Vec2TensorI64(uniqueKeys) :
                                                 Vec2TensorI32(uniqueKeys) }, channelId, embName);
    LOG_DEBUG("channelId:{} batchId:{}, sendUniqueKeysSyncTC(ms):{}",
              channelId, batchId, sendUniqueKeysSyncTC.ElapsedMS());

    TimeCost sendRestoreVecSecSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE_SECOND, {Vec2TensorI32(restoreVecSec) }, channelId, embName);
    LOG_DEBUG("channelId:{} batchId:{}, sendRestoreVecSecSyncTC(ms):{}",
              channelId, batchId, sendRestoreVecSecSyncTC.ElapsedMS());
}

/// 发送H2D和接收D2H向量
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \param start
void HybridMgmt::EmbHDTransWrap(int channelId, const int& batchId, int start)
{
    LOG_INFO(MGMT + "start:{} channelId:{} batchId:{}, EmbHDTransWrap start.", start, channelId, batchId);
    TimeCost embHDTransWrapTC;
    TimeCost hostEmbsTC;
    hostEmbs->Join(channelId);
    LOG_DEBUG("channelId:{} batchId:{}, hostEmbs Join end, hostEmbsTC(ms):{}",
              channelId, batchId, hostEmbsTC.ElapsedMS());
    if (!isRunning) {
        return;
    }
    EmbHDTrans(channelId, batchId);
    LOG_DEBUG("channelId:{} batchId:{}, EmbHDTransWrap end, embHDTransWrapTC(ms):{}",
              channelId, batchId, embHDTransWrapTC.ElapsedMS());
}

/// 发送H2D和接收D2H向量，并更新host emb
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
void HybridMgmt::EmbHDTrans(const int channelId, const int batchId)
{
    EASY_FUNCTION(profiler::colors::Blue)
    EASY_VALUE("mgmtProcess", batchId)
    LOG_DEBUG(MGMT + "channelId:{} batchId:{}, EmbHDTrans start.", channelId, batchId);
    TimeCost h2dTC;
    // 发送host需要换出的emb
    for (const auto& embInfo: mgmtEmbInfo) {
        const auto& missingKeys = EmbeddingMgmt::Instance()->GetMissingKeys(embInfo.name);
        vector<Tensor> h2dEmb;
        hostEmbs->GetH2DEmb(missingKeys, embInfo.name, h2dEmb); // order!
        hdTransfer->Send(TransferChannel::H2D, h2dEmb, channelId, embInfo.name, batchId);
    }
    LOG_DEBUG("channelId:{} batchId:{}, EmbHDTrans h2d end, h2dTC(ms):{}", channelId, batchId, h2dTC.ElapsedMS());

    TimeCost d2hTC;
    // 接收device换出的emb，并更新到host上
    for (const auto& embInfo: mgmtEmbInfo) {
        const auto& missingKeys = EmbeddingMgmt::Instance()->GetMissingKeys(embInfo.name);
        hostEmbs->UpdateEmbV2(missingKeys, channelId, embInfo.name); // order!
        EmbeddingMgmt::Instance()->ClearMissingKeys(embInfo.name);
    }
    LOG_DEBUG("channelId:{} batchId:{}, EmbHDTrans d2h end, d2hTC(ms):{}", channelId, batchId, d2hTC.ElapsedMS());
}
#endif

/// hook通过时间或者step数触发淘汰
/// \return
bool HybridMgmt::Evict()
{
#ifndef GTEST
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    // 配置了淘汰选项，则触发
    FeatureAdmitAndEvict& featAdmitNEvict = KEY_PROCESS_INSTANCE->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        featAdmitNEvict.FeatureEvict(evictKeyMap);
    } else {
        LOG_WARN(MGMT + "Hook can not trigger evict, cause AdmitNEvict is not open");
        return false;
    }
    LOG_DEBUG(MGMT + "evict triggered by hook, evict TableNum {}", evictKeyMap.size());

    // 表为空，淘汰触发失败
    if (evictKeyMap.empty()) {
        LOG_WARN(MGMT + "evict triggered by hook before dataset in injected");
        return false;
    }

    if (!mgmtRankInfo.isDDR) {
        if (GlobalEnv::useCombineFaae) {
            EmbeddingMgmt::Instance()->EvictKeysCombine(evictKeyMap[COMBINE_HISTORY_NAME]);
        } else {
            for (const auto& evict : as_const(evictKeyMap)) {
                EmbeddingMgmt::Instance()->EvictKeys(evict.first, evict.second);
            }
        }
    } else {
        if (GlobalEnv::useCombineFaae) {
            for (auto& map : hostHashMaps->embHashMaps) {
                EmbeddingMgmt::Instance()->EvictKeys(map.first, evictKeyMap[COMBINE_HISTORY_NAME]);
            }
        } else {
            for (const auto& evict : as_const(evictKeyMap)) {
                EvictKeys(evict.first, evict.second);
                EvictSSDKeys(evict.first, evict.second);
            }
        }
    }
    evictKeyMap.clear();
    return true;
#endif
}

/// DDR模式下的淘汰：删除映射表、初始化host表、发送dev淘汰位置
/// \param embName
/// \param keys
void HybridMgmt::EvictKeys(const string& embName, const vector<emb_key_t>& keys)
{
    std::shared_ptr<EmbeddingTable> table = EmbeddingMgmt::Instance()->GetTable(embName);

    table->EvictKeys(keys);

    const vector<int64_t>& evictOffsetDev = table->GetEvictedKeys();
    const vector<int64_t>& evictOffsetHost = table->GetHostEvictedKeys();

    vector<int64_t> evictOffsetHostx(evictOffsetHost);

    size_t devVocabSize = table->GetDevVocabSize();
    for (int64_t& key: evictOffsetHostx) {
        key -= static_cast<int64_t>(devVocabSize);
    };

    /* 淘汰Host侧 */
    if (!evictOffsetHost.empty()) {
        hostEmbs->EvictInitEmb(embName, evictOffsetHost);
    }

    vector<Tensor> tmpDataOut;
    Tensor tmpData = Vec2TensorI32(evictOffsetDev);
    tmpDataOut.emplace_back(tmpData);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));

    auto evictLen = tmpDataOut.back().flat<int32>();
    auto evictSize = static_cast<int>(evictOffsetDev.size());
    evictLen(0) = evictSize;

    hdTransfer->Send(TransferChannel::EVICT, tmpDataOut, TRAIN_CHANNEL_ID, embName);
}

inline void HybridMgmt::PrepareDDRData(std::shared_ptr<EmbeddingTable> table,
                                       const vector<emb_key_t>& keys, int channelId, int batchId) const
{
    if (!isSSDEnabled) {
        return;
    }
    LOG_DEBUG("channelId:{} batchId:{}, embTableName:{}, PrepareDDRData start.", channelId, batchId, table->name);
    TimeCost prepareDDRDataTc;
    TableInfo ti = table->GetTableInfo();
    TransferRet ret = cacheManager->TransferDDREmbWithSSD(ti, keys, channelId);
    if (ret != TransferRet::TRANSFER_OK) {
        HandlePrepareDDRDataRet(ret);
    }
    LOG_DEBUG("channelId:{} batchId:{}, embTableName:{}, PrepareDDRData end, prepareDDRDataTc(ms):{}",
              channelId, batchId, table->name, prepareDDRDataTc.ElapsedMS());
}

void HybridMgmt::EvictSSDKeys(const string& embName, const vector<emb_key_t>& keys) const
{
    if (!isSSDEnabled) {
        return;
    }
    vector<emb_key_t> ssdKeys;
    for (auto& key : keys) {
        if (cacheManager->IsKeyInSSD(embName, key)) {
            ssdKeys.emplace_back(key);
        }
    }
    cacheManager->EvictSSDEmbedding(embName, ssdKeys);
}

int HybridMgmt::GetStepFromPath(const string& loadPath) const
{
    regex pattern("sparse-model-(\\d+)");
    smatch match;
    if (regex_search(loadPath, match, pattern)) {
        int res = 0;
        unsigned int minSize = 2;
        if (match.size() < minSize) {
            return res;
        }
        try {
            res = stoi(match[1]);
        } catch (const std::invalid_argument& e) {
            LOG_ERROR(e.what());
        } catch (const std::out_of_range& e) {
            LOG_ERROR(e.what());
        }
        return res;
    }
    return 0;
}

/// 通过pyBind在python侧调用，通知hybridMgmt上层即将进行图的执行，需要进行唤醒
/// \param channelID 通道id
/// \param steps 运行的步数，由于可能存在循环下沉，所以1个session run 对应N步
void HybridMgmt::NotifyBySessionRun(int channelID) const
{
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    hybridMgmtBlock->CheckAndNotifyWake(channelID);
}

/// 通过pyBind在python侧调用，通知hybridMgmt上层即将进行图的执行
/// \param channelID 通道id
/// \param steps 运行的步数，由于可能存在循环下沉，所以1个session run 对应N步
void HybridMgmt::CountStepBySessionRun(int channelID, int steps) const
{
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    hybridMgmtBlock->CountPythonStep(channelID, steps);
}

/// 获取table表使用大小
/// \param embName 表名
/// \return 表使用大小
int64_t HybridMgmt::GetTableSize(const string& embName) const
{
#ifndef GTEST
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    if (mgmtRankInfo.useDynamicExpansion) {
        int64_t size = EmbeddingMgmt::Instance()->GetSize(embName);
        LOG_INFO(MGMT + "dynamic expansion mode, get emb:[{}] size:{}", embName, size);
        return size;
    }
    if (!mgmtRankInfo.isDDR) {
        size_t maxOffset = EmbeddingMgmt::Instance()->GetMaxOffset(embName);
        int64_t size = static_cast<int64_t>(maxOffset);
        LOG_INFO(MGMT + "HBM mode, get emb:[{}] size:{}", embName, size);
        return size;
    }
    int64_t ssdSize = 0;
    if (mgmtRankInfo.isSSDEnabled) {
        ssdSize = cacheManager->GetTableEmbeddingSize(embName);
    }

    const auto& iter = hostHashMaps->embHashMaps.find(embName);
    if (iter == hostHashMaps->embHashMaps.end()) {
        LOG_ERROR(MGMT + "get maxOffset, wrong embName:{} ", embName);
        return -1;
    }
    auto maxOffset = hostHashMaps->embHashMaps.at(embName).maxOffset;
    int64_t size = static_cast<int64_t>(maxOffset) + ssdSize;

    LOG_INFO(MGMT + "DDR/SSD mode, get emb:[{}] size:{}", embName, size);
    return size;
#endif
}

/// 获取table表容量大小
/// \param embName 表名
/// \return 表容量大小
int64_t HybridMgmt::GetTableCapacity(const string& embName) const
{
#ifndef GTEST
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }

    if (mgmtRankInfo.useDynamicExpansion) {
        int64_t capacity = EmbeddingMgmt::Instance()->GetCapacity(embName);
        LOG_INFO(MGMT + "dynamic expansion mode, get emb:[{}] capacity:{}", embName, capacity);
        return capacity;
    }
    LOG_WARN(MGMT + "no dynamic expansion mode, get emb:[{}] capacity failed", embName);
    return -1;
#endif
}

/// 设置表的优化器信息
/// \param embName 表名
/// \param optimInfo 优化器信息
/// \return
void HybridMgmt::SetOptimizerInfo(const string& embName, OptimizerInfo optimInfo) const
{
    if (!isInitialized) {
        throw runtime_error("HybridMgmt not initialized. Call Initialize first.");
    }
    EmbeddingMgmt::Instance()->SetOptimizerInfo(embName, optimInfo);
}

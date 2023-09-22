/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */
#include "hybrid_mgmt.h"

#include "utils/time_cost.h"
#include "utils/log.h"
#include "checkpoint/checkpoint.h"


using namespace MxRec;
using namespace std;

/// 启动数据处理线程
/// \param rankInfo 当前rank基本配置信息
/// \param embInfos 表信息list
/// \param thresholdValues 准入淘汰相关配置
/// \param seed 随机种子
/// \return bool类型 启动成功/失败
bool HybridMgmt::InitKeyProcess(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos,
                                const vector<ThresholdValue>& thresholdValues, int seed)
{
#ifndef GTEST
    // 初始化数据处理类，配置相关信息，启动处理线程
    preprocess = Singleton<KeyProcess>::GetInstance();
    preprocess->Initialize(rankInfo, embInfos, thresholdValues, seed);
    preprocess->Start();
#endif
    return true;
}

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
    if (totHostVocabSize == 0) {
        rankInfo.noDDR = true;
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
    g_statOn = GlobalEnv::statOn;

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
    bool rc = InitKeyProcess(rankInfo, embInfos, thresholdValues, seed);
    if (!rc) {
        return false;
    }

    isRunning = true;

    // DDR模式，初始化hashmap和host emb
    if (!rankInfo.noDDR) {
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
    }
    isLoad = ifLoad;
    if (!isLoad) {
        Start();
    }

    for (const auto& info: embInfos) {
        LOG_INFO(MGMT + "emb[{}] vocab size {}+{} sc:{}",
                 info.name, info.devVocabSize, info.hostVocabSize, info.sendCount);
    }
    LOG_INFO(MGMT + "end initialize, noDDR:{}, maxStep:[{}, {}], rank:{}", rankInfo.noDDR,
             rankInfo.maxStep.at(TRAIN_CHANNEL_ID), rankInfo.maxStep.at(EVAL_CHANNEL_ID), rankInfo.rankId);
#endif
    return true;
}

// 比较hostHashMap和cacheManager的数据是否一致
void HybridMgmt::AddCacheManagerTraceLog(CkptData& saveData)
{
    if (Log::GetLevel() != Log::TRACE) {
        return;
    }
    auto& embHashMaps = saveData.embHashMaps;
    auto& ddrKeyFreqMap = saveData.ddrKeyFreqMaps;
    for (auto& it : embHashMaps) {
        string embTableName = it.first;
        auto& hostMap = it.second.hostHashMap;
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
    // 数据处理线程上锁
    preprocess->LoadSaveLock();

    CkptData saveData;
    Checkpoint saveCkpt;
    if (!mgmtRankInfo.noDDR) {
        // DDR模式保存host的emb表以及hashmap
        LOG_DEBUG(MGMT + "Start host side save: ddr mode hashmap");
        saveData.hostEmbs = hostEmbs->GetHostEmbs();
        saveData.embHashMaps = hostHashMaps->GetHashMaps();
    } else {
        // HBM模式保存最大偏移（真正使用了多少vocab容量），特征到偏移的映射
        LOG_DEBUG(MGMT + "Start host side save: no ddr mode hashmap");
        saveData.maxOffset = preprocess->GetMaxOffset();
        saveData.keyOffsetMap = preprocess->GetKeyOffsetMap();
    }

    if (isSSDEnabled) {
        for (auto& it : cacheManager->ddrKeyFreqMap) {
            saveData.ddrKeyFreqMaps[it.first] = it.second.GetFreqTable();
        }
        saveData.excludeDDRKeyFreqMaps = cacheManager->excludeDDRKeyCountMap;
        RestoreFreq4Save(saveData);
        AddCacheManagerTraceLog(saveData);
        auto step = GetStepFromPath(savePath);
        cacheManager->SaveSSDEngine(step);
    }

    // 保存特征准入淘汰相关的数据
    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        LOG_DEBUG(MGMT + "Start host side save: feature admit and evict");
        saveData.table2Thresh = featAdmitNEvict.GetTableThresholds();
        saveData.histRec.timestamps = featAdmitNEvict.GetHistoryRecords().timestamps;
        saveData.histRec.historyRecords = featAdmitNEvict.GetHistoryRecords().historyRecords;
    }

    // 执行保存操作
    saveCkpt.SaveModel(savePath, saveData, mgmtRankInfo, mgmtEmbInfo);

    // 数据处理线程释放锁
    preprocess->LoadSaveUnlock();
#endif
    return true;
}

/// 加载模型
/// \param loadPath
/// \return
bool HybridMgmt::Load(const string& loadPath)
{
#ifndef GTEST
    // 数据处理线程上锁
    preprocess->LoadSaveLock();

    LOG_DEBUG(MGMT + "Start host side load process");

    CkptData loadData;
    Checkpoint loadCkpt;
    vector<CkptFeatureType> loadFeatures;

    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    SetFeatureTypeForLoad(loadFeatures, featAdmitNEvict);

    loadData.hostEmbs = hostEmbs->GetHostEmbs(); // 获取已经初始化好的host emb
    // 执行加载操作
    loadCkpt.LoadModel(loadPath, loadData, mgmtRankInfo, mgmtEmbInfo, loadFeatures);

    // 检查DDR模式保存的模型和当前训练配置是否一致，不一致则退出
    if (!mgmtRankInfo.noDDR && !LoadMatchesDDRSetup(loadData)) {
        preprocess->LoadSaveUnlock();
        return false;
    }

    if (!mgmtRankInfo.noDDR) {
        // DDR模式 将加载的hash map进行赋值
        LOG_DEBUG(MGMT + "Start host side load: ddr mode hashmap");
        hostHashMaps->LoadHashMap(loadData.embHashMaps);
    } else {
        // HBM模式 将加载的最大偏移（真正使用了多少vocab容量）、特征到偏移的映射，进行赋值
        LOG_DEBUG(MGMT + "Start host side load: no ddr mode hashmap");
        preprocess->LoadMaxOffset(loadData.maxOffset);
        preprocess->LoadKeyOffsetMap(loadData.keyOffsetMap);
    }

    // 将加载的特征准入淘汰记录进行赋值
    if (featAdmitNEvict.GetFunctionSwitch()) {
        LOG_DEBUG(MGMT + "Start host side load: feature admit and evict");
        featAdmitNEvict.LoadTableThresholds(loadData.table2Thresh);
        featAdmitNEvict.LoadHistoryRecords(loadData.histRec);
    }

    if (isSSDEnabled) {
        LOG_DEBUG(MGMT + "Start host side load: ssd key freq map");
        auto step = GetStepFromPath(loadPath);
        cacheManager->Load(loadData.ddrKeyFreqMaps, loadData.excludeDDRKeyFreqMaps, step);
    }

    LOG_DEBUG(MGMT + "Finish host side load process");

    preprocess->LoadSaveUnlock();

    // 执行训练
    if (isLoad) {
        Start();
    }
#endif
    return true;
}

void HybridMgmt::SetFeatureTypeForLoad(vector<CkptFeatureType>& loadFeatures,
                                       const FeatureAdmitAndEvict& featAdmitNEvict)
{
    if (!mgmtRankInfo.noDDR) {
        // DDR模式加载的类型为host的emb表以及hashmap
        loadFeatures.push_back(CkptFeatureType::HOST_EMB);
        loadFeatures.push_back(CkptFeatureType::EMB_HASHMAP);
    } else {
        // HBM模式加载的类型为最大偏移（真正使用了多少vocab容量），特征到偏移的映射
        loadFeatures.push_back(CkptFeatureType::MAX_OFFSET);
        loadFeatures.push_back(CkptFeatureType::KEY_OFFSET_MAP);
    }

    // 添加特征准入淘汰相关的数据类型的加载
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
key_offset_map_t HybridMgmt::SendHostMap(const string tableName)
{
#ifndef GTEST
    preprocess->LoadSaveLock();
    key_offset_mem_t keyOffsetMap;
    key_offset_map_t sendKeyOffsetMap;

    if (!mgmtRankInfo.noDDR) {
        LOG_DEBUG(MGMT + "Start send sparse data: ddr mode hashmap");
    } else {
        LOG_DEBUG(MGMT + "Start send sparse data: no ddr mode hashmap");
        keyOffsetMap = preprocess->GetKeyOffsetMap();
    }

    if ((!keyOffsetMap.empty()) && keyOffsetMap.count(tableName) > 0) {
        for (const auto& it : keyOffsetMap.at(tableName)) {
            sendKeyOffsetMap[it.first] = it.second;
        }
    }

    preprocess->LoadSaveUnlock();
    return sendKeyOffsetMap;
#endif
}

/// 加载key对应的offset，python侧调用；启动数据处理线程
/// \param ReceiveKeyOffsetMap
void HybridMgmt::ReceiveHostMap(all_key_offset_map_t receiveKeyOffsetMap)
{
#ifndef GTEST
    preprocess->LoadSaveLock();
    key_offset_mem_t loadKeyOffsetMap;
    offset_mem_t loadMaxOffset;
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
    if (!mgmtRankInfo.noDDR) {
        LOG_DEBUG(MGMT + "Start receive sparse data: ddr mode hashmap");
    } else {
        LOG_DEBUG(MGMT + "Start receive sparse data: no ddr mode hashmap");
        preprocess->LoadKeyOffsetMap(loadKeyOffsetMap);
        preprocess->LoadMaxOffset(loadMaxOffset);
    }

    preprocess->LoadSaveUnlock();
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
bool HybridMgmt::IsLoadDataMatches(emb_mem_t& loadHostEmbs, EmbInfo& setupHostEmbs, size_t& embTableCount) const
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
    if (mgmtRankInfo.noDDR) {
        InsertThreadForHBM();
    }

    if (!mgmtRankInfo.noDDR) {
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
    }
#endif
}

/// 启动HBM模式数据处理线程
void HybridMgmt::InsertThreadForHBM()
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
    LOG_INFO(MGMT + "start parse keys HBM, nBatch:{} , [{}]:{}", mgmtRankInfo.nBatch, channelId, batchId);

    // 循环处理每个表的数据
    for (const auto& embInfo: mgmtEmbInfo) {
        TimeCost parseKeysTc;
        // get
        TimeCost getTensorsSyncTC;

        // 获取各类向量，如果为空指针，退出当前函数
        auto infoVecs = preprocess->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::RESTORE);
        if (infoVecs == nullptr) {
            LOG_INFO(MGMT + "ParseKeys infoVecs empty ! batchId:{}, channelId:{}", batchId, channelId);
            return false;
        }

        // 动态shape场景下，获取all2all向量（通信量矩阵）
        unique_ptr<vector<Tensor>> all2all = nullptr;
        if (!mgmtRankInfo.useStatic) {
            all2all = preprocess->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::ALL2ALL);
        }
        LOG_DEBUG("getTensorsSyncTC(ms):{}", getTensorsSyncTC.ElapsedMS());

        // 动态shape场景下，发送all2all向量（通信量矩阵）
        TimeCost sendTensorsSyncTC;
        if (!mgmtRankInfo.useStatic) {
            TimeCost sendAll2AllScSyncTC;
            hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embInfo.name);
            LOG_DEBUG("sendAll2AllScSyncTC(ms):{}", sendAll2AllScSyncTC.ElapsedMS());
        }

        // 发送查询向量
        TimeCost sendLookupSyncTC;
        hdTransfer->Send(TransferChannel::LOOKUP, { infoVecs->back() }, channelId, embInfo.name);
        infoVecs->pop_back();
        LOG_DEBUG("sendLookupSyncTC(ms):{}", sendLookupSyncTC.ElapsedMS());

        // 训练时，使用全局去重聚合梯度，发送全局去重的key和对应的恢复向量
        if (GlobalEnv::applyGradientsStrategy == ApplyGradientsStrategyOptions::SUM_SAME_ID_GRADIENTS_AND_APPLY &&
            channelId == TRAIN_CHANNEL_ID) {
            TimeCost sendUnikeysSyncTC;
            hdTransfer->Send(TransferChannel::UNIQKEYS, { infoVecs->back() }, channelId, embInfo.name);
            infoVecs->pop_back();
            LOG_DEBUG("sendUnikeysSyncTC(ms):{}", sendUnikeysSyncTC.ElapsedMS());

            TimeCost sendRestoreVecSecSyncTC;
            hdTransfer->Send(TransferChannel::RESTORE_SECOND, { infoVecs->back() }, channelId, embInfo.name);
            infoVecs->pop_back();
            LOG_DEBUG("sendRestoreVecSecSyncTC(ms):{}", sendRestoreVecSecSyncTC.ElapsedMS());
        }

        // 发送恢复向量
        TimeCost sendRestoreSyncTC;
        hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, channelId, embInfo.name);
        LOG_DEBUG("sendRestoreSyncTC(ms):{}, sendTensorsSyncTC(ms):{}, parseKeysTc HBM mode (ms):{}",
                  sendRestoreSyncTC.ElapsedMS(), sendTensorsSyncTC.ElapsedMS(), parseKeysTc.ElapsedMS());
    }
    batchId++;
    return true;
}
#endif

/// 当前处理的batch是否是最后一个batch
/// \param batchId 已处理的batch数
/// \param channelId 通道索引（训练/推理）
/// \return
bool HybridMgmt::EndBatch(int batchId, int channelId) const
{
    return (batchId % mgmtRankInfo.maxStep[channelId] == 0 && mgmtRankInfo.maxStep[channelId] != -1);
}

/// DDR模式下，发送key process线程已处理好的各类型向量到指定通道中
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \return
bool HybridMgmt::ParseKeys(int channelId, int& batchId)
{
#ifndef GTEST
    LOG_INFO(MGMT + "DDR mode, start parse keys, [{}]:{}", channelId, batchId);
    TimeCost parseKeyTC;
    int start = batchId;
    bool remainBatch = true; // 是否从通道获取了数据

    LOG_INFO(MGMT + "parse keys, [{}]:{}", channelId, batchId);
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
    TimeCost embHdTrans2TC;
    EmbHDTransWrap(channelId, batchId - 1, start);
    LOG_DEBUG("embHdTrans2TC TimeCost(ms):{}", embHdTrans2TC.ElapsedMS());
    LOG_DEBUG("[{}]-{}, parseKeyTC TimeCost(ms):{}", channelId, batchId, parseKeyTC.ElapsedMS());
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
    auto& embHashMap = hostHashMaps->embHashMaps.at(embName);

    // 计数初始化
    embHashMap.SetStartCount();

    // 获取查询向量
    auto lookupKeys = preprocess->GetLookupKeys(batchId, embName, channelId);
    if (lookupKeys.empty()) {
        remainBatchOut = false;
        return false;
    }

    // 获取各类向量，如果为空指针，退出当前函数
    auto infoVecs = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::RESTORE);
    if (infoVecs == nullptr) { return false; }
    LOG_DEBUG("getTensorsTC(ms):{}", getTensorsTC.ElapsedMS());

    TimeCost sendRestoreSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, channelId, embName);
    LOG_DEBUG("sendRestoreSyncTC(ms):{}", sendRestoreSyncTC.ElapsedMS());

    // 调用SSD cache缓存处理流程
    PrepareDDRData(embName, embHashMap, lookupKeys, channelId);

    // 计算查询向量；记录需要被换出的HBM偏移
    vector<Tensor> tmpData;
    vector<int32_t> offsetsOut;
    DDRParam ddrParam(tmpData, offsetsOut);
    TimeCost hostHashMapProcessTC;
    hostHashMaps->Process(embName, lookupKeys, ddrParam, channelId);
    LOG_DEBUG("hostHashMapProcessTC(ms):{}", hostHashMapProcessTC.ElapsedMS());

    if (GlobalEnv::applyGradientsStrategy == ApplyGradientsStrategyOptions::SUM_SAME_ID_GRADIENTS_AND_APPLY &&
        channelId == TRAIN_CHANNEL_ID && remainBatchOut) {
        vector<int32_t> uniqueKeys;
        vector<int32_t> restoreVecSec;
        preprocess->GlobalUnique(offsetsOut, uniqueKeys, restoreVecSec);

        TimeCost sendUnikeysSyncTC;
        hdTransfer->Send(TransferChannel::UNIQKEYS, { mgmtRankInfo.useDynamicExpansion ? Vec2TensorI64(uniqueKeys) :
                                                      Vec2TensorI32(uniqueKeys) }, channelId, embName);

        TimeCost sendRestoreVecSecSyncTC;
        hdTransfer->Send(TransferChannel::RESTORE_SECOND, { Vec2TensorI32(restoreVecSec) }, channelId, embName);
        LOG_DEBUG("sendUnikeysSyncTC(ms):{}sendRestoreVecSecSyncTC(ms):{}",
                  sendUnikeysSyncTC.ElapsedMS(), sendRestoreVecSecSyncTC.ElapsedMS());
    }

    TimeCost sendTensorsTC;
    hdTransfer->Send(TransferChannel::LOOKUP, { ddrParam.tmpDataOut.front() }, channelId, embName);
    ddrParam.tmpDataOut.erase(ddrParam.tmpDataOut.cbegin());
    hdTransfer->Send(TransferChannel::SWAP, ddrParam.tmpDataOut, channelId, embName);
    if (!mgmtRankInfo.useStatic) {
        auto all2all = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::ALL2ALL);
        hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embName);
    }
    LOG_DEBUG("sendTensorsTC(ms):{} getAndSendTensorsTC(ms):{}, channelId:{}",
              sendTensorsTC.ElapsedMS(), getAndSendTensorsTC.ElapsedMS(), channelId);

    if (!isSSDEnabled && embHashMap.HasFree(lookupKeys.size())) { // check free > next one batch
        LOG_WARN(MGMT + "embName {}[{}]{}, freeSize not enough, {}", embName, channelId, batchId, lookupKeys.size());
        return false;
    }
    return true;
}

/// 发送H2D和接收D2H向量
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \param start
void HybridMgmt::EmbHDTransWrap(int channelId, const int& batchId, int start)
{
    LOG_INFO(MGMT + "trans emb, batchId:[{}-{}], channelId:{}", start, batchId, channelId);
    TimeCost hostEmbsTC;
    hostEmbs->Join(channelId);
    LOG_DEBUG("hostEmbsTC(ms):{}", hostEmbsTC.ElapsedMS());

    EmbHDTrans(channelId, batchId);
}

/// 发送H2D和接收D2H向量，并更新host emb
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
void HybridMgmt::EmbHDTrans(const int channelId, const int batchId)
{
    EASY_FUNCTION(profiler::colors::Blue)
    EASY_VALUE("mgmtProcess", batchId)
    LOG_DEBUG(MGMT + "trans emb, batchId:{}, channelId:{}", batchId, channelId);
    TimeCost tr;
    TimeCost h2dTC;
    // 发送host需要换出的emb
    for (const auto& embInfo: mgmtEmbInfo) {
        auto& missingKeys = hostHashMaps->GetMissingKeys(embInfo.name);
        vector<Tensor> h2dEmb;
        hostEmbs->GetH2DEmb(missingKeys, embInfo.name, h2dEmb); // order!
        hdTransfer->Send(TransferChannel::H2D, h2dEmb, channelId, embInfo.name, batchId);
    }
    LOG_DEBUG("h2dTC(ms):{}", h2dTC.ElapsedMS());

    TimeCost d2hTC;
    // 接收device换出的emb，并更新到host上
    for (const auto& embInfo: mgmtEmbInfo) {
        const auto& missingKeys = hostHashMaps->GetMissingKeys(embInfo.name);
        if (GlobalEnv::updateEmbV2) {
            hostEmbs->UpdateEmbV2(missingKeys, channelId, embInfo.name); // order!
        } else {
            hostEmbs->UpdateEmb(missingKeys, channelId, embInfo.name); // order!
        }
        hostHashMaps->ClearMissingKeys(embInfo.name);
    }
    LOG_DEBUG("D2HTC(ms):{} EmbHDTrans TimeCost(ms):{} batchId: {} channelId:{}",
              d2hTC.ElapsedMS(), tr.ElapsedMS(), batchId, channelId);
}
#endif

/// hook通过时间或者step数触发淘汰
/// \return
bool HybridMgmt::Evict()
{
#ifndef GTEST
    // 配置了淘汰选项，则触发
    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
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

    if (mgmtRankInfo.noDDR) {
        for (const auto& evict : as_const(evictKeyMap)) {
            preprocess->EvictKeys(evict.first, evict.second);
        }
    } else {
        for (const auto& evict : as_const(evictKeyMap)) {
            EvictKeys(evict.first, evict.second);
            EvictSSDKeys(evict.first, evict.second);
        }
    }
    return true;
#endif
}

/// DDR模式下的淘汰：删除映射表、初始化host表、发送dev淘汰位置
/// \param embName
/// \param keys
void HybridMgmt::EvictKeys(const string& embName, const vector<emb_key_t>& keys)
{
#ifndef GTEST
    LOG_DEBUG(MGMT + "ddr mode, delete emb: [{}]! evict keySize:{}", embName.c_str(), keys.size());
    // 删除映射关系
    if (keys.size() != 0) {
        hostHashMaps->EvictDeleteEmb(embName, keys);
    }

    // 初始化host侧的emb
    auto& evictOffset = hostHashMaps->GetEvictPos(embName);
    vector<size_t> evictOffset4Ddr;
    auto devVocabSize = hostHashMaps->embHashMaps.at(embName).devVocabSize;
    for (auto& offsetInHostHashMap : evictOffset) {
        evictOffset4Ddr.emplace_back(offsetInHostHashMap - devVocabSize);
    }
    if (!evictOffset4Ddr.empty()) {
        LOG_DEBUG(MGMT + "ddr mode, delete emb: [{}]! evict size on host:{}", embName, evictOffset4Ddr.size());
        hostEmbs->EvictInitEmb(embName, evictOffset4Ddr);
    } else {
        LOG_INFO(MGMT + "ddr mode, evict size on host is empty");
    }

    // 发送dev侧的淘汰pos，以便dev侧初始化emb
    auto evictDevOffset = hostHashMaps->embHashMaps.at(embName).evictDevPos;
    LOG_DEBUG(MGMT + "ddr mode, init dev emb: [{}]! evict size on dev :{}", embName, evictDevOffset.size());

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

inline void HybridMgmt::PrepareDDRData(const string& embTableName, EmbHashMapInfo& embHashMap,
                                       const vector<emb_key_t>& keys, int channelId) const
{
    if (!isSSDEnabled) {
        return;
    }
    LOG_DEBUG("PrepareDDRData start.");
    TimeCost prepareDDRDataTc;
    TransferRet ret = cacheManager->TransferDDREmbWithSSD(embTableName, embHashMap, keys, channelId);
    if (ret != TransferRet::TRANSFER_OK) {
        HandlePrepareDDRDataRet(ret);
    }
    LOG_DEBUG("PrepareDDRData end, TimeCost(ms):{}", prepareDDRDataTc.ElapsedMS());
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
    regex pattern("sparse-model-\\d+-(\\d+)");
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
    hybridMgmtBlock->CheckAndNotifyWake(channelID);
}

/// 通过pyBind在python侧调用，通知hybridMgmt上层即将进行图的执行
/// \param channelID 通道id
/// \param steps 运行的步数，由于可能存在循环下沉，所以1个session run 对应N步
void HybridMgmt::CountStepBySessionRun(int channelID, int steps) const
{
    hybridMgmtBlock->CountPythonStep(channelID, steps);
}
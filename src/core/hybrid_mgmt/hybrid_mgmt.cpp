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

#include <mpi.h>

#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "checkpoint/checkpoint.h"
#include "emb_table/embedding_mgmt.h"
#include "hd_transfer/hd_transfer.h"
#include "hybrid_mgmt/hybrid_mgmt_block.h"
#include "key_process/feature_admit_and_evict.h"
#include "key_process/key_process.h"
#include "utils/common.h"
#include "utils/error.h"
#include "utils/logger.h"
#include "utils/time_cost.h"

using namespace MxRec;
using namespace std;
using namespace ock::ctr;

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
    size_t totalL3StorageVocabSize = 0;
    for (const auto& emb : embInfos) {
        totHostVocabSize += emb.hostVocabSize;
        totalL3StorageVocabSize += emb.ssdVocabSize;
    }

    // 根据DDR的key数量，配置存储模式HBM/DDR
    if (totHostVocabSize != 0) {
        rankInfo.isDDR = true;
    }
    if (totalL3StorageVocabSize != 0) {
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
                            const vector<ThresholdValue>& thresholdValues, bool ifLoad, bool isIncrementalCheckpoint)
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

    // create factory for fastUnique and embeddingCache
    int result = ock::ctr::Factory::Create(factory);
    if (result != 0) {
        auto error = Error(ModuleName::M_OCK_CTR, ErrorType::CONSTRUCT_ERROR,
                           StringFormat("Create fast factory failed, error code: %d.", result));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }

    // InitPool need to be before Start().
    threadPool = make_unique<ThreadPool>(embInfos.size() * MAX_CHANNEL_NUM);

    InitRankInfo(rankInfo, embInfos);
    GlogConfig::gStatOn = GlobalEnv::statOn;

    LOG_INFO(MGMT + "begin initialize, localRankSize:{}, localRankId:{}, rank:{}", rankInfo.localRankSize,
             rankInfo.localRankId, rankInfo.rankId);

    mgmtRankInfo = rankInfo;
    mgmtEmbInfo = embInfos;
    isIncrementalCkpt = isIncrementalCheckpoint;

    // 进行acl资源初始化，设置当前训练进程的device，为每张表创建数据传输通道
    hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    hdTransfer->Init(embInfos, rankInfo.deviceId, isIncrementalCheckpoint);

    hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
    hybridMgmtBlock->SetRankInfo(rankInfo);

    // 启动数据处理线程
    KEY_PROCESS_INSTANCE->Initialize(rankInfo, embInfos, thresholdValues, seed, isIncrementalCheckpoint);

    isRunning = true;
    isL3StorageEnabled = rankInfo.isSSDEnabled;
    EmbeddingMgmt::Instance()->Init(rankInfo, embInfos, seed);

    if (rankInfo.isDDR) {
        InitEmbeddingCache(embInfos);
    }

    if (isL3StorageEnabled) {
        cacheManager = Singleton<MxRec::CacheManager>::GetInstance();
        // 用户可实现L3Storage接口替换SSDEngine以对接外部存储服务
        auto ssdEngine = std::make_shared<SSDEngine>();
        cacheManager->Init(embCache, mgmtEmbInfo, ssdEngine);
        EmbeddingMgmt::Instance()->SetCacheManagerForEmbTable(cacheManager);
    }
    isLoad = ifLoad;
    if (!isLoad) {
        Start();
    }

    // 启动接收python侧发回的key的线程
    if (isIncrementalCheckpoint) {
        ReceiveKey();
    }

    for (const auto& info : embInfos) {
        LOG_INFO(MGMT + "table:{}, vocab size dev+host:{}+{}, send count:{}", info.name, info.devVocabSize,
                 info.hostVocabSize, info.sendCount);
    }
    LOG_INFO(MGMT + "end initialize, rankId:{}, isDDR:{}, "
                    "step[train_interval, eval_interval, save_interval, max_train_step]:[{}, {}, {}, {}]",
             rankInfo.rankId, rankInfo.isDDR, rankInfo.ctrlSteps.at(TRAIN_CHANNEL_ID),
             rankInfo.ctrlSteps.at(EVAL_CHANNEL_ID), rankInfo.ctrlSteps.at(SAVE_STEP_INDEX),
             rankInfo.ctrlSteps.at(MAX_TRAIN_STEP_INDEX));
#endif
    isInitialized = true;

    return true;
}

/// 保存模型
/// \param savePath 保存路径
/// \return
void HybridMgmt::Save(const string& savePath, bool saveDelta)
{
#ifndef GTEST
    if (!isInitialized) {
        auto error = Error(ModuleName::M_CHECK_POINT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before save.");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    string saveModelType =
        saveDelta ? TransferModelType2Str(SaveModelType::DELTA) : TransferModelType2Str(SaveModelType::BASE);
    LOG_INFO(MGMT + "Start to save {} model to {}.", saveModelType, savePath);

    // 数据处理线程上锁
    KEY_PROCESS_INSTANCE->LoadSaveLock();

    CkptData saveData;
    Checkpoint saveCkpt;
    saveData.keyCountMap = KEY_PROCESS_INSTANCE->GetKeyCountMap();

    map<string, map<emb_key_t, KeyInfo>> keyInfoMap;
    GetDeltaModelKeys(savePath, saveDelta, keyInfoMap);
    EmbeddingMgmt::Instance()->Save(savePath, hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID], saveDelta, keyInfoMap);
    // after save key、embedding, reset deltaMap, isChanged->false, recentCount->0
    if (isIncrementalCkpt) {
        ResetDeltaInfo();
    }
    if (!mgmtRankInfo.isDDR) {
        // hbm模式只保存必要的offset对应的内容
        offsetMapToSend = EmbeddingMgmt::Instance()->GetDeviceOffsets();
    }

    if (isL3StorageEnabled) {
        LOG_DEBUG(MGMT + "start save L3Storage data");
        auto step = GetStepFromPath(savePath);
        cacheManager->Save(step);
    }

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
    isFirstSave = false;
    LOG_INFO(MGMT + "End to save {} model.", saveModelType);
    // 数据处理线程释放锁
    KEY_PROCESS_INSTANCE->LoadSaveUnlock();
#endif
}

/// 加载模型
/// \param loadPath
/// \return
bool HybridMgmt::Load(const string& loadPath, vector<string> warmStartTables)
{
#ifndef GTEST
    if (!isInitialized) {
        auto error = Error(ModuleName::M_CHECK_POINT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before load.");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }

    // 数据处理线程上锁
    KEY_PROCESS_INSTANCE->LoadSaveLock();

    LOG_DEBUG(MGMT + "Start host side load process");

    CkptData loadData;
    Checkpoint loadCkpt;
    vector<CkptFeatureType> loadFeatures;
    SetFeatureTypeForLoad(loadFeatures);
    BackUpTrainStatus();

    if (warmStartTables.size() == 0) {
        EmbeddingMgmt::Instance()->Load(loadPath, trainKeysSet);
    } else {
        for (auto& tableName : warmStartTables) {
            EmbeddingMgmt::Instance()->Load(tableName, loadPath, trainKeysSet);
        }
    }

    if (!mgmtRankInfo.isDDR) {
        // hbm模式只保存必要的offset对应的内容
        loadOffsetToSend = EmbeddingMgmt::Instance()->GetLoadOffsets();
    }

    // 执行加载操作
    loadCkpt.LoadModel(loadPath, loadData, mgmtRankInfo, mgmtEmbInfo, loadFeatures);

    KEY_PROCESS_INSTANCE->LoadKeyCountMap(loadData.keyCountMap);
    if (!mgmtRankInfo.isDDR) {
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

    int& theTrainBatchId = hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID];
    if (isL3StorageEnabled) {
        LOG_DEBUG(MGMT + "Start host side load: L3Storage key freq map");
        auto step = GetStepFromPath(loadPath);
        // When in load and train mode or predict mode, SSD needs to actually execute loading
        // When in the train and eval modes, loading before eval should be directly skipped
        if (theTrainBatchId == 0) {
            cacheManager->Load(mgmtEmbInfo, step, trainKeysSet);
        }
    }

    LOG_DEBUG(MGMT + "Finish host side load process");

    KEY_PROCESS_INSTANCE->LoadSaveUnlock();

    // 执行训练
    if (isLoad && procThreads.empty()) {
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
        auto error = Error(ModuleName::M_CHECK_POINT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before load offset.");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
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
    if (isLoad && procThreads.empty()) {
        Start();
    }
#endif
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

    auto embeddingProcessTask = [this]() {
        EmbeddingTask();
        LOG_INFO("embeddingProcessTask done");
    };
    procThreads.emplace_back(std::make_unique<std::thread>(embeddingProcessTask));
#endif
}

void HybridMgmt::Destroy()
{
    LOG_DEBUG(MGMT + "start Destroy hybrid_mgmt module");
    if (!isInitialized) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. No need to call [terminate].");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }

    if (!isRunning) {
        return;
    }
    // 先发送停止信号mgmt，先停止新lookup查询, 解除queue的限制防止卡住
    isRunning = false;
    mutexDestroy = true;

    {
        // 获取锁 避免KeyProcess中手动发送结束信息时通道关闭
        std::unique_lock<std::mutex> lockGuard(KEY_PROCESS_INSTANCE->destroyMutex);
        // 先发送停止信号给KEY_PROCESS_INSTANCE，用于停止查询中lookup卡住状态
        KEY_PROCESS_INSTANCE->isRunning = false;
        // 停止hdTransfer，用于停止mgmt的recv中卡住状态
        hdTransfer->Destroy();
        LOG_DEBUG(MGMT + "destroy hdTransfer end.");
    }

    JoinEmbeddingCacheThread();
    LOG_DEBUG(MGMT + "destroy EmbeddingCacheThread end.");

    hybridMgmtBlock->Destroy();
    for (auto& t : procThreads) {
        t->join();
    }
    procThreads.clear();
    LOG_DEBUG(MGMT + "destroy parseKeyThread end.");

    if (cacheManager != nullptr) {
        cacheManager = nullptr;
    }

    // 等待并销毁接收key的线程
    for (auto& t : receiveKeyThreads) {
        t.join();
    }
    receiveKeyThreads.clear();
    // 停止预处理
    KEY_PROCESS_INSTANCE->Destroy();
    // stop embCache, even if the host emb is still allocating
    if (embCache != nullptr) {
        embCache->Destroy();
    }
    LOG_DEBUG(MGMT + "Destroy hybrid_mgmt module end.");
}

/// 启动hybrid处理任务
/// \param type
void HybridMgmt::TrainTask(TaskType type)
{
#ifndef GTEST
    int& theTrainBatchId = hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID];
    do {
        hybridMgmtBlock->CheckAndSetBlock(TRAIN_CHANNEL_ID);
        if (hybridMgmtBlock->GetBlockStatus(TRAIN_CHANNEL_ID)) {
            hybridMgmtBlock->DoBlock(TRAIN_CHANNEL_ID);
        }
        if (!isRunning) {
            return;
        }
        LOG_INFO(HYBRID_BLOCKING + "hybrid start task channel {} batch {}", TRAIN_CHANNEL_ID, theTrainBatchId);
        if (isBackUpTrainStatus) {
            RecoverTrainStatus();
        }
        ParseKeys(TRAIN_CHANNEL_ID, theTrainBatchId, type);
    } while (true);
#endif
}

/// 推理数据处理：数据处理状态正常，处理的batch数小于用户预设值或者设为-1时，会循环处理；
/// \param type 存储模式
/// \return
void HybridMgmt::EvalTask(TaskType type)
{
#ifndef GTEST
    int& evalBatchId = hybridMgmtBlock->hybridBatchId[EVAL_CHANNEL_ID];
    do {
        hybridMgmtBlock->CheckAndSetBlock(EVAL_CHANNEL_ID);
        if (hybridMgmtBlock->GetBlockStatus(EVAL_CHANNEL_ID)) {
            hybridMgmtBlock->DoBlock(EVAL_CHANNEL_ID);
        }
        if (!isRunning) {
            return;
        }
        LOG_INFO(HYBRID_BLOCKING + "hybrid start task channel {} batch {}", EVAL_CHANNEL_ID, evalBatchId);

        ParseKeys(EVAL_CHANNEL_ID, evalBatchId, type);
    } while (true);
#endif
}

void HybridMgmt::SendUniqKeysAndRestoreVecHBM(const EmbBaseInfo& info, const unique_ptr<vector<Tensor>>& infoVecs,
                                              bool isGrad) const
{
    TimeCost sendUniqueKeysSyncTC;
    LOG_DEBUG("channelId:{} batchId:{}, global unique, table name: {}, is grad: {}", info.channelId, info.batchId,
              info.name, isGrad);
    if (isGrad) {
        hdTransfer->Send(TransferChannel::UNIQKEYS, {infoVecs->back()}, info.channelId, info.name, info.batchId);
    }
    infoVecs->pop_back();
    LOG_DEBUG("channelId:{} batchId:{}, sendUniqueKeysSyncTC(ms):{}", info.channelId, info.batchId,
              sendUniqueKeysSyncTC.ElapsedMS());

    TimeCost sendUniqueRestoreVecSyncTC;
    if (isGrad) {
        hdTransfer->Send(TransferChannel::RESTORE_SECOND, {infoVecs->back()}, info.channelId, info.name, info.batchId);
    }
    infoVecs->pop_back();
    LOG_DEBUG("channelId:{} batchId:{}, sendUniqueRestoreVecSyncTC(ms):{}", info.channelId, info.batchId,
              sendUniqueRestoreVecSyncTC.ElapsedMS());
}

/// DDR模式下，发送key process线程已处理好的各类型向量到指定通道中
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \return
bool HybridMgmt::ParseKeys(int channelId, int& batchId, TaskType type)
{
#ifndef GTEST
    LOG_INFO(MGMT + "channelId:{} batchId:{}, ParseKeys start.", channelId, batchId);
    TimeCost parseKeyTC;

    std::vector<std::future<bool>> remainResult;
    for (const auto& embInfo : mgmtEmbInfo) {
        EmbBaseInfo info = {.batchId = batchId, .channelId = channelId, .name = embInfo.name, .isDp = embInfo.isDp};
        switch (type) {
            case TaskType::HBM: {
                std::future<bool> remainBatch = threadPool->enqueueWithFuture(
                    [this, info, embInfo]() { return ProcessEmbInfoHBM(info, embInfo.isGrad); });
                remainResult.push_back(std::move(remainBatch));
            } break;
            case TaskType::DDR:
                if (!isL3StorageEnabled) {
                    std::future<bool> remainBatch =
                        threadPool->enqueueWithFuture([this, info]() { return ProcessEmbInfoDDR(info); });
                    remainResult.push_back(std::move(remainBatch));
                } else {
                    std::future<bool> remainBatch =
                        threadPool->enqueueWithFuture([this, info]() { return ProcessEmbInfoL3Storage(info); });
                    remainResult.push_back(std::move(remainBatch));
                }
                break;
            default: {
                auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::INVALID_ARGUMENT, "Invalid TaskType Type.");
                LOG_ERROR(error.ToString());
                throw runtime_error(error.ToString().c_str());
            }
        }
    }

    bool isRemainAll = true;
    for (auto& remain : remainResult) {
        bool isRemain = remain.get();
        LOG_DEBUG(MGMT + "channelId:{} batchId:{}, ParseKeys thread get future:{}", channelId, batchId, isRemain);
        isRemainAll = isRemainAll && isRemain;
    }
    // 通道数据已空
    if (!isRemainAll) {
        LOG_DEBUG("last batch ending");
        return false;
    }

    if (!isRunning) {
        return false;
    }
    LOG_DEBUG(MGMT + "channelId:{} batchId:{}, ParseKeys end, parseKeyTC(ms):{}", channelId, batchId,
              parseKeyTC.ElapsedMS());
    batchId++;
#endif
    return true;
}

/// 构造训练所需的各种向量数据
/// \param info 表名、batch数、通道索引（训练/推理）
/// \param isGrad 是否需要发送反向需要的tensor
/// \return remainBatchOut 是否从通道获取了数据
bool HybridMgmt::ProcessEmbInfoHBM(const EmbBaseInfo& info, bool isGrad)
{
    bool remainBatchOut = true;
    TimeCost parseKeysTc;
    LOG_DEBUG("ProcessEmbInfoHBM table:{}, batchId:{}, channel:{}", info.name, info.batchId, info.channelId);

    // 获取各类向量，如果为空指针，退出当前函数
    bool isEos = false;
    auto infoVecs = KEY_PROCESS_INSTANCE->GetInfoVec(info, ProcessedInfo::RESTORE, isEos);
    if (isEos) {
        KEY_PROCESS_INSTANCE->SendEos(info.name, info.batchId, info.channelId);
        return false;
    }
    if (infoVecs == nullptr) {
        LOG_WARN(MGMT + "table:{}, channelId:{} batchId:{}, ParseKeys infoVecs empty !", info.name, info.channelId,
                 info.batchId);
        return false;
    }
    LOG_DEBUG("table:{}, channelId:{} batchId:{}, ParseKeysHBM GetInfoVec end", info.name, info.channelId,
              info.batchId);

    // 动态shape场景下，获取all2all向量（通信量矩阵）
    SendAll2AllVec(info, remainBatchOut);
    if (!remainBatchOut) {
        return false;
    }

    // 发送查询向量
    TimeCost sendLookupSyncTC;
    hdTransfer->Send(TransferChannel::LOOKUP, {infoVecs->back()}, info.channelId, info.name, info.batchId);
    infoVecs->pop_back();
    LOG_DEBUG("table:{}, channelId:{} batchId:{}, sendLookupSyncTC(ms):{}", info.name, info.channelId, info.batchId,
              sendLookupSyncTC.ElapsedMS());

    // 训练时，使用全局去重聚合梯度，发送全局去重的key和对应的恢复向量
    // In the DP mode, the second USS is used to align the length of the grad in the allreduce.
    if ((mgmtRankInfo.useSumSameIdGradients && info.channelId == TRAIN_CHANNEL_ID) ||
        (info.isDp && info.channelId == TRAIN_CHANNEL_ID)) {
        SendUniqKeysAndRestoreVecHBM(info, infoVecs, isGrad);
    }

    // 发送恢复向量和hotPos
    TimeCost sendRestoreSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, info.channelId, info.name, info.batchId);
    LOG_DEBUG("table:{}, sendRestoreSyncTC(ms):{}, parseKeysTc HBM mode (ms):{}", info.name,
              sendRestoreSyncTC.ElapsedMS(), parseKeysTc.ElapsedMS());

    LOG_INFO(MGMT + "table:{}, channelId:{} batchId:{}, embName:{}, ParseKeys with HBM mode end.", info.name,
             info.channelId, info.batchId, info.name);

    if (info.channelId == TRAIN_CHANNEL_ID) {
        alreadyTrainOnce = true;
    }
    return remainBatchOut;
}

/// 构造训练所需的各种向量数据
/// \param info 表名、batch数、通道索引（训练/推理）
/// \return remainBatchOut 是否从通道获取了数据
bool HybridMgmt::ProcessEmbInfoDDR(const EmbBaseInfo& info)
{
    bool remainBatchOut = true;
#ifndef GTEST
    TimeCost getAndSendTensorsTC;
    LOG_DEBUG("ProcessEmbInfoDDR start, table:{}, channel:{}, batchId:{}", info.name, info.channelId, info.batchId);

    // 只有在每次GetUniqueKeys的时候才知道上游是否已经EOS
    // 注意GetUniqueKeys与EOS关联，需要在ProcessEmbInfoDDR最先调用，如需调整位置，请参考并适配其他函数
    // 获取GlobalUnique向量
    bool isEos = false;
    auto uniqueKeys = GetUniqueKeys(info, remainBatchOut, isEos);
    if (isEos) {
        EosL1Que[info.name][info.channelId].Pushv(true);
        LOG_DEBUG("Enqueue on EosL1Que, eos status! table:{}, batchId:{}, channelId:{}, EosL1Que size: {}", info.name,
                  info.batchId, info.channelId, EosL1Que[info.name][info.channelId].Size());
    }
    if (uniqueKeys.empty()) {
        return remainBatchOut;
    }

    // 获取GlobalUnique对应的restoreVectorSec
    auto restoreVecSec = GetRestoreVecSec(info, remainBatchOut);
    if (restoreVecSec.empty()) {
        return remainBatchOut;
    }

    SendAll2AllVec(info, remainBatchOut);
    if (!remainBatchOut) {
        return remainBatchOut;
    }

    SendRestoreVec(info, remainBatchOut);
    if (!remainBatchOut) {
        return remainBatchOut;
    }

    std::pair<vector<uint64_t>, vector<uint64_t>> swapInKoPair;
    std::pair<vector<uint64_t>, vector<uint64_t>> swapOutKoPair;
    GetSwapPairsAndKey2Offset(info, uniqueKeys, swapInKoPair, swapOutKoPair);

    SendLookupOffsets(info, uniqueKeys, restoreVecSec);

    SendGlobalUniqueVec(info, uniqueKeys, restoreVecSec);

    TimeCost swapProcessTC;
    EnqueueSwapInfo(info, swapInKoPair, swapOutKoPair);

    auto& swapInPos = swapInKoPair.second;
    auto& swapOutPos = swapOutKoPair.second;
    SendTensorForSwap(info, swapInPos, swapOutPos);
    if (info.channelId == TRAIN_CHANNEL_ID) {
        alreadyTrainOnce = true;
    }

    LOG_DEBUG("ProcessEmbInfoDDR end, table:{}, channel:{}, batchId:{} swapProcessTC(ms):{} getAndSendTensorsTC(ms):{}",
              info.name, info.channelId, info.batchId, swapProcessTC.ElapsedMS(), getAndSendTensorsTC.ElapsedMS());
#endif
    return remainBatchOut;
}

/// hook通过时间或者step数触发淘汰
/// \return
bool HybridMgmt::Evict()
{
#ifndef GTEST
    std::lock_guard<std::mutex> lk(evictMut);
    if (!isInitialized) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before evict hook.");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
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
            vector<std::string> allTableNames;
            int retCode = embCache->GetEmbTableNames(allTableNames);
            if (retCode != H_OK) {
                LOG_ERROR("GetEmbTableNames failed!");
                return false;
            }
            for (const string& embName : allTableNames) {
                EvictKeys(embName, evictKeyMap[COMBINE_HISTORY_NAME]);
                EvictL3StorageKeys(embName, evictKeyMap[COMBINE_HISTORY_NAME]);
            }
        } else {
            for (const auto& evict : as_const(evictKeyMap)) {
                EvictKeys(evict.first, evict.second);
                EvictL3StorageKeys(evict.first, evict.second);
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
void HybridMgmt::EvictKeys(const string& embName, const vector<emb_cache_key_t>& keys)
{
    if (keys.empty()) {
        return;
    }
    int retCode = embCache->RemoveEmbsByKeys(embName, keys);
    if (retCode != H_OK) {
        LOG_ERROR("RemoveEmbsByKeys failed!");
        return;
    }
}

void HybridMgmt::EvictL3StorageKeys(const string& embName, const vector<emb_cache_key_t>& keys) const
{
    if (!isL3StorageEnabled) {
        return;
    }
    cacheManager->EvictL3StorageEmbedding(embName, keys);
}

/// 通过pyBind在python侧调用，通知hybridMgmt上层即将进行图的执行，需要进行唤醒
/// \param channelID 通道id
/// \param steps 运行的步数，由于可能存在循环下沉，所以1个session run 对应N步
void HybridMgmt::NotifyBySessionRun(int channelID) const
{
    if (!isInitialized) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before sess run.");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }

    hybridMgmtBlock->CheckAndNotifyWake(channelID);
}

/// 通过pyBind在python侧调用，通知hybridMgmt上层即将进行图的执行
/// \param channelID 通道id
/// \param steps 运行的步数，由于可能存在循环下沉，所以1个session run 对应N步
void HybridMgmt::CountStepBySessionRun(int channelID, int steps) const
{
    if (!isInitialized) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before sess run.");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }

    hybridMgmtBlock->CountPythonStep(channelID, steps);
}

/// 获取table表使用大小
/// \param embName 表名
/// \return 表使用大小
int64_t HybridMgmt::GetTableSize(const string& embName) const
{
    int64_t size = -1;
#ifndef GTEST
    if (!isInitialized) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before [table.size()].");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }

    if (mgmtRankInfo.useDynamicExpansion) {
        size = EmbeddingMgmt::Instance()->GetSize(embName);
        LOG_INFO(MGMT + "dynamic expansion mode, get emb:[{}] size:{}", embName, size);
        return size;
    }
    if (!mgmtRankInfo.isDDR) {
        size_t maxOffset = EmbeddingMgmt::Instance()->GetMaxOffset(embName);
        size = static_cast<int64_t>(maxOffset);
        LOG_INFO(MGMT + "HBM mode, get emb:[{}] size:{}", embName, size);
        return size;
    }
    int64_t l3StorageUsage = 0;
    if (isL3StorageEnabled) {
        l3StorageUsage = cacheManager->GetTableUsage(embName);
    }

    uint32_t ddrSize = embCache->GetUsage(embName);
    size = static_cast<int64_t>(ddrSize) + l3StorageUsage;
    LOG_INFO(MGMT + "DDR/L3Storage mode, get emb:[{}] size:{}", embName, size);
#endif
    return size;
}

/// 获取table表容量大小
/// \param embName 表名
/// \return 表容量大小
int64_t HybridMgmt::GetTableCapacity(const string& embName) const
{
#ifndef GTEST
    if (!isInitialized) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before [table.capacity()].");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }

    if (mgmtRankInfo.useDynamicExpansion) {
        int64_t capacity = EmbeddingMgmt::Instance()->GetCapacity(embName);
        LOG_INFO(MGMT + "dynamic expansion mode, get emb:[{}] capacity:{}", embName, capacity);
        return capacity;
    }
    LOG_WARN(MGMT + "no dynamic expansion mode, get emb:[{}] capacity failed", embName);
#endif
    return -1;
}

/// 设置表的优化器信息
/// \param embName 表名
/// \param optimInfo 优化器信息
/// \return
void HybridMgmt::SetOptimizerInfo(const string& embName, OptimizerInfo optimInfo) const
{
    if (!isInitialized) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::EXECUTION_ORDER_ERROR,
                           "HybridMgmt not initialized. Call [start_asc_pipeline] before [save/restore].");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    EmbeddingMgmt::Instance()->SetOptimizerInfo(embName, optimInfo);
}

// L3Storage
void HybridMgmt::LookUpAndRemoveAddrs(const EmbTaskInfo& info)
{
    uint64_t memSize = info.extEmbeddingSize * sizeof(float);
    const std::string hbmSwapKeyQueName = "HBMSwapKeyQue";
    const std::string ddrSwapKeyQueName = "DDRSwapKeyQue";
    auto lookUpFunc = [this, memSize, info](
                          std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]>& fromQue,
                          std::map<std::string, TaskQueue<std::vector<float*>>[MAX_CHANNEL_NUM]>& toQue,
                          const string& swapStr, const string& fromQueName) {
        std::vector<uint64_t> keys = fromQue[info.name + swapStr][info.channelId].WaitAndPop();
        if (!isRunning) {
            return;
        }
        std::vector<float*> addrs;
        TimeCost lookupAddrsTC;
        int rc = embCache->EmbeddingLookupAddrs(info.name, keys, addrs);
        if (rc != H_OK) {
            auto error =
                Error(ModuleName::M_OCK_CTR, ErrorType::UNKNOWN,
                      StringFormat("LookUpAddrs, error code: %d. table: %s, fromQue: %s, swapStr: %s, keys.size: %d, "
                                   "addrs.size: %d, lookUpSwapAddrsPushId: %d, channelId: %d.",
                                   rc, info.name, fromQueName, swapStr, keys.size(), addrs.size(),
                                   hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId], info.channelId));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
        if (&fromQue == &DDRSwapKeyQue && swapStr == SWAP_OUT_STR) {
            for (auto& addr : addrs) {
                auto* newAddr = (float*)malloc(memSize);
                rc = memcpy_s(newAddr, memSize, addr, memSize);
                if (rc != 0) {
                    auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::UNKNOWN,
                                       StringFormat("Memcpy_s failed when DDR swap out, error code: %d. MemSize: %d. "
                                                    "You can query the meaning of security function error code.",
                                                    rc, memSize));
                    LOG_ERROR(error.ToString());
                    throw runtime_error(error.ToString().c_str());
                }
                addr = newAddr;
            }
            rc = embCache->EmbeddingRemove(info.name, keys);
            if (rc != H_OK) {
                auto error = Error(
                    ModuleName::M_OCK_CTR, ErrorType::UNKNOWN,
                    StringFormat("Remove, error code: %d. table: %s, fromQue: %s, swapStr: %s, keys.size: %d, "
                                 "lookUpSwapAddrsPushId: %d, channelId: %d",
                                 rc, info.name, fromQueName, swapStr, keys.size(),
                                 hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId], info.channelId));
                LOG_ERROR(error.ToString());
                throw runtime_error(error.ToString().c_str());
            }
        }
        LOG_DEBUG("table:{}, fromQue:{}, swapStr:{}, keys.size:{}, addrs.size:{}, lookUpSwapAddrsPushId:{}, "
                  "channelId:{}, lookupAddrsTC(ms):{}",
                  info.name, fromQueName, swapStr, keys.size(), addrs.size(),
                  hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId], info.channelId,
                  lookupAddrsTC.ElapsedMS());
        toQue[info.name + swapStr][info.channelId].Pushv(addrs);
    };

    lookUpFunc(DDRSwapKeyQue, DDRSwapAddrsQue, SWAP_OUT_STR, ddrSwapKeyQueName);
    lookUpFunc(DDRSwapKeyQue, DDRSwapAddrsQue, SWAP_IN_STR, ddrSwapKeyQueName);
    lookUpFunc(HBMSwapKeyQue, HBMSwapAddrsQue, SWAP_IN_STR, hbmSwapKeyQueName);
    lookUpFunc(HBMSwapKeyQue, HBMSwapAddrsQue, SWAP_OUT_STR, hbmSwapKeyQueName);
    LOG_DEBUG("LookUpAndRemoveAddrs, table:{}, accumulate pushId:{}, lookUpSwapAddrsPushId:{}", info.name, info.batchId,
              hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId]);

    hybridMgmtBlock->lookUpSwapAddrsPushId[info.name][info.channelId]++;
}

// DDR
void HybridMgmt::LookUpSwapAddrs(const string& embName, int channelId)
{
    int id = 0;
    std::string swapInName = embName + SWAP_IN_STR;
    std::string swapOutName = embName + SWAP_OUT_STR;
    std::vector<float*> addrs;
    while (isRunning && lookupAddrSuccess) {
        bool isEos = EosL1Que[embName][channelId].WaitAndPop();
        if (!isRunning) {
            return;
        }
        EosL2Que[embName][channelId].Pushv(isEos);
        if (isEos) {
            LOG_DEBUG("Enqueue on EosL2Que, eos status! table:{}, batchId:{}, channelId:{}, EosL1Que size:{}, "
                      "EosL2Que.size: {}",
                      embName, hybridMgmtBlock->lookUpSwapAddrsPushId[embName][channelId], channelId,
                      EosL1Que[embName][channelId].Size(), EosL2Que[embName][channelId].Size());
            continue;
        }

        // swap in
        std::vector<uint64_t> keys = HBMSwapKeyQue[swapInName][channelId].WaitAndPop();
        if (!isRunning) {
            return;
        }
        TimeCost lookupAddrsInTC;
        int rc = embCache->EmbeddingLookupAddrs(embName, keys, addrs);
        if (rc != H_OK) {
            lookupAddrSuccess = false;
            auto error = Error(ModuleName::M_OCK_CTR, ErrorType::UNKNOWN,
                               StringFormat("LookUpAddrs swap in, error code: %d. embName: %s, keys.size: %d, "
                                            "addrs.size: %d, lookUpSwapAddrsPushId: %d, channelId: %d.",
                                            rc, embName, keys.size(), addrs.size(),
                                            hybridMgmtBlock->lookUpSwapAddrsPushId[embName][channelId], channelId));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
        LOG_DEBUG("table:{}, swapStr:{}, keys.size:{}, addrs.size:{}, lookUpSwapAddrsPushId:{}, channelId:{}, "
                  "lookupAddrsInTC(ms):{}",
                  embName, SWAP_IN_STR, keys.size(), addrs.size(),
                  hybridMgmtBlock->lookUpSwapAddrsPushId[embName][channelId], channelId, lookupAddrsInTC.ElapsedMS());
        HBMSwapAddrsQue[swapInName][channelId].Pushv(addrs);

        // swap out
        keys = HBMSwapKeyQue[swapOutName][channelId].WaitAndPop();
        TimeCost lookupAddrsOutTC;
        rc = embCache->EmbeddingLookupAddrs(embName, keys, addrs);
        if (!isRunning) {
            return;
        }
        if (rc != H_OK) {
            lookupAddrSuccess = false;
            auto error = Error(ModuleName::M_OCK_CTR, ErrorType::UNKNOWN,
                               StringFormat("LookUpAddrs swap out, error code: %d. embName: %s, keys.size: %d, "
                                            "addrs.size: %d, lookUpSwapAddrsPushId: %d, channelId: %d.",
                                            rc, embName, keys.size(), addrs.size(),
                                            hybridMgmtBlock->lookUpSwapAddrsPushId[embName][channelId], channelId));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
        LOG_DEBUG("table:{}, swapStr:{}, keys.size:{}, addrs.size:{}, lookUpSwapAddrsPushId:{}, channelId:{}, "
                  "lookupAddrsOutTC(ms):{}",
                  embName, SWAP_OUT_STR, keys.size(), addrs.size(),
                  hybridMgmtBlock->lookUpSwapAddrsPushId[embName][channelId], channelId, lookupAddrsOutTC.ElapsedMS());
        HBMSwapAddrsQue[swapOutName][channelId].Pushv(addrs);

        // statistic step
        LOG_DEBUG("LookUpSwapAddrs, table:{}, channelId:{}, accumulate pushId:{}, lookUpSwapAddrsPushId:{}", embName,
                  channelId, id, hybridMgmtBlock->lookUpSwapAddrsPushId[embName][channelId]);

        hybridMgmtBlock->lookUpSwapAddrsPushId[embName][channelId]++;
        id++;
    }
}

/// 导出npu的embedding
void HybridMgmt::FetchDeviceEmb()
{
    // 数据处理线程上锁
    KEY_PROCESS_INSTANCE->LoadSaveLock();

    if (mgmtRankInfo.isDDR) {
        // DDR模式保存host的emb表以及hashmap
        LOG_DEBUG(MGMT + "start host side save: ddr mode");
        for (const auto& embInfo : mgmtEmbInfo) {
            std::vector<std::pair<uint64_t, uint64_t>> koVec;
            embCache->ExportDeviceKeyOffsetPairs(embInfo.name, koVec);
            std::vector<uint64_t> swapOutPos;
            for (const auto& p : koVec) {
                swapOutPos.push_back(p.second);
            }

            vector<Tensor> swapTensor;
            swapTensor.emplace_back(Vec2TensorI32(swapOutPos));
            swapTensor.emplace_back(Tensor(tensorflow::DT_INT32, {1}));
            auto swapOutLen = swapTensor.back().flat<int32>();
            swapOutLen(0) = swapOutPos.size();
            LOG_DEBUG(MGMT + "save swapOutPos size:{}", swapOutPos.size());
            // 发送SwapOutPos信息
            hdTransfer->Send(TransferChannel::SAVE_H2D, swapTensor, TRAIN_CHANNEL_ID, embInfo.name);
        }
    }
    KEY_PROCESS_INSTANCE->LoadSaveUnlock();
}

// 这里就是新增的embedding处理线程
void HybridMgmt::EmbeddingTask()
{
    for (const auto& embInfo : mgmtEmbInfo) {
        hybridMgmtBlock->lastUpdateFinishStep[embInfo.name][TRAIN_CHANNEL_ID] = 0;
        hybridMgmtBlock->lastUpdateFinishStep[embInfo.name][EVAL_CHANNEL_ID] = 0;

        hybridMgmtBlock->lastLookUpFinishStep[embInfo.name][TRAIN_CHANNEL_ID] = 0;
        hybridMgmtBlock->lastLookUpFinishStep[embInfo.name][EVAL_CHANNEL_ID] = 0;

        hybridMgmtBlock->lastSendFinishStep[embInfo.name][TRAIN_CHANNEL_ID] = 0;
        hybridMgmtBlock->lastSendFinishStep[embInfo.name][EVAL_CHANNEL_ID] = 0;

        hybridMgmtBlock->lastRecvFinishStep[embInfo.name][TRAIN_CHANNEL_ID] = 0;
        hybridMgmtBlock->lastRecvFinishStep[embInfo.name][EVAL_CHANNEL_ID] = 0;

        hybridMgmtBlock->lookUpAndSendTableBatchId[embInfo.name][TRAIN_CHANNEL_ID] = 0;
        hybridMgmtBlock->lookUpAndSendTableBatchId[embInfo.name][EVAL_CHANNEL_ID] = 0;
        hybridMgmtBlock->receiveAndUpdateTableBatchId[embInfo.name][TRAIN_CHANNEL_ID] = 0;
        hybridMgmtBlock->receiveAndUpdateTableBatchId[embInfo.name][EVAL_CHANNEL_ID] = 0;
    }

    TimeCost embHDTransTC;
    MultiThreadEmbHDTransWrap();
    LOG_DEBUG("embHDTransTC(ms):{}", embHDTransTC.ElapsedMS());
}

void HybridMgmt::MultiThreadEmbHDTransWrap()
{
    for (int index = 0; index < EMBEDDING_THREAD_NUM; index++) {
        for (const auto& embInfo : mgmtEmbInfo) {
            CreateEmbeddingLookUpAndSendThread(index, embInfo, TRAIN_CHANNEL_ID);
            CreateEmbeddingReceiveAndUpdateThread(index, embInfo, TRAIN_CHANNEL_ID);

            CreateEmbeddingLookUpAndSendThread(index, embInfo, EVAL_CHANNEL_ID);
            CreateEmbeddingReceiveAndUpdateThread(index, embInfo, EVAL_CHANNEL_ID);
        }
    }
}

void HybridMgmt::ReceiveKey()
{
    for (const auto& embInfo : mgmtEmbInfo) {
        ReceiveKeyThread(embInfo);
    }
}

void HybridMgmt::ReceiveKeyThread(const EmbInfo& embInfo)
{
    receiveKeyThreads.emplace_back([embInfo, this]() {
        while (isRunning) {
            TransferChannel transferName = TransferChannel::KEY_D2H;
            size_t ret = hdTransfer->RecvOffsetsAcl(transferName, TRAIN_CHANNEL_ID, embInfo.name);
            if (ret == 0) {
                LOG_WARN("Receive empty data.");
                return;
            }
            LOG_INFO("Receive data success, get {} data size: {}.", embInfo.name, ret);
            auto aclData = acltdtGetDataItem(hdTransfer->aclDatasetsForIncrementalCkpt[embInfo.name], 0);
            if (aclData == nullptr) {
                auto error = Error(ModuleName::M_CHECK_POINT, ErrorType::ACL_ERROR,
                                   "Acl get tensor data failed in [ReceiveKeyThread].");
                LOG_ERROR(error.ToString());
                throw runtime_error(error.ToString());
            }

            auto ptr = static_cast<int64_t*>(acltdtGetDataAddrFromItem(aclData));
            if (ptr == nullptr || (ptr + 1) == nullptr) {
                auto error = Error(ModuleName::M_CHECK_POINT, ErrorType::NULL_PTR,
                                   "Failed to parse ACL passing data to timestamp and global step [ReceiveKeyThread].");
                LOG_ERROR(error.ToString());
                throw runtime_error(error.ToString());
            }
            auto timeStamp = *ptr;
            auto globalStep = *(ptr + 1);

            LOG_INFO("Receive {} timeStamp: {}, global step: {}.", embInfo.name, timeStamp, globalStep);
            // tensorflow获取的global step是从1开始的，但是在key process中batch
            // id则是从0开始，因此，下面的info中的batchId需要用 globalStep - 1
            EmbBaseInfo info = {.batchId = static_cast<int>(globalStep - 1),
                                .channelId = TRAIN_CHANNEL_ID,
                                .name = embInfo.name};
            unique_ptr<vector<Tensor>> keyCountVecInfo = KEY_PROCESS_INSTANCE->GetKCInfoVec(info);
            if (keyCountVecInfo == nullptr) {
                auto error = Error(ModuleName::M_CHECK_POINT, ErrorType::NOT_FOUND,
                                   "Get key count info vector is empty in [ReceiveKeyThread].");
                LOG_ERROR(error.ToString());
                throw runtime_error(error.ToString());
            }
            auto keyCountVecTmp = keyCountVecInfo->at(0).flat<int64>();
            vector<int64_t> keyCountVec;
            int64 keyCountSize = keyCountVecTmp.size();
            keyCountVec.reserve(keyCountSize);
            for (int64 i = 0; i < keyCountSize; ++i) {
                keyCountVec.push_back(static_cast<int64_t>(keyCountVecTmp(i)));
            }
            LOG_INFO("Emb table: {}, channel: {}, size is: {}, data: {}", embInfo.name, TRAIN_CHANNEL_ID, keyCountSize,
                     VectorToString(keyCountVec));

            // 更新delta表
            std::lock_guard<std::mutex> lock(keyCountUpdateMtx);
            UpdateDeltaInfo(embInfo.name, keyCountVec, timeStamp, globalStep);
            keyBatchIdMap[embInfo.name]++;
            keyCountUpdateCv.notify_all();
        }
    });
}

void HybridMgmt::UpdateDeltaInfo(const string& embName, vector<int64_t>& keyCountVec, int64_t timeStamp,
                                 int64_t batchId)
{
    auto keyCountSize = keyCountVec.size();
    auto& embMap = deltaMap[embName];
    for (int i = 0; i < keyCountSize; i += KEY_COUNT_ELEMENT_NUM) {
        emb_key_t key = keyCountVec[i];
        int64_t recentCount = keyCountVec[i + 1];
        KeyInfo& keyInfo = embMap[key];
        keyInfo.totalCount += recentCount;
        keyInfo.recentCount += recentCount;
        keyInfo.isChanged = true;
        keyInfo.batchID = batchId;
        keyInfo.lastUseTime = timeStamp;
    }
    LOG_INFO("Batch id: {}, delta map size: {}, emb {} size: {}", batchId, deltaMap.size(), embName,
             deltaMap[embName].size());
}

void HybridMgmt::ResetDeltaInfo()
{
    for (const auto& embInfo : mgmtEmbInfo) {
        auto& embKeyCountInfo = deltaMap[embInfo.name];
        for (auto& it : embKeyCountInfo) {
            it.second.recentCount = 0;
            it.second.isChanged = false;
        }
    }
}

void HybridMgmt::EmbeddingLookUpAndSendDDR(int batchId, int index, const EmbInfo& embInfo, int channelId)
{
    int cvNotifyIndex = 0;
    if (index + 1 != EMBEDDING_THREAD_NUM) {
        cvNotifyIndex = index + 1;
    }

    EmbTaskInfo info = {.batchId = batchId,
                        .threadIdx = index,
                        .cvNotifyIndex = cvNotifyIndex,
                        .extEmbeddingSize = embInfo.extEmbeddingSize,
                        .channelId = channelId,
                        .name = embInfo.name};

    vector<Tensor> h2dEmb;
    auto isSuccess = EmbeddingLookUpDDR(info, h2dEmb);
    if (!isSuccess) {
        LOG_DEBUG("HybridMgmt is not running when [LookUpAndSendDDR], table:{}, batchId:{}, channel:{}", embInfo.name,
                  batchId, channelId);
        return;
    }
    EmbeddingSendDDR(info, h2dEmb);
}

void HybridMgmt::EmbeddingReceiveAndUpdateDDR(int batchId, int index, const EmbInfo& embInfo, int channelId)
{
    int cvNotifyIndex = 0;
    if (index + 1 != EMBEDDING_THREAD_NUM) {
        cvNotifyIndex = index + 1;
    }

    EmbTaskInfo info = {.batchId = batchId,
                        .threadIdx = index,
                        .cvNotifyIndex = cvNotifyIndex,
                        .extEmbeddingSize = embInfo.extEmbeddingSize,
                        .channelId = channelId,
                        .name = embInfo.name};

    float* ptr = nullptr;
    vector<float*> swapOutAddrs;
    auto isSuccess = EmbeddingReceiveDDR(info, ptr, swapOutAddrs);
    if (!isSuccess) {
        LOG_DEBUG("HybridMgmt is not running or receive empty data when [EmbeddingReceiveDDR], table:{}, batchId:{}, "
                  "channel:{}",
                  embInfo.name, batchId, channelId);
        return;
    }
    EmbeddingUpdateDDR(info, ptr, swapOutAddrs);
}

void HybridMgmt::EmbeddingLookUpAndSendL3Storage(int batchId, int index, const EmbInfo& embInfo, int channelId)
{
    int cvNotifyIndex = 0;
    if (index + 1 != EMBEDDING_THREAD_NUM) {
        cvNotifyIndex = index + 1;
    }

    EmbTaskInfo info = {.batchId = batchId,
                        .threadIdx = index,
                        .cvNotifyIndex = cvNotifyIndex,
                        .extEmbeddingSize = embInfo.extEmbeddingSize,
                        .channelId = channelId,
                        .name = embInfo.name};
    vector<Tensor> h2dEmb;

    auto isSuccess = EmbeddingLookUpL3Storage(info, h2dEmb);
    if (!isSuccess) {
        LOG_DEBUG("HybridMgmt is not running when [LookUpAndSendL3Storage], table:{}, batchId:{}, channel:{}",
                  embInfo.name, batchId, channelId);
        return;
    }

    EmbeddingSendL3Storage(info, h2dEmb);
}

void HybridMgmt::EmbeddingReceiveAndUpdateL3Storage(int batchId, int index, const EmbInfo& embInfo, int channelId)
{
    int cvNotifyIndex = 0;
    if (index + 1 != EMBEDDING_THREAD_NUM) {
        cvNotifyIndex = index + 1;
    }

    EmbTaskInfo info = {.batchId = batchId,
                        .threadIdx = index,
                        .cvNotifyIndex = cvNotifyIndex,
                        .extEmbeddingSize = embInfo.extEmbeddingSize,
                        .channelId = channelId,
                        .name = embInfo.name};

    float* ptr = nullptr;
    vector<float*> swapOutAddrs;
    int64_t dims0 = 0;

    auto isSuccess = EmbeddingReceiveL3Storage(info, ptr, swapOutAddrs, dims0);
    if (!isSuccess) {
        LOG_DEBUG(
            "HybridMgmt is not running or receive empty data when [EmbeddingReceiveL3Storage], table:{}, batchId:{}, "
            "channel:{}",
            embInfo.name, batchId, channelId);
        return;
    }
    EmbeddingUpdateL3Storage(info, ptr, swapOutAddrs, dims0);
}

/// 构造训练所需的各种向量数据
/// \param info 表名、已处理的batch数、通道索引（训练/推理）
/// \return remainBatchOut 是否从通道获取了数据
bool HybridMgmt::ProcessEmbInfoL3Storage(const EmbBaseInfo& info)
{
    bool remainBatchOut = true;
#ifndef GTEST
    TimeCost getAndSendTensorsTC;
    LOG_DEBUG("ProcessEmbInfoL3Storage table:{}, channel:{}, batchId:{}", info.name, info.channelId, info.batchId);

    // 只有在每次GetUniqueKeys的时候才知道上游是否已经EOS
    // 注意GetUniqueKeys与EOS关联，需要在ProcessEmbInfoL3Storage最先调用，如需调整位置，请参考并适配其他函数
    // 获取GlobalUnique向量
    bool isEos = false;
    auto uniqueKeys = GetUniqueKeys(info, remainBatchOut, isEos);
    if (isEos) {
        EosL1Que[info.name][info.channelId].Pushv(true);
        LOG_DEBUG("Enqueue on EosL1Que L3Storage, eos status! table:{}, batchId:{}, channelId:{}", info.name,
                  info.batchId, info.channelId);
    }

    if (uniqueKeys.empty()) {
        return remainBatchOut;
    }

    // 获取GlobalUnique对应的restoreVectorSec
    auto restoreVecSec = GetRestoreVecSec(info, remainBatchOut);
    if (restoreVecSec.empty()) {
        return remainBatchOut;
    }

    SendAll2AllVec(info, remainBatchOut);
    if (!remainBatchOut) {
        return remainBatchOut;
    }

    SendRestoreVec(info, remainBatchOut);
    if (!remainBatchOut) {
        return remainBatchOut;
    }

    std::pair<vector<uint64_t>, vector<uint64_t>> swapInKoPair;
    std::pair<vector<uint64_t>, vector<uint64_t>> swapOutKoPair;
    GetSwapPairsAndKey2Offset(info, uniqueKeys, swapInKoPair, swapOutKoPair);

    SendLookupOffsets(info, uniqueKeys, restoreVecSec);

    SendGlobalUniqueVec(info, uniqueKeys, restoreVecSec);

    TimeCost swapProcessTC;
    auto& swapInKeys = swapInKoPair.first;
    auto& swapInPos = swapInKoPair.second;
    auto& swapOutKeys = swapOutKoPair.first;
    auto& swapOutPos = swapOutKoPair.second;

    HandleDataSwapForL3Storage(info, swapInKeys, swapOutKeys);

    SendTensorForSwap(info, swapInPos, swapOutPos);

    if (info.channelId == TRAIN_CHANNEL_ID) {
        alreadyTrainOnce = true;
    }

    LOG_DEBUG("ProcessEmbInfoL3Storage end, table:{}, batchId:{}, swapProcessTC(ms):{}, getAndSendTensorsTC(ms):{}",
              info.name, info.batchId, swapProcessTC.ElapsedMS(), getAndSendTensorsTC.ElapsedMS());
#endif
    return remainBatchOut;
}

void HybridMgmt::SendTensorForSwap(const EmbBaseInfo& info, const vector<uint64_t>& swapInPosUint,
                                   const vector<uint64_t>& swapOutPosUint)
{
#ifndef GTEST
    vector<Tensor> swapTensor;
    swapTensor.emplace_back(Vec2TensorI32(swapInPosUint));
    swapTensor.emplace_back(Vec2TensorI32(swapOutPosUint));
    swapTensor.emplace_back(Tensor(tensorflow::DT_INT32, {1}));
    auto swapInLen = swapTensor.back().flat<int32>();
    swapInLen(0) = swapInPosUint.size();
    swapTensor.emplace_back(Tensor(tensorflow::DT_INT32, {1}));
    auto swapOutLen = swapTensor.back().flat<int32>();
    swapOutLen(0) = swapOutPosUint.size();

    hdTransfer->Send(TransferChannel::SWAP, swapTensor, info.channelId, info.name, info.batchId);
#endif
}

void HybridMgmt::InitDataPipelineForDDR(const string& embName)
{
    // 初始化公共队列
    HBMSwapKeyQue[embName + SWAP_IN_STR];
    HBMSwapKeyQue[embName + SWAP_OUT_STR];
    HBMSwapAddrsQue[embName + SWAP_IN_STR];
    HBMSwapAddrsQue[embName + SWAP_OUT_STR];

    EosL1Que[embName];
    EosL2Que[embName];
    // 初始化lookup线程
    hybridMgmtBlock->lookUpSwapAddrsPushId[embName][TRAIN_CHANNEL_ID] = 0;  // 此处初始化，避免多线程竞争导致计数错误
    hybridMgmtBlock->lookUpSwapAddrsPushId[embName][EVAL_CHANNEL_ID] = 0;

    // train and eval
    lookUpSwapAddrsThreads.emplace_back(
        std::async(std::launch::async, [=] { LookUpSwapAddrs(embName, TRAIN_CHANNEL_ID); }));
    lookUpSwapAddrsThreads.emplace_back(
        std::async(std::launch::async, [=] { LookUpSwapAddrs(embName, EVAL_CHANNEL_ID); }));

    LOG_DEBUG("data pipeline for ddr init");
}

void HybridMgmt::InitDataPipelineForL3Storage(const string& embName, int extEmbeddingSize)
{
    // 初始化公共队列
    HBMSwapKeyQue[embName + SWAP_IN_STR];
    HBMSwapKeyQue[embName + SWAP_OUT_STR];
    HBMSwapAddrsQue[embName + SWAP_IN_STR];
    HBMSwapAddrsQue[embName + SWAP_OUT_STR];

    EosL1Que[embName];
    EosL2Que[embName];

    HBMSwapKeyQue[embName + ADDR_STR];
    HBMSwapKeyForL3StorageQue[embName + SWAP_IN_STR];
    HBMSwapKeyForL3StorageQue[embName + ADDR_STR];
    HBMSwapKeyForL3StorageQue[embName + SWAP_OUT_STR];

    DDRSwapKeyQue[embName + SWAP_OUT_STR];
    DDRSwapKeyQue[embName + SWAP_IN_STR];
    DDRSwapKeyForL3StorageQue[embName + SWAP_OUT_STR];
    DDRSwapKeyForL3StorageQue[embName + SWAP_IN_STR];
    DDRSwapAddrsQue[embName + SWAP_OUT_STR];
    DDRSwapAddrsQue[embName + SWAP_IN_STR];

    // 初始化lookup线程
    LOG_DEBUG("data pipeline for L3Storage init");
}

void HybridMgmt::InitEmbeddingCache(const vector<EmbInfo>& embInfos)
{
    factory->SetExternalLogFuncInner(CTRLog);
    factory->CreateEmbCacheManager(embCache);
    EmbeddingMgmt::Instance()->SetEmbCacheForEmbTable(embCache);
    EmbeddingMgmt::Instance()->SetHDTransferForEmbTable(hdTransfer);

    for (auto embInfo : embInfos) {
        // Init mutex and condition_variable.
        InitPipelineMutexAndCV(embInfo.name);
        if (isL3StorageEnabled) {
            InitDataPipelineForL3Storage(embInfo.name, embInfo.extEmbeddingSize);
        } else {
            InitDataPipelineForDDR(embInfo.name);
        }

        // 初始化embedding cache
        LOG_INFO("create cache for table:{}, hostVocabSize:{}, extEmbeddingSize:{}, maxCacheSize(devVocabSize):{}",
                 embInfo.name, embInfo.hostVocabSize, embInfo.extEmbeddingSize, embInfo.devVocabSize);
        EmbCache::EmbCacheInfo embCacheInfo(embInfo.name, embInfo.hostVocabSize, embInfo.embeddingSize,
                                            embInfo.extEmbeddingSize, embInfo.devVocabSize);
        size_t prefill = std::max(embInfo.hostVocabSize / HOST_TO_PREFILL_RATIO, embInfo.devVocabSize);
        int ret = embCache->CreateCacheForTable(embCacheInfo, embInfo.initializeInfos, INVALID_KEY_VALUE, prefill,
                                                EMBEDDING_THREAD_NUM);
        if (ret != H_OK) {
            auto error = Error(ModuleName::M_OCK_CTR, ErrorType::CONSTRUCT_ERROR,
                               StringFormat("Create cache for table %s failed, error code: %d", embInfo.name, ret));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
    }
}

void HybridMgmt::JoinEmbeddingCacheThread()
{
    for (int channelId = 0; channelId < MAX_CHANNEL_NUM; channelId++) {
        // Let ReceiveAndUpdate & LookupAndSend thread stop.
        for (const auto& embInfo : mgmtEmbInfo) {
            for (int index = 0; index < EMBEDDING_THREAD_NUM; index++) {
                string key = MakeSwapCVName(index, embInfo.name, channelId);
                lastUpdateFinishCV[key].notify_all();
                lastLookUpFinishCV[key].notify_all();
                lastSendFinishCV[key].notify_all();
                lastRecvFinishCV[key].notify_all();
            }
        }

        for (auto& p : EosL1Que) {
            p.second[channelId].DestroyQueue();
        }
        for (auto& p : EosL2Que) {
            p.second[channelId].DestroyQueue();
        }
        for (auto& p : HBMSwapAddrsQue) {
            p.second[channelId].DestroyQueue();
        }
        for (auto& p : HBMSwapKeyQue) {
            p.second[channelId].DestroyQueue();
        }
        for (auto& p : HBMSwapKeyForL3StorageQue) {
            p.second[channelId].DestroyQueue();
        }
        for (auto& p : DDRSwapKeyQue) {
            p.second[channelId].DestroyQueue();
        }
        for (auto& p : DDRSwapKeyForL3StorageQue) {
            p.second[channelId].DestroyQueue();
        }
        for (auto& p : DDRSwapAddrsQue) {
            p.second[channelId].DestroyQueue();
        }
    }

    for (auto& t : EmbeddingLookUpAndSendThreadPool) {
        t.join();
    }
    for (auto& t : EmbeddingReceiveAndUpdateThreadPool) {
        t.join();
    }
    for (auto& t : lookUpSwapAddrsThreads) {
        t.wait();
    }
}

bool HybridMgmt::EmbeddingReceiveDDR(const EmbTaskInfo& info, float*& ptr, vector<float*>& swapOutAddrs)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastRecvFinishLocker(lastRecvFinishMutex[currentKey]);
    lastRecvFinishCV[currentKey].wait(lastRecvFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastRecvFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });
    if (!isRunning) {
        return false;
    }
    bool isEos = EosL2Que[info.name][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    if (isEos) {
        LOG_DEBUG("EmbeddingReceiveDDR get eos, table:{}, accumulate batchId:{}, channel: {}", info.name, info.batchId,
                  info.channelId);
        // It cannot return here after send eos, otherwise it will block the next round of switching.
        KEY_PROCESS_INSTANCE->SendEos(info.name, info.batchId, info.channelId);
        // Once eos is sent, it will be blocked in [EosL2Que WaitAndPop]. For train mode, it will be finished, but for
        // eval mode, it will be waked when normal data comes in next turn.
        isEos = EosL2Que[info.name][info.channelId].WaitAndPop();
        if (!isRunning) {
            return false;
        }
    }

    TimeCost EmbeddingRecvTC = TimeCost();

    swapOutAddrs = HBMSwapAddrsQue[info.name + SWAP_OUT_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    // 等待图执行发送d2h embedding过来

    // 区分通道接收
    auto size = hdTransfer->RecvAcl(TransferChannel::D2H, info.channelId, info.name, info.threadIdx, info.batchId);
    if (size == 0) {
        LOG_WARN(HOSTEMB + "recv empty data");
        return false;
    }

    auto aclData = acltdtGetDataItem(hdTransfer->aclDatasets[info.name][info.threadIdx], 0);
    if (aclData == nullptr) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::ACL_ERROR,
                           "Acl get tensor data from dataset failed in [EmbeddingReceiveDDR].");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    ptr = reinterpret_cast<float*>(acltdtGetDataAddrFromItem(aclData));

    // 判断拿到的embedding个数是否与swapOutKeys个数相等
    size_t dimNum = acltdtGetDimNumFromItem(aclData);
    int64_t dims[dimNum];
    acltdtGetDimsFromItem(aclData, dims, dimNum);

    LOG_DEBUG(MGMT + "In swap thread, finish receive d2h embedding, table:{}, channelId:{}, accumulate batchId:{}, "
                     "thread:{}, dims[0]:{}, swapOutAddrs size:{}, EmbeddingRecvTC(ms):{}",
              info.name, info.channelId, info.batchId, info.threadIdx, dims[0], swapOutAddrs.size(),
              EmbeddingRecvTC.ElapsedMS());

    if (dims[0] != static_cast<int64_t>(swapOutAddrs.size())) {
        auto error =
            Error(ModuleName::M_HYBRID_MGMT, ErrorType::LOGIC_ERROR,
                  StringFormat(
                      "Receive swap-out emb num %d does not equal to swap-out addrs num %d in [EmbeddingReceiveDDR].",
                      dims[0], swapOutAddrs.size()));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    hybridMgmtBlock->lastRecvFinishStep[info.name][info.channelId]++;

    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastRecvFinishCV[nextKey].notify_all();

    return true;
}

void HybridMgmt::EmbeddingUpdateDDR(const EmbTaskInfo& info, const float* embPtr, vector<float*>& swapOutAddrs)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastUpdateFinishLocker(lastUpdateFinishMutex[currentKey]);
    lastUpdateFinishCV[currentKey].wait(lastUpdateFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastUpdateFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });
    TimeCost EmbeddingUpdateTC = TimeCost();

    uint64_t memSize = info.extEmbeddingSize * sizeof(float);
    uint64_t extEmbeddingSize = info.extEmbeddingSize;
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) \
    shared(swapOutAddrs, embPtr, extEmbeddingSize, memSize)
    for (uint64_t i = 0; i < swapOutAddrs.size(); i++) {
        auto rc = memcpy_s(swapOutAddrs[i], memSize, embPtr + i * extEmbeddingSize, memSize);
        if (rc != 0) {
            auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::UNKNOWN,
                               StringFormat("Memcpy_s failed when emb update ddr, error code: %d. MemSize: %d. You can "
                                            "query the meaning of security function error code.",
                                            rc, memSize));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
    }
    if (MxRec::Logger::GetLevel() <= MxRec::Logger::DEBUG) {
        string sample;
        if (!swapOutAddrs.empty()) {
            sample = FloatPtrToLimitStr(swapOutAddrs.front(), info.extEmbeddingSize);  // print first element
        }
        LOG_DEBUG(MGMT + "In swap thread, finish update d2h embedding, table:{}, channelId:{}, accumulate batchId:{}, "
                         "thread:{}, ext emb:{}, emb size:{}, emb samples:{}, EmbeddingUpdateTC(ms):{}",
                  info.name, info.channelId, info.batchId, info.threadIdx, info.extEmbeddingSize, swapOutAddrs.size(),
                  sample, EmbeddingUpdateTC.ElapsedMS());
    }

    hybridMgmtBlock->lastUpdateFinishStep[info.name][info.channelId]++;
    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastUpdateFinishCV[nextKey].notify_all();
}

bool HybridMgmt::EmbeddingLookUpDDR(const EmbTaskInfo& info, vector<Tensor>& h2dEmb)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastUpdateFinishLocker(lastUpdateFinishMutex[currentKey]);
    lastUpdateFinishCV[currentKey].wait(lastUpdateFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastUpdateFinishStep[info.name][info.channelId] >= info.batchId) || mutexDestroy;
    });
    if (!isRunning) {
        return false;
    }

    std::unique_lock<std::mutex> lastLookUpFinishLocker(lastLookUpFinishMutex[currentKey]);
    lastLookUpFinishCV[currentKey].wait(lastLookUpFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastLookUpFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });
    if (!isRunning) {
        return false;
    }

    bool isSuccess = BuildH2DEmbedding(info, h2dEmb);
    if (!isSuccess) {
        return false;
    }

    hybridMgmtBlock->lastLookUpFinishStep[info.name][info.channelId]++;
    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastLookUpFinishCV[nextKey].notify_all();

    LOG_DEBUG(MGMT + "In swap thread, finish embedding lookup, table:{}, channelId:{}, accumulate batchId:{}, "
                     "thread:{}",
              info.name, info.channelId, info.batchId, info.threadIdx);
    return true;
}

void HybridMgmt::EmbeddingSendDDR(const EmbTaskInfo& info, vector<Tensor>& h2dEmb)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastSendFinishLocker(lastSendFinishMutex[currentKey]);
    lastSendFinishCV[currentKey].wait(lastSendFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastSendFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });
    TimeCost SendTC = TimeCost();
    // 区分通道发送
    hdTransfer->Send(TransferChannel::H2D, h2dEmb, info.channelId, info.name, info.batchId);
    hybridMgmtBlock->lastSendFinishStep[info.name][info.channelId]++;
    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastSendFinishCV[nextKey].notify_all();

    LOG_DEBUG(MGMT + "In swap thread, finish send h2d embedding, table:{}, channelId:{}, batchId:{}, accumulate "
                     "batchId:{}, thread:{}, SendH2DEmbTC(ms):{}",
              info.name, info.channelId, hybridMgmtBlock->h2dSendBatchId[info.name][info.channelId], info.batchId,
              info.threadIdx, SendTC.ElapsedMS());
    hybridMgmtBlock->h2dSendBatchId[info.name][info.channelId]++;
}

void HybridMgmt::CreateEmbeddingLookUpAndSendThread(int index, const EmbInfo& embInfo, int channelId)
{
    auto fn = [index, embInfo, channelId, this]() {
        LOG_DEBUG(MGMT + "Create LookUpAndSendThread, table:{}, index:{}, channel:{}", embInfo.name, index, channelId);
        while (true) {
            lookUpAndSendBatchIdMtx[channelId].lock();
            if (hybridMgmtBlock->lookUpAndSendTableBatchId[embInfo.name][channelId] % EMBEDDING_THREAD_NUM == index) {
                int curBatchId = hybridMgmtBlock->lookUpAndSendTableBatchId[embInfo.name][channelId];
                hybridMgmtBlock->lookUpAndSendTableBatchId[embInfo.name][channelId]++;
                lookUpAndSendBatchIdMtx[channelId].unlock();
                if (!isL3StorageEnabled) {
                    EmbeddingLookUpAndSendDDR(curBatchId, index, embInfo, channelId);
                } else {
                    EmbeddingLookUpAndSendL3Storage(curBatchId, index, embInfo, channelId);
                }
            } else {
                lookUpAndSendBatchIdMtx[channelId].unlock();
            }
            if (!isRunning) {
                LOG_DEBUG(MGMT + "Destroy LookUpAndSendThread, table:{}, index:{}, channel:{}, batchId:{}",
                          embInfo.name, index, channelId,
                          hybridMgmtBlock->receiveAndUpdateTableBatchId[embInfo.name][channelId]);
                return;
            }
        }
    };
    EmbeddingLookUpAndSendThreadPool.emplace_back(fn);
}

void HybridMgmt::CreateEmbeddingReceiveAndUpdateThread(int index, const EmbInfo& embInfo, int channelId)
{
    auto fn = [index, embInfo, channelId, this]() {
        LOG_DEBUG(MGMT + "Create ReceiveAndUpdateThread, table:{}, index:{}, channel:{}", embInfo.name, index,
                  channelId);
        while (true) {
            receiveAndUpdateBatchIdMtx[channelId].lock();
            if (hybridMgmtBlock->receiveAndUpdateTableBatchId[embInfo.name][channelId] % EMBEDDING_THREAD_NUM ==
                index) {
                int curBatchId = hybridMgmtBlock->receiveAndUpdateTableBatchId[embInfo.name][channelId];
                hybridMgmtBlock->receiveAndUpdateTableBatchId[embInfo.name][channelId]++;
                receiveAndUpdateBatchIdMtx[channelId].unlock();
                if (!isL3StorageEnabled) {
                    EmbeddingReceiveAndUpdateDDR(curBatchId, index, embInfo, channelId);
                } else {
                    EmbeddingReceiveAndUpdateL3Storage(curBatchId, index, embInfo, channelId);
                }
            } else {
                receiveAndUpdateBatchIdMtx[channelId].unlock();
            }
            if (!isRunning) {
                LOG_DEBUG(MGMT + "Destroy ReceiveAndUpdateThread, table:{}, index:{}, channel:{}, batchId:{}",
                          embInfo.name, index, channelId,
                          hybridMgmtBlock->receiveAndUpdateTableBatchId[embInfo.name][channelId]);
                return;
            }
        }
    };
    EmbeddingReceiveAndUpdateThreadPool.emplace_back(fn);
}

bool HybridMgmt::EmbeddingReceiveL3Storage(const EmbTaskInfo& info, float*& ptr, vector<float*>& swapOutAddrs,
                                           int64_t& dims0)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastRecvFinishLocker(lastRecvFinishMutex[currentKey]);
    lastRecvFinishCV[currentKey].wait(lastRecvFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastRecvFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });
    if (!isRunning) {
        return false;
    }
    bool isEos = EosL1Que[info.name][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    if (isEos) {
        LOG_DEBUG("EmbeddingReceiveL3Storage get eos, table:{}, accumulate batchId:{}, channel: {}", info.name,
                  info.batchId, info.channelId);
        // It cannot return here after send eos, otherwise it will block the next round of switching.
        KEY_PROCESS_INSTANCE->SendEos(info.name, info.batchId, info.channelId);
        // Once eos is sent, it will be blocked in [EosL2Que WaitAndPop]. For train mode, it will be finished, but for
        // eval mode, it will be waked when normal data comes in next turn.
        isEos = EosL1Que[info.name][info.channelId].WaitAndPop();
        if (!isRunning) {
            return false;
        }
    }

    // DDR swap out key need to be removed
    LookUpAndRemoveAddrs(info);

    TimeCost EmbeddingRecvTC = TimeCost();
    // finish时会pop空vector，因此需要额外判定isRunning
    swapOutAddrs = HBMSwapAddrsQue[info.name + SWAP_OUT_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    // 等待图执行发送d2h embedding过来
    // 区分通道接收
    auto size = hdTransfer->RecvAcl(TransferChannel::D2H, info.channelId, info.name, info.threadIdx, info.batchId);
    if (size == 0) {
        LOG_WARN(HOSTEMB + "recv empty data");
        return false;
    }

    auto aclData = acltdtGetDataItem(hdTransfer->aclDatasets[info.name][info.threadIdx], 0);
    if (aclData == nullptr) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::ACL_ERROR,
                           "Acl get tensor data from dataset failed in [EmbeddingReceiveL3Storage].");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    ptr = reinterpret_cast<float*>(acltdtGetDataAddrFromItem(aclData));

    // 判断拿到的embedding个数是否与swapOutKeys个数相等
    size_t dimNum = acltdtGetDimNumFromItem(aclData);
    int64_t dims[dimNum];
    acltdtGetDimsFromItem(aclData, dims, dimNum);
    dims0 = dims[0];

    LOG_DEBUG(MGMT + "In swap thread, finish receive d2h embedding, table:{}, channelId:{}, accumulate batchId:{}, "
                     "thread:{}, dims[0]:{}, swapOutAddrs size:{}, EmbeddingRecvTC(ms):{}",
              info.name, info.channelId, info.batchId, info.threadIdx, dims[0], swapOutAddrs.size(),
              EmbeddingRecvTC.ElapsedMS());
    hybridMgmtBlock->lastRecvFinishStep[info.name][info.channelId]++;

    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastRecvFinishCV[nextKey].notify_all();
    return true;
}

void HybridMgmt::EmbeddingUpdateL3Storage(const EmbTaskInfo& info, float* embPtr, vector<float*>& swapOutAddrs,
                                          int64_t& dims0)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastUpdateFinishLocker(lastUpdateFinishMutex[currentKey]);
    lastUpdateFinishCV[currentKey].wait(lastUpdateFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastUpdateFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });

    TimeCost EmbeddingUpdateTC = TimeCost();
    std::vector<uint64_t> swapOutDDRAddrOffs = HBMSwapKeyQue[info.name + ADDR_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return;
    }
    uint64_t memSize = info.extEmbeddingSize * sizeof(float);
    uint64_t extEmbeddingSize = info.extEmbeddingSize;
    // DDR更新
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) \
    shared(swapOutAddrs, swapOutDDRAddrOffs, embPtr, extEmbeddingSize, memSize)
    for (uint64_t i = 0; i < swapOutAddrs.size(); i++) {
        auto rc = memcpy_s(swapOutAddrs[i], memSize, embPtr + swapOutDDRAddrOffs[i] * extEmbeddingSize, memSize);
        if (rc != 0) {
            auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::UNKNOWN,
                               StringFormat("Memcpy_s failed when emb update L3Storage, error code: %d. MemSize: %d. "
                                            "You can query the meaning of security function error code.",
                                            rc, memSize));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
    }

    LOG_DEBUG(MGMT + "In swap thread, finish update d2h DDR embedding, table:{}, channelId:{}, accumulate batchId:{}, "
                     "thread:{}, EmbeddingUpdateTC(ms):{}",
              info.name, info.channelId, info.batchId, info.threadIdx, EmbeddingUpdateTC.ElapsedMS());
    // L3Storage更新
    TimeCost L3StorageUpdateTC = TimeCost();
    std::vector<uint64_t> swapOutL3StorageAddrOffs =
        HBMSwapKeyForL3StorageQue[info.name + ADDR_STR][info.channelId].WaitAndPop();
    std::vector<uint64_t> swapOutL3StorageKeys =
        HBMSwapKeyForL3StorageQue[info.name + SWAP_OUT_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return;
    }

    if (dims0 != static_cast<int64_t>(swapOutAddrs.size() + swapOutL3StorageKeys.size())) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::LOGIC_ERROR,
                           StringFormat("Receive swap-out emb num %d does not equal to addrs num %d for swap-out ddr "
                                        "and %d for swap-out L3Storage in [EmbeddingUpdateL3Storage].",
                                        dims0, swapOutAddrs.size(), swapOutL3StorageKeys.size()));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    cacheManager->UpdateL3StorageEmb(info.name, embPtr, extEmbeddingSize, swapOutL3StorageKeys,
                                     swapOutL3StorageAddrOffs);

    LOG_DEBUG(
        MGMT + "In swap thread, finish update d2h L3Storage embedding, table:{}, channelId:{}, accumulate batchId:{}, "
               "thread:{}, L3StorageUpdateTC(ms):{}",
        info.name, info.channelId, info.batchId, info.threadIdx, L3StorageUpdateTC.ElapsedMS());

    hybridMgmtBlock->lastUpdateFinishStep[info.name][info.channelId]++;
    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastUpdateFinishCV[nextKey].notify_all();
}

bool HybridMgmt::EmbeddingLookUpL3Storage(const EmbTaskInfo& info, vector<Tensor>& h2dEmb)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastUpdateFinishLocker(lastUpdateFinishMutex[currentKey]);
    lastUpdateFinishCV[currentKey].wait(lastUpdateFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastUpdateFinishStep[info.name][info.channelId] >= info.batchId) || mutexDestroy;
    });
    if (!isRunning) {
        return false;
    }

    std::unique_lock<std::mutex> lastLookUpFinishLocker(lastLookUpFinishMutex[currentKey]);
    lastLookUpFinishCV[currentKey].wait(lastLookUpFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastLookUpFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });
    if (!isRunning) {
        return false;
    }

    TimeCost transferDDR2L3StorageTC = TimeCost();
    // DDR腾空间
    std::vector<uint64_t> DDR2L3StorageKeys =
        DDRSwapKeyForL3StorageQue[info.name + SWAP_OUT_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    std::vector<float*> DDR2L3StorageAddrs = DDRSwapAddrsQue[info.name + SWAP_OUT_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    cacheManager->TransferDDR2L3Storage(info.name, info.extEmbeddingSize, DDR2L3StorageKeys, DDR2L3StorageAddrs);
    LOG_DEBUG("table:{}, accumulate batchId:{}, channelId:{}, thread:{}, transferDDR2L3StorageTC(ms):{}",
              info.name.c_str(), info.batchId, info.channelId, info.threadIdx, transferDDR2L3StorageTC.ElapsedMS());

    TimeCost fetchL3StorageEmb2DDRTC = TimeCost();
    // swapInKeys中在L3Storage的挪到DDR
    std::vector<uint64_t> L3Storage2DDRKeys =
        DDRSwapKeyForL3StorageQue[info.name + SWAP_IN_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    std::vector<float*> L3Storage2DDRAddrs = DDRSwapAddrsQue[info.name + SWAP_IN_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    cacheManager->FetchL3StorageEmb2DDR(info.name, info.extEmbeddingSize, L3Storage2DDRKeys, L3Storage2DDRAddrs);
    LOG_DEBUG("table:{}, accumulate batchId:{}, channelId:{}, thread:{}, fetchL3StorageEmb2DDRTC(ms):{}",
              info.name.c_str(), info.batchId, info.channelId, info.threadIdx, fetchL3StorageEmb2DDRTC.ElapsedMS());

    bool isSuccess = BuildH2DEmbedding(info, h2dEmb);
    if (!isSuccess) {
        return false;
    }
    LOG_DEBUG(MGMT + "In swap thread, finish embedding lookup, table:{}, channelId:{}, accumulate batchId:{}, "
                     "thread:{}",
              info.name, info.channelId, info.batchId, info.threadIdx);
    hybridMgmtBlock->lastLookUpFinishStep[info.name][info.channelId]++;
    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastLookUpFinishCV[nextKey].notify_all();

    return true;
}

void HybridMgmt::EmbeddingSendL3Storage(const EmbTaskInfo& info, vector<Tensor>& h2dEmb)
{
    string currentKey = MakeSwapCVName(info.threadIdx, info.name, info.channelId);
    std::unique_lock<std::mutex> lastSendFinishLocker(lastSendFinishMutex[currentKey]);
    lastSendFinishCV[currentKey].wait(lastSendFinishLocker, [info, this] {
        return (hybridMgmtBlock->lastSendFinishStep[info.name][info.channelId] == info.batchId) || mutexDestroy;
    });
    TimeCost SendTC = TimeCost();
    // 区分通道发送
    hdTransfer->Send(TransferChannel::H2D, h2dEmb, info.channelId, info.name, info.batchId);
    hybridMgmtBlock->lastSendFinishStep[info.name][info.channelId]++;
    string nextKey = MakeSwapCVName(info.cvNotifyIndex, info.name, info.channelId);
    lastSendFinishCV[nextKey].notify_all();
    LOG_DEBUG(MGMT + "In swap thread, finish send h2d embedding, table:{}, channelId:{}, batchId:{}, accumulate "
                     "batchId:{}, thread:{}, SendH2DEmbTC(ms):{}",
              info.name, info.channelId, hybridMgmtBlock->h2dSendBatchId[info.name][info.channelId], info.batchId,
              info.threadIdx, SendTC.ElapsedMS());

    hybridMgmtBlock->h2dSendBatchId[info.name][info.channelId]++;
}

void HybridMgmt::HandleDataSwapForL3Storage(const EmbBaseInfo& info, vector<uint64_t>& swapInKeys,
                                            vector<uint64_t>& swapOutKeys)
{
    TimeCost ProcessSwapInKeysTC;
    vector<emb_cache_key_t> L3StorageToDDRKeys;
    vector<emb_cache_key_t> DDRToL3StorageKeys;
    cacheManager->ProcessSwapInKeys(info.name, swapInKeys, DDRToL3StorageKeys, L3StorageToDDRKeys);
    LOG_DEBUG("ProcessSwapInKeysTC(ms):{} ", ProcessSwapInKeysTC.ElapsedMS());

    TimeCost ProcessSwapOutKeysTC;
    HBMSwapOutInfo hbmSwapInfo;
    cacheManager->ProcessSwapOutKeys(info.name, swapOutKeys, hbmSwapInfo);
    LOG_DEBUG("ProcessSwapOutKeysTC(ms):{} ", ProcessSwapOutKeysTC.ElapsedMS());

    LOG_DEBUG("table:{}, batchId:{}, channelId:{}, swapInSize:{}, swapOutSize:{}", info.name, info.batchId,
              info.channelId, swapInKeys.size(), swapOutKeys.size());
    LOG_DEBUG("table:{}, batchId:{}, channelId:{}, swap out, HBM2DDR Keys:{}, HBM2DDR AddrOffs:{}, "
              "HBM2L3Storage Keys:{}, HBM2L3Storage AddrOff:{}",
              info.name, info.batchId, info.channelId, hbmSwapInfo.swapOutDDRKeys.size(),
              hbmSwapInfo.swapOutDDRAddrOffs.size(), hbmSwapInfo.swapOutL3StorageKeys.size(),
              hbmSwapInfo.swapOutL3StorageAddrOffs.size());
    LOG_DEBUG("table:{}, batchId:{}, channelId:{}, DDR2L3Storage Keys:{}, L3Storage2DDR Keys:{}", info.name,
              info.batchId, info.channelId, DDRToL3StorageKeys.size(), L3StorageToDDRKeys.size());

    auto DDRToL3StorageKeysForL3S = DDRToL3StorageKeys;
    auto L3StorageToDDRKeysForL3S = L3StorageToDDRKeys;
    // DDR<->L3Storage
    DDRSwapKeyQue[info.name + SWAP_OUT_STR][info.channelId].Pushv(DDRToL3StorageKeys);
    DDRSwapKeyQue[info.name + SWAP_IN_STR][info.channelId].Pushv(L3StorageToDDRKeys);

    DDRSwapKeyForL3StorageQue[info.name + SWAP_OUT_STR][info.channelId].Pushv(DDRToL3StorageKeysForL3S);
    DDRSwapKeyForL3StorageQue[info.name + SWAP_IN_STR][info.channelId].Pushv(L3StorageToDDRKeysForL3S);

    // HBM<->DDR
    HBMSwapKeyQue[info.name + SWAP_OUT_STR][info.channelId].Pushv(hbmSwapInfo.swapOutDDRKeys);
    HBMSwapKeyQue[info.name + ADDR_STR][info.channelId].Pushv(hbmSwapInfo.swapOutDDRAddrOffs);
    HBMSwapKeyQue[info.name + SWAP_IN_STR][info.channelId].Pushv(swapInKeys);

    // HBM->L3Storage
    HBMSwapKeyForL3StorageQue[info.name + SWAP_OUT_STR][info.channelId].Pushv(hbmSwapInfo.swapOutL3StorageKeys);
    HBMSwapKeyForL3StorageQue[info.name + ADDR_STR][info.channelId].Pushv(hbmSwapInfo.swapOutL3StorageAddrOffs);

    // normal status
    EosL1Que[info.name][info.channelId].Pushv(false);
}

bool HybridMgmt::BuildH2DEmbedding(const EmbTaskInfo& info, vector<Tensor>& h2dEmb)
{
    std::vector<float*> swapInAddrs = HBMSwapAddrsQue[info.name + SWAP_IN_STR][info.channelId].WaitAndPop();
    if (!isRunning) {
        return false;
    }
    h2dEmb.emplace_back(
        Tensor(tensorflow::DT_FLOAT, {int(swapInAddrs.size()), static_cast<long long>(info.extEmbeddingSize)}));
    auto& tmpTensor = h2dEmb.back();
    float* h2dEmbAddr = tmpTensor.flat<float>().data();
    TimeCost embeddingLookupTC = TimeCost();

    uint64_t memSize = info.extEmbeddingSize * sizeof(float);
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(swapInAddrs, h2dEmbAddr, info, memSize)
    for (uint64_t i = 0; i < swapInAddrs.size(); i++) {
        auto rc = memcpy_s(h2dEmbAddr + i * info.extEmbeddingSize, memSize, swapInAddrs[i], memSize);
        if (rc != 0) {
            auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::UNKNOWN,
                               StringFormat("Memcpy_s failed when emb lookup, error code: %d. MemSize: %d. You can "
                                            "query the meaning of security function error code.",
                                            rc, memSize));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString().c_str());
        }
    }
    LOG_DEBUG(
        "[BuildH2DEmbedding] table:{}, channel:{}, thread:{}, accumulate batchId:{}, emb size:{}, emb samples:{}, "
        "embeddingLookupTC(ms):{}",
        info.name.c_str(), info.channelId, info.threadIdx, info.batchId, swapInAddrs.size(),
        FloatPtrToLimitStr(h2dEmbAddr, swapInAddrs.size() * info.extEmbeddingSize), embeddingLookupTC.ElapsedMS());
    return true;
}

vector<uint64_t> HybridMgmt::GetUniqueKeys(const EmbBaseInfo& info, bool& remainBatchOut, bool& isEos)
{
    auto uniqueKeys = KEY_PROCESS_INSTANCE->GetUniqueKeys(info, isEos);
    // DDR eos send in swap pipeline.
    if (isEos) {
        remainBatchOut = false;
        return uniqueKeys;
    }
    if (uniqueKeys.empty()) {
        remainBatchOut = false;
        LOG_WARN("table:{}, channelId:{} batchId:{}, UniqueKeys result is empty", info.name, info.channelId,
                 info.batchId);
        return uniqueKeys;
    }

    if (info.channelId == TRAIN_CHANNEL_ID) {
        TimeCost KeyMaintainTC;
        trainKeysSet[info.name].insert(uniqueKeys.begin(), uniqueKeys.end());
        LOG_DEBUG("table:{}, batchId:{}, KeyMaintainTC(ms):{}", info.name, info.batchId, KeyMaintainTC.ElapsedMS());
    } else {
        for (auto& key : uniqueKeys) {
            if (trainKeysSet[info.name].find(key) == trainKeysSet[info.name].end()) {
                key = INVALID_KEY_VALUE;
                LOG_TRACE("find key not train before, set as invalid key");
            }
        }
    }

    LOG_DEBUG("table:{}, channelId:{} batchId:{}, GetUniqueKeys end", info.name, info.channelId, info.batchId);
    return uniqueKeys;
}

vector<int32_t> HybridMgmt::GetRestoreVecSec(const EmbBaseInfo& info, bool& remainBatchOut)
{
    auto restoreVecSec = KEY_PROCESS_INSTANCE->GetRestoreVecSec(info);
    if (restoreVecSec.empty()) {
        remainBatchOut = false;
        LOG_WARN("table:{}, channelId:{} batchId:{}, restoreVecSec result is empty", info.name, info.channelId,
                 info.batchId);
        return restoreVecSec;
    }
    LOG_DEBUG("table:{}, channelId:{} batchId:{}, GetRestoreVecSec end", info.name, info.channelId, info.batchId);
    return restoreVecSec;
}

void HybridMgmt::SendAll2AllVec(const EmbBaseInfo& info, bool& remainBatchOut)
{
    // The static shape and dp cases do not require all2all.
    if (!mgmtRankInfo.useStatic && !info.isDp) {
        bool isEos = false;  // useless, adapt to HBM mode
        TimeCost getAll2AllTC;
        unique_ptr<vector<Tensor>> all2all = KEY_PROCESS_INSTANCE->GetInfoVec(info, ProcessedInfo::ALL2ALL, isEos);
        LOG_DEBUG("table:{}, channelId:{}, batchId:{}, GetInfoVec all2all end, GetAll2AllTC(ms):{}", info.name,
                  info.channelId, info.batchId, getAll2AllTC.ElapsedMS());
        if (all2all == nullptr) {
            remainBatchOut = false;
            LOG_WARN("Information vector is nullptr!");
            return;
        }
        TimeCost sendAll2AllTC;
        hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, info.channelId, info.name, info.batchId);
        LOG_DEBUG("table:{}, channelId:{}, batchId:{}, send all2all end, sendAll2AllTC(ms):{}", info.name,
                  info.channelId, info.batchId, sendAll2AllTC.ElapsedMS());
    }
}

void HybridMgmt::SendRestoreVec(const EmbBaseInfo& info, bool& remainBatchOut)
{
    bool isEos = false;  // useless, adapt to HBM mode
    TimeCost getRestoreTC;
    unique_ptr<vector<Tensor>> infoVecs = KEY_PROCESS_INSTANCE->GetInfoVec(info, ProcessedInfo::RESTORE, isEos);
    if (infoVecs == nullptr) {
        remainBatchOut = false;
        if (isRunning) {
            LOG_ERROR("Information vector is nullptr!");
        }
        return;
    }
    LOG_DEBUG("table:{}, channelId:{}, batchId:{}, get restore end, getRestoreTC(ms):{}", info.name, info.channelId,
              info.batchId, getRestoreTC.ElapsedMS());

    TimeCost sendRestoreSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, info.channelId, info.name, info.batchId);
    LOG_DEBUG("table:{}, channelId:{}, batchId:{}, send restore end, sendRestoreSyncTC(ms):{}", info.name,
              info.channelId, info.batchId, sendRestoreSyncTC.ElapsedMS());
}

void HybridMgmt::SendLookupOffsets(const EmbBaseInfo& info, vector<uint64_t>& uniqueKeys,
                                   vector<int32_t>& restoreVecSec)
{
    // uniqueKeys already transfer to offset in GetSwapPairsAndKey2Offset
    // graph will filter out invalid offset(-1). see function _set_specific_value_for_non_valid_key
    TimeCost sendLookupOffsetsTC;
    std::vector<uint64_t> lookupOffsets;
    for (const auto& index : restoreVecSec) {
        if (index == INVALID_INDEX_VALUE) {
            lookupOffsets.emplace_back(static_cast<uint64_t>(INVALID_KEY_VALUE));
            continue;
        }
        lookupOffsets.emplace_back(uniqueKeys[index]);
    }
    hdTransfer->Send(TransferChannel::LOOKUP, {Vec2TensorI32(lookupOffsets)}, info.channelId, info.name, info.batchId);
    LOG_DEBUG("table:{}, channelId:{}, batchId:{}, send lookupOffset, sendLookupOffsetsTC(ms):{}", info.name,
              info.channelId, info.batchId, sendLookupOffsetsTC.ElapsedMS());
}

void HybridMgmt::SendGlobalUniqueVec(const EmbBaseInfo& info, vector<uint64_t>& uniqueKeys,
                                     vector<int32_t>& restoreVecSec)
{
    // In the DP mode, the second USS is used to align the length of the grad in the allreduce.
    if (!((info.channelId == TRAIN_CHANNEL_ID && mgmtRankInfo.useSumSameIdGradients) ||
          (info.channelId == TRAIN_CHANNEL_ID && info.isDp))) {
        return;
    }
    TimeCost sendUniqueKeysSyncTC;
    hdTransfer->Send(TransferChannel::UNIQKEYS,
                     {mgmtRankInfo.useDynamicExpansion ? Vec2TensorI64(uniqueKeys) : Vec2TensorI32(uniqueKeys)},
                     info.channelId, info.name, info.batchId);
    LOG_DEBUG("table:{}, channelId:{}, batchId:{}, sendUniqueKeysSyncTC(ms):{}", info.name, info.channelId,
              info.batchId, sendUniqueKeysSyncTC.ElapsedMS());

    TimeCost sendRestoreVecSecSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE_SECOND, {Vec2TensorI32(restoreVecSec)}, info.channelId, info.name,
                     info.batchId);
    LOG_DEBUG("table:{}, channelId:{}, batchId:{}, sendRestoreVecSecSyncTC(ms):{}", info.name, info.channelId,
              info.batchId, sendRestoreVecSecSyncTC.ElapsedMS());
}

void HybridMgmt::CheckLookupAddrSuccessDDR()
{
    if (lookupAddrSuccess) {
        return;
    }
    // lookup失败，从future捞出异常
    for (auto& t : lookUpSwapAddrsThreads) {
        t.get();
    }
}

void HybridMgmt::GetSwapPairsAndKey2Offset(const EmbBaseInfo& info, vector<uint64_t>& uniqueKeys,
                                           pair<vector<uint64_t>, vector<uint64_t>>& swapInKoPair,
                                           pair<vector<uint64_t>, vector<uint64_t>>& swapOutKoPair)
{
    TimeCost GetSwapPairsAndKey2OffsetTC;
    int swapInCode = embCache->GetSwapPairsAndKey2Offset(info.name, uniqueKeys, swapInKoPair, swapOutKoPair);
    if (swapInCode != H_OK) {
        auto error = Error(ModuleName::M_OCK_CTR, ErrorType::UNKNOWN,
                           StringFormat("Table:%s, [GetSwapPairsAndKey2Offset] failed! error code:%d.",
                                        info.name.c_str(), swapInCode));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString().c_str());
    }
    LOG_DEBUG("table:{}, channel:{}, batchId:{}, GetSwapPairsAndKey2OffsetTC(ms):{}", info.name, info.channelId,
              info.batchId, GetSwapPairsAndKey2OffsetTC.ElapsedMS());

    LOG_DEBUG("table:{}, channel:{}, batchId:{}, swapIn keys:{}, swapIn pos:{}, swapOut keys:{}, swapOut pos:{}",
              info.name, info.channelId, info.batchId, VectorToString(swapInKoPair.first),
              VectorToString(swapInKoPair.second), VectorToString(swapOutKoPair.first),
              VectorToString(swapOutKoPair.second));
}

void HybridMgmt::EnqueueSwapInfo(const EmbBaseInfo& info, pair<vector<uint64_t>, vector<uint64_t>>& swapInKoPair,
                                 pair<vector<uint64_t>, vector<uint64_t>>& swapOutKoPair)
{
    auto& swapInKeys = swapInKoPair.first;
    auto& swapOutKeys = swapOutKoPair.first;
    HBMSwapKeyQue[info.name + SWAP_OUT_STR][info.channelId].Pushv(swapOutKeys);
    HBMSwapKeyQue[info.name + SWAP_IN_STR][info.channelId].Pushv(swapInKeys);

    CheckLookupAddrSuccessDDR();

    EosL1Que[info.name][info.channelId].Pushv(false);
    LOG_DEBUG("Enqueue on HBMSwapKeyQue and EosL1Que, table:{}, batchId:{}, channelId:{}, swapInSize:{}, "
              "swapOutSize:{}, EosL1Que.size: {}",
              info.name, info.batchId, info.channelId, swapInKeys.size(), swapOutKeys.size(),
              EosL1Que[info.name][info.channelId].Size());
}

void HybridMgmt::BackUpTrainStatus()
{
    int channelID = TRAIN_CHANNEL_ID;
    int& theTrainBatchId = hybridMgmtBlock->pythonBatchId[channelID];
    if (theTrainBatchId == 0) {
        return;
    }

    LOG_INFO("On Estimator train and eval mode, start to backup train status, "
             "current train batchId: {} .",
             theTrainBatchId);
    // When in the train and eval mode of estimator, backup training states before loading.
    EmbeddingMgmt::Instance()->BackUpTrainStatusBeforeLoad();

    if (isL3StorageEnabled) {
        cacheManager->BackUpTrainStatus();
    }
    isBackUpTrainStatus = true;
}

void HybridMgmt::RecoverTrainStatus()
{
    if (isBackUpTrainStatus) {
        EmbeddingMgmt::Instance()->RecoverTrainStatus();
    }

    if (isL3StorageEnabled) {
        cacheManager->RecoverTrainStatus();
    }
    isBackUpTrainStatus = false;
}

void HybridMgmt::GetDeltaModelKeys(const string& savePath, bool saveDelta,
                                   map<string, map<emb_key_t, KeyInfo>>& keyInfoMap)
{
    int saveStep = GetStepFromPath(savePath);
    if (isFirstSave) {
        for (auto& embInfo : mgmtEmbInfo) {
            keyBatchIdMap[embInfo.name] = saveStep;
        }
    }
    if (isIncrementalCkpt) {
        std::unique_lock<std::mutex> lock(keyCountUpdateMtx);
        checkConditionMet = false;
        while (!checkConditionMet) {
            keyCountUpdateCv.wait(lock, [this, saveStep] {
                for (const auto& it : keyBatchIdMap) {
                    if (it.second != saveStep) {
                        return false;
                    }
                }
                checkConditionMet = true;
                return true;
            });
        }

        if (saveDelta) {
            for (auto& delta : deltaMap) {
                auto& deltaInfo = delta.second;
                for (auto& it : deltaInfo) {
                    if (it.second.isChanged) {
                        keyInfoMap[delta.first][it.first] = it.second;
                    }
                }
            }
        }
    }
}

void HybridMgmt::InitPipelineMutexAndCV(const string& embTableName)
{
    for (int channelId = 0; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadIndex = 0; threadIndex < EMBEDDING_THREAD_NUM; ++threadIndex) {
            string key = MakeSwapCVName(threadIndex, embTableName, channelId);
            lastUpdateFinishMutex[key];
            lastUpdateFinishCV[key];
            lastLookUpFinishMutex[key];
            lastLookUpFinishCV[key];
            lastSendFinishMutex[key];
            lastSendFinishCV[key];
            lastRecvFinishMutex[key];
            lastRecvFinishCV[key];
        }
    }
}

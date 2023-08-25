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
    // 是否设置全局去重（相同key的梯度先累加），默认为false
    if (getenv("APPLY_GRADIENTS_STRATEGY") != nullptr) {
        bool strategy = (!strcmp(getenv("APPLY_GRADIENTS_STRATEGY"), SUM_SAME_ID));
        PerfConfig::gradientStrategy = strategy;
        LOG(INFO) << StringFormat("config GRADIENTS_STRATEGY:%d", strategy);
    }

    // 设置当前进程用于数据处理的线程数，默认为6，取值1-10；取值不在范围内，则数据处理线程启动失败退出
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

    // 设置AccCTR去重线程数，默认为8，取值1-8；取值不在范围内，则数据处理线程启动失败退出
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

    // 设置是否使用AccCTR库提供的去重、分桶功能，默认关闭
    PerfConfig::fastUnique = GetEnv("FAST_UNIQUE");
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
void HybridMgmt::InitRankInfo(RankInfo& rankInfo, const vector<EmbInfo>& embInfos)
{
#ifndef GTEST
    MPI_Comm_size(MPI_COMM_WORLD, &rankInfo.rankSize);
    rankInfo.localRankId = rankInfo.deviceId;

    // 计算训练任务涉及的所有表在DDR中需要分配的key数量
    size_t totHostVocabSize = 0;
    for (const auto& emb : embInfos) {
        totHostVocabSize += emb.hostVocabSize;
    }

    // 根据DDR的key数量，配置存储模式HBM/DDR
    if (totHostVocabSize == 0) {
        rankInfo.noDDR = true;
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
    // 判断是否已经拉起特征处理线程（key process）
    if (isRunning) {
        return true;
    }

    // 设置日志的级别，对日志格式进行配置
    SetLog(rankInfo.rankId);
    InitRankInfo(rankInfo, embInfos);

    LOG(INFO) << StringFormat(
        MGMT + "begin initialize, localRankSize:%d, localRankId:%d, rank:%d",
        rankInfo.localRankSize, rankInfo.localRankId, rankInfo.rankId);

    mgmtRankInfo = rankInfo;
    mgmtEmbInfo = embInfos;

    // 进行acl资源初始化，设置当前训练进程的device，为每张表创建数据传输通道
    hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    hdTransfer->Init(embInfos, rankInfo.deviceId);

    // 启动数据处理线程
    bool rc = InitKeyProcess(rankInfo, embInfos, thresholdValues, seed);
    if (!rc) {
        return false;
    }

    isRunning = true;

    // DDR模式，初始化hashmap和host emb
    if (!rankInfo.noDDR) {
        hostEmbs = make_unique<HostEmb>();
        hostHashMaps = make_unique<EmbHashMap>();
        hostEmbs->Initialize(embInfos, seed);
        hostHashMaps->Init(rankInfo, embInfos, ifLoad);
    }

    // 非断点续训模式，启动数据传输
    isLoad = ifLoad;
    if (!isLoad) {
        Start();
    }

    for (const auto& info: embInfos) {
        LOG(INFO) << StringFormat(
            MGMT + "emb[%s] vocab size %d+%d sc:%d",
            info.name.c_str(), info.devVocabSize, info.hostVocabSize, info.sendCount);
    }
    LOG(INFO) << StringFormat(
        MGMT + "end initialize, noDDR:%d, maxStep:[%d, %d], rank:%d", rankInfo.noDDR,
        rankInfo.maxStep.at(TRAIN_CHANNEL_ID), rankInfo.maxStep.at(EVAL_CHANNEL_ID), rankInfo.rankId);
#endif
    return true;
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
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side save: ddr mode hashmap");
        saveData.hostEmbs = hostEmbs->GetHostEmbs();
        saveData.embHashMaps = hostHashMaps->GetHashMaps();
    } else {
        // HBM模式保存最大偏移（真正使用了多少vocab容量），特征到偏移的映射
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side save: no ddr mode hashmap");
        saveData.maxOffset = preprocess->GetMaxOffset();
        saveData.keyOffsetMap = preprocess->GetKeyOffsetMap();
    }

    // 保存特征准入淘汰相关的数据
    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side save: feature admit and evict");
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

    VLOG(GLOG_DEBUG) << (MGMT + "Start host side load process");

    CkptData loadData;
    Checkpoint loadCkpt;
    vector<CkptFeatureType> loadFeatures;
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
    auto& featAdmitNEvict = preprocess->GetFeatAdmitAndEvict();
    if (featAdmitNEvict.GetFunctionSwitch()) {
        loadFeatures.push_back(CkptFeatureType::FEAT_ADMIT_N_EVICT);
    }

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
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side load: ddr mode hashmap");
        hostHashMaps->LoadHashMap(loadData.embHashMaps);
    } else {
        // HBM模式 将加载的最大偏移（真正使用了多少vocab容量）、特征到偏移的映射，进行赋值
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side load: no ddr mode hashmap");
        preprocess->LoadMaxOffset(loadData.maxOffset);
        preprocess->LoadKeyOffsetMap(loadData.keyOffsetMap);
    }

    // 将加载的特征准入淘汰记录进行赋值
    if (featAdmitNEvict.GetFunctionSwitch()) {
        VLOG(GLOG_DEBUG) << (MGMT + "Start host side load: feature admit and evict");
        featAdmitNEvict.LoadTableThresholds(loadData.table2Thresh);
        featAdmitNEvict.LoadHistoryRecords(loadData.histRec);
    }

    VLOG(GLOG_DEBUG) << (MGMT + "Finish host side load process");

    preprocess->LoadSaveUnlock();

    // 执行训练
    if (isLoad) {
        Start();
    }
#endif
    return true;
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

/// 加载key对应的offset，python侧调用；启动数据处理线程
/// \param ReceiveKeyOffsetMap
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

/// 对DDR模式保存的模型和训练配置进行一致性校验
/// \param loadData
/// \return 是否一致
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

/// 根据HBM/DDR模式，启动数据处理线程
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

/// 启动HBM模式数据处理线程
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
/// 启动训练数据处理线程
/// \param type 存储模式
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

/// 启动推理数据处理线程
/// \param type 存储模式
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

/// 训练数据处理：数据处理状态正常，处理的batch数小于用户预设值或者设为-1时，会循环处理；
/// \param type 存储模式
/// \return
bool HybridMgmt::TrainTask(TaskType type)
{
    bool isContinue;
    bool status;
    do {
        if (!isRunning) {
            return false;
        }

        switch (type) {
            case TaskType::HBM:
                status = ParseKeysHBM(TRAIN_CHANNEL_ID, trainBatchId);
                isContinue = !EndBatch(trainBatchId, TRAIN_CHANNEL_ID);
                LOG(INFO) << StringFormat(MGMT + "ParseKeysHBMBatchId = %d", trainBatchId);
                break;
            case TaskType::DDR:
                status =  ParseKeys(TRAIN_CHANNEL_ID, trainBatchId);
                isContinue = !EndBatch(trainBatchId, TRAIN_CHANNEL_ID);
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

/// 推理数据处理：数据处理状态正常，处理的batch数小于用户预设值或者设为-1时，会循环处理；
/// \param type 存储模式
/// \return
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
    } while (!EndBatch(evalBatchId, EVAL_CHANNEL_ID));

    return true;
}

/// HBM模式下，发送key process线程已处理好的各类型向量到指定通道中
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \return
bool HybridMgmt::ParseKeysHBM(int channelId, int& batchId)
{
    LOG(INFO) << StringFormat(
        MGMT + "start parse keys HBM, nBatch:%d , [%d]:%d", mgmtRankInfo.nBatch, channelId, batchId);

    // 循环处理每个表的数据
    for (const auto& embInfo: mgmtEmbInfo) {
        TimeCost ParseKeysTC;
        // get
        TimeCost getTensorsSyncTC;

        // 获取各类向量，如果为空指针，退出当前函数
        auto infoVecs = preprocess->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::RESTORE);
        if (infoVecs == nullptr) {
            LOG(INFO) << StringFormat(
                MGMT + "ParseKeys infoVecs empty ! batchId:%d, channelId:%d", batchId, channelId);
            return false;
        }

        // 动态shape场景下，获取all2all向量（通信量矩阵）
        unique_ptr<vector<Tensor>> all2all = nullptr;
        if (!mgmtRankInfo.useStatic) {
            all2all = preprocess->GetInfoVec(batchId, embInfo.name, channelId, ProcessedInfo::ALL2ALL);
        }
        VLOG(GLOG_DEBUG) << StringFormat("getTensorsSyncTC(ms):%d", getTensorsSyncTC.ElapsedMS());

        // 动态shape场景下，发送all2all向量（通信量矩阵）
        TimeCost sendTensorsSyncTC;
        if (!mgmtRankInfo.useStatic) {
            TimeCost sendAll2AllScSyncTC;
            hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embInfo.name);
            VLOG(GLOG_DEBUG) << StringFormat("sendAll2AllScSyncTC(ms):%d", sendAll2AllScSyncTC.ElapsedMS());
        }

        // 发送查询向量
        TimeCost sendLookupSyncTC;
        hdTransfer->Send(TransferChannel::LOOKUP, { infoVecs->back() }, channelId, embInfo.name);
        infoVecs->pop_back();
        VLOG(GLOG_DEBUG) << StringFormat("sendLookupSyncTC(ms):%d", sendLookupSyncTC.ElapsedMS());

        // 训练时，使用全局去重聚合梯度，发送全局去重的key和对应的恢复向量
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

        // 发送恢复向量
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
    LOG(INFO) << StringFormat(
        MGMT + "DDR mode, start parse keys, nBatch:%d , [%d]:%d",
        mgmtRankInfo.nBatch, channelId, batchId);
    TimeCost parseKeyTC;
    int start = batchId;
    int iBatch = 0; // 预取数据处理计数
    bool ifHashmapFree = true;
    bool remainBatch = true; // 是否从通道获取了数据
    while (true) {
        LOG(INFO) << StringFormat(MGMT + "parse keys, [%d]:%d", channelId, batchId);
        for (const auto& embInfo : mgmtEmbInfo) {
            ifHashmapFree = ProcessEmbInfo(embInfo.name, batchId, channelId, iBatch, remainBatch);

            // 通道数据已空
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
/// 构造训练所需的各种向量数据
/// \param embName 表名
/// \param batchId 已处理的batch数
/// \param channelId 通道索引（训练/推理）
/// \param iBatch 预取数据处理计数
/// \param remainBatchOut 是否从通道获取了数据
/// \return HBM是否还有剩余空间
bool HybridMgmt::ProcessEmbInfo(const std::string& embName, int batchId,
                                int channelId, int iBatch, bool& remainBatchOut)
{
    TimeCost getAndSendTensorsTC;
    TimeCost getTensorsTC;
    auto& embHashMap = hostHashMaps->embHashMaps.at(embName);

    // 进行新一批预取数据时，计数初始化
    if (iBatch == 0) {
        embHashMap.SetStartCount();
    }

    // 获取查询向量
    auto lookupKeys = preprocess->GetLookupKeys(batchId, embName, channelId);
    if (lookupKeys.empty()) { remainBatchOut = false; }

    // 获取各类向量，如果为空指针，退出当前函数
    auto infoVecs = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::RESTORE);
    if (infoVecs == nullptr) { return false; }
    VLOG(GLOG_DEBUG) << StringFormat("getTensorsTC(ms):%d", getTensorsTC.ElapsedMS());

    TimeCost sendRestoreSyncTC;
    hdTransfer->Send(TransferChannel::RESTORE, *infoVecs, channelId, embName);
    VLOG(GLOG_DEBUG) << StringFormat("sendRestoreSyncTC(ms):%d", sendRestoreSyncTC.ElapsedMS());

    // 计算查询向量；记录需要被换出的HBM偏移
    vector<Tensor> tmpData;
    vector<int32_t> offsetsOut;
    DDRParam ddrParam(tmpData, offsetsOut);
    TimeCost hostHashMapProcessTC;
    hostHashMaps->Process(embName, lookupKeys, iBatch, ddrParam, channelId);
    VLOG(GLOG_DEBUG) << StringFormat("hostHashMapProcessTC(ms):%d", hostHashMapProcessTC.ElapsedMS());

    if (PerfConfig::gradientStrategy && channelId == TRAIN_CHANNEL_ID && remainBatchOut) {
        vector<int32_t> uniqueKeys, restoreVecSec;
        preprocess->GlobalUnique(offsetsOut, uniqueKeys, restoreVecSec);

        TimeCost sendUnikeysSyncTC;
        hdTransfer->Send(TransferChannel::UNIQKEYS, { mgmtRankInfo.useDynamicExpansion ? Vec2TensorI64(uniqueKeys) :
                                                                    Vec2TensorI32(uniqueKeys) }, channelId, embName);
        VLOG(GLOG_DEBUG) << StringFormat("sendUnikeysSyncTC(ms):%d", sendUnikeysSyncTC.ElapsedMS());

        TimeCost sendRestoreVecSecSyncTC;
        hdTransfer->Send(TransferChannel::RESTORE_SECOND, { Vec2TensorI32(restoreVecSec) }, channelId, embName);
        VLOG(GLOG_DEBUG) << StringFormat("sendRestoreVecSecSyncTC(ms):%d", sendRestoreVecSecSyncTC.ElapsedMS());
    }

    TimeCost sendTensorsTC;
    hdTransfer->Send(TransferChannel::LOOKUP, { ddrParam.tmpDataOut.front() }, channelId, embName);
    ddrParam.tmpDataOut.erase(ddrParam.tmpDataOut.begin());
    hdTransfer->Send(TransferChannel::SWAP, ddrParam.tmpDataOut, channelId, embName);
    if (!mgmtRankInfo.useStatic) {
        auto all2all = preprocess->GetInfoVec(batchId, embName, channelId, ProcessedInfo::ALL2ALL);
        hdTransfer->Send(TransferChannel::ALL2ALL, *all2all, channelId, embName);
    }
    VLOG(GLOG_DEBUG) << StringFormat("sendTensorsTC(ms):%d", sendTensorsTC.ElapsedMS());

    VLOG(GLOG_DEBUG) << StringFormat(
        "getAndSendTensorsTC(ms):%d, channelId:%d", getAndSendTensorsTC.ElapsedMS(), channelId);

    if (embHashMap.HasFree(lookupKeys.size())) { // check free > next one batch
        LOG(WARNING) << StringFormat(MGMT + "embName %s[%d]%d,iBatch:%d freeSize not enough, %d",
                                     embName.c_str(), channelId, batchId, iBatch, lookupKeys.size());
        return false;
    }
    return true;
}

/// 发送H2D和接收D2H向量
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
/// \param start
/// \param iBatch 预取数据处理计数
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

/// 发送H2D和接收D2H向量，并更新host emb
/// \param channelId 通道索引（训练/推理）
/// \param batchId 已处理的batch数
void HybridMgmt::EmbHDTrans(const int channelId, const int batchId)
{
    EASY_FUNCTION(profiler::colors::Blue)
    EASY_VALUE("mgmtProcess", batchId)
    VLOG(GLOG_DEBUG) << StringFormat(MGMT + "trans emb, batchId:%d, channelId:%d", batchId, channelId);
    TimeCost tr;
    TimeCost h2dTC;
    // 发送host需要换出的emb
    for (const auto& embInfo: mgmtEmbInfo) {
        auto& missingKeys = hostHashMaps->GetMissingKeys(embInfo.name);
        vector<Tensor> h2dEmb;
        hostEmbs->GetH2DEmb(missingKeys, embInfo.name, h2dEmb); // order!
        hdTransfer->Send(TransferChannel::H2D, h2dEmb, channelId, embInfo.name, batchId);
    }
    VLOG(GLOG_DEBUG) << StringFormat("h2dTC(ms):%d", h2dTC.ElapsedMS());

    TimeCost d2hTC;
    // 接收device换出的emb，并更新到host上
    for (const auto& embInfo: mgmtEmbInfo) {
        const auto& missingKeys = hostHashMaps->GetMissingKeys(embInfo.name);
        auto updateEmbV2 = getenv("UpdateEmb_V2");
        if (updateEmbV2 != nullptr and atoi(updateEmbV2) == 1) {
            hostEmbs->UpdateEmbV2(missingKeys, channelId, embInfo.name); // order!
        } else {
            hostEmbs->UpdateEmb(missingKeys, channelId, embInfo.name); // order!
        }
        hostHashMaps->ClearMissingKeys(embInfo.name);
    }
    VLOG(GLOG_DEBUG) << StringFormat("d2hTC(ms):%d", d2hTC.ElapsedMS());

    VLOG(GLOG_DEBUG) << StringFormat(
        "EmbHDTrans TimeCost(ms):%d batchId: %d channelId:%d", tr.ElapsedMS(), batchId, channelId
    );
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
        LOG(WARNING) << (MGMT + "Hook can not trigger evict, cause AdmitNEvict is not open");
        return false;
    }
    VLOG(GLOG_DEBUG) << StringFormat(MGMT + "evict triggered by hook, evict TableNum %d ", evictKeyMap.size());

    // 表为空，淘汰触发失败
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

/// DDR模式下的淘汰：删除映射表、初始化host表、发送dev淘汰位置
/// \param embName
/// \param keys
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
    auto& evictOffset = hostHashMaps->GetEvictPos(embName);
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

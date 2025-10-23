/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef MX_REC_EMB_MGMT_H
#define MX_REC_EMB_MGMT_H

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "emb_table/embedding_table.h"
#include "hd_transfer/hd_transfer.h"
#include "pcie_through/rma_shm_svm.h"
#include "hybrid_mgmt_block.h"
#include "l3_storage/cache_manager.h"
#include "ock_ctr_common/include/embedding_cache.h"
#include "ock_ctr_common/include/error_code.h"
#include "ock_ctr_common/include/factory.h"
#include "utils/common.h"
#include "utils/config.h"
#include "utils/singleton.h"
#include "utils/task_queue.h"
#include "utils/thread_pool.h"
#include "utils/time_cost.h"

namespace MxRec {
using namespace std;
using namespace tensorflow;
using namespace Common;

enum class TaskType {
    HBM,
    DDR
};

enum class SaveModelType {
    DELTA,
    BASE
};

inline string TransferModelType2Str(SaveModelType t)
{
    switch (t) {
        case SaveModelType::DELTA:
            return "delta";
        case SaveModelType::BASE:
            return "base";
        default:
            throw std::invalid_argument("Invalid ModelType.");
    }
}

struct EmbTaskInfo {
    int batchId;
    int threadIdx;
    int cvNotifyIndex;
    int extEmbeddingSize;
    int channelId;
    string name;
};

class HybridMgmt {
public:
    HybridMgmt() = default;

    ~HybridMgmt()
    {
        if (isRunning) {
            Destroy();
        }
    }

    HybridMgmt(const HybridMgmt&) = delete;

    HybridMgmt& operator=(const HybridMgmt&) = delete;

    bool Initialize(RankInfo rankInfo, const vector<EmbInfo>& embInfos, int seed,
                    const vector<ThresholdValue>& thresholdValues, bool ifLoad,
                    bool isIncrementalCheckpoint, bool useLccl);

    void Save(const string& savePath, bool saveDelta, bool isSaveL3Storage = true);

    bool Load(const string& loadPath, vector<string> warmStartTables);

    OffsetT SendHostMap(const string tableName);

    OffsetT SendLoadMap(const string tableName);

    void Start();

    void StartThreadForHBM();

    void StartThreadForDDR();

    void Destroy();

    bool ParseKeys(int channelId, int& batchId, TaskType type);

    bool Evict();

    void NotifyBySessionRun(int channelID) const;

    void CountStepBySessionRun(int channelID, int steps) const;

    int64_t GetTableSize(const string& embName) const;

    int64_t GetTableCapacity(const string& embName) const;

    void SetOptimizerInfo(const string& embName, OptimizerInfo optimInfo) const;

    void FetchDeviceEmb();

    bool ProcessEmbInfoHBM(const EmbBaseInfo& info, bool isGrad);

    bool ProcessEmbInfoDDR(const EmbBaseInfo& info);

    bool ProcessEmbInfoL3Storage(const EmbBaseInfo& info);

    void BackUpTrainStatus();

    void RecoverTrainStatus();

    void ReceiveKey();

    void ReceiveKeyThread(const EmbInfo& embInfo);

    void StartSyncThread();

    GTEST_PRIVATE
        : volatile bool mutexDestroy{false};  // LookupAndSend & ReceiveAndUpdate Condition_Variable_Wait stop.
    std::mutex lookUpAndSendBatchIdMtx[MAX_CHANNEL_NUM];  // train and eval
    std::mutex receiveAndUpdateBatchIdMtx[MAX_CHANNEL_NUM];

    std::unordered_map<std::string, std::mutex> lastUpdateFinishMutex;
    std::unordered_map<std::string, std::condition_variable> lastUpdateFinishCV;

    std::unordered_map<std::string, std::mutex> lastLookUpFinishMutex;
    std::unordered_map<std::string, std::condition_variable> lastLookUpFinishCV;

    std::unordered_map<std::string, std::mutex> lastSendFinishMutex;
    std::unordered_map<std::string, std::condition_variable> lastSendFinishCV;

    std::unordered_map<std::string, std::mutex> lastRecvFinishMutex;
    std::unordered_map<std::string, std::condition_variable> lastRecvFinishCV;

    std::vector<std::thread> EmbeddingLookUpAndSendThreadPool;
    std::vector<std::thread> EmbeddingReceiveAndUpdateThreadPool;
    std::vector<std::future<void>> lookUpSwapAddrsThreads;

    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> HBMSwapKeyQue;  // train and eval
    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> HBMSwapKeyForL3StorageQue;
    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> DDRSwapKeyQue;
    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> DDRSwapKeyForL3StorageQue;
    std::map<std::string, TaskQueue<std::vector<float*>>[MAX_CHANNEL_NUM]> HBMSwapAddrsQue;
    std::map<std::string, TaskQueue<std::vector<float*>>[MAX_CHANNEL_NUM]> DDRSwapAddrsQue;

    std::map<std::string, TaskQueue<bool>[MAX_CHANNEL_NUM]> EosL1Que;
    std::map<std::string, TaskQueue<bool>[MAX_CHANNEL_NUM]> EosL2Que;

    std::mutex evictMut;

    std::map<std::string, std::unordered_set<uint64_t>> trainKeysSet;
    const string SWAP_IN_STR = "SwapIn";
    const string SWAP_OUT_STR = "SwapOut";

    const string ADDR_STR = "Addr";
    ock::ctr::EmbCacheManagerPtr embCache = nullptr;
    std::map<std::string, std::vector<uint64_t>> lastSwapInPosMap{};
    std::map<std::string, std::vector<std::vector<uint64_t>>> trainTestSwitchInfoStore{};
    std::atomic<bool> lookupAddrSuccess{true};

    unique_ptr<ThreadPool> threadPool;

    void SetFeatureTypeForLoad(vector<CkptFeatureType>& loadFeatures);

    void EvictKeys(const string& embName, const vector<emb_cache_key_t>& keys);

    void InitRankInfo(RankInfo& rankInfo, const vector<EmbInfo>& embInfos) const;

    void EvictL3StorageKeys(const string& embName, const vector<emb_cache_key_t>& keys) const;

    void LookUpAndRemoveAddrs(const EmbTaskInfo& info);  // L3Storage, synchronous

    void LookUpSwapAddrs(const std::string& embName, int channelId);  // DDR, asynchronous

    void EmbeddingTask();

    void MultiThreadEmbHDTransWrap();

    void EmbeddingLookUpAndSendDDR(int batchId, int index, const EmbInfo& embInfo, int channelId);

    void EmbeddingReceiveAndUpdateDDR(int batchId, int index, const EmbInfo& embInfo, int channelId);

    void EmbeddingLookUpAndSendL3Storage(int batchId, int index, const EmbInfo& embInfo, int channelId);

    void EmbeddingReceiveAndUpdateL3Storage(int batchId, int index, const EmbInfo& embInfo, int channelId);

    void SendTensorForSwap(const EmbBaseInfo& info, const vector<uint64_t>& swapInPosUint,
                           const vector<uint64_t>& swapOutPosUint);

    void UpdateDeltaInfo(const string& embName, vector<int64_t>& keyCountVec, int64_t timeStamp, int64_t batchId);

    void ResetDeltaInfo();

    void GetDeltaModelKeys(const string& savePath, bool saveDelta, map<string, map<emb_key_t, KeyInfo>>& keyInfoMap);

    void InitPipelineMutexAndCV(const string& embTableName);

GTEST_PRIVATE:
    HybridMgmtBlock* hybridMgmtBlock;
    vector<EmbInfo> mgmtEmbInfo;
    RankInfo mgmtRankInfo;
    CacheManager* cacheManager;
    vector<std::unique_ptr<std::thread>> procThreads{};
    vector<std::thread> receiveKeyThreads{};
    map<string, vector<emb_cache_key_t>> evictKeyMap{};
    HDTransfer* hdTransfer;
    OffsetMapT offsetMapToSend;
    OffsetMapT loadOffsetToSend;
    bool isL3StorageEnabled{false};
    bool isRunning;
    bool isLoad{false};
    bool isInitialized{false};
    bool isBackUpTrainStatus = false;  // whether the train state has been backed up
    bool isIncrementalCkpt;
    map<string, absl::flat_hash_map<emb_key_t, KeyInfo>> deltaMap;
    absl::flat_hash_map<string, int> keyBatchIdMap;
    bool isFirstSave = true;
    std::mutex keyCountUpdateMtx;
    std::condition_variable keyCountUpdateCv;
    bool checkConditionMet = false;
    bool enableLccl = false;

    void TrainTask(TaskType type);

    void EvalTask(TaskType type);

    void SendUniqKeysAndRestoreVecHBM(const EmbBaseInfo& info, const unique_ptr<vector<Tensor>>& infoVecs,
                                      bool isGrad) const;

    void SendPaddingKeysMaskVecHBM(const EmbBaseInfo& info, const unique_ptr<vector<Tensor>>& infoVecs,
                                   bool isGrad) const;

    void SendPaddingKeysMaskVecDDRL3(const EmbBaseInfo& info, const vector<uint64_t>& offsetKeys) const;

    void InitEmbeddingCache(const vector<EmbInfo>& embInfos);

    void InitDataPipelineForDDR(const string& embName);

    void InitDataPipelineForL3Storage(const string& embName, int extEmbeddingSize);

    void JoinEmbeddingCacheThread();

    bool EmbeddingReceiveDDR(const EmbTaskInfo& info, float*& ptr, vector<float*>& swapOutAddrs);

    void EmbeddingUpdateDDR(const EmbTaskInfo& info, const float* embPtr, vector<float*>& swapOutAddrs);

    bool EmbeddingLookUpDDR(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    void EmbeddingSendDDR(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    bool EmbeddingBuildAndSendDDR(const EmbTaskInfo& info, float*& h2dEmb, std::array<int64_t, RMA_DIM_MAX>& dims);

    bool BuildAndSendH2DEmbedding(const EmbTaskInfo& info, float*& h2dEmb, std::array<int64_t, RMA_DIM_MAX>& dims);

    bool EmbeddingReceiveL3Storage(const EmbTaskInfo& info, float*& ptr, vector<float*>& swapOutAddrs, int64_t& dims0);

    void EmbeddingUpdateL3Storage(const EmbTaskInfo& info, float* embPtr, vector<float*>& swapOutAddrs, int64_t& dims0);

    bool EmbeddingLookUpL3Storage(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    void EmbeddingSendL3Storage(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    void CreateEmbeddingLookUpAndSendThread(int index, const EmbInfo& embInfo, int channelId);

    void CreateEmbeddingReceiveAndUpdateThread(int index, const EmbInfo& embInfo, int channelId);

    void HandleDataSwapForL3Storage(const EmbBaseInfo& info, vector<uint64_t>& swapInKeys,
                                    vector<uint64_t>& swapOutKeys);

    bool BuildH2DEmbedding(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    vector<uint64_t> GetUniqueKeys(const EmbBaseInfo& info, bool& remainBatchOut, bool& isEos);

    vector<int32_t> GetRestoreVecSec(const EmbBaseInfo& info, bool& remainBatchOut);

    void SendAll2AllVec(const EmbBaseInfo& info, bool& remainBatchOut);

    void SendRestoreVec(const EmbBaseInfo& info, bool& remainBatchOut);

    void SendLookupOffsets(const EmbBaseInfo& info, vector<uint64_t>& uniqueKeys, vector<int32_t>& restoreVecSec);

    void SendGlobalUniqueVec(const EmbBaseInfo& info, vector<uint64_t>& uniqueKeys, vector<int32_t>& restoreVecSec);

    void CheckLookupAddrSuccessDDR();

    void GetSwapPairsAndKey2Offset(const EmbBaseInfo& info, vector<uint64_t>& uniqueKeys,
                                   std::pair<vector<uint64_t>, vector<uint64_t>>& swapInKoPair,
                                   std::pair<vector<uint64_t>, vector<uint64_t>>& swapOutKoPair);

    void EnqueueSwapInfo(const EmbBaseInfo& info, std::pair<vector<uint64_t>, vector<uint64_t>>& swapInKoPair,
                         std::pair<vector<uint64_t>, vector<uint64_t>>& swapOutKoPair);

    vector<Tensor> BuildSaveSwapTensor(vector<uint64_t> swapOutPos, bool isSyncRemain);
};
}  // namespace MxRec
#endif  // MX_REC_EMB_MGMT_H

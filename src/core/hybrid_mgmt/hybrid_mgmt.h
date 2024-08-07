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

#ifndef MX_REC_EMB_MGMT_H
#define MX_REC_EMB_MGMT_H

#include <array>
#include <memory>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "emb_table/embedding_table.h"
#include "hd_transfer/hd_transfer.h"
#include "hybrid_mgmt_block.h"
#include "l3_storage/cache_manager.h"
#include "ock_ctr_common/include/embedding_cache.h"
#include "ock_ctr_common/include/error_code.h"
#include "ock_ctr_common/include/factory.h"
#include "utils/common.h"
#include "utils/config.h"
#include "utils/singleton.h"
#include "utils/task_queue.h"
#include "utils/time_cost.h"
#include "utils/thread_pool.h"

namespace MxRec {
using namespace std;
using namespace tensorflow;
using namespace Common;

enum class TaskType {
    HBM,
    DDR
};

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
                    const vector<ThresholdValue>& thresholdValues, bool ifLoad);

    void Save(const string& savePath);

    bool Load(const string& loadPath, vector<string> warmStartTables);

    OffsetT SendHostMap(const string tableName);

    OffsetT SendLoadMap(const string tableName);

    void ReceiveHostMap(AllKeyOffsetMapT receiveKeyOffsetMap);

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

    GTEST_PRIVATE : bool mutexDestroy{false};
    std::mutex lookUpAndSendBatchIdMtx[MAX_CHANNEL_NUM];  // train and eval
    std::mutex receiveAndUpdateBatchIdMtx[MAX_CHANNEL_NUM];

    std::map<std::string, std::mutex> lastUpdateFinishMutex;
    std::map<std::string, std::condition_variable> lastUpdateFinishCV;

    std::map<std::string, std::mutex> lastLookUpFinishMutex;
    std::map<std::string, std::condition_variable> lastLookUpFinishCV;

    std::map<std::string, std::mutex> lastSendFinishMutex;
    std::map<std::string, std::condition_variable> lastSendFinishCV;

    std::map<std::string, std::mutex> lastRecvFinishMutex;
    std::map<std::string, std::condition_variable> lastRecvFinishCV;

    std::vector<std::thread> EmbeddingLookUpAndSendThreadPool;
    std::vector<std::thread> EmbeddingReceiveAndUpdateThreadPool;
    std::vector<std::future<void>> lookUpSwapAddrsThreads;

    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> HBMSwapKeyQue;  // train and eval
    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> HBMSwapKeyForL3StorageQue;
    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> DDRSwapKeyQue;
    std::map<std::string, TaskQueue<std::vector<uint64_t>>[MAX_CHANNEL_NUM]> DDRSwapKeyForL3StorageQue;
    std::map<std::string, TaskQueue<std::vector<float*>>[MAX_CHANNEL_NUM]> HBMSwapAddrsQue;
    std::map<std::string, TaskQueue<std::vector<float*>>[MAX_CHANNEL_NUM]> DDRSwapAddrsQue;

    std::mutex evictMut;

    std::map<std::string, std::unordered_set<uint64_t>> trainKeysSet;
    const string SWAP_IN_STR = "SwapIn";
    const string SWAP_OUT_STR = "SwapOut";

    const string ADDR_STR = "Addr";
    ock::ctr::EmbCacheManagerPtr embCache = nullptr;
    std::map<std::string, std::vector<uint64_t>> lastSwapInPosMap{};
    std::map<std::string, std::vector<std::vector<uint64_t>>> trainTestSwitchInfoStore{};
    std::atomic<bool> lookupAddrSuccess{true};

    std::mutex saveMutex;
    std::condition_variable cvCheckSave;

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

private:
    HybridMgmtBlock* hybridMgmtBlock;
    vector<EmbInfo> mgmtEmbInfo;
    RankInfo mgmtRankInfo;
    CacheManager* cacheManager;
    vector<std::unique_ptr<std::thread>> procThreads{};
    map<string, vector<emb_cache_key_t>> evictKeyMap{};
    HDTransfer* hdTransfer;
    OffsetMapT offsetMapToSend;
    OffsetMapT loadOffsetToSend;
    bool isL3StorageEnabled{false};
    bool isRunning;
    bool isLoad{false};
    bool isInitialized{false};
    bool alreadyTrainOnce = false;     // 用于判断是否为predict模式
    bool isBackUpTrainStatus = false;  // whether the train state has been backed up

    void TrainTask(TaskType type);

    void EvalTask(TaskType type);

    void SendUniqKeysAndRestoreVecHBM(const EmbBaseInfo& info, const unique_ptr<vector<Tensor>>& infoVecs,
                                      bool isGrad) const;

    void InitEmbeddingCache(const vector<EmbInfo>& embInfos);

    void InitDataPipelineForDDR(const string& embName);

    void InitDataPipelineForL3Storage(const string& embName, int extEmbeddingSize);

    void JoinEmbeddingCacheThread();

    void HandleEosCase(const EmbBaseInfo& info, bool& remainBatchOut);

    bool EmbeddingReceiveDDR(const EmbTaskInfo& info, float*& ptr, vector<float*>& swapOutAddrs);

    void EmbeddingUpdateDDR(const EmbTaskInfo& info, const float* embPtr, vector<float*>& swapOutAddrs);

    bool EmbeddingLookUpDDR(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    void EmbeddingSendDDR(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    bool EmbeddingReceiveL3Storage(const EmbTaskInfo& info, float*& ptr, vector<float*>& swapOutAddrs, int64_t& dims0);

    void EmbeddingUpdateL3Storage(const EmbTaskInfo& info, float* embPtr, vector<float*>& swapOutAddrs, int64_t& dims0);

    bool EmbeddingLookUpL3Storage(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    void EmbeddingSendL3Storage(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    void CreateEmbeddingLookUpAndSendThread(int index, const EmbInfo& embInfo, int channelId);

    void CreateEmbeddingReceiveAndUpdateThread(int index, const EmbInfo& embInfo, int channelId);

    void HandleDataSwapForL3Storage(const EmbBaseInfo& info, vector<uint64_t>& swapInKeys,
                                    vector<uint64_t>& swapOutKeys);

    bool BuildH2DEmbedding(const EmbTaskInfo& info, vector<Tensor>& h2dEmb);

    vector<uint64_t> GetUniqueKeys(const EmbBaseInfo& info, bool& remainBatchOut);

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
};
}  // namespace MxRec
#endif  // MX_REC_EMB_MGMT_H

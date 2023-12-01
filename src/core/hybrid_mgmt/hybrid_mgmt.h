/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#ifndef MX_REC_EMB_MGMT_H
#define MX_REC_EMB_MGMT_H

#include <array>
#include <vector>
#include <memory>

#include "absl/container/flat_hash_map.h"

#include "utils/common.h"
#include "utils/config.h"

#include "host_emb/host_emb.h"
#include "emb_hashmap/emb_hashmap.h"
#include "hd_transfer/hd_transfer.h"
#include "ssd_cache/cache_manager.h"
#include "hybrid_mgmt_block.h"

namespace MxRec {
    using namespace std;
    using namespace tensorflow;

    enum class TaskType {
        HBM,
        DDR
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

        bool Save(const string savePath);

        bool Load(const string& loadPath);

        OffsetT SendHostMap(const string tableName);

        void ReceiveHostMap(AllKeyOffsetMapT receiveKeyOffsetMap);

        void Start();

        void StartThreadForHBM();

        void StartThreadForDDR();

        void Destroy();

        bool ParseKeys(int channelId, int& batchId);

        bool ParseKeysHBM(int channelId, int& batchId);

        bool ProcessEmbInfo(const std::string& embName, int batchId, int channelId, bool& remainBatchOut);

        void EmbHDTrans(const int channelId, const int batchId);

        bool Evict();

        void NotifyBySessionRun(int channelID) const;

        void CountStepBySessionRun(int channelID, int steps) const;

        int64_t GetTableSize(const string& embName) const;

        int64_t GetTableCapacity(const string& embName) const;

    GTEST_PRIVATE:

        void SetFeatureTypeForLoad(vector<CkptFeatureType>& loadFeatures);

        bool IsLoadDataMatches(const EmbMemT& loadHostEmbs, const EmbInfo& setupHostEmbs, size_t& embTableCount) const;

        void EvictKeys(const string& embName, const vector<emb_key_t>& keys);

        void InitRankInfo(RankInfo& rankInfo, const vector<EmbInfo>& embInfos) const;

        void EvictSSDKeys(const string& embName, const vector<emb_key_t>& keys) const;

        void PrepareDDRData(const std::string& embTableName, EmbHashMapInfo& embHashMap,
                            const vector<emb_key_t> &keys, int channelId, int batchId) const;

        int GetStepFromPath(const string& loadPath) const;

        static void AddCacheManagerTraceLog(CkptData& saveData);

        void RestoreFreq4Save(CkptData& saveData) const;
    private:
        int currentBatchId;
        int trainBatchId = 0; // 0-199, 200-
        int getInfoBatchId; // 0-199, 200-
        int sendBatchId;
        HybridMgmtBlock* hybridMgmtBlock;
        vector<EmbInfo> mgmtEmbInfo;
        RankInfo mgmtRankInfo;
        CacheManager* cacheManager;
        HostEmb* hostEmbs {};
        unique_ptr<EmbHashMap> hostHashMaps {};
        vector<std::unique_ptr<std::thread>> procThreads {};
        map<std::string, std::vector<emb_key_t>> evictKeyMap {};
        HDTransfer *hdTransfer;
        OffsetMapT offsetMapToSend;
        bool isSSDEnabled { false };
        bool isRunning;
        bool isLoad { false };
        bool isInitialized { false };

        void TrainTask(TaskType type);

        void EvalTask(TaskType type);

        bool EndBatch(int batchId, int channelId) const;

        void EmbHDTransWrap(int channelId, const int& batchId, int start);

        bool LoadMatchesDDRSetup(const CkptData& loadData);

        void HandlePrepareDDRDataRet(TransferRet prepareSSDRet) const;

        void SendUniqKeysAndRestoreVecHBM(int channelId, int& batchId, const EmbInfo &embInfo,
                                          const unique_ptr<vector<Tensor>> &infoVecs) const;

        void SendUniqKeysAndRestoreVecDDR(const string &embName, int &batchId, int &channelId, DDRParam &ddrParam);
    };
}
#endif // MX_REC_EMB_MGMT_H

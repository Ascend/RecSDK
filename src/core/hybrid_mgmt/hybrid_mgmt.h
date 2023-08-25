/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#ifndef MX_REC_EMB_MGMT_H
#define MX_REC_EMB_MGMT_H

#include <csignal>

#include <pthread.h>

#include <array>
#include <vector>
#include <memory>

#include "absl/container/flat_hash_map.h"

#include "utils/common.h"
#include "utils/singleton.h"

#include "host_emb/host_emb.h"
#include "emb_hashmap/emb_hashmap.h"
#include "hd_transfer/hd_transfer.h"
#include "key_process/key_process.h"

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

        key_offset_map_t SendHostMap(const string tableName);

        void ReceiveHostMap(all_key_offset_map_t keyOffsetMap);

        void Start();

        void InsertThreadForHBM();

    void Destroy()
    {
        if (!isRunning) {
            return;
        }
        // 先发送停止信号mgmt，先停止新lookup查询, 解除queue的限制防止卡住
        isRunning = false;

        // 先发送停止信号给preprocess，用于停止查询中lookup卡住状态
        preprocess->isRunning = false;
        // 停止hdTransfer，用于停止mgmt的recv中卡住状态
        hdTransfer->Destroy();
        for (auto& t : procThreads) {
            t->join();
        }
        if (hostEmbs != nullptr) {
            hostEmbs->Join(TRAIN_CHANNEL_ID);
            hostEmbs->Join(EVAL_CHANNEL_ID);
            hostEmbs = nullptr;
        }
        procThreads.clear();
        // 停止预处理
        if (preprocess != nullptr) {
            preprocess->Destroy();
            preprocess = nullptr;
        }
    };

        bool ParseKeys(int channelId, int& batchId);

        bool ParseKeysHBM(int channelId, int& batchId);

        bool ProcessEmbInfo(const std::string& embName, int batchId, int channelId, int iBatch, bool& remainBatchOut);

        void EmbHDTrans(const int channelId, const int batchId);

        bool Evict();

        void EvictKeys(const string& embName, const vector<emb_key_t>& keys);

        bool IsLoadDataMatches(emb_mem_t* loadHostEmbs, EmbInfo* setupHostEmbs, size_t& embTableCount);

    private:
        bool InitKeyProcess(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos,
                            const vector<ThresholdValue>& thresholdValues, int seed);

        void InitRankInfo(RankInfo& rankInfo, const vector<EmbInfo>& embInfos);

    private:
        int currentBatchId;
        int trainBatchId = 0; // 0-199, 200-
        int getInfoBatchId; // 0-199, 200-
        int sendBatchId;
        vector<EmbInfo> mgmtEmbInfo;
        RankInfo mgmtRankInfo;
        unique_ptr<HostEmb> hostEmbs {};
        unique_ptr<EmbHashMap> hostHashMaps {};
        vector<std::unique_ptr<std::thread>> procThreads {};
        map<std::string, std::vector<emb_key_t>> evictKeyMap {};
        KeyProcess *preprocess;
        HDTransfer *hdTransfer;
        bool isRunning;
        bool isLoad { false };

        void TaskForTrain(TaskType type);
        void TaskForEval(TaskType type);
        bool TrainTask(TaskType type);
        bool EvalTask(TaskType type);

        bool EndBatch(int batchId, int channelId) const;

        void EmbHDTransWrap(int channelId, const int& batchId, int start, int iBatch);

        bool LoadMatchesDDRSetup(const CkptData& loadData);
    };
}
#endif // MX_REC_EMB_MGMT_H

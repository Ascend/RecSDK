/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#ifndef MX_REC_KEY_PROCESS_H
#define MX_REC_KEY_PROCESS_H

#include <vector>
#include <deque>
#include <queue>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <iostream>
#include <shared_mutex>

#include <mpi.h>
#include <absl/container/flat_hash_map.h>

#include "ock_ctr_common/include/factory.h"
#include "ock_ctr_common/include/error_code.h"

#include "utils/common.h"
#include "utils/time_cost.h"
#include "utils/safe_queue.h"

#include "host_emb/host_emb.h"
#include "emb_table/emb_table.h"

#include "feature_admit_and_evict.h"
#include "hybrid_mgmt/hybrid_mgmt_block.h"

namespace MxRec {
    using namespace std;
    using namespace ock::ctr;

    template<class T>
    struct Cmp {
        bool operator()(const T& a, const T& b)
        {
            return get<int>(a) > get<int>(b); // batch id order
        }
    };

    template<class T>
    using heap_t = priority_queue<T, deque<T>, Cmp<T>>;

    template<class T>
    using info_list_t = map<emb_name_t, array<heap_t<T>, MAX_QUEUE_NUM>>;

    enum class ProcessedInfo {
        RESTORE,
        ALL2ALL,
        INVALID
    };

    class KeyProcess {
    public:
        bool Initialize(const RankInfo& rInfo, const vector<EmbInfo>& eInfos,
                       const vector<ThresholdValue>& thresholdValues = {}, int seed = 0);

        unique_ptr<vector<Tensor>> GetInfoVec(int batch, const string& embName, int channel, ProcessedInfo type);

        keys_t GetLookupKeys(int batch, const string& embName, int channel);

        int GetMaxStep(int channelId) const;

        int Start();

        auto GetMaxOffset() -> offset_mem_t;

        auto GetKeyOffsetMap() -> key_offset_mem_t;

        auto GetFeatAdmitAndEvict() -> FeatureAdmitAndEvict&;

        void LoadMaxOffset(offset_mem_t& loadData);

        void LoadKeyOffsetMap(key_offset_mem_t& loadData);

        void Destroy();

        void LoadSaveLock();

        void LoadSaveUnlock();

        void EvictKeys(const string& embName, const vector<emb_key_t>& keys);

        void SetupHotEmbUpdateStep();

        template <typename T>
        void GlobalUnique(T& lookupKeys, T& uniqueKeys, vector<int32_t>& restoreVecSec)
        {
            absl::flat_hash_map<emb_key_t, int32_t> umap;
            restoreVecSec.resize(lookupKeys.size(), -1);
            int32_t length = 0;

            for (size_t i = 0; i < lookupKeys.size(); ++i) {
                int64_t key = lookupKeys[i];
                if (rankInfo.useStatic && key == -1) {
                    continue;
                }
                auto result = umap.find(key);
                if (result == umap.end()) {
                    uniqueKeys.push_back(lookupKeys[i]);
                    umap[key] = length;
                    restoreVecSec[i] = length;
                    length++;
                } else {
                    restoreVecSec[i] = result->second;
                }
            }

            if (rankInfo.useStatic) {
                if (rankInfo.useDynamicExpansion) {
                    uniqueKeys.resize(lookupKeys.size(), 0);
                } else {
                    uniqueKeys.resize(lookupKeys.size(), -1);
                }
            }
        }

        bool isRunning { false };

        inline bool hasEmbName(const string &emb_name)
        {
            return embInfos.find(emb_name) != embInfos.end();
        };
    GTEST_PRIVATE:
        template<class T>
        T GetInfo(info_list_t<T>& list, int batch, const string& embName, int channel);

        RankInfo rankInfo;
        map<emb_name_t, EmbInfo> embInfos;
        MPI_Comm comm[MAX_CHANNEL_NUM][KEY_PROCESS_THREAD];
        std::mutex mut {};
        vector<std::unique_ptr<std::thread>> procThreads {};
        std::mutex loadSaveMut[MAX_CHANNEL_NUM][KEY_PROCESS_THREAD] {};
        info_list_t<lookup_key_t> lookupKeysList;
        list<unique_ptr<vector<Tensor>>> storage;
        info_list_t<tensor_info_t> infoList;
        info_list_t<tensor_info_t> all2AllList;
        map<emb_name_t, size_t> maxOffset {};
        map<emb_name_t, absl::flat_hash_map<emb_key_t, int64_t>> keyOffsetMap {};
        FeatureAdmitAndEvict m_featureAdmitAndEvict {};
        map<emb_name_t, std::vector<size_t>> evictPosMap {};
        map<emb_name_t, absl::flat_hash_map<emb_key_t, int>> hotKey {};
        map<emb_name_t, int> hotEmbTotCount;
        map<emb_name_t, EmbTable> embeddingTableMap {};

        FactoryPtr factory {};
        int hotEmbUpdateStep = HOT_EMB_UPDATE_STEP_DEFAULT;
        bool isWithFAAE;

        void InitHotEmbTotCount(const EmbInfo& info, const RankInfo& rInfo);

        void KeyProcessTask(int channel, int threadId);

        void KeyProcessTaskWithFastUnique(int channel, int threadId);

        bool KeyProcessTaskHelper(unique_ptr<emb_batch_t>& batch, int channel, int threadId);

        bool KeyProcessTaskHelperWithFastUnique(unique_ptr<emb_batch_t> &batch, UniquePtr& unique,
                                            int channel, int threadId);

        auto ProcessSplitKeys(const unique_ptr<emb_batch_t>& batch, int id,
                              vector<keys_t>& splitKeys) -> tuple<keys_t, vector<int>, vector<int>>;

        void GetUniqueConfig(UniqueConf& uniqueConf);

        void InitializeUnique(UniqueConf& uniqueConf, size_t& preBatchSize, bool& uniqueInitialize,
                                  const unique_ptr <emb_batch_t>& batch, UniquePtr& unique);

        void ProcessBatchWithFastUnique(const unique_ptr<emb_batch_t> &batch, UniquePtr& unique,
                                           int id, UniqueInfo& uniqueInfoOut);

        size_t GetKeySize(const unique_ptr<emb_batch_t> &batch);

        void All2All(vector<int>& sc, int id, int channel, KeySendInfo& keySendInfo,
                     All2AllInfo& all2AllInfoOut);

        auto HashSplit(const unique_ptr<emb_batch_t>& batch) const -> tuple<vector<keys_t>, vector<int32_t>>;

        auto HotHashSplit(const unique_ptr<emb_batch_t>& batch) -> tuple<vector<keys_t>, vector<int32_t>, vector<int>>;

        auto HashSplit_withFAAE(const unique_ptr<emb_batch_t>& batch) const
        -> tuple<vector<keys_t>, vector<int32_t>, vector<vector<uint32_t>>>;

        vector<int> GetScAll(const vector<int>& keyScLocal, int commId, int channel) const;

        void GetScAllForUnique(const vector<int>& keyScLocal, int commId, int channel, vector<int> &scAllOut) const;

        void Key2Offset(const emb_name_t& embName, keys_t& splitKey, int channel);

        void Key2OffsetDynamicExpansion(const emb_name_t& embName, keys_t& splitKey, int channel);

        unique_ptr<emb_batch_t> GetBatchData(int channel, int commId);

        void BuildRestoreVec(const unique_ptr<emb_batch_t>& batch, const vector<int>& rs,
                             vector<int>& restoreVec, int hotPosSize = 0) const;

        void SendA2A(const vector<int>& a2aInfo, const string& embName, int channel, int batch);

        void EvictDeleteDeviceEmb(const string& embName, const vector<emb_key_t>& keys);

        void EvictInitDeviceEmb(const string& embName, vector<size_t> offset);

        void UpdateHotMap(absl::flat_hash_map<emb_key_t, int>& keyCountMap, uint32_t count, bool refresh,
                          const string& embName);

        void UpdateHotMapForUnique(const keys_t &keySend, const vector<int32_t> &keyCount,
                                   uint32_t count, bool refresh, const string& embName);

        void HandleHotAndSendCount(const unique_ptr<emb_batch_t> &batch, UniqueInfo& uniqueInfoOut,
                                       KeySendInfo& keySendInfo, vector<int>& sc, vector<int>& splitSize);

        void PushResult(unique_ptr<emb_batch_t>& batch, unique_ptr<vector<Tensor>> tensors, keys_t& lookupKeys);

        void PushGlobalUniqueTensors(const unique_ptr<vector<Tensor>>& tensors, keys_t& lookupKeys, int chanel);

        void AddCountStartToHotPos(vector<keys_t>& splitKeys, vector<int>& hotPos, const vector<int>& hotPosDev,
                                   const unique_ptr<emb_batch_t>& batch);

        void ComputeHotPos(const unique_ptr<emb_batch_t> &batch, absl::flat_hash_map<emb_key_t, int> &hotMap,
                           vector<int> &hotPos, vector<int32_t> &restore, const int hotOffset);

        vector<uint32_t> GetCountRecv(const unique_ptr<emb_batch_t>& batch, int id,
                                      vector<vector<uint32_t>>& keyCount, vector<int> scAll, vector<int> ss);

        void HashSplitHelper(const unique_ptr <emb_batch_t>& batch, vector <keys_t>& splitKeys,
                             vector <int32_t>& restore, vector <int32_t>& hotPos,
                             vector <vector<uint32_t>>& keyCount);
    };
} // end namespace MxRec
#endif // MX_REC_KEY_PROCESS_H

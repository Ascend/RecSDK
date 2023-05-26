/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#ifndef MX_REC_KEY_PROCESS_H
#define MX_REC_KEY_PROCESS_H

#include <vector>
#include <thread>
#include <deque>
#include <queue>
#include <map>
#include <string>
#include <mpi.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include "utils/common.h"
#include "utils/safe_queue.h"
#include "utils/unique.h"
#include "utils/spinlock.h"
#include "utils/task_queue.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "host_emb/host_emb.h"
#include "feature_admit_and_evict.h"
#include "emb_table/emb_table.h"

namespace MxRec {
    using namespace std;

    constexpr int UNIQUE_BUCKET = 6;
    constexpr int MIN_UNIQUE_THREAD_NUM = 1;
    constexpr int MAX_UNIQUE_THREAD_NUM = 8;

    using a2a_info_t = vector<int>;
    using sharded_dedup = ShardedDedup<GroupMethod, UNIQUE_BUCKET>*;

    template <class T> struct Cmp {
        bool operator () (const T &a, const T &b)
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
        int Initialize(const RankInfo& rInfo, const vector<EmbInfo>& eInfos,
                       const vector<ThresholdValue>& thresholdValues = {}, bool ifLoad = false, int seed = 0);

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

        bool isRunning { false };

    GTEST_PRIVATE:

        template<class T>
        T GetInfo(array<info_list_t<T>, KEY_PROCESS_THREAD>& list, int batch, const string& embName, int channel);

        RankInfo rankInfo;
        map<emb_name_t, EmbInfo> embInfos;
        MPI_Comm comm[MAX_CHANNEL_NUM][KEY_PROCESS_THREAD];
        vector<std::thread> procThread {};
        std::mutex key2OffsetMut {};
        std::mutex loadSaveMut[MAX_CHANNEL_NUM][KEY_PROCESS_THREAD] {};
        std::mutex getInfoMut[KEY_PROCESS_THREAD] {};
        array<info_list_t<lookup_key_t>, KEY_PROCESS_THREAD> lookupKeysList;
        list<unique_ptr<vector<Tensor>>> storage[KEY_PROCESS_THREAD];
        array<info_list_t<tensor_info_t>, KEY_PROCESS_THREAD> infoList;
        array<info_list_t<tensor_info_t>, KEY_PROCESS_THREAD> all2AllList;
        map<emb_name_t, size_t> maxOffset {};
        map<emb_name_t, absl::flat_hash_map<emb_key_t, int64_t>> keyOffsetMap {};
        FeatureAdmitAndEvict m_featureAdmitAndEvict {};
        map<emb_name_t, std::vector<size_t>> evictPosMap {};
        map<emb_name_t, map<emb_key_t, int>> hotKey {};
        map<emb_name_t, int> hotEmbTotCount;
        map<emb_name_t, EmbTable> embeddingTableMap {};
        int hotEmbUpdateStep = HOT_EMB_UPDATE_STEP_DEFAULT;
        bool isWithFAAE;

        void InitHotEmbTotCount(const EmbInfo& info, const RankInfo& rInfo);

        auto GetSendCount(const string& name, const string& channelName, bool modifyGraph);

        void KeyProcessTask(int channel, int id);

        bool KeyProcessTaskHelper(unique_ptr<emb_batch_t>& batch, sharded_dedup unique_,
                                  int channel, int id, spdlog::stopwatch& sw);
        auto ProcessSplitKeys(const unique_ptr<emb_batch_t>& batch, int id,
                              vector<keys_t>& splitKeys) -> tuple<keys_t, vector<int>, vector<int>>;

        void ProcessBatchWithUniqueCompute(const unique_ptr<emb_batch_t> &batch, sharded_dedup unique, int id,
                                           UniqueInfo& uniqueInfo);

        size_t GetKeySize(const unique_ptr<emb_batch_t> &batch);

        void All2All(vector<int>& sc, int id, int channel, keys_t& keySend, vector<int32_t>& keyCount,
                     All2AllInfo& all2AllInfo);

        auto HashSplit(const unique_ptr<emb_batch_t>& batch) const -> tuple<vector<keys_t>, vector<int32_t>>;

        auto HotHashSplit(const unique_ptr<emb_batch_t>& batch) -> tuple<vector<keys_t>, vector<int32_t>, vector<int>>;

        auto HashSplit_withFAAE(const unique_ptr<emb_batch_t>& batch) const
        -> tuple<vector<keys_t>, vector<int32_t>, vector<vector<uint32_t>>>;
        void GetScAll(const vector<int>& keyScLocal, int commId, int channel, vector<int> &scAll) const;

        void Key2Offset(const emb_name_t& embName, keys_t& splitKey);

        unique_ptr<emb_batch_t> GetBatchData(int channel, int commId);

        void BuildRestoreVec(const unique_ptr<emb_batch_t>& batch, const vector<int>& rs,
                             vector<int>& restoreVec, int hotPosSize = 0) const;

        void SendA2A(const vector<int>& a2aInfo, const string& embName, int channel, int batch);

        void Key2OffsetInit(const emb_name_t& embName);

        void EvictDeleteDeviceEmb(const string& embName, const vector<emb_key_t>& keys);

        void EvictInitDeviceEmb(const string& embName, vector<size_t> offset);

        void UpdateHotMap(absl::flat_hash_map<emb_key_t, int>& keyCountMap, uint32_t count, bool refresh,
                          const string& embName);

        void PushResult(unique_ptr<emb_batch_t>& batch, unique_ptr<vector<Tensor>> tensors, keys_t& lookupKeys, int id);

        void AddCountStartToHotPos(vector<keys_t>& splitKeys, vector<int>& hotPos, const vector<int>& hotPosDev,
                                   const string& embName) const;

        vector<uint32_t> GetCountRecv(const unique_ptr<emb_batch_t>& batch, int id,
                                      vector<vector<uint32_t>>& keyCount, vector<int> scAll, vector<int> ss);
    };
}


#endif // MX_REC_KEY_PROCESS_H

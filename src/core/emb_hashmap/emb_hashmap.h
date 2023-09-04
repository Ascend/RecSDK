/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#ifndef MX_REC_EMB_HASHMAP_H
#define MX_REC_EMB_HASHMAP_H

#include <vector>
#include <memory>
#include <array>
#include "absl/container/flat_hash_map.h"
#include "host_emb/host_emb.h"
#include "ssd_cache/cache_manager.h"
#include "utils/common.h"
#include "utils/time_cost.h"

namespace MxRec {
    using namespace std;

    class EmbHashMap {
    public:
        EmbHashMap() = default;

        void Init(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos, bool ifLoad = false);

        void Process(const string& embName, std::vector<emb_key_t>& keys, size_t iBatch,
                     DDRParam& ddrParam, int channelId);

        void FindAndUpdateOffset(const string& embName, vector<emb_key_t>& keys, size_t currentBatchId,
                                 size_t keepBatchId, int channelId);

        auto GetHashMaps() -> absl::flat_hash_map<string, EmbHashMapInfo>;

        void LoadHashMap(absl::flat_hash_map<string, EmbHashMapInfo>& loadData);

        const std::vector<size_t>& GetMissingKeys(const string& embName)
        {
            return embHashMaps.at(embName).missingKeysHostPos;
        }

        void ClearMissingKeys(const string& embName)
        {
            embHashMaps.at(embName).missingKeysHostPos.clear();
        }

        void EvictDeleteEmb(const string& embName, const vector<emb_key_t>& keys);

        absl::flat_hash_map<string, EmbHashMapInfo> embHashMaps;

        void FindOffset(const string& embName, const vector<emb_key_t>& keys,
                        size_t currentBatchId, size_t keepBatchId, int channelId);

        bool FindOffsetHelper(const emb_key_t& key, EmbHashMapInfo& embHashMap, int channelId, size_t& offset);

        void UpdateBatchId(const vector<emb_key_t>& keys, size_t currentBatchId, size_t keySize,
                           EmbHashMapInfo& embHashMap) const;

        int FindSwapPosV2(const string& embName, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                          size_t keepBatchId);

        bool FindSwapPosOld(const string& embName, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                            size_t keepBatchId);

        std::vector<size_t>& GetEvictPos(const string& embName)
        {
            return embHashMaps.at(embName).evictPos;
        }

        bool isSSDEnabled { false };
        CacheManager* cacheManager;

    private:
        void FindAndUpdateBatchId(vector<emb_key_t>& keys, size_t currentBatchId, size_t keySize,
                                  EmbHashMapInfo& embHashMap) const;

        int32_t FindNewOffset(const emb_key_t& key, EmbHashMapInfo& embHashMap);

        void RefreshFreqInfoWithSwap(const string& embName, EmbHashMapInfo& embHashMap);

        void AddCacheManagerTraceLog(const string& embName, const EmbHashMapInfo& embHashMap) const;

        void AddKeyFreqInfo(const string& embTableName, const emb_key_t& key, RecordType type);

        RankInfo rankInfo;
        int swapId { 0 };
    };
}

#endif // MX_REC_EMB_HASHMAP_H

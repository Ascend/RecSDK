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

#ifndef MXREC_CACHE_MANAGER_H
#define MXREC_CACHE_MANAGER_H

#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <unordered_set>

#include "hd_transfer/hd_transfer.h"
#include "lfu_cache.h"
#include "ssd_engine/ssd_engine.h"
#include "utils/common.h"
#include "preprocess_mapper.h"
#include "ock_ctr_common/include/factory.h"
#include "l3_storage.h"

namespace MxRec {

    struct TableInfo {
        std::string name;
        size_t hostVocabSize;
        size_t devVocabSize;
        size_t& maxOffset;
        absl::flat_hash_map<emb_key_t, int64_t>& keyOffsetMap;
    };

    struct SwapOutInfo {
        vector<emb_cache_key_t> swapOutDDRKeys;
        vector<emb_cache_key_t> swapOutDDRAddrOffs;
        vector<emb_cache_key_t> swapOutL3StorageKeys;
        vector<emb_cache_key_t> swapOutL3StorageAddrOffs;
    };

    enum class TransferRet {
        TRANSFER_OK = 0, // 转移成功或无需处理
        TRANSFER_ERROR,
        L3STORAGE_SPACE_NOT_ENOUGH,
        DDR_SPACE_NOT_ENOUGH,
    };

    enum class TransferType {
        DDR_2_HBM = 0,
        DDR_2_EVICT,
        HBM_2_DDR,
        HBM_2_EVICT,
    };

    enum class RecordType {
        DDR = 0,
        NOT_DDR,
    };

    class CacheManager {
    public:
        CacheManager() = default;
        CacheManager(const CacheManager&) = delete;
        CacheManager& operator=(const CacheManager&) = delete;

        ~CacheManager();

        void Init(ock::ctr::EmbCacheManagerPtr embCachePtr, vector<EmbInfo>& mgmtEmbInfo,
                  shared_ptr<L3Storage> level3Storage);

        void Load(const std::vector<EmbInfo>& mgmtEmbInfo, int step,
                  map<string, unordered_set<emb_cache_key_t>>& trainKeySet);

        void Save(int step);

        bool IsKeyInL3Storage(const string& embTableName, emb_cache_key_t key);

        void EvictL3StorageEmbedding(const string& embTableName, const vector<emb_cache_key_t>& keys);

        void PutKey(const string& embTableName, const emb_key_t& key, RecordType type);

        void ProcessSwapOutKeys(const string& tableName, const vector<emb_cache_key_t>& swapOutKeys,
                                SwapOutInfo& info);

        void ProcessSwapInKeys(const string& tableName, const vector<emb_cache_key_t>& swapInKeys,
                               vector<emb_cache_key_t>& DDRToL3StorageKeys,
                               vector<emb_cache_key_t>& L3StorageToDDRKeys);

        void UpdateL3StorageEmb(string tableName, float* embPtr, uint32_t extEmbeddingSize, vector<emb_cache_key_t>& keys,
                                const vector<uint64_t>& swapOutL3StorageAddrOffs);

        void TransferDDR2L3Storage(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t>& keys,
                                   vector<float*>& addrs);

        void FetchL3StorageEmb2DDR(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t>& keys,
                                   const vector<float*>& addrs);

        int64_t GetTableUsage(const string& tableName);

        // DDR内每个表中emb数据频次缓存；map<embTableName, 频次缓存>
        unordered_map<std::string, LFUCache> ddrKeyFreqMap;
        // 每张表中非DDR内key的出现次数
        unordered_map<std::string, unordered_map<emb_cache_key_t, freq_num_t>> excludeDDRKeyCountMap;

        // 每一个table对应一个PreProcessMapper，预先推演HBM->DDR的情况
        std::unordered_map<std::string, PreProcessMapper> preProcessMapper;

        int preProcessStep = 0;
        int embeddingTaskStep = 0;
        std::mutex evictWaitMut;
        std::condition_variable evictWaitCond;

    private:
        struct EmbBaseInfo {
            uint64_t maxTableSize;
            vector<std::string> savePath;
            bool isExist;
        };

        void CreateL3StorageTableIfNotExist(const std::string& embTableName);

        unordered_map<std::string, EmbBaseInfo> embBaseInfos;

    GTEST_PRIVATE:
        shared_ptr<L3Storage> l3Storage;
        vector<std::thread> l3StorageEvictThreads;
        ock::ctr::EmbCacheManagerPtr embCache {};
    };
}

#endif // MXREC_CACHE_MANAGER_H

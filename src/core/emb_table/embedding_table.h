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

#ifndef MX_REC_EMBEDDING_TABLE_H
#define MX_REC_EMBEDDING_TABLE_H
#include <atomic>
#include <map>
#include <string>
#include <vector>

#include "utils/common.h"
#include "l3_storage/cache_manager.h"
#include "file_system/file_system_handler.h"

namespace MxRec {

class EmbeddingTable {
public:
    EmbeddingTable();
    EmbeddingTable(const EmbInfo& info, const RankInfo& rankInfo, int inSeed);
    virtual ~EmbeddingTable();

    /**
     * 从embedding表中查批量查找key
     * @param[in,out] keys 待查找的key，输出为找到的HBM偏移或者HBM地址
     * @param[in] channel 数据通道，主要区分train和eval
     */
    virtual void Key2Offset(std::vector<emb_key_t>& keys, int channel);

    /**
     * Dp模式下，从embedding表中查批量查找key，查不到即为error
     * @param[in,out] keys 待查找的key，输出为找到的HBM偏移或者HBM地址
     * @param[in] channel 数据通道，主要区分train和eval
     */
    virtual void Key2OffsetForDp(std::vector<emb_key_t>& keys, int channel);

    /**
     * 淘汰key,  配合GetEvictedKeys一起使用GetEvictedKeys
     * EvictKeys执行，通过GetEvictedKeys, GetEvictedKeys拿结果
     */
    virtual void EvictKeys(const std::vector<emb_cache_key_t>& keys);

    /**
     * 获取设备侧淘汰的key的偏移或者地址
     * @return  HBM模式为偏移, 动态扩容时为地址
     */
    virtual const std::vector<int64_t>& GetEvictedKeys();

    /**
     * 获取host侧淘汰的key的偏移。只有Host侧扩容DDR使用
     * @return host侧淘汰key的偏移
     */
    virtual const std::vector<int64_t>& GetHostEvictedKeys();

    virtual void EvictInitDeviceEmb();

    size_t GetMaxOffset();

    virtual int64_t capacity() const;

    virtual size_t size() const;

    absl::flat_hash_map<emb_key_t, int64_t> GetKeyOffsetMap();

    void SetFileSystemPtr(const string& savePath);

    void UnsetFileSystemPtr();

    virtual void Load(const string& savePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet);

    virtual void Save(const string& savePath, const int pythonBatchId);

    void MakeDir(const string& dirName);

    virtual void BackUpTrainStatus();

    virtual void RecoverTrainStatus();

    virtual vector<int64_t> GetDeviceOffset();

    vector<int64_t> GetLoadOffset();

    virtual void SetOptimizerInfo(OptimizerInfo& optimizerInfo);

    virtual void SetCacheManager(CacheManager* cacheManager);

    virtual TableInfo GetTableInfo();

    virtual void SetHDTransfer(HDTransfer *hdTransfer);

    virtual void SetEmbCache(ock::ctr::EmbCacheManagerPtr embCache);

    std::string name;
    size_t hostVocabSize;
    size_t devVocabSize;
    size_t ssdVocabSize;
    size_t maxOffset;
    absl::flat_hash_map<emb_key_t, int64_t> keyOffsetMap;
    absl::flat_hash_map<emb_key_t, int64_t> keyOffsetMapBackUp;
    std::vector<int64_t> evictDevPos;     // 记录HBM内被淘汰的key
    std::vector<int64_t> evictHostPos; // 记录Host内淘汰列表

#ifdef NDEBUG
protected:
#endif

    EmbeddingTable& operator=(const EmbeddingTable& table) = delete;

    size_t freeSize_;
    bool isDynamic_;
    std::mutex mut_;
    std::vector<InitializeInfo> initializeInfos_;
    EmbInfo embInfo_;
    size_t embSize_;
    size_t extEmbSize_;
    int seed_;
    std::atomic<int64_t> capacity_{0};
    size_t rankId_;
    size_t rankSize_;
    vector<int64_t> loadOffset;

    std::vector<size_t> missingKeysHostPos_; // 用于记录当前batch在host上需要换出的偏移
    CacheManager* cacheManager_;
    bool isSSDEnabled_ = false;

    unique_ptr<FileSystem> fileSystemPtr_;
};

}

#endif // MX_REC_EMBEDDING_TABLE_H

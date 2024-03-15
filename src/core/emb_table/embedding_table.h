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
#include <map>
#include <string>
#include <vector>

#include "utils/common.h"
#include "ssd_cache/cache_manager.h"

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
     * DDR模式使用
     */
    virtual void FindOffset(const vector<emb_key_t>& keys,
                            size_t currentBatchId, size_t keepBatchId, int channelId);

    virtual std::vector<int32_t> FindOffset(const vector<emb_key_t>& keys,
                                            size_t batchId, int channelId,
                                            std::vector<size_t>& swapPos);

    /**
     * 淘汰key,  配合GetEvictedKeys一起使用GetEvictedKeys
     * EvictKeys执行，通过GetEvictedKeys, GetEvictedKeys拿结果
     */
    virtual void EvictKeys(const std::vector<emb_key_t>& keys);

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

    void ClearMissingKeys();

    virtual const std::vector<size_t>& GetMissingKeys();

    absl::flat_hash_map<emb_key_t, int64_t> GetKeyOffsetMap();

    virtual void SetStartCount();

    virtual void ClearLookupAndSwapOffset();

    virtual void Load(const string& savePath);

    virtual void Save(const string& savePath);

    size_t GetDevVocabSize();

    size_t GetHostVocabSize();

    static void MakeDir(const string& dirName);

    virtual vector<int64_t> GetDeviceOffset();

    vector<int64_t> GetLoadOffset();

    virtual void SetOptimizerInfo(OptimizerInfo& optimizerInfo);

    virtual void SetCacheManager(CacheManager* cacheManager);

    void EnableSSD();

    virtual void RefreshFreqInfoWithSwap();

    virtual TableInfo GetTableInfo();

    std::string name;
    size_t hostVocabSize;
    size_t devVocabSize;
    size_t maxOffset;
    absl::flat_hash_map<emb_key_t, int64_t> keyOffsetMap;
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
    int64_t capacity_;
    size_t rankId_;
    size_t rankSize_;
    vector<int64_t> loadOffset;

    std::vector<size_t> missingKeysHostPos_; // 用于记录当前batch在host上需要换出的偏移
    CacheManager* cacheManager_;
    bool isSSDEnabled_ = false;
};

}

#endif // MX_REC_EMBEDDING_TABLE_H

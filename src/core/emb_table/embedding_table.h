/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: emb table
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#ifndef MX_REC_EMBEDDING_TABLE_H
#define MX_REC_EMBEDDING_TABLE_H
#include <map>
#include <string>
#include <vector>

#include "utils/common.h"

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

    virtual int Load(const string& savePath);

    virtual int Save(const string& savePath);

    size_t GetDevVocabSize();

    size_t GetHostVocabSize();

    static void MakeDir(const string& dirName);

#ifdef NDEBUG
protected:
#endif

    EmbeddingTable& operator=(const EmbeddingTable& table) = delete;

    std::string name_;
    size_t hostVocabSize_;
    size_t devVocabSize_;
    size_t freeSize_;
    size_t maxOffset_;
    bool isDynamic_;
    absl::flat_hash_map<emb_key_t, int64_t> keyOffsetMap_;
    std::vector<int64_t> evictPos_;     // 记录HBM内被淘汰的key
    std::vector<int64_t> evictHostPos_; // 记录Host内淘汰列表
    std::mutex mut_;
    std::vector<InitializeInfo> initializeInfos_;
    EmbInfo embInfo_;
    size_t embSize_;
    size_t extEmbSize_;
    int seed_;
    int64_t capacity_;
    size_t rankId_;

    std::vector<size_t> missingKeysHostPos_; // 用于记录当前batch在host上需要换出的偏移
};

}

#endif // MX_REC_EMBEDDING_TABLE_H

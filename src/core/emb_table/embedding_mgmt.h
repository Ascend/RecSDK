/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: EmbeddingMgmt管理类
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#ifndef MX_REC_EMBEDDING_MGMT_H
#define MX_REC_EMBEDDING_MGMT_H

#include <map>
#include <string>
#include <memory>
#include "utils/common.h"
#include "emb_table/embedding_table.h"

namespace MxRec {

/**
 * Embedding管理类
 */
class EmbeddingMgmt {
public:

    /**
     * @param[in] rInfo 从python侧传过了的rank信息
     * @param[in] eInfos 从python侧传过了的embedding表信息
     */
    void Init(const RankInfo& rInfo, const vector<EmbInfo>& eInfos,
              const vector<ThresholdValue>& thresholdValues = {}, int seed = 0);

    /**
     * 从embedding表中查批量查找key
     * @param[in] name embedding表名
     * @param[in,out] splitKey 待查找的key，输出为找到的HBM偏移或者HBM地址
     * @param[in] channel 数据通道，主要区分train和eval
     */
    void Key2Offset(const std::string& name, std::vector<emb_key_t>& keys, int channel);

    void FindOffset(const std::string& name, const vector<emb_key_t>& keys,
                    size_t currentBatchId, size_t keepBatchId, int channel);

    /**
     * 在指定的embedding表中淘汰key
     * @param[in] name embedding表名
     * @param[in] keys 待淘汰的key
     */
    void EvictKeys(const std::string& name, const vector<emb_key_t>& keys);

    /**
     * 在全部的embedding表中淘汰key
     * @param[in] keys 待淘汰的key
     */
    void EvictKeysCombine(const vector<emb_key_t>& keys);

    const std::vector<size_t>& GetMissingKeys(const std::string& name);

    void ClearMissingKeys(const std::string& name);

    void LoadMaxOffset(OffsetMemT& loadData);

    void LoadKeyOffsetMap(KeyOffsetMemT& loadData);

    size_t GetMaxOffset(const std::string& name);

    int64_t GetSize(const std::string &name);

    int64_t GetCapacity(const std::string &name);

    std::map<EmbNameT, size_t> GetMaxOffset();

    KeyOffsetMemT GetKeyOffsetMap();

    static EmbeddingMgmt* Instance();

    std::shared_ptr<EmbeddingTable> GetTable(const string& name);

    /**
     * 加载
     */
    int Load(const string& name, const string& filePath);

    int Load(const string& filePath);

    /**
     * 保存单个表
     */
    int Save(const string& name, const string& filePath);

    /**
     * 保存所有表
     */
    int Save(const string& filePath);

    void SetCacheManagerForEmbTable(CacheManager* cacheManager);

    void EnableSSD();

    EmbHashMemT GetEmbHashMaps();
private:

    EmbeddingMgmt();

    EmbeddingMgmt(const EmbeddingMgmt& mgmt) = delete;

    std::unordered_map<std::string, std::shared_ptr<EmbeddingTable>> embeddings;
};

}

#endif // MX_REC_EMBEDDING_MGMT_H

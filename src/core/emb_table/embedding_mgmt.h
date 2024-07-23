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
    void Init(const RankInfo& rInfo, const vector<EmbInfo>& eInfos, int seed = 0);

    /**
     * 从embedding表中查批量查找key
     * @param[in] name embedding表名
     * @param[in,out] splitKey 待查找的key，输出为找到的HBM偏移或者HBM地址
     * @param[in] channel 数据通道，主要区分train和eval
     */
    void Key2Offset(const std::string& name, std::vector<emb_key_t>& keys, int channel);

    /**
     * Dp模式下，从embedding表中查批量查找key，查不到即为error
     * @param[in] name embedding表名
     * @param[in,out] splitKey 待查找的key，输出为找到的HBM偏移或者HBM地址
     * @param[in] channel 数据通道，主要区分train和eval
     */
    void Key2OffsetForDp(const std::string& name, std::vector<emb_key_t>& keys, int channel);

    /**
     * 在指定的embedding表中淘汰key
     * @param[in] name embedding表名
     * @param[in] keys 待淘汰的key
     */
    void EvictKeys(const std::string& name, const vector<emb_cache_key_t>& keys);

    /**
     * 在全部的embedding表中淘汰key
     * @param[in] keys 待淘汰的key
     */
    void EvictKeysCombine(const vector<emb_cache_key_t>& keys);

    size_t GetMaxOffset(const std::string& name);

    int64_t GetSize(const std::string &name);

    int64_t GetCapacity(const std::string &name);

    std::map<EmbNameT, size_t> GetMaxOffset();

    KeyOffsetMemT GetKeyOffsetMap();

    static EmbeddingMgmt* Instance();

    /**
     * 加载单个表
     */
    void Load(const string& name, const string& filePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet);

    /**
     * 加载所有表
     */
    void Load(const string& filePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet);

    /**
     * 保存单个表
     */
    void Save(const string& name, const string& filePath);

    /**
     * 保存所有表
     */
    void Save(const string& filePath);

    /**
    * 获取所有表对应的DeviceOffsets，该偏移用于python侧保存embedding时抽取key对应的embedding
    */
    OffsetMapT GetDeviceOffsets();

    /**
    * 获取所有表对应的LoadOffsets，该偏移用于python侧加载embedding文件对应的行偏移，仅加载本卡key所对应的embedding
    */
    OffsetMapT GetLoadOffsets();

    /**
    * 设置某张表的优化器信息
    */
    void SetOptimizerInfo(const string& name, OptimizerInfo& optimizerInfo);

    void SetCacheManagerForEmbTable(CacheManager* cacheManager);

    void SetHDTransferForEmbTable(HDTransfer* hdTransfer);

    void SetEmbCacheForEmbTable(const ock::ctr::EmbCacheManagerPtr& embCache);
private:

    EmbeddingMgmt();

    EmbeddingMgmt(const EmbeddingMgmt& mgmt) = delete;

    std::unordered_map<std::string, std::shared_ptr<EmbeddingTable>> embeddings;
};

}

#endif // MX_REC_EMBEDDING_MGMT_H

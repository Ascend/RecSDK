/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#include "emb_hashmap.h"
#include <fstream>
#include <iomanip>
#include <mpi.h>

#include "checkpoint/checkpoint.h"
#include "hd_transfer/hd_transfer.h"
#include "hybrid_mgmt/hybrid_mgmt_block.h"
#include "utils/common.h"

using namespace MxRec;

void EmbHashMap::Init(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos, bool ifLoad)
{
#ifndef GTEST
    this->rankInfo = rankInfo;
    if (!ifLoad) {
        EmbHashMapInfo embHashMapInfo;
        LOG_INFO("init emb hash map from scratch");
        for (const auto& embInfo: embInfos) {
            embHashMapInfo.devOffset2Batch.resize(embInfo.devVocabSize);
            embHashMapInfo.devOffset2Key.resize(embInfo.devVocabSize);
            embHashMapInfo.hostVocabSize = embInfo.hostVocabSize;
            embHashMapInfo.devVocabSize = embInfo.devVocabSize;
            embHashMapInfo.currentUpdatePos = 0;
            fill(embHashMapInfo.devOffset2Batch.begin(), embHashMapInfo.devOffset2Batch.end(), -1);
            fill(embHashMapInfo.devOffset2Key.begin(), embHashMapInfo.devOffset2Key.end(), -1);
            embHashMaps[embInfo.name] = embHashMapInfo;

            LOG_TRACE("devOffset2Key, {}", VectorToString(embHashMaps.at(embInfo.name).devOffset2Key));
            LOG_TRACE("devOffset2Batch, {}", VectorToString(embHashMaps.at(embInfo.name).devOffset2Batch));
        }
    }
#endif
}

inline void ClearLookupAndSwapOffset(EmbHashMapInfo& embHashMap)
{
    embHashMap.swapPos.clear();
    embHashMap.lookUpVec.clear();
}

/// DDR模型下处理特征的offset、swap信息等
/// \param embName 表名
/// \param keys 查询向量
/// \param iBatch 预取数据处理计数
/// \param tmpDataOut 临时向量
/// \param channelId 通道索引（训练/推理）
void EmbHashMap::Process(const string& embName, vector<emb_key_t>& keys, size_t iBatch,
                         DDRParam& ddrParam, int channelId)
{
#ifndef GTEST
    EASY_FUNCTION(profiler::colors::Pink)
    TimeCost swapTimeCost;
    auto& embHashMap = embHashMaps.at(embName);
    embHashMap.devOffset2KeyOld.clear();
    embHashMap.oldSwap.clear();
    embHashMap.maxOffsetOld = embHashMap.maxOffset;

    auto keepBatch = swapId - iBatch; // 处理batch的次数，多个预取一起处理算一次
    bool findOffsetV2 = GetEnv("FIND_OFFSET_V2");

    LOG_DEBUG("FindOffset version:{}", findOffsetV2);

    // 找到所有key的偏移；dev和host需要交换的位置
    if (findOffsetV2) {
        FindAndUpdateOffset(embName, keys, swapId, keepBatch, channelId);
    } else {
        FindOffset(embName, keys, swapId, keepBatch, channelId);
    }
    LOG_DEBUG("FindOffset end");

    // 调用刷新频次数据方法
    RefreshFreqInfoWithSwap(embName, embHashMap);

    EASY_BLOCK("hostHashMaps->tdt")

    std::copy(embHashMap.lookUpVec.begin(), embHashMap.lookUpVec.end(), std::back_inserter(ddrParam.offsetsOut));

    // 构造查询向量tensor
    auto lookUpVecSize = static_cast<int>(embHashMap.lookUpVec.size());
    ddrParam.tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { lookUpVecSize }));

    auto lookupTensorData = ddrParam.tmpDataOut.back().flat<int32>();
    for (int i = 0; i < lookUpVecSize; i++) {
        lookupTensorData(i) = static_cast<int32_t>(embHashMap.lookUpVec[i]);
    }
    LOG_TRACE("lookupTensor, {}", VectorToString(embHashMap.lookUpVec));

    // 构造交换向量tensor
    auto swapSize = static_cast<int>(embHashMap.swapPos.size());
    ddrParam.tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { swapSize }));

    auto swapTensorData = ddrParam.tmpDataOut.back().flat<int32>();
    for (int i = 0; i < swapSize; i++) {
        swapTensorData(i) = static_cast<int>(embHashMap.swapPos[i]);
    }
    if (swapSize > 0) {
        LOG_DEBUG("swap num: {}", swapSize);
    }
    LOG_TRACE("swapTensor, {}", VectorToString(embHashMap.swapPos));
    // 清空本次记录的查询偏移和交换偏移
    ClearLookupAndSwapOffset(embHashMap);
    LOG_INFO("current ddr emb:{}, usage:{}/[{}+{}]", embName, embHashMap.maxOffset,
        embHashMap.devVocabSize, embHashMap.hostVocabSize);
    ddrParam.tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));
    auto swapLen = ddrParam.tmpDataOut.back().flat<int32>();
    swapLen(0) = swapSize;

    if (g_statOn) {
        LOG_INFO(STAT_INFO "channel_id {} batch_id {} rank_id {} swap_key_size {} swap_time_cost {}",
            channelId, swapId, rankInfo.rankId, swapSize, swapTimeCost.ElapsedMS());
    }
    
    swapId++;
    EASY_END_BLOCK
#endif
}

/*
 * 从embHashMaps获取key对应的位置，并更新devOffset2Batch
 */
#ifndef GTEST
void EmbHashMap::FindAndUpdateOffset(const string& embName, vector<emb_key_t>& keys,
                                     size_t currentBatchId, size_t keepBatchId, int channelId)
{
    EASY_FUNCTION()
    size_t keySize = keys.size();
    auto& embHashMap = embHashMaps.at(embName);
    FindAndUpdateBatchId(keys, currentBatchId, keySize, embHashMap);
    const int devVocabSize = static_cast<int>(embHashMap.devVocabSize);
    for (size_t i = 0; i < keySize; i++) {
        auto key = keys[i];
        if (key == -1) {
            continue;
        }
        auto& offset = embHashMap.lookUpVec[i];
        if (offset == INVALID_KEY_VALUE && channelId == TRAIN_CHANNEL_ID) {
            offset = FindNewOffset(key, embHashMap);
            if (offset < devVocabSize) {
                embHashMap.devOffset2KeyOld.emplace_back(offset, embHashMap.devOffset2Key[offset]);
                embHashMap.devOffset2Key[offset] = key;
                embHashMap.devOffset2Batch[offset] = static_cast<int>(currentBatchId);
            }
        }
        if (offset >= devVocabSize) {
            embHashMap.missingKeysHostPos.emplace_back(offset - embHashMap.devVocabSize);
            offset = FindSwapPosV2(embName, key, offset, currentBatchId, keepBatchId);
        }
    }
}

int32_t EmbHashMap::FindNewOffset(const emb_key_t& key, EmbHashMapInfo& embHashMap)
{
    int32_t offset;
    const auto& iter = embHashMap.hostHashMap.find(key);
    if (iter != embHashMap.hostHashMap.end()) { // 由于未全局去重，需要再次查询确保是新key
        offset = static_cast<int32_t>(iter->second);
    } else if (embHashMap.evictDevPos.size() != 0) { // 优先复用hbm表
        offset = static_cast<int32_t>(embHashMap.evictDevPos.back());
        embHashMap.hostHashMap[key] = offset;
        LOG_TRACE("ddr mode, dev evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, embHashMap.evictDevPos.size());
        embHashMap.evictDevPos.pop_back();
    } else if (embHashMap.evictPos.size() != 0) { // hbm不足，再复用ddr表
        offset = static_cast<int32_t>(embHashMap.evictPos.back());
        embHashMap.hostHashMap[key] = offset;
        LOG_TRACE("ddr mode, host evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, embHashMap.evictPos.size());
        embHashMap.evictPos.pop_back();
    } else {
        embHashMap.hostHashMap[key] = embHashMap.maxOffset;
        offset = static_cast<int32_t>(embHashMap.maxOffset);
        embHashMap.maxOffset++;
        if (embHashMap.maxOffset == embHashMap.devVocabSize) {
            LOG_INFO("start using host vocab!");
        }
        if (embHashMap.maxOffset > embHashMap.hostVocabSize + embHashMap.devVocabSize) {
            LOG_ERROR("hostVocabSize too small! dev:{} host:{}", embHashMap.devVocabSize,
                embHashMap.hostVocabSize);
            throw runtime_error("hostVocabSize too small");
        }
    }
    return offset;
}

void EmbHashMap::FindAndUpdateBatchId(vector<emb_key_t>& keys, size_t currentBatchId, size_t keySize,
                                      EmbHashMapInfo& embHashMap) const
{
    EASY_FUNCTION()
    bool findOffsetV3 = GetEnv("FIND_OFFSET_V3");
    for (size_t i = 0; i < keySize; i++) {
        int offset;
        auto& key = keys[i];
        if (key == -1) {
            continue;
        }
        const auto& iter = embHashMap.hostHashMap.find(key);
        if (iter != embHashMap.hostHashMap.end()) { // found
            if (findOffsetV3) {
                key = -1;
            }
            offset = static_cast<int>(iter->second);
            embHashMap.lookUpVec.emplace_back(offset); // convert to offset(current)

            if (offset < static_cast<int>(embHashMap.devVocabSize)) {
                embHashMap.devOffset2Batch[offset] = static_cast<int>(currentBatchId);
            }
        } else {
            embHashMap.lookUpVec.emplace_back(INVALID_KEY_VALUE);
        }
    }
}

auto EmbHashMap::GetHashMaps() -> absl::flat_hash_map<string, EmbHashMapInfo>
{
    LOG_DEBUG(HYBRID_BLOCKING + " start GetHashMaps");
    HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
    auto embHashMapsOld = embHashMaps;
    int checkResult = hybridMgmtBlock->CheckSaveEmbdMapValid();
    if (checkResult == 0) {
        // 检查是否需要回退
        return embHashMapsOld;
    }
    if (checkResult == 1) {
        // 回退一步
        for (auto& temp: embHashMapsOld) {
            auto &embHashMap = temp.second;
            for (auto &swapKeys: embHashMap.oldSwap) {
                emb_key_t oldKey = swapKeys.first;
                emb_key_t key = swapKeys.second;
                int tempOffset = static_cast<int>(embHashMap.hostHashMap[key]);
                embHashMap.hostHashMap[key] = embHashMap.hostHashMap[oldKey];
                embHashMap.hostHashMap[oldKey] = static_cast<int>(tempOffset);
            }
            embHashMap.maxOffset = embHashMap.maxOffsetOld;
            for (auto &Offset2Key: embHashMap.devOffset2KeyOld) {
                embHashMap.devOffset2Key[Offset2Key.first] = Offset2Key.second;
            }
        }
        return embHashMapsOld;
    }
    // 此时需要回退2步，无法满足此条件，保存的东西错误，直接回退
    if (not rankInfo.noDDR) {
        throw HybridMgmtBlockingException("EmbHashMap::GetHashMaps() ");
    }
    return embHashMapsOld;
}

void EmbHashMap::LoadHashMap(emb_hash_mem_t& loadData)
{
    embHashMaps = std::move(loadData);
}

/// 对HBM剩余空间和更新位置进行初始化
void EmbHashMapInfo::SetStartCount()
{
    currentUpdatePosStart = currentUpdatePos;
    freeSize = devVocabSize;
}

/// 判断HBM是否有剩余空间
/// \param i 查询向量的大小
/// \return
bool EmbHashMapInfo::HasFree(size_t i)
{
    return freeSize < i;
}

/*
* 删除淘汰key的映射关系，并将其offset更新到evictPos，待后续复用
*/
void EmbHashMap::EvictDeleteEmb(const string& embName, const vector<emb_key_t>& keys)
{
    EASY_FUNCTION()
    size_t keySize = keys.size();
    auto& embHashMap = embHashMaps.at(embName);
    vector<emb_key_t> evictHBMKeys;
    vector<emb_key_t> evictDDRKeys;
    for (size_t i = 0; i < keySize; i++) {
        size_t offset;
        auto key = keys[i];
        if (key == -1) {
            LOG_WARN("evict key equal -1!");
            continue;
        }
        const auto& iter = embHashMap.hostHashMap.find(key);
        if (iter != embHashMap.hostHashMap.end()) {
            offset = iter->second;
            embHashMap.hostHashMap.erase(iter);
            LOG_TRACE("evict embName {}, offset {}", embName, offset);
        } else {
            // 淘汰依据keyProcess中的history，hashmap映射关系创建于ParseKey；两者异步，造成淘汰的值在hashmap里可能未创建
            continue;
        }

        if (offset < embHashMap.devVocabSize) {
            embHashMap.devOffset2Batch[offset] = -1;
            embHashMap.devOffset2KeyOld.emplace_back(offset, embHashMap.devOffset2Key[offset]);
            embHashMap.devOffset2Key[offset] = -1;
            embHashMap.evictDevPos.emplace_back(offset);
            evictHBMKeys.emplace_back(key);
        } else {
            embHashMap.evictPos.emplace_back(offset);
            evictDDRKeys.emplace_back(key);
        }
    }
    if (isSSDEnabled) {
        cacheManager->RefreshFreqInfoCommon(embName, evictHBMKeys, TransferType::HBM_2_EVICT);
        cacheManager->RefreshFreqInfoCommon(embName, evictDDRKeys, TransferType::DDR_2_EVICT);
    }

    LOG_INFO("ddr EvictDeleteEmb, emb: [{}], hostEvictSize: {}, devEvictSize: {}",
        embName, embHashMap.evictPos.size(), embHashMap.evictDevPos.size());
    LOG_TRACE("hostHashMap, {}", MapToString(embHashMaps[embName].hostHashMap));
}

/// 从embHashMaps获取key对应的位置，构造查询向量；更新devOffset2Batch；记录dev与host需要交换的偏移
/// \param embName 表名
/// \param keys 查询向量
/// \param currentBatchId 已处理的batch数
/// \param keepBatchId 处理batch的次数，多个预取一起处理算一次
/// \param channelId 通道索引（训练/推理）
void EmbHashMap::FindOffset(const string& embName, const vector<emb_key_t>& keys,
                            size_t currentBatchId, size_t keepBatchId, int channelId)
{
    EASY_FUNCTION()
    size_t keySize = keys.size();
    auto& embHashMap = embHashMaps.at(embName);
    UpdateBatchId(keys, currentBatchId, keySize, embHashMap);
    for (size_t i = 0; i < keySize; i++) {
        auto key = keys[i];
        if (key == -1) {
            embHashMap.lookUpVec.emplace_back(INVALID_KEY_VALUE);
            continue;
        }
        size_t offset;
        auto isOffsetValid = FindOffsetHelper(key, embHashMap, channelId, offset);
        if (!isOffsetValid) {
            embHashMap.lookUpVec.emplace_back(INVALID_KEY_VALUE);
            continue;
        }

        if (offset < embHashMap.devVocabSize) {
            // 偏移小于等于HBM容量：直接放入查询向量；更新偏移之前关联的key和当前关联的key
            embHashMap.lookUpVec.emplace_back(offset);
            embHashMap.devOffset2KeyOld.emplace_back(offset, static_cast<int>(embHashMap.devOffset2Key[offset]));
            embHashMap.devOffset2Key[offset] = key;
            AddKeyFreqInfo(embName, key, RecordType::NOT_DDR);
        } else {
            // 偏移大于HBM容量：记录在host emb上的偏移；找到需要交换的HBM偏移
            embHashMap.missingKeysHostPos.emplace_back(offset - embHashMap.devVocabSize);
            FindSwapPosOld(embName, key, offset, currentBatchId, keepBatchId);
            AddKeyFreqInfo(embName, key, RecordType::DDR);
        }
    }
    if (currentBatchId == 0) {
        LOG_INFO("max offset {}", embHashMap.maxOffset);
    }
    LOG_TRACE("hostHashMap, {}", MapToString(embHashMaps[embName].hostHashMap));
}


/// 查找key对应的偏移；1. 已在hash map中，直接返回对应的offset；2. 开启淘汰的情况下，复用淘汰的位置；3. 没有则新分配
/// \param key 输入特征
/// \param embHashMap hash map实例
/// \param channelId 通道索引（训练/推理）
/// \param offset 未初始化变量，用于记录
/// \return
bool EmbHashMap::FindOffsetHelper(const emb_key_t& key, EmbHashMapInfo& embHashMap, int channelId, size_t& offset)

{
    const auto& iter = embHashMap.hostHashMap.find(key);
    if (iter != embHashMap.hostHashMap.end()) {
        offset = iter->second;
        LOG_TRACE("devVocabSize, {} , offset , {}", embHashMap.devVocabSize, offset);
    } else if (embHashMap.evictDevPos.size() != 0 && channelId == TRAIN_CHANNEL_ID) { // 优先复用hbm表
        offset = embHashMap.evictDevPos.back();
        embHashMap.hostHashMap[key] = offset;
        LOG_TRACE("ddr mode, dev evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, embHashMap.evictDevPos.size());
        embHashMap.evictDevPos.pop_back();
    } else if (embHashMap.evictPos.size() != 0 && channelId == TRAIN_CHANNEL_ID) { // hbm不足，再复用ddr表
        offset = embHashMap.evictPos.back();
        embHashMap.hostHashMap[key] = offset;
        LOG_TRACE("ddr mode, host evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, embHashMap.evictPos.size());
        embHashMap.evictPos.pop_back();
    } else {
        if (channelId == TRAIN_CHANNEL_ID) {
            embHashMap.hostHashMap[key] = embHashMap.maxOffset;
            offset = embHashMap.maxOffset;
            embHashMap.maxOffset++;
            if (embHashMap.maxOffset == embHashMap.devVocabSize) {
                LOG_INFO("start using host vocab!");
            }
            if (embHashMap.maxOffset > embHashMap.hostVocabSize + embHashMap.devVocabSize) {
                LOG_ERROR("hostVocabSize too small! dev:{} host:{}", embHashMap.devVocabSize, embHashMap.hostVocabSize);
                throw runtime_error("hostVocabSize too small");
            }
        } else {
            return false;
        }
    }
    return true;
}

/// 更新HBM中的key相应offset最近出现的batch步数，用于跟踪哪些offset是最近在使用的
/// \param keys 查询向量
/// \param currentBatchId 已处理的batch数
/// \param keySize 查询向量长度
/// \param embHashMap hash map实例
void EmbHashMap::UpdateBatchId(const vector<emb_key_t>& keys, size_t currentBatchId, size_t keySize,
                               EmbHashMapInfo& embHashMap) const
{
    for (size_t i = 0; i < keySize; i++) {
        size_t offset;
        auto key = keys[i];
        if (key == -1) {
            continue;
        }
        const auto& iter = embHashMap.hostHashMap.find(key);
        if (iter != embHashMap.hostHashMap.end()) {
            offset = iter->second;

            LOG_TRACE("key will be used, {} , offset , {}", key, offset);
            if (offset < embHashMap.devVocabSize) {
                embHashMap.devOffset2Batch[offset] = static_cast<int>(currentBatchId);
            }
        }
    }
}

/*
 * 利用devOffset2Batch上key最近使用的batchId，来选择需要淘汰的key，记录淘汰位置和device侧所需的keys
 */
int EmbHashMap::FindSwapPosV2(const string& embName, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                              size_t keepBatchId)
{
    bool notFind = true;
    auto& embHashMap = embHashMaps.at(embName);
    int newDevOffset;
    while (notFind) {
        if (embHashMap.devOffset2Batch[embHashMap.currentUpdatePos] < static_cast<int>(keepBatchId)) {
            embHashMap.devOffset2Batch[embHashMap.currentUpdatePos] = static_cast<int>(currentBatchId);
            embHashMap.swapPos.emplace_back(embHashMap.currentUpdatePos);
            newDevOffset = static_cast<int>(embHashMap.currentUpdatePos);
            embHashMap.hostHashMap[key] = embHashMap.currentUpdatePos;
            embHashMap.devOffset2KeyOld.emplace_back(embHashMap.currentUpdatePos,
                                                     embHashMap.devOffset2Key[embHashMap.currentUpdatePos]);
            auto& oldKey = embHashMap.devOffset2Key[embHashMap.currentUpdatePos];
            embHashMap.oldSwap.emplace_back(oldKey, key);
            embHashMap.hostHashMap[oldKey] = hostOffset;
            oldKey = key;
            notFind = false;
        }
        embHashMap.currentUpdatePos++;
        embHashMap.freeSize--;
        if (embHashMap.currentUpdatePos == embHashMap.devVocabSize) {
            embHashMap.currentUpdatePos = 0;
        }
        if (embHashMap.currentUpdatePos == embHashMap.currentUpdatePosStart) {
            LOG_ERROR("devVocabSize is too small");
            throw runtime_error("devVocabSize is too small");
        }
    }
    return newDevOffset;
}

/// 利用devOffset2Batch上key最近使用的batchId，来选择需要淘汰的key，记录淘汰位置和device侧所需的keys
/// \param embName 表名
/// \param key 输入特征
/// \param hostOffset 全局偏移
/// \param currentBatchId 已处理的batch数
/// \param keepBatchId 处理batch的次数，多个预取一起处理算一次
/// \return 是否找到需要交换的位置
bool EmbHashMap::FindSwapPosOld(const string& embName, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                                size_t keepBatchId)
{
    bool notFind = true;
    auto& embHashMap = embHashMaps.at(embName);
    while (notFind) {
        // 找到本次预取之前的偏移（保证所有预取batch的key都在HBM中）
        if (embHashMap.devOffset2Batch[embHashMap.currentUpdatePos] < static_cast<int>(keepBatchId)) {
            embHashMap.devOffset2Batch[embHashMap.currentUpdatePos] = static_cast<int>(currentBatchId);
            embHashMap.swapPos.emplace_back(embHashMap.currentUpdatePos); // 记录需要被换出的HBM偏移
            embHashMap.lookUpVec.emplace_back(embHashMap.currentUpdatePos); // 交换的位置就是该key查询的偏移
            embHashMap.hostHashMap[key] = embHashMap.currentUpdatePos;  // 更新key对应的HBM偏移
            // 记录HBM偏移之前的key
            embHashMap.devOffset2KeyOld.emplace_back(embHashMap.currentUpdatePos,
                                                     embHashMap.devOffset2Key[embHashMap.currentUpdatePos]);
            auto& oldKey = embHashMap.devOffset2Key[embHashMap.currentUpdatePos];
            embHashMap.oldSwap.emplace_back(oldKey, key); // 记录交换的两个key
            embHashMap.hostHashMap[oldKey] = hostOffset; // 更新被替换的key的偏移
            oldKey = key;
            notFind = false;
        }
        embHashMap.currentUpdatePos++; // 查找位置+1
        embHashMap.freeSize--; // HBM可用空间-1

        // 遍历完一遍整个HBM表后，从头开始遍历
        if (embHashMap.currentUpdatePos == embHashMap.devVocabSize) {
            embHashMap.currentUpdatePos = 0;
        }

        // 已经找完了整个HBM空间，没有找到可用位置，表示HBM空间不足以放下整个batch（预取batch数）的key，无法正常执行训练，固运行时错误退出
        if (embHashMap.currentUpdatePos == embHashMap.currentUpdatePosStart) {
            LOG_ERROR("devVocabSize is too small");
            throw runtime_error("devVocabSize is too small");
        }
    }
    return true;
}

/// HBM-DDR换入换出时刷新频次信息
/// \param embName emb表名
/// \param embHashMap emb hash map
void EmbHashMap::RefreshFreqInfoWithSwap(const string& embName, EmbHashMapInfo& embHashMap)
{
    if (!isSSDEnabled) {
        return;
    }
    // 换入换出key列表，元素为pair: pair<oldKey, key> oldKey为从HBM移出的key, key为从DDR移出的key
    auto& oldSwap = embHashMap.oldSwap;
    LOG_DEBUG("RefreshFreqInfoWithSwap:oldSwap Size:{}", oldSwap.size());
    vector<emb_key_t> enterDDRKeys;
    vector<emb_key_t> leaveDDRKeys;
    for (auto keyPair : oldSwap) {
        enterDDRKeys.emplace_back(keyPair.first);
        leaveDDRKeys.emplace_back(keyPair.second);
    }
    cacheManager->RefreshFreqInfoCommon(embName, enterDDRKeys, TransferType::HBM_2_DDR);
    cacheManager->RefreshFreqInfoCommon(embName, leaveDDRKeys, TransferType::DDR_2_HBM);

    AddCacheManagerTraceLog(embName, embHashMap);
}

/// 记录日志：HBM和DDR换入换出后，比较hostHashMap中DDR内key和表对应的lfuCache对象中的key内容
void EmbHashMap::AddCacheManagerTraceLog(const string& embTableName, const EmbHashMapInfo& embHashMap) const
{
    if (Log::GetLevel() != Log::TRACE) {
        return;
    }
    auto& hostMap = embHashMap.hostHashMap;
    auto& devSize = embHashMap.devVocabSize;
    auto& lfu = cacheManager->ddrKeyFreqMap[embTableName];
    const auto& lfuTab = lfu.GetFreqTable();
    if (lfuTab.empty()) {
        return;
    }
    size_t tableKeyInDdr = 0;
    vector<emb_key_t> ddrKeys; // 获取hostHashMap中保存在DDR的key
    for (const auto& item : hostMap) {
        if (item.second < devSize) {
            continue;
        }
        ddrKeys.emplace_back(item.first);
        ++tableKeyInDdr;
    }
    vector<emb_key_t> lfuKeys;
    for (const auto& it : lfuTab) {
        lfuKeys.emplace_back(it.first);
    }
    std::sort(ddrKeys.begin(), ddrKeys.end());
    std::sort(lfuKeys.begin(), lfuKeys.end());
    std::string ddrKeysString = VectorToString(ddrKeys);
    std::string lfuKeysString = VectorToString(lfuKeys);
    if (ddrKeysString != lfuKeysString) {
        LOG_ERROR("swap HBM with DDR step error, key string not equal, ddrKeysString:{}, lfuKeysString:{}",
            ddrKeysString, lfuKeysString);
    } else {
        LOG_INFO("swap HBM with DDR step OK, table:{}, ddrKeysString == lfuKeysString, string length:{}",
            embTableName, lfuKeysString.length());
    }

    LOG_INFO("swap HBM with DDR step end, table:{}, tableKeyInDdr:{}, tableKeyInLfu:{}",
        embTableName, tableKeyInDdr, lfu.keyTable.size());
}

/// 记录key频次数据
/// \param embTableName emb表名
/// \param key key
/// \param type 记录类型枚举
void EmbHashMap::AddKeyFreqInfo(const string& embTableName, const emb_key_t& key, RecordType type)
{
    if (!isSSDEnabled) {
        return;
    }
    cacheManager->PutKey(embTableName, key, type);
}

#endif

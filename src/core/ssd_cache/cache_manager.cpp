/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: ssd cache module
 * Author: MindX SDK
 * Date: 2023/8/10
 */
#include "cache_manager.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "utils/common.h"
#include "utils/time_cost.h"

using namespace MxRec;

inline void CacheManager::GetExternalKeys(EmbHashMapInfo &embHashMap, vector<emb_key_t> &externalKeys,
                                          vector<emb_key_t> &internalKeys, const vector<emb_key_t> &keys)
{
    for (const emb_key_t key : keys) {
        if (embHashMap.hostHashMap.find(key) == embHashMap.hostHashMap.end()) {
            externalKeys.emplace_back(key);
        } else {
            internalKeys.emplace_back(key);
        }
    }
}

void CacheManager::AddDebugAndTraceLog(size_t batchKeySize, vector<emb_key_t> &externalKeys,
                                       vector<emb_key_t> &externalSSDKeys)
{
    LOG_DEBUG("TransferDDREmbWithSSD: batchKeySize:{}, externalKeys size:{}, externalSSDKeys size:{}",
        batchKeySize, externalKeys.size(), externalSSDKeys.size());
    LOG_TRACE("TransferDDREmbWithSSD: externalKeys:{}, externalSSDKeys:{}",
        VectorToString(externalKeys), VectorToString(externalSSDKeys));
}

/// 去重和过滤无效key
/// \param originalKeys 原有keys
/// \param keys 处理后的keys
void CacheManager::HandleRepeatAndInvalidKey(const vector<emb_key_t>& originalKeys, vector<emb_key_t>& keys)
{
    // 去重并保持原key的顺序 结果可测试
    unordered_set<emb_key_t> keySet;
    for (auto& key : originalKeys) {
        if (key == INVALID_KEY_VALUE) {
            continue;
        }
        if (keySet.find(key) == keySet.end()) {
            keySet.emplace(key);
            keys.emplace_back(key);
        }
    }
}

/// DDR与SSD数据转移，使DDR内剩余空间能放置当前批次key
/// \param embTableName emb表名
/// \param embHashMap emb表
/// \param originalKeys 当前批次key
/// \param channelId 通道id
/// \return 转移结果枚举
TransferRet CacheManager::TransferDDREmbWithSSD(const std::string& embTableName, EmbHashMapInfo& embHashMap,
                                                const vector<emb_key_t>& originalKeys, int channelId)
{
    vector<emb_key_t> keys; // 去重和删除无效key
    HandleRepeatAndInvalidKey(originalKeys, keys);
    // 区分HBM+DDR内key，和HBM+DDR外的key(新key或保存在SSD中的key)
    vector<emb_key_t> externalKeys;
    vector<emb_key_t> internalKeys;
    GetExternalKeys(embHashMap, externalKeys, internalKeys, keys);
    if (externalKeys.empty()) { return TransferRet::TRANSFER_OK; }

    // 判断剩余内存空间是否足够; 可用内存空间计算：HBM+DDR-已占用; 若是训练，再加DDR已淘汰;
    // SSD仅与DDR交互，不考虑HBM淘汰位置；由于maxOffset比实际使用大1，所以虽然从0开始也不用再减1
    size_t ddrAvailableSize = embHashMap.devVocabSize + embHashMap.hostVocabSize - embHashMap.maxOffset;
    if (channelId == TRAIN_CHANNEL_ID) {
        ddrAvailableSize += embHashMap.evictPos.size();
    }
    LOG_DEBUG("TransferDDREmbWithSSD, maxOffset:{}, evictPos size:{}, ddrAvailableSize:{}",
        embHashMap.maxOffset, embHashMap.evictPos.size(), ddrAvailableSize);
    CreateSSDTableIfNotExist(embTableName);

    // 调用ssdEngine查询当前批次key中保存在SSD中的key
    vector<emb_key_t> externalSSDKeys;
    GetSSDKeys(embTableName, externalKeys, externalSSDKeys);
    // 后续判断maxOffset是否超出范围时，maxOffset=devVocabSize+hostVocabSize时可用,此处包含等于
    bool isDDRSpaceEnough = ddrAvailableSize >= externalKeys.size();
    bool ddrSpaceEnoughOrEval = channelId != TRAIN_CHANNEL_ID || isDDRSpaceEnough;
    if (ddrSpaceEnoughOrEval && externalSSDKeys.empty()) {
        // 部分场景后续不用处理，在此处返回
        return TransferRet::TRANSFER_OK;
    }

    AddDebugAndTraceLog(keys.size(), externalKeys, externalSSDKeys);
    /*
     * 前面 externalSSDKeys = 0 ，评估场景的 ddr空间可用、不可用已返回； 训练的可用已返回；
     * 剩下的情况如下：
     * 评估：
     *   externalSSDKeys > 0, 可用 & 不可用操作一样；
     *     可选：Ddr->ssd, 腾出 externalSSDKeys 大小空间；
     *     Ssd->ddr, 需要移动 externalSSDKeys ；
     *   externalSSDKeys = 0  --已返回
     * 训练：
     *   externalSSDKeys > 0
     *     可用：
     *       可选：Ddr->ssd, 腾出 externalSSDKeys 大小空间；
     *       Ssd->ddr, 需要移动 externalSSDKeys ；
     *     不可用：
     *       必选：Ddr->ssd, 腾出 externalKeys 大小空间；
     *         需要计算ssd剩余空间：externalKeys - externalSSDKeys
     *         (注: 当前策略均转移externalKeys)
     *       Ssd->ddr, 需要移动 externalSSDKeys ；
     *   externalSSDKeys = 0
     *     可用： --已返回
     *     不可用：
     *       Ddr->ssd, 腾出 externalKeys 大小的空间；
     *         需要计算ssd剩余空间： externalKeys
     *  因cache每次只转移DDR最小空间，上述可选动作也需执行，避免SSD移入DDR时空间不足
     */
    // 训练场景检查SSD剩余空间 评估不考虑新key
    if (channelId == TRAIN_CHANNEL_ID) {
        size_t needSSDSize = externalKeys.size() - externalSSDKeys.size() - ddrAvailableSize;
        const int64_t ssdAvailableSize = ssdEngine->GetTableAvailableSpace(embTableName);
        if (int64_t(needSSDSize) > ssdAvailableSize) {
            LOG_ERROR("TransferDDREmbWithSSD: ssd available space is not enough to transfer DDR emb data. "
                      "needSSDSize:{}, ssdAvailableSize:{}", needSSDSize, ssdAvailableSize);
            return TransferRet::SSD_SPACE_NOT_ENOUGH;
        }
    }

    // 从SSD获取emb数据并从SSD删除; 避免DDR->SSD时空间不够
    vector<vector<float>> ssdEmbData;
    if (!externalSSDKeys.empty()) {
        ssdEmbData = ssdEngine->FetchEmbeddings(embTableName, externalSSDKeys);
        ssdEngine->DeleteEmbeddings(embTableName, externalSSDKeys);
    }

    // 从ddr转移到ssd的key个数
    size_t ddrSwapOutSizeTmp = ddrSpaceEnoughOrEval ? externalSSDKeys.size() : externalKeys.size();
    auto ddrSwapOutSize = static_cast<int64_t>(ddrSwapOutSizeTmp - ddrAvailableSize);
    LOG_DEBUG("TransferDDREmbWithSSD: ddrSwapOutSize:{}", ddrSwapOutSize);

    /*
     * 转移DDR中数据到SSD
     */
    // 记录要从DDR转移到SSD的key对应的offset(相对值，需减去devVocabSize)
    vector<size_t> ddrTransferPos;
    TransferRet ddr2SsdRet = TransferDDREmb2SSD(embTableName, embHashMap, ddrSwapOutSize, internalKeys, ddrTransferPos);
    if (ddr2SsdRet == TransferRet::DDR_SPACE_NOT_ENOUGH) {
        ssdEngine->InsertEmbeddings(embTableName, externalSSDKeys, ssdEmbData);
        return ddr2SsdRet;
    }

    HandleDDRTransferPos(ddrTransferPos, externalSSDKeys, embHashMap);

    /*
     * 转移SSD中保存的当前批次key的emb数据到DDR
     */
    return TransferSSDEmb2DDR(embTableName, embHashMap, externalSSDKeys, ddrTransferPos, ssdEmbData);
}

/// SSD数据转移到DDR中后刷新映射和频次信息
/// \param embTableName emb表名
/// \param embHashMap emb hash表
/// \param externalSSDKeys 存储在SSD中的key列表
/// \param ddrTransferPos
void CacheManager::RefreshRelateInfoWithSSD2DDR(const std::string& embTableName, EmbHashMapInfo& embHashMap,
                                                vector<emb_key_t>& externalSSDKeys, vector<size_t>& ddrTransferPos)
{
    for (size_t i = 0; i < externalSSDKeys.size(); ++i) {
        // 映射关系 ddrTransferPos是在ddrEmbHash中的位置，记录映射时需加上devVocabSize
        auto& key = externalSSDKeys[i];
        embHashMap.hostHashMap[key] = ddrTransferPos[i] + embHashMap.devVocabSize;
        // 频次
        ddrKeyFreqMap[embTableName].PutWithInit(key, excludeDDRKeyCountMap[embTableName][key]);
        excludeDDRKeyCountMap[embTableName].erase(key);
    }
}

void CacheManager::GetDDREmbInfo(vector<emb_key_t>& keys, const std::string& embTableName, EmbHashMapInfo& embHashMap,
                                 vector<size_t>& ddrTransferPos, vector<vector<float>>& ddrEmbData) const
{
    // 根据offset 获取对应Emb数据
    for (auto& key : keys) {
        ddrTransferPos.emplace_back(embHashMap.hostHashMap[key] - embHashMap.devVocabSize);
    }

    LOG_TRACE("DDR keys:{}", VectorToString(keys));
    LOG_TRACE("DDR key positions:{}", VectorToString(ddrTransferPos));

    ddrEmbData.resize(keys.size());
    const auto& emb = hostEmbs->GetEmb(embTableName);
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(ddrTransferPos, emb, ddrEmbData)
    for (size_t i = 0; i < ddrTransferPos.size(); ++i) {
        auto& missingKeyPo = ddrTransferPos[i];
        const auto& src = emb.embData[missingKeyPo];
        ddrEmbData[i] = src;
    }
}

/// 使用ssdEmbData更新DDR内emb数据
/// \param embTableName emb表名
/// \param ddrTransferPos 需要更新的DDR内的offset
/// \param ssdEmbData SSD对应的emb数据
void CacheManager::UpdateDDREmbInfo(const std::string& embTableName,
                                    vector<size_t>& ddrTransferPos,
                                    vector<vector<float>>& ssdEmbData) const
{
    auto& emb = hostEmbs->GetEmb(embTableName);
    auto& embData = emb.embData;
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(ddrTransferPos, embData, ssdEmbData)
    for (size_t i = 0; i < ddrTransferPos.size(); ++i) {
        embData[ddrTransferPos[i]] = ssdEmbData[i];
    }
}

/// DDR_2_SSD场景数据刷新: 仅刷新映射和频次，ddr转移出去的offset信息后续统一处理
/// \param embTableName emb表名
/// \param embHashMap emb map
/// \param ddrSwapOutKeys 从DDR中转移到SSD中key列表
/// \param ddrSwapOutCounts 从DDR中转移到SSD中key频次数据
void CacheManager::RefreshRelateInfoWithDDR2SSD(const string& embTableName, EmbHashMapInfo& embHashMap,
                                                vector<emb_key_t>& ddrSwapOutKeys,
                                                vector<freq_num_t>& ddrSwapOutCounts)
{
    auto& excludeFreqMap = excludeDDRKeyCountMap[embTableName];
    for (size_t i = 0; i < ddrSwapOutKeys.size(); ++i) {
        auto& key = ddrSwapOutKeys[i];
        embHashMap.hostHashMap.erase(key);
        excludeFreqMap[key] = ddrSwapOutCounts[i];
    }
}

/// key从DDR移入、移出、HBM淘汰时刷新频次信息；仅刷新频次信息
/// \param embTableName emb表名
/// \param keys 操作的key集合
/// \param type TransferType
void CacheManager::RefreshFreqInfoCommon(const string& embTableName, vector<emb_key_t>& keys, TransferType type)
{
    if (type == TransferType::DDR_2_HBM) {
        for (auto& key : keys) {
            // 频次数据记录到 excludeDDRKeyCountMap,并删除ddrKeyFreqMap中频次数据
            // 进入findOffset时记录的key次数 + ddr内key次数
            auto tmpCount = excludeDDRKeyCountMap[embTableName][key];
            excludeDDRKeyCountMap[embTableName][key] = ddrKeyFreqMap[embTableName].Get(key) + tmpCount;
            ddrKeyFreqMap[embTableName].Pop(key);
        }
    } else if (type == TransferType::HBM_2_DDR) {
        for (auto& key : keys) {
            // excludeDDRKeyCountMap 中次数转移到 ddrKeyFreqMap, 并删除原记录
            ddrKeyFreqMap[embTableName].PutWithInit(key, excludeDDRKeyCountMap[embTableName][key]);
            excludeDDRKeyCountMap[embTableName].erase(key);
        }
    } else if (type == TransferType::DDR_2_EVICT) {
        for (auto& key : keys) {
            ddrKeyFreqMap[embTableName].Pop(key);
        }
    } else {
        // TransferType::HBM_2_EVICT
        for (auto& key : keys) {
            excludeDDRKeyCountMap[embTableName].erase(key);
        }
    }
}

void CacheManager::Init(HostEmb* hostEmbPtr, vector<EmbInfo>& mgmtEmbInfo)
{
    this->hostEmbs = hostEmbPtr;
    for (auto& emb : mgmtEmbInfo) {
        EmbBaseInfo baseInfo {emb.ssdVocabSize, emb.ssdDataPath, false};
        embBaseInfos.emplace(emb.name, baseInfo);
    }
    ssdEngine->Start();
    LOG_INFO("CacheManager Init method end.");
}

bool CacheManager::IsKeyInSSD(const string& embTableName, emb_key_t key)
{
    return ssdEngine->IsKeyExist(embTableName, key);
}

/// 淘汰SSD中Emb信息
/// \param embTableName emb表名
/// \param keys 淘汰key列表
void CacheManager::EvictSSDEmbedding(const string& embTableName, vector<emb_key_t>& keys)
{
    if (keys.empty()) {
        return;
    }
    // 1 删除缓存中记录的key的次数 2 删除SSD中保存的Emb数据
    for (auto& key : keys) {
        excludeDDRKeyCountMap[embTableName].erase(key);
    }
    ssdEngine->DeleteEmbeddings(embTableName, keys);
}

/// 放入key，新增/更新(次数+1)次数
/// \param embTableName emb表名
/// \param key key
/// \param type 记录类型
void CacheManager::PutKey(const string& embTableName, const emb_key_t& key, RecordType type)
{
    if (type == RecordType::DDR) {
        ddrKeyFreqMap[embTableName].Put(key);
        return;
    }
    auto& hashMap = excludeDDRKeyCountMap[embTableName];
    const auto& it = hashMap.find(key);
    freq_num_t count = it == hashMap.end() ? 1 : it->second + 1;
    hashMap[key] = count;
}

/// DDR->SSD与SSD->DDR的key个数可能不一致，手动补齐/截取
/// \param ddrTransferPos DDR->SSD的offset列表(hostEmb表内的偏移值)
/// \param externalSSDKeys SSD->DDR的key列表
/// \param embHashMap emb hash表
void CacheManager::HandleDDRTransferPos(vector<size_t>& ddrTransferPos, vector<emb_key_t>& externalSSDKeys,
                                        EmbHashMapInfo& embHashMap)
{
    if (ddrTransferPos.size() == externalSSDKeys.size()) {
        return;
    }
    LOG_DEBUG("TransferDDREmbWithSSD: operate length is not equal, will padding or clipping, "
              "ddrTransferPos size:{}, externalSSDKeys size:{}",
              ddrTransferPos.size(), externalSSDKeys.size());
    // ddrTransferPos中是DDR内偏移位置，存入evictPos时，需加上devVocabSize;取出时需减去
    if (ddrTransferPos.size() > externalSSDKeys.size()) {
        while (ddrTransferPos.size() > externalSSDKeys.size()) {
            embHashMap.evictPos.emplace_back(ddrTransferPos.back() + embHashMap.devVocabSize);
            ddrTransferPos.pop_back();
        }
        return;
    }
    // 补齐offset
    while (ddrTransferPos.size() < externalSSDKeys.size() && !embHashMap.evictPos.empty()) {
        ddrTransferPos.emplace_back(embHashMap.evictPos.back() - embHashMap.devVocabSize);
        embHashMap.evictPos.pop_back();
    }
    auto allSize = embHashMap.devVocabSize + embHashMap.hostVocabSize;
    // 还不够继续使用maxOffset
    while (ddrTransferPos.size() < externalSSDKeys.size() && embHashMap.maxOffset < allSize) {
        auto nextPos = embHashMap.maxOffset++;
        ddrTransferPos.emplace_back(nextPos - embHashMap.devVocabSize);
    }
    LOG_DEBUG("HandleDDRTransferPos: handle end, pos len:{}, keys len:{}",
        ddrTransferPos.size(), externalSSDKeys.size());
}

void CacheManager::GetSSDKeys(const std::string& embTableName, vector<emb_key_t>& externalKeys,
                              vector<emb_key_t>& externalSSDKeys)
{
    for (auto& key : externalKeys) {
        if (ssdEngine->IsKeyExist(embTableName, key)) {
            externalSSDKeys.emplace_back(key);
        }
    }
}

TransferRet CacheManager::TransferDDREmb2SSD(const string& embTableName, EmbHashMapInfo& embHashMap,
                                             int64_t ddrSwapOutSize,
                                             const vector<emb_key_t>& keys, vector<size_t>& ddrTransferPos)
{
    if (ddrSwapOutSize <= 0) {
        // 此时不需要转移数据
        return TransferRet::TRANSFER_OK;
    }

    TimeCost ddr2SsdTc;
    LOG_DEBUG("TransferDDREmbWithSSD: get ddr least freq keys, ddrSwapOutSize:{}", ddrSwapOutSize);
    // 获取DDR中指定数量的最低频次key，并获取相应emb数据，执行DDR换出到SSD
    vector<emb_key_t> ddrSwapOutKeys;
    vector<freq_num_t> ddrSwapOutCounts;
    ddrKeyFreqMap[embTableName].GetAndDeleteLeastFreqKeyInfo(ddrSwapOutSize, keys, ddrSwapOutKeys,
                                                             ddrSwapOutCounts);
    if (static_cast<int64_t>(ddrSwapOutKeys.size()) != ddrSwapOutSize) {
        auto keyTableSize = ddrKeyFreqMap[embTableName].keyTable.size();
        // 获取的最低频次key数量和预期不一致，DDR空间不足，不能放置当前批次数据
        LOG_ERROR("TransferDDREmbWithSSD, vector length is not equal, ddrSwapOutKeys size:{}, "
                  "ddrSwapOutSize:{}, ddr lfu keyTable size:{}",
                  ddrSwapOutKeys.size(), ddrSwapOutSize, keyTableSize);
        RestoreLeastFreqInfo(embTableName, ddrSwapOutKeys, ddrSwapOutCounts);
        return TransferRet::DDR_SPACE_NOT_ENOUGH;
    }
    LOG_DEBUG("TransferDDREmbWithSSD: get DDR embeddings and save to SSD, size:{}", ddrSwapOutKeys.size());
    // 获取DDR中emb数据
    vector<vector<float>> ddrEmbData;
    GetDDREmbInfo(ddrSwapOutKeys, embTableName, embHashMap, ddrTransferPos, ddrEmbData);
    // 调用SSDEngine接口，将DDR Emb数据保存到SSD
    ssdEngine->InsertEmbeddings(embTableName, ddrSwapOutKeys, ddrEmbData);

    // 初始化DDR内被转移出去的位置
    hostEmbs->EvictInitEmb(embTableName, ddrTransferPos);

    // 更新记录的DDR中key频次信息
    RefreshRelateInfoWithDDR2SSD(embTableName, embHashMap, ddrSwapOutKeys, ddrSwapOutCounts);
    LOG_DEBUG("TransferDDREmbWithSSD: ddr2SsdTc TimeCost(ms):{}", ddr2SsdTc.ElapsedMS());
    return TransferRet::TRANSFER_OK;
}

TransferRet CacheManager::TransferSSDEmb2DDR(const string& embTableName, EmbHashMapInfo& embHashMap,
                                             vector<emb_key_t>& externalSSDKeys, vector<size_t>& ddrTransferPos,
                                             vector<vector<float>>& ssdEmbData)
{
    if (externalSSDKeys.empty()) {
        return TransferRet::TRANSFER_OK;
    }
    TimeCost ssd2DdrTc;
    LOG_DEBUG("TransferDDREmbWithSSD: get SSD embeddings and save to DDR, size:{}", externalSSDKeys.size());
    if (ddrTransferPos.size() != externalSSDKeys.size() || externalSSDKeys.size() != ssdEmbData.size()) {
        LOG_ERROR("TransferDDREmbWithSSD, vector length is not equal, ddrTransferPos len:{}, externalSSDKeys len:{}, "
            "ssdEmbData len:{}", ddrTransferPos.size(), externalSSDKeys.size(), ssdEmbData.size());
        return TransferRet::TRANSFER_ERROR;
    }
    // 将SSD emb存储到DDR中 刷新频次信息
    UpdateDDREmbInfo(embTableName, ddrTransferPos, ssdEmbData);
    RefreshRelateInfoWithSSD2DDR(embTableName, embHashMap, externalSSDKeys, ddrTransferPos);
    LOG_DEBUG("TransferDDREmbWithSSD: ssd2DdrTc TimeCost(ms):{}", ssd2DdrTc.ElapsedMS());
    return TransferRet::TRANSFER_OK;
}

void CacheManager::CreateSSDTableIfNotExist(const std::string& embTableName)
{
    if (embBaseInfos[embTableName].isExist) {
        return;
    }
    if (!ssdEngine->IsTableExist(embTableName)) {
        ssdEngine->CreateTable(embTableName, embBaseInfos[embTableName].savePath,
                               embBaseInfos[embTableName].maxTableSize);
        embBaseInfos[embTableName].isExist = true;
        LOG_INFO("create ssd table end, embTableName:" + embTableName);
        return;
    }
    // 续训场景：embBaseInfos 没有保存，不会初始化；SSD表会初始化，此时表已存在
    embBaseInfos[embTableName].isExist = true;
    LOG_INFO("ssd table is exist, embTableName:" + embTableName);
}

void CacheManager::RestoreLeastFreqInfo(const std::string& embTableName, vector<emb_key_t>& ddrSwapOutKeys,
                                        vector<freq_num_t>& ddrSwapOutCounts)
{
    auto& lfuCache = ddrKeyFreqMap[embTableName];
    for (size_t i = 0; i < ddrSwapOutKeys.size(); ++i) {
        lfuCache.PutWithInit(ddrSwapOutKeys[i], ddrSwapOutCounts[i]);
    }
}

CacheManager::~CacheManager()
{
    hostEmbs = nullptr;
    ssdEngine->Stop();
    ddrKeyFreqMap.clear();
    excludeDDRKeyCountMap.clear();
}

/// 加载数据到CacheManager
/// \param ddrFreqInitMap ddr内key频次数据
/// \param excludeDdrFreqInitMap 非DDR key频次数据
/// \param step 加载SSDEngine传入步数
void CacheManager::Load(unordered_map<std::string, unordered_map<emb_key_t, freq_num_t>>& ddrFreqInitMap,
                        unordered_map<std::string, unordered_map<emb_key_t, freq_num_t>>& excludeDdrFreqInitMap,
                        int step)
{
    // 加载CacheManager数据
    for (auto& it : ddrFreqInitMap) {
        auto& embTableName = it.first;
        auto& freqMap = it.second;
        for (auto& freqIt : freqMap) {
            ddrKeyFreqMap[embTableName].PutWithInit(freqIt.first, freqIt.second);
        }
    }
    for (auto& it : excludeDdrFreqInitMap) {
        auto& embTableName = it.first;
        auto& freqMap = it.second;
        for (auto& freqIt : freqMap) {
            excludeDDRKeyCountMap[embTableName].emplace(freqIt.first, freqIt.second);
        }
    }
    // 加载SSDEngine数据
#ifndef GTEST
    for (auto& it : embBaseInfos) {
        string embTableName = it.first;
        EmbBaseInfo& embBase = it.second;
        ssdEngine->Load(embTableName, embBase.savePath, embBase.maxTableSize, step);
    }
#endif
}

void CacheManager::SaveSSDEngine(int step)
{
#ifndef GTEST
    ssdEngine->Save(step);
#endif
}

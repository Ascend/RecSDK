/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#include "emb_hashmap.h"
#include <fstream>
#include <spdlog/spdlog.h>
#include <iomanip>
#include <mpi.h>
#include <spdlog/fmt/bundled/ranges.h>
#include "hd_transfer/hd_transfer.h"
#include "checkpoint/checkpoint.h"

using namespace MxRec;

void EmbHashMap::Init(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos, bool ifLoad)
{
    this->rankInfo = rankInfo;
    if (!ifLoad) {
        EmbHashMapInfo embHashMap;
        spdlog::info("init emb hash map from scratch");
        for (const auto& embInfo: embInfos) {
            embHashMap.devOffset2Batch.resize(embInfo.devVocabSize);
            embHashMap.devOffset2Key.resize(embInfo.devVocabSize);
            embHashMap.hostVocabSize = embInfo.hostVocabSize;
            embHashMap.devVocabSize = embInfo.devVocabSize;
            embHashMap.currentUpdatePos = 0;
            fill(embHashMap.devOffset2Batch.begin(), embHashMap.devOffset2Batch.end(), -1);
            fill(embHashMap.devOffset2Key.begin(), embHashMap.devOffset2Key.end(), -1);
            embHashMaps[embInfo.name] = embHashMap;
            spdlog::trace("devOffset2Key, {}", embHashMaps.at(embInfo.name).devOffset2Key);
            spdlog::trace("devOffset2Batch, {}", embHashMaps.at(embInfo.name).devOffset2Batch);
        }
    }
}

void EmbHashMap::Process(const string& embName, const vector<emb_key_t>& keys, size_t iBatch,
                         vector<Tensor>& tmpDataOut)
{
    EASY_FUNCTION(profiler::colors::Pink)
    auto keepBatch = swapId - iBatch;
    FindAndUpdateOffset(embName, keys, swapId, keepBatch);
    swapId++;
    EASY_BLOCK("hostHashMaps->tdt")

    auto& embHashMap = embHashMaps.at(embName);
    auto lookUpVecSize = static_cast<int>(embHashMap.lookUpVec.size());
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { lookUpVecSize }));

    auto lookupTensorData = tmpDataOut.back().flat<int32>();
    for (int i = 0; i < lookUpVecSize; i++) {
        lookupTensorData(i) = static_cast<int32_t>(embHashMap.lookUpVec[i]);
    }
    spdlog::trace("lookupTensor, {}", embHashMap.lookUpVec);
    auto swapSize = static_cast<int>(embHashMap.swapPos.size());
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { swapSize }));

    auto swapTensorData = tmpDataOut.back().flat<int32>();
    for (int i = 0; i < swapSize; i++) {
        swapTensorData(i) = static_cast<int>(embHashMap.swapPos[i]);
    }
    if (swapSize > 0) {
        spdlog::debug("swap num: {}", swapSize);
    }
    spdlog::trace("swapTensor, {}", embHashMap.swapPos);
    embHashMap.swapPos.clear();
    spdlog::info("current dev emb usage:{}-{}/[{}+{}]", embName, embHashMap.maxOffset, embHashMap.devVocabSize,
                 embHashMap.hostVocabSize);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));
    auto swapLen = tmpDataOut.back().flat<int32>();
    swapLen(0) = swapSize;
    EASY_END_BLOCK
}

/*
 * 从embHashMaps获取key对应的位置，并更新devOffset2Batch
 */
void EmbHashMap::FindAndUpdateOffset(const string& embName, const vector<emb_key_t>& keys,
                                     size_t currentBatchId, size_t keepBatchId)
{
    EASY_FUNCTION()
    size_t keySize = keys.size();
    auto& embHashMap = embHashMaps.at(embName);
    embHashMap.lookUpVec.resize(keySize);
    std::fill(embHashMap.lookUpVec.begin(), embHashMap.lookUpVec.end(), INVALID_KEY_VALUE);

    FindAndUpdateBatchId(keys, currentBatchId, keySize, embHashMap);
    EASY_BLOCK("FindNewOffset")
    vector<pair<emb_key_t, int32_t>> KeysAndOffset;

    for (size_t i = 0; i < keySize; i++) {
        auto key = keys[i];
        if (key == -1) {
            continue;
        }
        auto& offset = embHashMap.lookUpVec[i];
        if (offset == INVALID_KEY_VALUE) {
            offset = FindNewOffset(key, embHashMap);
        }
        if (offset >= static_cast<int>(embHashMap.devVocabSize)) {
            embHashMap.missingKeysHostPos.emplace_back(offset - embHashMap.devVocabSize);
            KeysAndOffset.emplace_back(key, i);
        }
    }
    EASY_END_BLOCK
    EASY_BLOCK("FindPos")
    size_t swapSize = KeysAndOffset.size();
    FindPos(embHashMap, swapSize, currentBatchId, keepBatchId);
    EASY_END_BLOCK
    EASY_BLOCK("ChangeInfo")
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) \
                         shared(swapSize, KeysAndOffset, embHashMap, currentBatchId)
    for (size_t i = 0; i < swapSize; i++) {
        auto[key, j] = KeysAndOffset[i];
        int pos = static_cast<int>(embHashMap.swapPos[i]);
        ChangeSwapInfo(embHashMap, key, embHashMap.missingKeysHostPos[i] + embHashMap.devVocabSize,
                       currentBatchId, pos);
        embHashMap.lookUpVec[j] = pos;
    }
    EASY_END_BLOCK
}

void EmbHashMap::ChangeSwapInfo(EmbHashMapInfo& embHashMap, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                                int pos)
{
    embHashMap.devOffset2Batch[pos] = static_cast<int>(currentBatchId);
    embHashMap.hostHashMap[key] = pos;
    auto& oldKey = embHashMap.devOffset2Key[pos];
    if (oldKey != -1) {
        embHashMap.hostHashMap[oldKey] = hostOffset;
    }
    oldKey = key;
}

int32_t EmbHashMap::FindNewOffset(const emb_key_t& key, EmbHashMapInfo& embHashMap)
{
    int offset;
    const auto& iter = embHashMap.hostHashMap.find(key);
    if (iter != embHashMap.hostHashMap.end()) { // 由于未全局去重，需要再次查询确保是新key
        offset = iter->second;
    } else if (embHashMap.evictDevPos.size() != 0) { // 优先复用hbm表
        offset = embHashMap.evictDevPos.back();
        embHashMap.hostHashMap[key] = offset;
        spdlog::trace("ddr mode, dev evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
                      key, offset, embHashMap.evictDevPos.size());
        embHashMap.evictDevPos.pop_back();
    } else if (embHashMap.evictPos.size() != 0) { // hbm不足，再复用ddr表
        offset = embHashMap.evictPos.back();
        embHashMap.hostHashMap[key] = offset;
        spdlog::trace("ddr mode, host evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
                      key, offset, embHashMap.evictPos.size());
        embHashMap.evictPos.pop_back();
    } else {
        embHashMap.hostHashMap[key] = embHashMap.maxOffset;
        offset = embHashMap.maxOffset;
        embHashMap.maxOffset++;
        if (embHashMap.maxOffset == embHashMap.devVocabSize) {
            spdlog::info("start using host vocab!");
        }
        if (embHashMap.maxOffset > embHashMap.hostVocabSize + embHashMap.devVocabSize) {
            spdlog::error("hostVocabSize too small! dev:{} host:{}", embHashMap.devVocabSize,
                          embHashMap.hostVocabSize);
            throw runtime_error("hostVocabSize too small");
        }
    }
    return offset;
}

void EmbHashMap::FindAndUpdateBatchId(const vector<emb_key_t>& keys, size_t currentBatchId, size_t keySize,
                                      EmbHashMapInfo& embHashMap) const
{
    EASY_FUNCTION()
    for (size_t i = 0; i < keySize; i++) {
        int offset;
        auto key = keys[i];
        if (key == -1) {
            continue;
        }
        const auto& iter = embHashMap.hostHashMap.find(key);
        if (iter != embHashMap.hostHashMap.end()) { // found
            offset = static_cast<int>(iter->second);
            embHashMap.lookUpVec[i] = offset; // convert to offset(current)
            spdlog::trace("key will be used, {} , offset , {}", key, offset);
            if (offset < static_cast<int>(embHashMap.devVocabSize)) {
                embHashMap.devOffset2Batch[offset] = currentBatchId;
                embHashMap.devOffset2Key[offset] = key;
            }
        }
    }
}

void EmbHashMap::FindPos(EmbHashMapInfo& embHashMap, int num, size_t currentBatchId,
                         size_t keepBatchId)
{
    while (num != 0) {
        if (embHashMap.devOffset2Batch[embHashMap.currentUpdatePos] < static_cast<int>(keepBatchId)) {
            embHashMap.swapPos.emplace_back(embHashMap.currentUpdatePos);
            num -= 1;
        }
        embHashMap.currentUpdatePos++;
        embHashMap.freeSize--;
        if (embHashMap.currentUpdatePos == embHashMap.devVocabSize) {
            embHashMap.currentUpdatePos = 0;
        }
        if (embHashMap.currentUpdatePos == embHashMap.currentUpdatePosStart) {
            spdlog::error("devVocabSize is too small");
            throw runtime_error("devVocabSize is too small");
        }
    }
}

auto EmbHashMap::GetHashMaps() -> absl::flat_hash_map<string, EmbHashMapInfo>
{
    return embHashMaps;
}

void EmbHashMap::LoadHashMap(emb_hash_mem_t& loadData)
{
    embHashMaps = std::move(loadData);
}

void EmbHashMapInfo::SetStartCount()
{
    currentUpdatePosStart = currentUpdatePos;
    freeSize = devVocabSize;
}

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

    for (size_t i = 0; i < keySize; i++) {
        size_t offset;
        auto key = keys[i];
        if (key == -1) {
            spdlog::error("evict key equal -1!");
            continue;
        }
        const auto& iter = embHashMap.hostHashMap.find(key);
        if (iter != embHashMap.hostHashMap.end()) {
            offset = iter->second;
            embHashMap.hostHashMap.erase(iter);
            spdlog::trace("evict embName {} , offset , {}", embName, offset);
        } else {
            // 淘汰依据keyProcess中的history，hashmap映射关系创建于ParseKey；两者异步，造成淘汰的值在hashmap里可能未创建
            continue;
        }

        if (offset < embHashMap.devVocabSize) {
            embHashMap.devOffset2Batch[offset] = -1;
            embHashMap.devOffset2Key[offset] = -1;
            embHashMap.evictDevPos.emplace_back(offset);
        } else {
            embHashMap.evictPos.emplace_back(offset - embHashMap.devVocabSize);
        }
    }

    spdlog::info("ddr EvictDeleteEmb, emb: [{}], hostEvictSize: {}, devEvictSize: {} ",
                 embName, embHashMap.evictPos.size(), embHashMap.evictDevPos.size());
    spdlog::trace("hostHashMap, {}", embHashMaps[embName].hostHashMap);
}
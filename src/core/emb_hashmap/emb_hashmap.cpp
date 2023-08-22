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
#include "hd_transfer/hd_transfer.h"
#include "checkpoint/checkpoint.h"
#include "utils/common.h"

using namespace MxRec;

void EmbHashMap::Init(const RankInfo& rankInfo, const vector<EmbInfo>& embInfos, bool ifLoad)
{
#ifndef GTEST
    this->rankInfo = rankInfo;
    if (!ifLoad) {
        EmbHashMapInfo embHashMap;
        LOG(INFO) << "init emb hash map from scratch";
        for (const auto& embInfo: embInfos) {
            embHashMap.devOffset2Batch.resize(embInfo.devVocabSize);
            embHashMap.devOffset2Key.resize(embInfo.devVocabSize);
            embHashMap.hostVocabSize = embInfo.hostVocabSize;
            embHashMap.devVocabSize = embInfo.devVocabSize;
            embHashMap.currentUpdatePos = 0;
            fill(embHashMap.devOffset2Batch.begin(), embHashMap.devOffset2Batch.end(), -1);
            fill(embHashMap.devOffset2Key.begin(), embHashMap.devOffset2Key.end(), -1);
            embHashMaps[embInfo.name] = embHashMap;

            if (VLOG_IS_ON(GLOG_TRACE)) {
                VLOG(GLOG_TRACE) << StringFormat(
                    "devOffset2Key, %s", VectorToString(embHashMaps.at(embInfo.name).devOffset2Key).c_str()
                );
                VLOG(GLOG_TRACE) << StringFormat(
                    "devOffset2Batch, %s", VectorToString(embHashMaps.at(embInfo.name).devOffset2Batch).c_str()
                );
            }
        }
    }
#endif
}

void EmbHashMap::Process(const string& embName, vector<emb_key_t>& keys, size_t iBatch,
                         vector<Tensor>& tmpDataOut, int channelId, vector<int32_t>& offsetsOut)
{
#ifndef GTEST
    EASY_FUNCTION(profiler::colors::Pink)
    auto& embHashMap = embHashMaps.at(embName);
    embHashMap.devOffset2KeyOld.clear();
    embHashMap.oldSwap.clear();
    embHashMap.maxOffsetOld = embHashMap.maxOffset;

    auto keepBatch = swapId - iBatch;
    bool findOffsetV2 = getenv("FIND_OFFSET_V2") != nullptr;
    VLOG(GLOG_DEBUG) << StringFormat("FindOffset version:%d", findOffsetV2);

    if (findOffsetV2) {
        FindAndUpdateOffset(embName, keys, swapId, keepBatch, channelId);
    } else {
        FindOffset(embName, keys, swapId, keepBatch, channelId);
    }
    VLOG(GLOG_DEBUG) << "FindOffset end";

    swapId++;
    EASY_BLOCK("hostHashMaps->tdt")

//    std::copy(embHashMap.lookUpVec.begin(), embHashMap.lookUpVec.end(), offsetsOut.begin());
    std::copy(embHashMap.lookUpVec.begin(), embHashMap.lookUpVec.end(), std::back_inserter(offsetsOut));

    auto lookUpVecSize = static_cast<int>(embHashMap.lookUpVec.size());
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { lookUpVecSize }));

    auto lookupTensorData = tmpDataOut.back().flat<int32>();
    for (int i = 0; i < lookUpVecSize; i++) {
        lookupTensorData(i) = static_cast<int32_t>(embHashMap.lookUpVec[i]);
    }
    if (VLOG_IS_ON(GLOG_TRACE)) {
        VLOG(GLOG_TRACE) << StringFormat("lookupTensor, %s", VectorToString(embHashMap.lookUpVec).c_str());
    }
    auto swapSize = static_cast<int>(embHashMap.swapPos.size());
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { swapSize }));

    auto swapTensorData = tmpDataOut.back().flat<int32>();
    for (int i = 0; i < swapSize; i++) {
        swapTensorData(i) = static_cast<int>(embHashMap.swapPos[i]);
    }
    if (swapSize > 0) {
        VLOG(GLOG_DEBUG) << StringFormat("swap num: %d", swapSize);
    }
    if (VLOG_IS_ON(GLOG_TRACE)) {
        VLOG(GLOG_TRACE) << StringFormat("swapTensor, %s", VectorToString(embHashMap.swapPos).c_str());
    }
    embHashMap.swapPos.clear();
    embHashMap.lookUpVec.clear();
    LOG(INFO) << StringFormat("current ddr emb:%s, usage:%d/[%d+%d]", embName.c_str(), embHashMap.maxOffset,
                              embHashMap.devVocabSize, embHashMap.hostVocabSize);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));
    auto swapLen = tmpDataOut.back().flat<int32>();
    swapLen(0) = swapSize;
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

void EmbHashMap::ChangeSwapInfo(EmbHashMapInfo& embHashMap, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                                int pos)
{
    embHashMap.devOffset2Batch[pos] = static_cast<int>(currentBatchId);
    embHashMap.hostHashMap[key] = pos;
    auto& oldKey = embHashMap.devOffset2Key[pos];
    if (oldKey != -1) {
        embHashMap.oldSwap.emplace_back(oldKey, key);
        embHashMap.hostHashMap[oldKey] = hostOffset;
    }
    oldKey = key;
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
        VLOG(GLOG_TRACE) << StringFormat(
            "ddr mode, dev evictPos is not null, key [%d] reuse offset [%d], evictSize [%d]",
            key, offset, embHashMap.evictDevPos.size());
        embHashMap.evictDevPos.pop_back();
    } else if (embHashMap.evictPos.size() != 0) { // hbm不足，再复用ddr表
        offset = static_cast<int32_t>(embHashMap.evictPos.back());
        embHashMap.hostHashMap[key] = offset;
        VLOG(GLOG_TRACE) << StringFormat(
            "ddr mode, host evictPos is not null, key [%d] reuse offset [%d], evictSize [%d]",
            key, offset, embHashMap.evictPos.size());
        embHashMap.evictPos.pop_back();
    } else {
        embHashMap.hostHashMap[key] = embHashMap.maxOffset;
        offset = static_cast<int32_t>(embHashMap.maxOffset);
        embHashMap.maxOffset++;
        if (embHashMap.maxOffset == embHashMap.devVocabSize) {
            LOG(INFO) << "start using host vocab!";
        }
        if (embHashMap.maxOffset > embHashMap.hostVocabSize + embHashMap.devVocabSize) {
            LOG(ERROR) << StringFormat("hostVocabSize too small! dev:%d host:%d", embHashMap.devVocabSize,
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
    bool findOffsetV3 = getenv("FIND_OFFSET_V3") != nullptr;
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

void EmbHashMap::FindPos(EmbHashMapInfo& embHashMap, int num, size_t keepBatchId)
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
            LOG(ERROR) << "devVocabSize is too small";
            throw runtime_error("devVocabSize is too small");
        }
    }
}


auto EmbHashMap::GetHashMaps() -> absl::flat_hash_map<string, EmbHashMapInfo>
{
    auto embHashMapsOld = embHashMaps;
    for (auto& temp: embHashMapsOld) {
        auto& embHashMap = temp.second;
        for (auto& swapKeys: embHashMap.oldSwap) {
            emb_key_t oldKey = swapKeys.first;
            emb_key_t key = swapKeys.second;
            int tempOffset = static_cast<int>(embHashMap.hostHashMap[key]);
            embHashMap.hostHashMap[key] = embHashMap.hostHashMap[oldKey];
            embHashMap.hostHashMap[oldKey] = static_cast<int>(tempOffset);
        }
        embHashMap.maxOffset = embHashMap.maxOffsetOld;
        for (auto& Offset2Key: embHashMap.devOffset2KeyOld) {
            embHashMap.devOffset2Key[Offset2Key.first] = Offset2Key.second;
        }
    }
    return embHashMapsOld;
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
            LOG(WARNING) << "evict key equal -1!";
            continue;
        }
        const auto& iter = embHashMap.hostHashMap.find(key);
        if (iter != embHashMap.hostHashMap.end()) {
            offset = iter->second;
            embHashMap.hostHashMap.erase(iter);
            VLOG(GLOG_TRACE) << StringFormat("evict embName %s , offset , %d", embName.c_str(), offset);
        } else {
            // 淘汰依据keyProcess中的history，hashmap映射关系创建于ParseKey；两者异步，造成淘汰的值在hashmap里可能未创建
            continue;
        }

        if (offset < embHashMap.devVocabSize) {
            embHashMap.devOffset2Batch[offset] = -1;
            embHashMap.devOffset2KeyOld.emplace_back(offset, embHashMap.devOffset2Key[offset]);
            embHashMap.devOffset2Key[offset] = -1;
            embHashMap.evictDevPos.emplace_back(offset);
        } else {
            embHashMap.evictPos.emplace_back(offset - embHashMap.devVocabSize);
        }
    }

    LOG(INFO) << StringFormat(
        "ddr EvictDeleteEmb, emb: [%s], hostEvictSize: %d, devEvictSize: %d ",
        embName.c_str(), embHashMap.evictPos.size(), embHashMap.evictDevPos.size()
    );
    if (VLOG_IS_ON(GLOG_TRACE)) {
        VLOG(GLOG_TRACE) << StringFormat("hostHashMap, %s", MapToString(embHashMaps[embName].hostHashMap).c_str());
    }
}

// old version
/*
 * 从embHashMaps获取key对应的位置，并更新devOffset2Batch
 */

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
            embHashMap.lookUpVec.emplace_back(offset);
            embHashMap.devOffset2KeyOld.emplace_back(offset, static_cast<int>(embHashMap.devOffset2Key[offset]));
            embHashMap.devOffset2Key[offset] = key;
        } else {
            embHashMap.missingKeysHostPos.emplace_back(offset - embHashMap.devVocabSize);
            FindSwapPosOld(embName, key, offset, currentBatchId, keepBatchId);
        }
    }
    if (currentBatchId == 0) {
        LOG(INFO) << StringFormat("max offset %d", embHashMap.maxOffset);
    }
    if (VLOG_IS_ON(GLOG_TRACE)) {
        VLOG(GLOG_TRACE) << StringFormat("hostHashMap, %s", MapToString(embHashMaps[embName].hostHashMap).c_str());
    }
}


bool EmbHashMap::FindOffsetHelper(const emb_key_t& key, EmbHashMapInfo& embHashMap, int channelId, size_t& offset)

{
    const auto& iter = embHashMap.hostHashMap.find(key);
    if (iter != embHashMap.hostHashMap.end()) {
        offset = iter->second;
        VLOG(GLOG_TRACE) << StringFormat("devVocabSize, %d , offset , %d", embHashMap.devVocabSize, offset);
    } else if (embHashMap.evictDevPos.size() != 0 && channelId == TRAIN_CHANNEL_ID) { // 优先复用hbm表
        offset = embHashMap.evictDevPos.back();
        embHashMap.hostHashMap[key] = offset;
        VLOG(GLOG_TRACE) << StringFormat(
            "ddr mode, dev evictPos is not null, key [%d] reuse offset [%d], evictSize [%d]",
            key, offset, embHashMap.evictDevPos.size()
        );
        embHashMap.evictDevPos.pop_back();
    } else if (embHashMap.evictPos.size() != 0 && channelId == TRAIN_CHANNEL_ID) { // hbm不足，再复用ddr表
        offset = embHashMap.evictPos.back();
        embHashMap.hostHashMap[key] = offset;
        VLOG(GLOG_TRACE) << StringFormat(
            "ddr mode, host evictPos is not null, key [%d] reuse offset [%d], evictSize [%d]",
            key, offset, embHashMap.evictPos.size());
        embHashMap.evictPos.pop_back();
    } else {
        if (channelId == TRAIN_CHANNEL_ID) {
            embHashMap.hostHashMap[key] = embHashMap.maxOffset;
            offset = embHashMap.maxOffset;
            embHashMap.maxOffset++;
            if (embHashMap.maxOffset == embHashMap.devVocabSize) {
                LOG(INFO) << ("start using host vocab!");
            }
            if (embHashMap.maxOffset > embHashMap.hostVocabSize + embHashMap.devVocabSize) {
                LOG(ERROR) << StringFormat(
                    "hostVocabSize too small! dev:%d host:%d", embHashMap.devVocabSize, embHashMap.hostVocabSize);
                throw runtime_error("hostVocabSize too small");
            }
        } else {
            return false;
        }
    }
    return true;
}

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

            VLOG(GLOG_TRACE) << StringFormat("key will be used, %d , offset , %d", key, offset);
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
            LOG(ERROR) << "devVocabSize is too small";
            throw runtime_error("devVocabSize is too small");
        }
    }
    return newDevOffset;
}

/*
 * 利用devOffset2Batch上key最近使用的batchId，来选择需要淘汰的key，记录淘汰位置和device侧所需的keys
 */
bool EmbHashMap::FindSwapPosOld(const string& embName, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                                size_t keepBatchId)
{
    bool notFind = true;
    auto& embHashMap = embHashMaps.at(embName);
    while (notFind) {
        if (embHashMap.devOffset2Batch[embHashMap.currentUpdatePos] < static_cast<int>(keepBatchId)) {
            embHashMap.devOffset2Batch[embHashMap.currentUpdatePos] = static_cast<int>(currentBatchId);
            embHashMap.swapPos.emplace_back(embHashMap.currentUpdatePos);
            embHashMap.lookUpVec.emplace_back(embHashMap.currentUpdatePos);
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
            LOG(ERROR) << "devVocabSize is too small";
            throw runtime_error("devVocabSize is too small");
        }
    }
    return true;
}
#endif

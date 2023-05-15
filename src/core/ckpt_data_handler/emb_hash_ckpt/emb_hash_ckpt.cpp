/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-14
 */

#include "emb_hash_ckpt.h"
#include <climits>

using namespace std;
using namespace MxRec;

// remember to comment that the function will take over the control on the mem space

void EmbHashCkpt::SetProcessData(CkptData& processData)
{
    saveEmbHashMaps.clear();
    loadEmbHashMaps.clear();
    saveEmbHashMaps = std::move(processData.embHashMaps);
}

void EmbHashCkpt::GetProcessData(CkptData& processData)
{
    processData.embHashMaps = std::move(loadEmbHashMaps);
    saveEmbHashMaps.clear();
    loadEmbHashMaps.clear();
}

vector<CkptDataType> EmbHashCkpt::GetDataTypes()
{
    return saveDataTypes;
}

vector<string> EmbHashCkpt::GetDirNames()
{
    return fileDirNames;
}

vector<string> EmbHashCkpt::GetEmbNames()
{
    vector<string> embNames;
    for (const auto& item : saveEmbHashMaps) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData EmbHashCkpt::GetDataset(CkptDataType dataType, string embName)
{
    map<CkptDataType, function<void()>> dataTransMap { { CkptDataType::EMB_HASHMAP,
        [=] { SetEmbHashMapTrans(embName); } },
        { CkptDataType::DEV_OFFSET, [=] { SetDevOffsetTrans(embName); } },
        { CkptDataType::EMB_CURR_STAT, [=] { SetEmbCurrStatTrans(embName); } } };

    CleanTransfer();
    dataTransMap.at(dataType)();
    return move(transferData);
}

void EmbHashCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    map<CkptDataType, function<void()>> dataLoadMap { { CkptDataType::EMB_HASHMAP, [=] { SetEmbHashMap(embName); } },
        { CkptDataType::DEV_OFFSET, [=] { SetDevOffset(embName); } },
        { CkptDataType::EMB_CURR_STAT, [=] { SetEmbCurrStat(embName); } } };

    CleanTransfer();
    transferData = move(loadedData);
    dataLoadMap.at(dataType)();
}

void EmbHashCkpt::SetEmbHashMapTrans(string embName)
{
    auto& transArr = transferData.int64Arr;
    auto& attribute = transferData.attribute;
    auto embHashMapSize = saveEmbHashMaps.at(embName).hostHashMap.size();

    attribute.push_back(embHashMapSize);
    embHashMapSize = embHashMapSize * embHashElmtNum;

    attribute.push_back(embHashElmtNum);
    attribute.push_back(eightBytes);
    transferData.datasetSize = embHashMapSize * eightBytes;
    transferData.attributeSize = attribute.size() * eightBytes;
    transArr.reserve(embHashMapSize);
    for (const auto& it : saveEmbHashMaps.at(embName).hostHashMap) {
        transArr.push_back(it.first);
        transArr.push_back(static_cast<int64_t>(it.second));
    }
}

void EmbHashCkpt::SetDevOffsetTrans(string embName)
{
    const auto& devOffset2Batch = saveEmbHashMaps.at(embName).devOffset2Batch;
    const auto& devOffset2Key = saveEmbHashMaps.at(embName).devOffset2Key;
    auto& transArr = transferData.int64Arr;
    auto& attribute = transferData.attribute;
    auto embDevOffsetSize = devOffset2Batch.size();
    embDevOffsetSize += devOffset2Key.size();

    attribute.push_back(devOffset2Batch.size());
    attribute.push_back(devOffset2Key.size());
    attribute.push_back(eightBytes);
    transferData.datasetSize = embDevOffsetSize * eightBytes;
    transferData.attributeSize = attribute.size() * eightBytes;

    transArr.reserve(embDevOffsetSize);
    transArr.insert(transArr.end(), devOffset2Batch.begin(), devOffset2Batch.end());
    transArr.insert(transArr.end(), devOffset2Key.begin(), devOffset2Key.end());
}

void EmbHashCkpt::SetEmbCurrStatTrans(string embName)
{
    auto& transArr = transferData.int32Arr;
    auto& attribute = transferData.attribute;
    auto embDevOffsetSize = embCurrStatNum;

    attribute.push_back(embCurrStatNum);
    attribute.push_back(fourBytes);
    transferData.datasetSize = embCurrStatNum * fourBytes;
    transferData.attributeSize = attribute.size() * eightBytes;

    transArr.reserve(embDevOffsetSize);
    transArr.push_back(static_cast<int>(saveEmbHashMaps.at(embName).currentUpdatePos));
    transArr.push_back(static_cast<int>(saveEmbHashMaps.at(embName).hostVocabSize));
    transArr.push_back(static_cast<int>(saveEmbHashMaps.at(embName).devVocabSize));
}

void EmbHashCkpt::SetEmbHashMap(string embName)
{
    auto& hostHashMap = loadEmbHashMaps[embName].hostHashMap;
    const auto& transArr = transferData.int64Arr;
    for (size_t i = 0; i < transArr.size(); i += embHashElmtNum) {
        if (i + embHashElmtNum > transArr.size()) {
            // this is an error, need to log this
        }

        hostHashMap[transArr.at(i)] = static_cast<size_t>(transArr.at(i + 1));
    }
}

void EmbHashCkpt::SetDevOffset(string embName)
{
    const auto& transArr = transferData.int64Arr;
    const auto& attribute = transferData.attribute;
    auto& dev2Batch = loadEmbHashMaps[embName].devOffset2Batch;
    auto& dev2Key = loadEmbHashMaps[embName].devOffset2Key;

    dev2Batch.resize(attribute.at(attrbDev2BatchIdx));
    dev2Key.reserve(attribute.at(attrbDev2KeyIdx));

    fill(dev2Batch.begin(), dev2Batch.end(), -1);
    dev2Key.insert(dev2Key.begin(), transArr.begin() + attribute.at(attrbDev2BatchIdx), transArr.end());
}

void EmbHashCkpt::SetEmbCurrStat(string embName)
{
    auto& embCurrStat = loadEmbHashMaps[embName];
    const auto& transArr = transferData.int32Arr;

    embCurrStat.currentUpdatePos = static_cast<size_t>(transArr.at(currUpdataPosIdx));
    embCurrStat.hostVocabSize = static_cast<size_t>(transArr.at(hostVocabIdx));
    embCurrStat.devVocabSize = static_cast<size_t>(transArr.at(devVocabIdx));
}
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-08-17
 */

#include "key_freq_map_ckpt.h"

using namespace std;
using namespace MxRec;

void KeyFreqMapCkpt::SetProcessData(CkptData& processData)
{
    saveDDRKeyFreqMaps.clear();
    loadDDRKeyFreqMaps.clear();
    saveExcludeDDRKeyFreqMaps.clear();
    loadExcludeDDRKeyFreqMaps.clear();

    saveDDRKeyFreqMaps = std::move(processData.ddrKeyFreqMaps);
    saveExcludeDDRKeyFreqMaps = std::move(processData.excludeDDRKeyFreqMaps);
}

void KeyFreqMapCkpt::GetProcessData(CkptData& processData)
{
    processData.ddrKeyFreqMaps = std::move(loadDDRKeyFreqMaps);
    processData.excludeDDRKeyFreqMaps = std::move(loadExcludeDDRKeyFreqMaps);

    saveDDRKeyFreqMaps.clear();
    loadDDRKeyFreqMaps.clear();
    saveExcludeDDRKeyFreqMaps.clear();
    loadExcludeDDRKeyFreqMaps.clear();
}

vector<CkptDataType> KeyFreqMapCkpt::GetDataTypes()
{
    return saveDataTypes;
}

vector<string> KeyFreqMapCkpt::GetDirNames()
{
    return fileDirNames;
}

vector<string> KeyFreqMapCkpt::GetEmbNames()
{
    vector<string> embNames;
    for (const auto& item :saveDDRKeyFreqMaps) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData KeyFreqMapCkpt::GetDataset(CkptDataType dataType, string embName)
{
    map<CkptDataType, function<void()>> dataTransMap {
        { CkptDataType::DDR_FREQ_MAP, [=] { SetDDRFreqMapTrans(embName); } },
        { CkptDataType::EXCLUDE_FREQ_MAP, [=] { SetExcludeDDRFreqMapTrans(embName); } } };

    CleanTransfer();
    dataTransMap.at(dataType)();
    return move(transferData);
}

void KeyFreqMapCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    map<CkptDataType, function<void()>> dataLoadMap {
        { CkptDataType::DDR_FREQ_MAP, [=] { SetDDRFreqMaps(embName); } },
        { CkptDataType::EXCLUDE_FREQ_MAP, [=] { SetExcludeDDRFreqMaps(embName); } } };

    CleanTransfer();
    transferData = move(loadedData);
    dataLoadMap.at(dataType)();
}

//  set DDRFreqMapTrans for save
void KeyFreqMapCkpt::SetDDRFreqMapTrans(string embName)
{
    auto& transArr = transferData.int64Arr;
    auto& attribute = transferData.attribute;
    auto ddrFreqMapSize = saveDDRKeyFreqMaps.at(embName).size();

    attribute.push_back(ddrFreqMapSize);
    ddrFreqMapSize = ddrFreqMapSize * freqMapElmtNum;

    attribute.push_back(freqMapElmtNum);
    attribute.push_back(eightBytes);

    transferData.datasetSize = ddrFreqMapSize * eightBytes;
    transferData.attributeSize = attribute.size() * eightBytes;

    transArr.reserve(ddrFreqMapSize);
    for (const auto& it : saveDDRKeyFreqMaps.at(embName)) {
        transArr.push_back(it.first);
        transArr.push_back(static_cast<int64_t>(it.second));
    }
}

void KeyFreqMapCkpt::SetExcludeDDRFreqMapTrans(string embName)
{
    auto& transArr = transferData.int64Arr;
    auto& attribute = transferData.attribute;
    auto excludeDDRFreqMapSize = saveExcludeDDRKeyFreqMaps.at(embName).size();

    attribute.push_back(excludeDDRFreqMapSize);
    excludeDDRFreqMapSize = excludeDDRFreqMapSize * freqMapElmtNum;

    attribute.push_back(freqMapElmtNum);
    attribute.push_back(eightBytes);

    transferData.datasetSize = excludeDDRFreqMapSize * eightBytes;
    transferData.attributeSize = attribute.size() * eightBytes;

    transArr.reserve(excludeDDRFreqMapSize);
    for (const auto& it : saveExcludeDDRKeyFreqMaps.at(embName)) {
        transArr.push_back(it.first);
        transArr.push_back(static_cast<int64_t>(it.second));
    }
}

void KeyFreqMapCkpt::SetDDRFreqMaps(string embName)
{
    auto& ddrKeyFreqMap = loadDDRKeyFreqMaps[embName];
    const auto& transArr = transferData.int64Arr;
    for (size_t i = 0; i < transArr.size(); i += freqMapElmtNum) {
        ddrKeyFreqMap[transArr.at(i)] = static_cast<size_t>(transArr.at(i + 1));
    }
}

void KeyFreqMapCkpt::SetExcludeDDRFreqMaps(string embName)
{
    auto& excludeDDRKeyFreqMap = loadExcludeDDRKeyFreqMaps[embName];
    const auto& transArr = transferData.int64Arr;
    for (size_t i = 0; i < transArr.size(); i += freqMapElmtNum) {
        excludeDDRKeyFreqMap[transArr.at(i)] = static_cast<size_t>(transArr.at(i + 1));
    }
}

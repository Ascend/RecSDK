/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-17
 */
#include "nddr_feat_map_ckpt.h"


using namespace std;
using namespace MxRec;

void NddrFeatMapCkpt::SetProcessData(CkptData& processData)
{
    saveKeyOffsetMap.clear();
    loadKeyOffsetMap.clear();
    saveKeyOffsetMap = std::move(processData.keyOffsetMap);
}

void NddrFeatMapCkpt::GetProcessData(CkptData& processData)
{
    processData.keyOffsetMap = std::move(loadKeyOffsetMap);
    saveKeyOffsetMap.clear();
    loadKeyOffsetMap.clear();
}

vector<CkptDataType> NddrFeatMapCkpt::GetDataTypes()
{
    return saveDataTypes;
}

vector<string> NddrFeatMapCkpt::GetDirNames()
{
    return fileDirNames;
}

vector<string> NddrFeatMapCkpt::GetEmbNames()
{
    vector<string> embNames;
    for (const auto& item : saveKeyOffsetMap) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData NddrFeatMapCkpt::GetDataset(CkptDataType dataType, string embName)
{
    CleanTransfer();

    auto& transArr = transferData.int64Arr;
    auto& attribute = transferData.attribute;
    auto embHashMapSize = saveKeyOffsetMap.at(embName).size();

    attribute.push_back(embHashMapSize);
    embHashMapSize = embHashMapSize * embHashElmtNum;

    attribute.push_back(embHashElmtNum);
    attribute.push_back(eightBytes);
    transferData.datasetSize = embHashMapSize * eightBytes;
    transferData.attributeSize = attribute.size() * eightBytes;

    transArr.reserve(embHashMapSize);
    for (const auto& it : saveKeyOffsetMap.at(embName)) {
        transArr.push_back(it.first);
        transArr.push_back(it.second);
    }
    LOG_INFO("CkptDataType::EMB_INFO:{}, dataType:{}", CkptDataType::EMB_INFO, dataType);
    return move(transferData);
}

void NddrFeatMapCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    CleanTransfer();
    transferData = move(loadedData);

    auto& hostHashMap = loadKeyOffsetMap[embName];
    const auto& transArr = transferData.int64Arr;

    for (size_t i { 0 }; i < transArr.size(); i += embHashElmtNum) {
        if (i + embHashElmtNum > transArr.size()) {
            // this is an error, need to log this
        }
        int64_t key { transArr.at(i) };
        hostHashMap[key] = transArr.at(i + 1);
    }
    LOG_INFO("dataType:{}", dataType);
}

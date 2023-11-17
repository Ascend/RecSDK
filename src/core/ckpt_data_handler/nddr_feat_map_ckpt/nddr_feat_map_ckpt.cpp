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
    // 传递 offsetMap的引用
    offsetMapPtr = processData.offsetMapPtr;
}

void NddrFeatMapCkpt::GetProcessData(CkptData& processData)
{
    processData.keyOffsetMap = std::move(loadKeyOffsetMap);
    processData.maxOffset = std::move(loadMaxOffset);
    saveKeyOffsetMap.clear();
    loadKeyOffsetMap.clear();
    loadMaxOffset.clear();
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
    for (const auto& item : as_const(saveKeyOffsetMap)) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData NddrFeatMapCkpt::GetDataset(CkptDataType dataType, string embName)
{
    CleanTransfer();

    auto& transArr = transferData.int64Arr;
    auto& addressArr = transferData.addressArr;
    auto& attribute = transferData.attribute;
    auto embHashMapSize = saveKeyOffsetMap.at(embName).size();

    attribute.push_back(embHashMapSize);
    embHashMapSize = embHashMapSize * embHashElmtNum;

    attribute.push_back(embHashElmtNum);
    attribute.push_back(eightBytes);
    transferData.datasetSize = embHashMapSize * eightBytes;
    transferData.attributeSize = attribute.size() * eightBytes;

    transArr.reserve(embHashMapSize);
    (*offsetMapPtr)[embName].clear();
    LOG_ERROR("build offset map : first key offset {}", saveKeyOffsetMap[embName][0]);
    for (const auto& it : saveKeyOffsetMap.at(embName)) {
        transArr.push_back(it.first);
        transArr.push_back(it.second);
        (*offsetMapPtr)[embName].push_back(it.second);
    }
    LOG_INFO("CkptDataType::EMB_INFO:{}, dataType:{}", CkptDataType::NDDR_FEATMAP, dataType);
    return move(transferData);
}

void NddrFeatMapCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    CleanTransfer();
    transferData = move(loadedData);
    auto& maxOffset = loadMaxOffset[embName];
    auto& hostHashMap = loadKeyOffsetMap[embName];
    const auto& transArr = transferData.int64Arr;
    const auto& addressArr = transferData.addressArr;
    int64_t offset { 0 };
    for (size_t i { 0 }; i < transArr.size(); i += embHashElmtNum) {
        if (i + embHashElmtNum > transArr.size()) {
            // this is an error, need to log this
        }
        int64_t key { transArr.at(i) };
        if (addressArr.size() == 0) {
            // no dynamic expansion
            hostHashMap[key] = offset;
        } else{
            //  dynamic expansion
            hostHashMap[key] = addressArr.at(i);
        }
        offset++;
    }
    maxOffset = offset;
    LOG_INFO("dataType:{}", dataType);
}

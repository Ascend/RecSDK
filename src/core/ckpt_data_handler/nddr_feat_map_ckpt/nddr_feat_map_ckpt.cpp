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
    for (const auto& it : saveKeyOffsetMap.at(embName)) {
        transArr.push_back(it.first);
        (*offsetMapPtr)[embName].push_back(it.second);
        addressArr.push_back(it.second);
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
        } else {
            //  dynamic expansion
            hostHashMap[key] = addressArr.at(i);
        }
        offset++;
    }
    maxOffset = offset;
    LOG_INFO("dataType:{}", dataType);
}

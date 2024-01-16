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
#include "key_count_map_ckpt.h"

using namespace std;
using namespace MxRec;

void KeyCountMapCkpt::SetProcessData(CkptData& processData)
{
    saveKeyCountMap.clear();
    loadKeyCountMap.clear();
    saveKeyCountMap = std::move(processData.keyCountMap);
}

void KeyCountMapCkpt::GetProcessData(CkptData& processData)
{
    processData.keyCountMap = std::move(loadKeyCountMap);
    saveKeyCountMap.clear();
    loadKeyCountMap.clear();
}

vector<CkptDataType> KeyCountMapCkpt::GetDataTypes()
{
    return saveDataTypes;
}

vector<string> KeyCountMapCkpt::GetDirNames()
{
    return fileDirNames;
}

vector<string> KeyCountMapCkpt::GetEmbNames()
{
    vector<string> embNames;
    for (const auto& item : as_const(saveKeyCountMap)) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData KeyCountMapCkpt::GetDataset(CkptDataType dataType, string embName)
{
    CleanTransfer();

    auto& transArr = transferData.int64Arr;
    auto& attribute = transferData.attribute;
    auto embHashMapSize = saveKeyCountMap.at(embName).size();

    attribute.push_back(embHashMapSize);
    embHashMapSize = embHashMapSize * embHashElmtNum;

    attribute.push_back(embHashElmtNum);
    attribute.push_back(eightBytes);
    transferData.datasetSize = embHashMapSize * eightBytes;
    transferData.attributeSize = attribute.size() * eightBytes;

    transArr.reserve(embHashMapSize);
    for (const auto& it : saveKeyCountMap.at(embName)) {
        transArr.push_back(it.first);
        transArr.push_back(it.second);
    }
    LOG_INFO("CkptDataType::EMB_INFO:{}, dataType:{}", CkptDataType::EMB_INFO, dataType);
    return move(transferData);
}

void KeyCountMapCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    CleanTransfer();
    transferData = move(loadedData);

    auto& singleKeyCountMap = loadKeyCountMap[embName];
    const auto& transArr = transferData.int64Arr;

    for (size_t i = 0; i < transArr.size(); i += embHashElmtNum) {
        if (i + embHashElmtNum > transArr.size()) {
            // this is an error, need to log this
        }
        int64_t key { transArr.at(i) };
        singleKeyCountMap[key] = transArr.at(i + 1);
    }
    LOG_INFO("dataType:{}", dataType);
}
